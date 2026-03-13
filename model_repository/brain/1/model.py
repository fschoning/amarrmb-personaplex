"""
brain/1/model.py — Triton Python backend for LLM Brain Reasoner

Runs Qwen 2.5 27B AWQ via vLLM inside the Triton container.
enforce_eager=True because CUDA graph capture OOMs on the memory-
constrained GB10 (96GB shared with 2× PersonaPlex).

Set BRAIN_MODEL env var to point at any compatible model directory.
"""

import os
import time
import json
import re
import numpy as np
import triton_python_backend_utils as pb_utils

# ── Config from environment ──────────────────────────────────────────────────
_ENGINE_DIR  = os.environ.get("BRAIN_ENGINE", "/mnt/models/brain-engine")
_MODEL_DIR   = os.environ.get("BRAIN_MODEL",  "/mnt/models/qwen35-27b-awq")
_HF_MODEL_ID = os.environ.get("BRAIN_HF_ID",  "Qwen/Qwen2.5-27B-Instruct-AWQ")
_DEVICE      = "cuda"


def _detect_quantization(model_dir: str, logger) -> str | None:
    """Detect quantization type. Returns 'awq', 'gptq', 'SKIP_FP4', or None."""
    for fname in ("hf_quant_config.json", "quantize_config.json"):
        p = os.path.join(model_dir, fname)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            cfg = json.load(f)
        algo = cfg.get("quantization", {}).get("quant_algo", "")
        if not algo:
            algo = cfg.get("quant_type", "") or cfg.get("quant_method", "")
        algo = algo.upper()
        if "FP4" in algo or "NVFP4" in algo:
            logger.log_info(
                "brain: NVFP4 NOT supported on GB10 (sm_121). "
                "Set BRAIN_MODEL to an AWQ model."
            )
            return "SKIP_FP4"
        if "AWQ" in algo:
            logger.log_info("brain: detected AWQ quantization")
            return "awq"
        if "GPTQ" in algo:
            logger.log_info("brain: detected GPTQ quantization")
            return "gptq"
    return None


def _try_trtllm(engine_dir: str, logger):
    if not os.path.isdir(engine_dir):
        logger.log_info(f"brain: TRT-LLM engine not found at {engine_dir}")
        return None
    try:
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        runner = ModelRunner.from_dir(engine_dir=engine_dir, rank=0)
        logger.log_info(f"brain: loaded TRT-LLM engine from {engine_dir}")
        return runner
    except Exception as e:
        logger.log_info(f"brain: TRT-LLM failed ({e})")
        return None


def _try_vllm(model_dir: str, logger):
    """Load via vLLM with enforce_eager (CUDA graphs OOM on GB10)."""
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        quant = _detect_quantization(model_dir, logger)
        if quant == "SKIP_FP4":
            return None

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        llm = LLM(
            model=model_dir,
            quantization=quant,
            dtype="bfloat16",
            gpu_memory_utilization=0.38,
            max_model_len=1024,
            max_num_seqs=1,             # single request at a time
            trust_remote_code=True,
            enforce_eager=True,         # CUDA graph capture OOMs on GB10
            enable_prefix_caching=True, # reuse KV cache across chunked generate() calls
        )
        logger.log_info(f"brain: loaded via vLLM from {model_dir} (quant={quant})")
        return llm, tokenizer, SamplingParams
    except Exception as e:
        logger.log_info(f"brain: vLLM failed ({e})")
        return None


def _try_transformers(model_dir: str, hf_id: str, logger):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = _detect_quantization(model_dir, logger)
        if quant == "SKIP_FP4":
            return None

        src = model_dir if os.path.isdir(model_dir) else hf_id
        logger.log_info(f"brain: loading via transformers from {src} ...")
        tokenizer = AutoTokenizer.from_pretrained(src)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            src, quantization_config=bnb_cfg,
            device_map="cuda", trust_remote_code=True,
        )
        model.eval()
        logger.log_info("brain: loaded via transformers (4-bit BnB)")
        return model, tokenizer
    except Exception as e:
        logger.log_info(f"brain: transformers failed ({e})")
        return None


class TritonPythonModel:

    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info(f"brain: initializing from {_MODEL_DIR}...")

        self._backend     = None
        self._trtllm      = None
        self._vllm        = None
        self._vllm_params = None
        self._hf_model    = None
        self._hf_tok      = None

        runner = _try_trtllm(_ENGINE_DIR, self.logger)
        if runner is not None:
            self._trtllm  = runner
            self._backend = "trtllm"

        if self._backend is None:
            result = _try_vllm(_MODEL_DIR, self.logger)
            if result is not None:
                self._vllm, self._hf_tok, self._vllm_params = result
                self._backend = "vllm"

        if self._backend is None:
            result = _try_transformers(_MODEL_DIR, _HF_MODEL_ID, self.logger)
            if result is not None:
                self._hf_model, self._hf_tok = result
                self._backend = "transformers"

        if self._backend is None:
            raise RuntimeError(
                f"brain: all backends failed for model '{_MODEL_DIR}'. "
                "Set BRAIN_MODEL to a compatible model (AWQ recommended)."
            )

        self.logger.log_info(f"brain: ready (backend={self._backend}, model={_MODEL_DIR})")

        # --- Warmup: must exercise FLA linear attention code paths ---
        # The first vLLM generate triggers triton kernel JIT for Qwen 3.5's
        # hybrid attention layers. Use a realistic-length prompt to cover
        # both standard attention and FLA code paths.
        if self._backend == "vllm":
            self.logger.log_info("brain: warming up vLLM (JIT compiling FLA kernels)...")
            t0 = time.monotonic()
            try:
                warmup_prompt = self._format_prompt(
                    "This is a warmup prompt to trigger kernel compilation. "
                    "The brain model uses hybrid attention with both standard "
                    "multi-head attention and flash linear attention layers. "
                    "Each unique sequence length may trigger JIT compilation "
                    "of triton kernels for the FLA layers."
                )
                params = self._vllm_params(temperature=0.7, top_p=0.9, max_tokens=32)
                self._vllm.generate([warmup_prompt], params)
                self.logger.log_info(f"brain: warmup done in {time.monotonic() - t0:.1f}s")
            except Exception as e:
                self.logger.log_info(f"brain: warmup failed ({e}), first request will be slow")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._process_one(request))
            except Exception as e:
                self.logger.log_error(f"brain error: {e}")
                import traceback; traceback.print_exc()
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))
        return responses

    def _process_one(self, request):
        prompt_tensor  = pb_utils.get_input_tensor_by_name(request, "PROMPT")
        max_tok_tensor = pb_utils.get_input_tensor_by_name(request, "MAX_TOKENS")

        prompt_bytes = prompt_tensor.as_numpy().flat[0]
        prompt = prompt_bytes.decode("utf-8") if isinstance(prompt_bytes, bytes) else str(prompt_bytes)
        max_tokens = min(max(int(max_tok_tensor.as_numpy().flat[0]), 32), 1024)

        self.logger.log_info(
            f"brain: generating ({self._backend}, max_tokens={max_tokens}, "
            f"prompt_len={len(prompt)})"
        )
        t0 = time.monotonic()

        if self._backend == "trtllm":
            response_text = self._generate_trtllm(prompt, max_tokens)
        elif self._backend == "vllm":
            response_text = self._generate_vllm(prompt, max_tokens)
        else:
            response_text = self._generate_hf(prompt, max_tokens)

        elapsed = time.monotonic() - t0
        n_words = len(response_text.split())
        self.logger.log_info(
            f"brain: done in {elapsed:.2f}s, {n_words} words, "
            f"~{n_words / max(elapsed, 0.01):.1f} words/s"
        )

        out_arr = np.array([[response_text.encode("utf-8")]], dtype=object)
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("RESPONSE", out_arr),
        ])

    def _generate_trtllm(self, prompt: str, max_tokens: int) -> str:
        import torch
        input_ids = self._trtllm.tokenizer.encode(
            self._format_prompt(prompt), return_tensors="pt"
        ).to(_DEVICE)
        with torch.no_grad():
            output = self._trtllm.generate(
                batch_input_ids=[input_ids[0]],
                max_new_tokens=max_tokens,
                temperature=0.7, top_p=0.9,
            )
        raw = self._trtllm.tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        return self._strip_thinking(raw)

    def _generate_vllm(self, prompt: str, max_tokens: int) -> str:
        """Chunked generation: generate `chunk_size` tokens at a time with
        pauses between chunks.  This yields GPU time so PersonaPlex frames
        can process with exclusive GPU access (via Triton rate limiter mutex).

        With prefix caching ON, vLLM reuses the KV cache for the prompt
        prefix — each chunk only prefills the new tokens from the previous
        chunk, keeping overhead minimal.
        """
        chunk_size = 8  # tokens per generate() call
        pause_s = 0.20  # seconds to yield GPU between chunks
        formatted = self._format_prompt(prompt)
        full_output = ""
        total_output_toks = 0
        t_start = time.monotonic()

        for chunk_idx in range(0, max_tokens, chunk_size):
            remaining = min(chunk_size, max_tokens - total_output_toks)
            if remaining <= 0:
                break

            params = self._vllm_params(
                temperature=0.7, top_p=0.9, max_tokens=remaining
            )

            # Extend prompt with previous output for prefix cache hit
            current_prompt = formatted + full_output
            outputs = self._vllm.generate([current_prompt], params)
            result = outputs[0]

            chunk_text = result.outputs[0].text
            n_chunk_toks = len(result.outputs[0].token_ids) if result.outputs[0].token_ids else 0
            total_output_toks += n_chunk_toks
            full_output += chunk_text

            # Check for natural stop (EOS)
            if result.outputs[0].finish_reason == "stop":
                break

            # YIELD GPU: sleep so PP can process frames uncontested.
            # With gpu_compute mutex, this releases the resource and Triton
            # schedules pending PP frames with exclusive GPU access.
            time.sleep(pause_s)

        elapsed = time.monotonic() - t_start
        n_prompt = len(result.prompt_token_ids) if result.prompt_token_ids else 0
        n_chunks = (total_output_toks + chunk_size - 1) // chunk_size
        decode_tps = total_output_toks / max(elapsed, 0.01)
        self.logger.log_info(
            f"brain: vllm chunked — {n_chunks} chunks of {chunk_size}, "
            f"prompt={n_prompt} toks, output={total_output_toks} toks, "
            f"decode={decode_tps:.1f} tok/s (with {pause_s*1000:.0f}ms pauses)"
        )

        return self._strip_thinking(full_output.strip())

    def _generate_hf(self, prompt: str, max_tokens: int) -> str:
        import torch
        formatted = self._format_prompt(prompt)
        inputs = self._hf_tok(formatted, return_tensors="pt").to(_DEVICE)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self._hf_model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
                pad_token_id=self._hf_tok.eos_token_id,
            )
        raw = self._hf_tok.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        return self._strip_thinking(raw)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> chain-of-thought blocks (Qwen3 reasoning)."""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
        return text.strip()

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using tokenizer's chat template with thinking disabled."""
        prompt = prompt.strip()
        if "<|im_start|>" in prompt or "[INST]" in prompt:
            return prompt  # already formatted

        messages = [
            {"role": "system", "content": "You are a concise AI assistant. Respond directly."},
            {"role": "user", "content": prompt},
        ]

        # Use tokenizer's native chat template if available (handles thinking mode)
        if self._hf_tok is not None and hasattr(self._hf_tok, 'apply_chat_template'):
            try:
                return self._hf_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Qwen 3.5 native thinking control
                )
            except TypeError:
                pass  # tokenizer doesn't support enable_thinking kwarg

        # Fallback: manual Qwen chat format
        return (
            "<|im_start|>system\n"
            "You are a concise AI assistant. Respond directly. /no_think\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def finalize(self):
        self._trtllm   = None
        self._vllm     = None
        self._hf_model = None
        self.logger.log_info("brain: finalized.")
