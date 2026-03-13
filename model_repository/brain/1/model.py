"""
brain/1/model.py — Triton Python backend for LLM Brain Reasoner

SmolLM3-3B via TRT-LLM (NVFP4) or vLLM fallback.
3B model is ~9× lighter than Qwen 27B — sub-second inference with
TRT-LLM enables always-on brain with zero GPU contention.

Set BRAIN_ENGINE to the TRT-LLM engine dir (fastest path).
Set BRAIN_MODEL to the HF weights dir (vLLM/transformers fallback).
"""

import os
import time
import json
import re
import numpy as np
import triton_python_backend_utils as pb_utils

# ── Config from environment ──────────────────────────────────────────────────
_ENGINE_DIR  = os.environ.get("BRAIN_ENGINE", "/mnt/models/smollm3-3b-nvfp4-engine")
_MODEL_DIR   = os.environ.get("BRAIN_MODEL",  "/mnt/models/smollm3-3b")
_HF_MODEL_ID = os.environ.get("BRAIN_HF_ID",  "HuggingFaceTB/SmolLM3-3B")
_DEVICE      = "cuda"


# ── TRT-LLM C++ engine loader ────────────────────────────────────────────────
def _try_trtllm(engine_dir: str, model_dir: str, logger):
    """Load SmolLM3 via TRT-LLM C++ backend — fastest path."""
    if not os.path.isdir(engine_dir):
        logger.log_info(f"brain: TRT-LLM engine not found at {engine_dir}")
        return None

    # Check engine file exists
    engine_file = os.path.join(engine_dir, "rank0.engine")
    if not os.path.isfile(engine_file):
        logger.log_info(f"brain: no rank0.engine in {engine_dir}")
        return None

    try:
        # Use the C++ backend API (avoids PyTorch fallback crashes)
        from tensorrt_llm._tensorrt_engine import LLM as TrtLLM
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        llm = TrtLLM(model=engine_dir)
        logger.log_info(f"brain: loaded TRT-LLM C++ engine from {engine_dir}")
        return llm, tokenizer
    except ImportError:
        logger.log_info("brain: tensorrt_llm._tensorrt_engine not available, trying ModelRunner...")
    except Exception as e:
        logger.log_info(f"brain: TRT-LLM C++ engine failed ({e}), trying ModelRunner...")

    # Fallback: older ModelRunner API
    try:
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        runner = ModelRunner.from_dir(engine_dir=engine_dir, rank=0)
        logger.log_info(f"brain: loaded TRT-LLM ModelRunner from {engine_dir}")
        return runner, tokenizer
    except Exception as e:
        logger.log_info(f"brain: TRT-LLM ModelRunner failed ({e})")
        return None


# ── vLLM loader ──────────────────────────────────────────────────────────────
def _try_vllm(model_dir: str, logger):
    """Load SmolLM3 via vLLM — 3B model needs minimal VRAM."""
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        llm = LLM(
            model=model_dir,
            dtype="bfloat16",
            gpu_memory_utilization=0.30,    # 3B bf16 ≈ 6GB + KV cache
            max_model_len=1024,
            max_num_seqs=1,
            trust_remote_code=True,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        logger.log_info(f"brain: loaded via vLLM from {model_dir}")
        return llm, tokenizer, SamplingParams
    except Exception as e:
        logger.log_info(f"brain: vLLM failed ({e})")
        return None


# ── HF Transformers loader ───────────────────────────────────────────────────
def _try_transformers(model_dir: str, hf_id: str, logger):
    """Load SmolLM3 via HF transformers — simplest fallback."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        src = model_dir if os.path.isdir(model_dir) else hf_id
        logger.log_info(f"brain: loading via transformers from {src} ...")
        tokenizer = AutoTokenizer.from_pretrained(src)
        model = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True,
        )
        model.eval()
        logger.log_info("brain: loaded via transformers (bf16)")
        return model, tokenizer
    except Exception as e:
        logger.log_info(f"brain: transformers failed ({e})")
        return None


class TritonPythonModel:

    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info(f"brain: initializing SmolLM3...")
        self.logger.log_info(f"brain:   engine_dir = {_ENGINE_DIR}")
        self.logger.log_info(f"brain:   model_dir  = {_MODEL_DIR}")

        self._backend     = None
        self._trtllm      = None
        self._trtllm_tok  = None
        self._vllm        = None
        self._vllm_params = None
        self._hf_model    = None
        self._hf_tok      = None

        # Priority: TRT-LLM (fastest) → vLLM → transformers
        result = _try_trtllm(_ENGINE_DIR, _MODEL_DIR, self.logger)
        if result is not None:
            self._trtllm, self._trtllm_tok = result
            self._hf_tok = self._trtllm_tok   # for chat template
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
                f"brain: all backends failed. "
                f"Tried: TRT-LLM ({_ENGINE_DIR}), vLLM ({_MODEL_DIR}), "
                f"transformers ({_HF_MODEL_ID})."
            )

        self.logger.log_info(f"brain: ready (backend={self._backend})")

        # Warmup: first inference is always slower (JIT, cache warmup)
        self.logger.log_info("brain: warming up...")
        t0 = time.monotonic()
        try:
            warmup = self._generate("Say hello.", 8)
            self.logger.log_info(
                f"brain: warmup done in {time.monotonic() - t0:.1f}s → '{warmup[:50]}'"
            )
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
        response_text = self._generate(prompt, max_tokens)
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

    # ── Generate dispatch ────────────────────────────────────────────────────
    def _generate(self, prompt: str, max_tokens: int) -> str:
        if self._backend == "trtllm":
            return self._strip_thinking(self._generate_trtllm(prompt, max_tokens))
        elif self._backend == "vllm":
            return self._strip_thinking(self._generate_vllm(prompt, max_tokens))
        else:
            return self._strip_thinking(self._generate_hf(prompt, max_tokens))

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> chain-of-thought blocks."""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
        return text.strip()

    def _generate_trtllm(self, prompt: str, max_tokens: int) -> str:
        formatted = self._format_prompt(prompt)
        t0 = time.monotonic()

        # Try C++ LLM API first (newer tensorrt_llm)
        if hasattr(self._trtllm, 'generate_async') or hasattr(self._trtllm, 'generate'):
            try:
                # C++ LLM.generate() accepts text directly
                outputs = self._trtllm.generate([formatted], sampling_params=dict(
                    max_new_tokens=max_tokens, temperature=0.7, top_p=0.9,
                ))
                elapsed = time.monotonic() - t0
                # Handle different output formats
                if hasattr(outputs[0], 'text'):
                    raw = outputs[0].text
                elif hasattr(outputs[0], 'outputs'):
                    raw = outputs[0].outputs[0].text
                else:
                    raw = str(outputs[0])

                self.logger.log_info(f"brain: trtllm generate in {elapsed:.2f}s")
                return raw.strip()
            except Exception as e:
                self.logger.log_info(f"brain: trtllm generate API failed ({e}), trying tokenizer path")

        # Fallback: ModelRunner with manual tokenization
        import torch
        input_ids = self._trtllm_tok.encode(formatted, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            output = self._trtllm.generate(
                batch_input_ids=[input_ids[0]],
                max_new_tokens=max_tokens,
                temperature=0.7, top_p=0.9,
            )
        elapsed = time.monotonic() - t0
        raw = self._trtllm_tok.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        self.logger.log_info(f"brain: trtllm ModelRunner in {elapsed:.2f}s")
        return raw

    def _generate_vllm(self, prompt: str, max_tokens: int) -> str:
        params = self._vllm_params(temperature=0.7, top_p=0.9, max_tokens=max_tokens)
        t0 = time.monotonic()
        outputs = self._vllm.generate([self._format_prompt(prompt)], params)
        elapsed = time.monotonic() - t0
        result = outputs[0]

        out_text = result.outputs[0].text.strip()
        n_prompt = len(result.prompt_token_ids) if result.prompt_token_ids else 0
        n_output = len(result.outputs[0].token_ids) if result.outputs[0].token_ids else 0
        decode_tps = n_output / max(elapsed, 0.01)
        self.logger.log_info(
            f"brain: vllm — prompt={n_prompt} toks, output={n_output} toks, "
            f"elapsed={elapsed:.2f}s, decode={decode_tps:.1f} tok/s"
        )
        return out_text

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
        return raw

    # ── Chat template ────────────────────────────────────────────────────────
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using tokenizer's chat template."""
        prompt = prompt.strip()

        # Already formatted — pass through
        if "<|im_start|>" in prompt or "[INST]" in prompt or "<|system|>" in prompt:
            return prompt

        messages = [
            {"role": "system", "content": "You are a concise AI assistant. Respond directly."},
            {"role": "user", "content": prompt},
        ]

        # Use tokenizer's native chat template
        if self._hf_tok is not None and hasattr(self._hf_tok, 'apply_chat_template'):
            try:
                return self._hf_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback: generic ChatML format (works for most models)
        return (
            "<|im_start|>system\n"
            "You are a concise AI assistant. Respond directly.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def finalize(self):
        self._trtllm   = None
        self._vllm     = None
        self._hf_model = None
        self.logger.log_info("brain: finalized.")
