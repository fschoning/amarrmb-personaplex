"""
mixtral_brain/1/model.py — Triton Python backend for LLM Brain Reasoner

Initialization strategy (in order of preference):
  1. TensorRT-LLM engine  — fastest, requires prior trtllm-build step
  2. vLLM                 — fast, Blackwell-compatible (AWQ/GPTQ/standard)
  3. HF Transformers      — slowest fallback (requires bitsandbytes for 4-bit)

Key: NVFP4 models are NOT supported on GB10 (sm_121) due to sm120 cutlass
kernel incompatibility. Use AWQ or standard models instead.
Set MIXTRAL_MODEL env var to point at any compatible model directory.

Recommended models (already on /mnt/models):
  /mnt/models/qwen35-27b-awq   — AWQ 4-bit, ~14GB, works well on GB10
"""

import os
import time
import json
import numpy as np
import triton_python_backend_utils as pb_utils

# ── Config from environment ──────────────────────────────────────────────────
_ENGINE_DIR  = os.environ.get("MIXTRAL_ENGINE", "/mnt/models/mixtral-engine")
_MODEL_DIR   = os.environ.get("MIXTRAL_MODEL",  "/mnt/models/qwen35-27b-awq")
_HF_MODEL_ID = os.environ.get("MIXTRAL_HF_ID",  "Qwen/Qwen2.5-27B-Instruct-AWQ")
_DEVICE      = "cuda"
_BACKEND     = None   # set during initialize()


def _detect_quantization(model_dir: str, logger) -> str | None:
    """Detect quantization type. Returns 'awq', 'gptq', 'SKIP_FP4', or None."""
    for fname in ("hf_quant_config.json", "quantize_config.json"):
        p = os.path.join(model_dir, fname)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            cfg = json.load(f)
        # NVFP4 format (nvidia modelopt)
        algo = cfg.get("quantization", {}).get("quant_algo", "")
        # AWQ format
        if not algo:
            algo = cfg.get("quant_type", "") or cfg.get("quant_method", "")
        algo = algo.upper()
        if "FP4" in algo or "NVFP4" in algo:
            logger.log_info(
                "mixtral_brain: NVFP4 model detected — NOT supported on GB10 (sm_121). "
                "Set MIXTRAL_MODEL to an AWQ or standard model."
            )
            return "SKIP_FP4"
        if "AWQ" in algo:
            logger.log_info(f"mixtral_brain: detected AWQ quantization")
            return "awq"
        if "GPTQ" in algo:
            logger.log_info(f"mixtral_brain: detected GPTQ quantization")
            return "gptq"
    return None  # no quantization / standard bf16


def _try_trtllm(engine_dir: str, logger):
    """Attempt to load TRT-LLM engine. Returns runner or None."""
    if not os.path.isdir(engine_dir):
        logger.log_info(f"mixtral_brain: TRT-LLM engine not found at {engine_dir}")
        return None
    try:
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        runner = ModelRunner.from_dir(engine_dir=engine_dir, rank=0)
        logger.log_info(f"mixtral_brain: loaded TRT-LLM engine from {engine_dir}")
        return runner
    except Exception as e:
        logger.log_info(f"mixtral_brain: TRT-LLM failed ({e}), trying next backend")
        return None


def _try_vllm(model_dir: str, logger):
    """Attempt to load via vLLM. Returns (llm, tokenizer, SamplingParams) or None."""
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        quant = _detect_quantization(model_dir, logger)
        if quant == "SKIP_FP4":
            return None  # skip FP4 — won't work on sm_121

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        llm = LLM(
            model=model_dir,
            quantization=quant,             # "awq"/"gptq"/None
            dtype="bfloat16",
            gpu_memory_utilization=0.38,    # ~36GB of 96GB total; ~40GB free after 2× PersonaPlex
            max_model_len=1024,             # brain prompts are short; limits KV cache size
            trust_remote_code=True,
            enforce_eager=True,             # skip CUDA graph capture
            enable_prefix_caching=False,
        )
        logger.log_info(f"mixtral_brain: loaded via vLLM from {model_dir} (quant={quant})")
        return llm, tokenizer, SamplingParams
    except Exception as e:
        logger.log_info(f"mixtral_brain: vLLM failed ({e}), trying next backend")
        return None


def _try_transformers(model_dir: str, hf_id: str, logger):
    """Load via HF Transformers (slowest fallback). Returns (model, tokenizer) or None."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = _detect_quantization(model_dir, logger)
        if quant == "SKIP_FP4":
            return None

        src = model_dir if os.path.isdir(model_dir) else hf_id
        logger.log_info(f"mixtral_brain: loading via transformers from {src} ...")
        tokenizer = AutoTokenizer.from_pretrained(src)

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            src,
            quantization_config=bnb_cfg,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()
        logger.log_info("mixtral_brain: loaded via transformers (4-bit BnB)")
        return model, tokenizer
    except Exception as e:
        logger.log_info(f"mixtral_brain: transformers failed ({e})")
        return None


class TritonPythonModel:

    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info(f"mixtral_brain: initializing from {_MODEL_DIR}...")

        self._backend     = None
        self._trtllm      = None
        self._vllm        = None
        self._vllm_params = None
        self._hf_model    = None
        self._hf_tok      = None

        # --- Try TRT-LLM first ---
        runner = _try_trtllm(_ENGINE_DIR, self.logger)
        if runner is not None:
            self._trtllm  = runner
            self._backend = "trtllm"

        # --- Try vLLM ---
        if self._backend is None:
            result = _try_vllm(_MODEL_DIR, self.logger)
            if result is not None:
                self._vllm, self._hf_tok, self._vllm_params = result
                self._backend = "vllm"

        # --- Transformers fallback ---
        if self._backend is None:
            result = _try_transformers(_MODEL_DIR, _HF_MODEL_ID, self.logger)
            if result is not None:
                self._hf_model, self._hf_tok = result
                self._backend = "transformers"

        if self._backend is None:
            raise RuntimeError(
                f"mixtral_brain: all backends failed for model '{_MODEL_DIR}'. "
                "If using NVFP4 model: switch to AWQ model (set MIXTRAL_MODEL). "
                "If using AWQ model: ensure vLLM is installed. "
                "Install bitsandbytes for transformers fallback."
            )

        self.logger.log_info(f"mixtral_brain: ready (backend={self._backend})")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                resp = self._process_one(request)
                responses.append(resp)
            except Exception as e:
                self.logger.log_error(f"mixtral_brain error: {e}")
                import traceback; traceback.print_exc()
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))
        return responses

    def _process_one(self, request):
        prompt_tensor   = pb_utils.get_input_tensor_by_name(request, "PROMPT")
        max_tok_tensor  = pb_utils.get_input_tensor_by_name(request, "MAX_TOKENS")

        prompt_bytes = prompt_tensor.as_numpy().flat[0]
        prompt = prompt_bytes.decode("utf-8") if isinstance(prompt_bytes, bytes) else str(prompt_bytes)
        max_tokens = min(max(int(max_tok_tensor.as_numpy().flat[0]), 32), 2048)

        self.logger.log_info(
            f"mixtral_brain: generating ({self._backend}, max_tokens={max_tokens}, "
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
        self.logger.log_info(f"mixtral_brain: done in {elapsed:.2f}s, {len(response_text.split())} words")

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
        return self._trtllm.tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    def _generate_vllm(self, prompt: str, max_tokens: int) -> str:
        params = self._vllm_params(temperature=0.7, top_p=0.9, max_tokens=max_tokens)
        outputs = self._vllm.generate([self._format_prompt(prompt)], params)
        return outputs[0].outputs[0].text.strip()

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
        return self._hf_tok.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    @staticmethod
    def _format_prompt(prompt: str) -> str:
        """Use chat template format. Falls back to [INST]...[/INST] for Mixtral."""
        prompt = prompt.strip()
        # Qwen / generic chat format — wrap in user role tags if not already formatted
        if "<|im_start|>" not in prompt and "[INST]" not in prompt:
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif not prompt.startswith("[INST]"):
            prompt = f"[INST] {prompt} [/INST]"
        return prompt

    def finalize(self):
        self._trtllm   = None
        self._vllm     = None
        self._hf_model = None
        self.logger.log_info("mixtral_brain: finalized.")
