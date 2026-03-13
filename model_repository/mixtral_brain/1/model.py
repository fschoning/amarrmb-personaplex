"""
mixtral_brain/1/model.py — Triton Python backend for Mixtral 8x7B Reasoner

Initialization strategy (in order of preference):
  1. TensorRT-LLM engine  — fastest, requires prior trtllm-build step
  2. vLLM                 — fast, good Blackwell support
  3. HF Transformers      — slowest fallback, always available

The chosen backend is logged at startup so you can verify which is running.
"""

import os
import time
import json
import numpy as np
import triton_python_backend_utils as pb_utils

# ── Config from environment ──────────────────────────────────────────────────
_ENGINE_DIR  = os.environ.get("MIXTRAL_ENGINE", "/mnt/models/mixtral-engine")
_MODEL_DIR   = os.environ.get("MIXTRAL_MODEL",  "/mnt/models/Mixtral-8x7B-Instruct-v0.1-NVFP4")
_HF_MODEL_ID = os.environ.get("MIXTRAL_HF_ID",  "mistralai/Mixtral-8x7B-Instruct-v0.1")
_DEVICE      = "cuda"
_BACKEND     = None   # set during initialize()


def _try_trtllm(engine_dir: str, logger):
    """Attempt to load TRT-LLM engine. Returns runner or None."""
    if not os.path.isdir(engine_dir):
        logger.log_info(f"mixtral_brain: TRT-LLM engine not found at {engine_dir}")
        return None
    try:
        import tensorrt_llm
        from tensorrt_llm.runtime import ModelRunner
        runner = ModelRunner.from_dir(
            engine_dir=engine_dir,
            rank=0,
        )
        logger.log_info(f"mixtral_brain: loaded TRT-LLM engine from {engine_dir}")
        return runner
    except Exception as e:
        logger.log_info(f"mixtral_brain: TRT-LLM failed ({e}), trying next backend")
        return None


def _try_vllm(model_dir: str, logger):
    """Attempt to load via vLLM. Returns (llm, tokenizer) or None."""
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Detect NVFP4 quantization from hf_quant_config.json
        import json, os
        quant = None
        quant_cfg_path = os.path.join(model_dir, "hf_quant_config.json")
        if os.path.exists(quant_cfg_path):
            with open(quant_cfg_path) as f:
                cfg = json.load(f)
            algo = cfg.get("quantization", {}).get("quant_algo", "")
            if "FP4" in algo.upper() or "NVFP4" in algo.upper():
                quant = "fp4"
                logger.log_info(f"mixtral_brain: detected NVFP4 quantization")

        llm = LLM(
            model=model_dir,
            quantization=quant,           # "fp4" for NVFP4, None for standard
            dtype="bfloat16",             # compute dtype
            gpu_memory_utilization=0.28,  # leave ~72% for 2× PersonaPlex
            max_model_len=6144,
            trust_remote_code=True,
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

        # Try the NVFP4 directory first, fall back to HF hub ID
        src = model_dir if os.path.isdir(model_dir) else hf_id
        logger.log_info(f"mixtral_brain: loading via transformers from {src} ...")

        tokenizer = AutoTokenizer.from_pretrained(src)

        # Use 4-bit quantization to fit alongside two PersonaPlex instances
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
        self.logger.log_info("mixtral_brain: initializing...")

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
                "mixtral_brain: all backends failed. "
                "Install tensorrt_llm, vllm, or transformers+bitsandbytes."
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
        # --- Decode inputs ---
        prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
        max_tok_tensor = pb_utils.get_input_tensor_by_name(request, "MAX_TOKENS")

        prompt_bytes = prompt_tensor.as_numpy().flat[0]
        if isinstance(prompt_bytes, bytes):
            prompt = prompt_bytes.decode("utf-8")
        else:
            prompt = str(prompt_bytes)

        max_tokens = int(max_tok_tensor.as_numpy().flat[0])
        max_tokens = min(max(max_tokens, 32), 2048)  # clamp

        self.logger.log_info(
            f"mixtral_brain: generating ({self._backend}, max_tokens={max_tokens}, "
            f"prompt_len={len(prompt)})"
        )
        t0 = time.monotonic()

        # --- Generate ---
        if self._backend == "trtllm":
            response_text = self._generate_trtllm(prompt, max_tokens)
        elif self._backend == "vllm":
            response_text = self._generate_vllm(prompt, max_tokens)
        else:
            response_text = self._generate_hf(prompt, max_tokens)

        elapsed = time.monotonic() - t0
        self.logger.log_info(
            f"mixtral_brain: done in {elapsed:.2f}s, "
            f"{len(response_text.split())} words"
        )

        # --- Encode output ---
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
                temperature=0.7,
                top_p=0.9,
            )
        input_len = input_ids.shape[1]
        return self._trtllm.tokenizer.decode(
            output[0][input_len:], skip_special_tokens=True
        ).strip()

    def _generate_vllm(self, prompt: str, max_tokens: int) -> str:
        params = self._vllm_params(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens,
        )
        outputs = self._vllm.generate([self._format_prompt(prompt)], params)
        return outputs[0].outputs[0].text.strip()

    def _generate_hf(self, prompt: str, max_tokens: int) -> str:
        import torch
        formatted = self._format_prompt(prompt)
        inputs = self._hf_tok(formatted, return_tensors="pt").to(_DEVICE)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._hf_tok.eos_token_id,
            )
        new_tokens = output_ids[0][input_len:]
        return self._hf_tok.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _format_prompt(prompt: str) -> str:
        """Wrap in Mixtral Instruct format: [INST] ... [/INST]"""
        prompt = prompt.strip()
        if not prompt.startswith("[INST]"):
            prompt = f"[INST] {prompt} [/INST]"
        return prompt

    def finalize(self):
        self._trtllm  = None
        self._vllm    = None
        self._hf_model = None
        self.logger.log_info("mixtral_brain: finalized.")
