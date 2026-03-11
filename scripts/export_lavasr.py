#!/usr/bin/env python3
# scripts/export_lavasr.py
# Exports LavaSR v2 from PyTorch → ONNX → TensorRT engine for DGX Spark (SM 12.1)
#
# Usage:
#   python scripts/export_lavasr.py [--hf-repo declinator/lava-sr-v2] \
#                                   [--out-dir model_repository/lavasr_v2/1]

import argparse
import os
import subprocess
import sys
import tempfile

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo",  default="declinator/lava-sr-v2")
    ap.add_argument("--out-dir",  default="model_repository/lavasr_v2/1")
    ap.add_argument("--input-len", type=int, default=1920,
                    help="Number of 24kHz samples per frame (1920 = 80ms)")
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="Export with FP16 (recommended for Blackwell)")
    ap.add_argument("--skip-trt", action="store_true",
                    help="Stop after ONNX, skip TensorRT conversion")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    onnx_path = os.path.join(args.out_dir, "lavasr_v2.onnx")
    plan_path = os.path.join(args.out_dir, "model.plan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading LavaSR v2 from {args.hf_repo!r}...")
    try:
        from lavasr import LavaSR  # type: ignore
        model = LavaSR.from_pretrained(args.hf_repo).to(device).eval()
    except ImportError:
        model = torch.hub.load(args.hf_repo, "lavasr_v2", pretrained=True
                               ).to(device).eval()

    if args.fp16:
        model = model.half()

    dtype  = torch.float16 if args.fp16 else torch.float32
    dummy  = torch.zeros(1, 1, args.input_len, dtype=dtype, device=device)

    # --- Trace / Script ---
    print("Wrapping for ONNX export...")

    class LavaSRWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, pcm: torch.Tensor) -> torch.Tensor:
            return self.m(pcm, input_sr=24000, output_sr=48000)

    wrapped = LavaSRWrapper(model)

    # --- Export ONNX ---
    print(f"Exporting ONNX → {onnx_path}")
    torch.onnx.export(
        wrapped,
        (dummy,),
        onnx_path,
        opset_version=17,
        input_names=["PCM_24K"],
        output_names=["OUTPUT_PCM_48K"],
        dynamic_axes={
            "PCM_24K":        {0: "batch"},
            "OUTPUT_PCM_48K": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"ONNX saved: {onnx_path}")

    if args.skip_trt:
        print("Skipping TensorRT conversion (--skip-trt).")
        return

    # --- Convert to TensorRT via trtexec ---
    precision = "--fp16" if args.fp16 else ""
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        f"--minShapes=PCM_24K:1x1x{args.input_len}",
        f"--optShapes=PCM_24K:1x1x{args.input_len}",
        f"--maxShapes=PCM_24K:1x1x{args.input_len}",
        "--buildOnly",
        "--noDataTransfers",
        precision,
    ]
    cmd = [c for c in cmd if c]  # strip empty strings
    print("Running:", " ".join(cmd))
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print("ERROR: trtexec failed. Check CUDA / TensorRT installation.")
        sys.exit(ret.returncode)

    print(f"\nTensorRT engine saved: {plan_path}")
    print("\nTo activate TensorRT backend:")
    print("  1. Edit model_repository/lavasr_v2/config.pbtxt")
    print('     Change: backend: "python"  →  backend: "tensorrt"')
    print("  2. Remove or rename model_repository/lavasr_v2/1/model.py")
    print("  3. Restart Triton")


if __name__ == "__main__":
    main()
