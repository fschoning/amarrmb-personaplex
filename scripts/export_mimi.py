#!/usr/bin/env python3
# scripts/export_mimi.py
# Reference script for exporting Mimi encoder/decoder to ONNX.
# NOTE: Full TensorRT conversion is Phase 2 (optional, high-risk).
# The Triton Python backends in model_repository/ work without this.
#
# Run only if pursuing full Python elimination (see design_document.md §8).

import argparse
import os
import torch

def main():
    ap = argparse.ArgumentParser(
        description="Export Mimi encoder/decoder to ONNX (Phase 2 optional step)")
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--out-dir", default="model_repository/mimi_onnx")
    ap.add_argument("--component", choices=["encoder", "decoder", "both"], default="both")
    args = ap.parse_args()

    print("WARNING: Mimi ONNX export is an advanced Phase 2 task.")
    print("The streaming state (reset_streaming, streaming_forever) must be")
    print("exposed as explicit input/output tensors. This requires significant")
    print("refactoring of compression.py. See design_document.md §8.")
    print()

    from moshi.models import loaders
    device = torch.device("cuda")
    mimi_weight = loaders.get_mimi_weight_path(args.hf_repo or loaders.DEFAULT_REPO)
    mimi = loaders.get_mimi(mimi_weight, device).eval()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.component in ("encoder", "both"):
        print("Exporting encoder... (stateful streaming — see compression.py:MimiModel.encode)")
        # TODO: refactor MimiModel.encode to accept/return streaming state tensors
        # then call torch.onnx.export() here
        print("  Encoder export: NOT YET IMPLEMENTED — see Phase 2 notes")

    if args.component in ("decoder", "both"):
        print("Exporting decoder... (stateful streaming — see compression.py:MimiModel.decode)")
        # TODO: same as encoder
        print("  Decoder export: NOT YET IMPLEMENTED — see Phase 2 notes")

    print("\nFor Phase 1 (current), use the Python backends in model_repository/.")


if __name__ == "__main__":
    main()
