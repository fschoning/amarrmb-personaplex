#!/usr/bin/env python3
"""
tools/test_brain.py — Standalone test for the Mixtral brain Triton model.

Usage (from the Spark):
  # Single question
  python3 tools/test_brain.py --host 192.168.2.117 --prompt "What is quantum entanglement?"

  # Interactive mode
  python3 tools/test_brain.py --host 192.168.2.117 --interactive

  # With custom max tokens
  python3 tools/test_brain.py --host 192.168.2.117 --prompt "Summarize AI" --max-tokens 128
"""

import argparse
import sys
import time
import numpy as np

try:
    import tritonclient.grpc as triton_grpc
except ImportError:
    print("ERROR: tritonclient not installed. Run: pip install tritonclient[grpc]")
    sys.exit(1)


def query_brain(client, prompt: str, max_tokens: int = 256) -> str:
    """Send a prompt to the brain Triton model and return the response."""

    # Build inputs — dims: [1] means 1-D tensor of length 1
    prompt_arr  = np.array([prompt.encode("utf-8")], dtype=object)   # shape (1,)
    max_tok_arr = np.array([max_tokens], dtype=np.int32)              # shape (1,)

    prompt_inp = triton_grpc.InferInput("PROMPT", [1], "BYTES")
    prompt_inp.set_data_from_numpy(prompt_arr)

    max_tok_inp = triton_grpc.InferInput("MAX_TOKENS", [1], "INT32")
    max_tok_inp.set_data_from_numpy(max_tok_arr)

    response_out = triton_grpc.InferRequestedOutput("RESPONSE")

    t0 = time.monotonic()
    result = client.infer(
        model_name="brain",
        inputs=[prompt_inp, max_tok_inp],
        outputs=[response_out],
    )
    elapsed = time.monotonic() - t0

    resp_bytes = result.as_numpy("RESPONSE").flat[0]
    if isinstance(resp_bytes, bytes):
        response = resp_bytes.decode("utf-8")
    else:
        response = str(resp_bytes)

    return response, elapsed


def main():
    parser = argparse.ArgumentParser(description="Test the PersonaPlex Mixtral brain")
    parser.add_argument("--host",        default="localhost", help="Triton gRPC host")
    parser.add_argument("--port",        type=int, default=8001, help="Triton gRPC port")
    parser.add_argument("--prompt",  "-p", default="",    help="Single prompt to test")
    parser.add_argument("--max-tokens", "-m", type=int, default=256)
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive prompt loop")
    args = parser.parse_args()

    print(f"\n  Mixtral Brain Test")
    print(f"  ──────────────────")
    print(f"  Triton : {args.host}:{args.port}")
    print(f"  Model  : brain")
    print()

    client = triton_grpc.InferenceServerClient(
        url=f"{args.host}:{args.port}", verbose=False
    )

    # Check model is ready
    if not client.is_model_ready("brain"):
        print("ERROR: brain model is not ready on Triton.")
        print("  Check: docker compose logs triton | grep mixtral")
        sys.exit(1)

    print("  ✅ brain is ready\n")

    if args.interactive:
        print("  Interactive mode — type your prompt and press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                prompt = input("  You: ").strip()
                if not prompt:
                    continue
                print("  Brain: ", end="", flush=True)
                response, elapsed = query_brain(client, prompt, args.max_tokens)
                print(response)
                print(f"  [{elapsed:.2f}s]\n")
            except KeyboardInterrupt:
                print("\n  Bye!")
                break

    elif args.prompt:
        print(f"  Prompt: {args.prompt}\n")
        response, elapsed = query_brain(client, args.prompt, args.max_tokens)
        print(f"  Response:\n  {response}")
        print(f"\n  Generated in {elapsed:.2f}s ({len(response.split())} words)")

    else:
        # Default smoke test
        prompts = [
            "In one sentence, what is the capital of France?",
            "You are a helpful AI assistant. The user said: 'tell me about space'. "
            "In 2-3 sentences, what should you say next?",
        ]
        for p in prompts:
            print(f"  Prompt: {p}")
            response, elapsed = query_brain(client, p, max_tokens=512)
            print(f"  Response: {response}")
            print(f"  Time: {elapsed:.2f}s\n")


if __name__ == "__main__":
    main()
