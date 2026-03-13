#!/usr/bin/env python3
"""
Phase 0: Transcript Quality Analysis Tool
==========================================
Decodes raw TEXT_TOKEN IDs logged by the gateway into human-readable text
using PersonaPlex's SentencePiece tokenizer.

Usage:
  # Capture gateway logs while having a conversation, then decode:
  docker compose logs gateway 2>&1 | grep '\[tok\]' > /tmp/tokens.txt
  python tools/decode_transcript.py --tokens /tmp/tokens.txt

  # Or stream live (on the Spark):
  docker compose logs -f gateway 2>&1 | python tools/decode_transcript.py --stream

  # Specify a custom tokenizer path:
  python tools/decode_transcript.py --tokens /tmp/tokens.txt \
      --tokenizer /mnt/models/hf-cache/.../tokenizer_spm_32k_3.model
"""

import sys
import argparse
import os
import re
from pathlib import Path


def find_tokenizer() -> Path:
    """Search common HF cache locations for the PersonaPlex SentencePiece model."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    search_roots = [
        Path(hf_home) / "hub",
        Path("/mnt/models/hf-cache/hub"),
        Path("/root/.cache/huggingface/hub"),
    ]

    tokenizer_name = "tokenizer_spm_32k_3.model"
    for root in search_roots:
        if not root.exists():
            continue
        for match in root.rglob(tokenizer_name):
            return match

    raise FileNotFoundError(
        f"Could not find {tokenizer_name} in HF cache. "
        "Use --tokenizer to specify the path directly."
    )


def load_tokenizer(path: Path):
    try:
        import sentencepiece
    except ImportError:
        print("ERROR: sentencepiece not installed. Run: pip install sentencepiece")
        sys.exit(1)

    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(str(path))
    print(f"  Tokenizer: {path}")
    print(f"  Vocab size: {sp.GetPieceSize()}")
    return sp


def decode_tokens(token_ids: list[int], sp) -> str:
    """Decode a list of token IDs to text, skipping special/padding tokens."""
    # SentencePiece decode handles the ▁ (word boundary) markers automatically
    return sp.Decode(token_ids)


def parse_token_line(line: str) -> tuple[str, int] | None:
    """
    Parse a [tok] log line.
    Format: "[tok] <session8> <token_id>"
    """
    m = re.match(r'\[tok\]\s+(\S+)\s+(\d+)', line)
    if m:
        return m.group(1), int(m.group(2))
    return None


def process_tokens(lines, sp, min_print_every: int = 20):
    """Accumulate tokens per session, print running transcript."""
    sessions: dict[str, list[int]] = {}
    counts: dict[str, int] = {}

    for line in lines:
        line = line.strip()
        parsed = parse_token_line(line)
        if not parsed:
            continue

        sess_prefix, token_id = parsed

        if sess_prefix not in sessions:
            sessions[sess_prefix] = []
            counts[sess_prefix] = 0
            print(f"\n{'─' * 60}")
            print(f"  Session: {sess_prefix}...")
            print(f"{'─' * 60}")

        sessions[sess_prefix].append(token_id)
        counts[sess_prefix] += 1

        # Print incremental transcript every N tokens
        if counts[sess_prefix] % min_print_every == 0:
            text = decode_tokens(sessions[sess_prefix], sp)
            print(f"\r  [{counts[sess_prefix]:4d} tokens] {text}", end="", flush=True)

    # Final decode for each session
    print("\n")
    for sess_prefix, token_ids in sessions.items():
        text = decode_tokens(token_ids, sp)
        n = len(token_ids)
        print(f"\n{'═' * 60}")
        print(f"  Session {sess_prefix} — {n} tokens")
        print(f"{'═' * 60}")
        print(text)
        print()

        # Quality metrics
        words = text.split()
        chars = len(text)
        unique_tokens = len(set(token_ids))
        print(f"  📊 Quality metrics:")
        print(f"     Words:          {len(words)}")
        print(f"     Characters:     {chars}")
        print(f"     Unique tokens:  {unique_tokens} / {n}")
        print(f"     Words/min:      {len(words) / (n * 0.08 / 60):.1f}")
        print()

        # Detect issues
        issues = []
        if unique_tokens < 10:
            issues.append("⚠️  Very low token variety — may be looping/stuck")
        if chars / max(n, 1) < 1.5:
            issues.append("⚠️  Short chars/token — may be special token spam")
        non_alpha = sum(1 for c in text if not c.isalpha() and c not in " ,.!?'-\n")
        if non_alpha / max(chars, 1) > 0.3:
            issues.append("⚠️  High non-alpha ratio — possible encoding issues")

        if issues:
            print("  Issues detected:")
            for issue in issues:
                print(f"     {issue}")
        else:
            print("  ✅ Transcript looks clean — suitable for Mixtral input")
        print()


def stream_mode(sp):
    """Read from stdin line-by-line (for piped live logs)."""
    print("  🎙  Streaming mode — pipe Docker logs here. Ctrl+C to stop.")
    print("  Tip: docker compose logs -f gateway 2>&1 | python tools/decode_transcript.py --stream")
    print()

    sessions: dict[str, list[int]] = {}
    counts: dict[str, int] = {}

    try:
        for line in sys.stdin:
            line = line.strip()
            parsed = parse_token_line(line)
            if not parsed:
                continue

            sess_prefix, token_id = parsed

            if sess_prefix not in sessions:
                sessions[sess_prefix] = []
                counts[sess_prefix] = 0
                print(f"\n  [Session {sess_prefix}...]")

            sessions[sess_prefix].append(token_id)
            counts[sess_prefix] += 1

            # Print current text every 10 new tokens
            if counts[sess_prefix] % 10 == 0:
                text = decode_tokens(sessions[sess_prefix], sp)
                print(f"\r  [{counts[sess_prefix]:4d}] {text[-120:]}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopped. Final transcripts:")
        for sess_prefix, token_ids in sessions.items():
            text = decode_tokens(token_ids, sp)
            print(f"\n  {sess_prefix}: {text}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Decode PersonaPlex TEXT_TOKEN logs into text (Phase 0 analysis)"
    )
    parser.add_argument("--tokens",  "-t", help="Path to grepped [tok] log file")
    parser.add_argument("--stream",  "-s", action="store_true",
                        help="Stream from stdin (pipe docker logs here)")
    parser.add_argument("--tokenizer", help="Path to tokenizer_spm_32k_3.model")
    parser.add_argument("--every", type=int, default=20,
                        help="Print transcript every N tokens (default: 20)")
    args = parser.parse_args()

    if not args.tokens and not args.stream:
        parser.print_help()
        print("\nExample:")
        print("  docker compose logs gateway 2>&1 | grep '[tok]' > /tmp/t.txt")
        print("  python tools/decode_transcript.py --tokens /tmp/t.txt")
        sys.exit(1)

    # Load tokenizer
    print("\n PersonaPlex Transcript Decoder (Phase 0)")
    print(" ─────────────────────────────────────────")
    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
    else:
        print("  Searching for tokenizer in HF cache...")
        try:
            tokenizer_path = find_tokenizer()
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

    sp = load_tokenizer(tokenizer_path)

    if args.stream:
        stream_mode(sp)
    else:
        token_file = Path(args.tokens)
        if not token_file.exists():
            print(f"  ERROR: File not found: {token_file}")
            sys.exit(1)
        print(f"  Decoding: {token_file}")
        with open(token_file) as f:
            lines = f.readlines()
        print(f"  Lines:    {len(lines)}")
        process_tokens(lines, sp, min_print_every=args.every)


if __name__ == "__main__":
    main()
