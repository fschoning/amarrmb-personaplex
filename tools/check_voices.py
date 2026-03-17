#!/usr/bin/env python3
"""
check_voices.py — Verify Moshi voice .pt files and tokenize prompts.

Run ON THE SPARK (inside or outside the Triton container):

  # Check what voice files exist:
  python3 tools/check_voices.py

  # Download missing voice files + show filler token IDs:
  python3 tools/check_voices.py --download

  # Just tokenize a phrase (to get token IDs for hardcoding):
  python3 tools/check_voices.py --tokenize "Just a moment, I'll be right with you."
"""
import os, sys, argparse

VOICE_NAMES = [
    "NATF0","NATF1","NATF2","NATF3",
    "NATM0","NATM1","NATM2","NATM3",
    "VARF0","VARF1","VARF2","VARF3","VARF4",
    "VARM0","VARM1","VARM2","VARM3","VARM4",
]

# HF repo for Moshi model files
DEFAULT_HF_REPO = os.environ.get("HF_REPO", "kyutai/moshi")

# The filler persona instruction — carefully worded so Moshi conditions
# on being a polite pause-holder that never answers questions.
FILLER_INSTRUCTION = (
    "You are briefly pausing. Say only short, warm filler phrases like "
    "'Just a moment', 'Hold on a sec', 'Bear with me', 'One moment please', "
    "'I'll be right with you', 'Just a second'. "
    "Never answer questions. Never volunteer information. "
    "If asked anything, respond only with another variation of 'just a moment'."
)

def find_hf_home():
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

def find_voices_dir():
    voices_base = os.path.join(find_hf_home(), "hub")
    print(f"Searching HF cache: {voices_base}")
    found_dirs = []
    for root, dirs, files in os.walk(voices_base):
        if os.path.basename(root) == "voices":
            found_dirs.append(root)
            pts = sorted(f for f in files if f.endswith(".pt"))
            print(f"  Found voices dir: {root}")
            print(f"    Contains: {pts if pts else '(empty)'}")
    if not found_dirs:
        print("  ✗ No 'voices/' directory found in HF cache!")
    return found_dirs

def check_voice(name, voices_dirs):
    for d in voices_dirs:
        p = os.path.join(d, f"{name}.pt")
        if os.path.exists(p):
            size = os.path.getsize(p)
            print(f"  ✓ {name:8s}  {size//1024:5d} KB  {p}")
            return True
    print(f"  ✗ {name:8s}  NOT FOUND")
    return False

def download_voices(hf_repo):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed: pip install huggingface_hub")
        return
    print(f"\nDownloading voice files from {hf_repo}...")
    for name in VOICE_NAMES:
        fname = f"voices/{name}.pt"
        try:
            path = hf_hub_download(hf_repo, fname)
            print(f"  ✓ {name} → {path}")
        except Exception as e:
            print(f"  ✗ {name} — {e}")

def find_tokenizer():
    """Find the Moshi SentencePiece tokenizer in HF cache."""
    hf_home = find_hf_home()
    # Walk hub looking for tokenizer file
    for root, dirs, files in os.walk(os.path.join(hf_home, "hub")):
        for f in files:
            if "tokenizer" in f.lower() and f.endswith(".model"):
                return os.path.join(root, f)
    return None

def download_tokenizer(hf_repo):
    try:
        from huggingface_hub import hf_hub_download
        # Moshi uses sentencepiece; the file is 'tokenizer_spm_32k_3.model'
        for candidate in [
            "tokenizer_spm_32k_3.model",
            "tokenizer.model",
        ]:
            try:
                path = hf_hub_download(hf_repo, candidate)
                print(f"  Tokenizer: {path}")
                return path
            except Exception:
                continue
    except ImportError:
        pass
    return None

def tokenize_phrase(phrase, hf_repo=DEFAULT_HF_REPO):
    """Tokenize a phrase with Moshi's SP tokenizer. Returns list of int IDs."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("sentencepiece not installed: pip install sentencepiece")
        return None

    tok_path = find_tokenizer()
    if not tok_path:
        print("  Tokenizer not in cache — downloading...")
        tok_path = download_tokenizer(hf_repo)
    if not tok_path:
        print("  ERROR: could not find or download Moshi tokenizer")
        return None

    sp = spm.SentencePieceProcessor(tok_path)
    ids = sp.Encode(phrase, out_type=int)
    return ids

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true",
                   help="Download missing voice files and tokenizer from HF")
    p.add_argument("--repo", default=DEFAULT_HF_REPO,
                   help=f"HF repo (default: {DEFAULT_HF_REPO})")
    p.add_argument("--tokenize", metavar="PHRASE", default=None,
                   help="Tokenize a phrase and print the token IDs")
    args = p.parse_args()

    print(f"\n=== PersonaPlex Voice & Tokenizer Check ===\n")
    print(f"HF_HOME : {find_hf_home()}")
    print(f"HF_REPO : {args.repo}\n")

    if args.download:
        download_voices(args.repo)

    print("\n--- Voice files ---")
    dirs = find_voices_dir()
    found = sum(1 for n in VOICE_NAMES if check_voice(n, dirs))
    print(f"\n{found}/{len(VOICE_NAMES)} voice files available.\n")

    # Always tokenize the standard filler instruction
    phrases = [FILLER_INSTRUCTION]
    if args.tokenize:
        phrases = [args.tokenize] + phrases

    print("--- Tokenizer ---")
    for phrase in phrases:
        label = "FILLER_INSTRUCTION" if phrase == FILLER_INSTRUCTION else "Custom"
        short = phrase[:60] + ("..." if len(phrase) > 60 else "")
        print(f"\n  Phrase ({label}): \"{short}\"")
        ids = tokenize_phrase(phrase, args.repo)
        if ids is not None:
            print(f"  Token IDs ({len(ids)}): {ids}")
            print(f"\n  # Paste this into brain_client.py FILLER_TEXT_TOKENS:")
            print(f"  FILLER_TEXT_TOKENS = {ids}")

if __name__ == "__main__":
    main()
