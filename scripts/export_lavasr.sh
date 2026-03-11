#!/usr/bin/env bash
# scripts/export_lavasr.py  — PyTorch shell wrapper calling the Python script
# Usage: bash scripts/export_lavasr.sh
#        (calls python scripts/export_lavasr.py internally)
exec python "$(dirname "$0")/export_lavasr.py" "$@"
