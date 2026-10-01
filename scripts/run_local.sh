#!/usr/bin/env bash
# Local PC run (Linux/macOS). Windows: see README "Run on your PC".
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull mistral && ollama pull nomic-embed-text && ollama pull llava:7b
APP_MODE=local streamlit run app.py
