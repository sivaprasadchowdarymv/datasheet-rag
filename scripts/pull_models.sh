#!/usr/bin/env bash
# Download the open models into the Ollama container (run once after `docker compose up -d`).
# Usage: ./scripts/pull_models.sh [llm-model]      e.g. ./scripts/pull_models.sh qwen2.5:3b
set -euo pipefail
LLM="${1:-${LLM_MODEL:-mistral}}"
docker compose exec ollama ollama pull "$LLM"
docker compose exec ollama ollama pull nomic-embed-text
if [ "${ENABLE_VISION:-false}" = "true" ]; then
  docker compose exec ollama ollama pull llava:7b
fi
docker compose exec ollama ollama list
