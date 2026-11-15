#!/usr/bin/env bash
# Owner WSL: stop on failure; retain each stage's original run directory.
set -euo pipefail
cd /home/sasha/Facade_Change_Benchmark
nvidia-smi
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic check \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-15-www-archive-20mp-001.local.json \
  --roles jina --load-model \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-jina-check-005
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www archive-index \
  --corpus /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-archive-v31-001/corpus.json \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-15-www-archive-20mp-001.local.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-archive-index-v31-002
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www run \
  --questions /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-public-smoke-001/questions.json \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-15-www-archive-20mp-001.local.json \
  --archive-index /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-archive-index-v31-002/index.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-gpu-smoke-003
