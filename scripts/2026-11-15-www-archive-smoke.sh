#!/usr/bin/env bash
# Owner WSL: Jina check-005 passed; index and QA use new 2026-11-16 run directories.
set -euo pipefail
cd /home/sasha/Facade_Change_Benchmark
nvidia-smi
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www archive-index \
  --corpus /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-archive-v31-001/corpus.json \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-15-www-archive-20mp-001.local.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-archive-index-v31-001
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www run \
  --questions /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-public-smoke-001/questions.json \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-15-www-archive-20mp-001.local.json \
  --archive-index /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-archive-index-v31-001/index.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-gpu-smoke-001
