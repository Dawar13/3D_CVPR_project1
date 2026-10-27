#!/usr/bin/env bash
# Stage A inspection. Prints everything Deliverable A needs; paste into the report.
set -u
echo "## date";            date -u
echo "## host";            hostname; uname -a
echo "## gpus";            nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv 2>/dev/null || echo "no nvidia-smi"
echo "## gpu processes";   nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null
echo "## disk";            df -h . | tail -1
echo "## python";          which python; python --version
python - <<'PY'
import importlib
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
except Exception as e:
    print("torch not importable:", e)
for m in ["mapanything", "numpy", "PIL", "pandas", "matplotlib"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"), mod.__file__)
    except Exception as e:
        print(m, "NOT importable:", e)
PY
REPO=${1:-third_party/map-anything}
echo "## map-anything repo ($REPO)"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --short | head
  head -5 "$REPO/LICENSE" 2>/dev/null
else
  echo "not cloned"
fi
echo "## HF cache"; ls ~/.cache/huggingface/hub 2>/dev/null | grep -i map || echo "no map-anything checkpoint cached"
