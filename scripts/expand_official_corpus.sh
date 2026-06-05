#!/usr/bin/env bash
# Fetch official docs (AWS + Docker + Google Drive/Workspace) and rebuild unified index.
set -euo pipefail
cd "$(dirname "$0")/.."

MAX_PAGES="${1:-100}"
echo "Fetching official docs (max ${MAX_PAGES} pages per source)…"
echo "This may take a long time (many HTTP requests)."

python scripts/fetch_official_docs.py --sources all --max-pages "$MAX_PAGES"

echo "Rebuilding unified index at data/index/official …"
python run_ingest.py --docs-dir data/docs --index-dir data/index/official

DOC_COUNT=$(find data/docs -type f \( -name '*.txt' -o -name '*.md' \) ! -path '*/.*' | wc -l | tr -d ' ')
CHUNKS=$(python -c "import json; print(json.load(open('data/index/official/meta.json'))['num_chunks'])")
echo "Done: ${DOC_COUNT} document files under data/docs/, ${CHUNKS} chunks in index."
