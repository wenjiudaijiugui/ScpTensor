#!/bin/bash
# Live preview server for ScpTensor documentation with auto-reload

cd "$(dirname "$0")"

echo "Starting live documentation server..."
echo "Press Ctrl+C to stop."
echo ""
echo "Documentation will be available at: http://127.0.0.1:8000"
echo ""

uv run sphinx-autobuild . _build/html --open-browser
