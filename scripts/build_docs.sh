#!/bin/bash
# Build script for ScpTensor documentation

set -e

# Change to docs directory
cd "$(dirname "$0")/.."
cd docs

echo "Building ScpTensor documentation..."
echo "===================================="

# Clean previous build
echo "Cleaning previous build..."
rm -rf _build

# Build HTML documentation
echo "Building HTML documentation..."
uv run sphinx-build -b html . _build/html "$@"

# Report success
echo ""
echo "===================================="
echo "Build complete!"
echo ""
echo "Documentation generated in: _build/html/"
echo "Open _build/html/index.html in your browser to view."
echo ""
echo "To serve locally:"
echo "  cd _build/html && python3 -m http.server 8000"
echo "  Then visit: http://localhost:8000"
echo ""
echo "For live reload while editing:"
echo "  make livehtml"
echo "  or"
echo "  ../scripts/serve_docs.sh"
echo ""
