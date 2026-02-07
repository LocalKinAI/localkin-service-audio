#!/usr/bin/env bash
# Show LocalKin Audio system status
set -euo pipefail

echo "=== LocalKin Audio Status ==="
kin audio status
echo ""
echo "=== Available Models ==="
kin audio models
echo ""
echo "=== Cache Info ==="
kin audio cache info
