#!/bin/bash
# Helper script to convert consistency model checkpoint for DARTS compatibility

set -e

if [[ $# -lt 1 ]]; then
    cat << 'EOF'
Usage:
    ./src/darts_superresolution/util/convert_checkpoint.sh /path/to/original.ckpt [/path/to/output.ckpt]

Converts a consistency model checkpoint from src.* module format to DARTS-compatible format.

This removes misleading pickle metadata so the checkpoint loads without module path errors.

Examples:
    # Convert with automatic output name
    ./src/darts_superresolution/util/convert_checkpoint.sh /p/scratch/hai_earth_04/lucas/Consistency_Model/checkpoint/consistency_wavelet*.ckpt

    # Convert to specific location
    ./src/darts_superresolution/util/convert_checkpoint.sh original.ckpt ./consistency_wavelet_converted.ckpt

EOF
    exit 1
fi

ORIGINAL_CKPT="$1"
OUTPUT_CKPT="${2:-${ORIGINAL_CKPT%.ckpt}.converted.ckpt}"

if [[ ! -f "$ORIGINAL_CKPT" ]]; then
    echo "❌ Original checkpoint not found: $ORIGINAL_CKPT"
    exit 1
fi

echo "Converting checkpoint for DARTS compatibility..."
echo "  Original: $ORIGINAL_CKPT"
echo "  Output:   $OUTPUT_CKPT"

# Run conversion from consistency repo
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
WORKTREE_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$WORKTREE_ROOT" ]]; then
    # Fallback: util -> darts_superresolution -> src -> darts-superresolution -> darts-nextgen
    WORKTREE_ROOT="$(realpath "$SCRIPT_DIR/../../../../")"
fi

CONSISTENCY_REPO="$(realpath "$WORKTREE_ROOT/../consistency_model_distillation_for_sr3")"
CONSISTENCY_PYTHON="$CONSISTENCY_REPO/CM/bin/python"

if [[ ! -x "$CONSISTENCY_PYTHON" ]]; then
    echo "❌ Consistency repo Python not found: $CONSISTENCY_PYTHON"
    exit 1
fi

cd "$CONSISTENCY_REPO"
"$CONSISTENCY_PYTHON" convert_checkpoint.py "$ORIGINAL_CKPT" "$OUTPUT_CKPT"

echo ""
echo "✅ Conversion complete!"
echo ""
echo "Next steps:"
echo "  1. Update infer.py to use: $OUTPUT_CKPT"
echo "  2. Run the test: python infer.py"
