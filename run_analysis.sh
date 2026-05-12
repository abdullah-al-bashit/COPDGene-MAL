#!/bin/bash
# =============================================================================
# run_analysis.sh  —  Execute all analyses and generate HTML reports.
# Usage:  bash run_analysis.sh
#         bash run_analysis.sh path/to/script.ipynb   (single file)
#         bash run_analysis.sh path/to/script.R        (single file)
# =============================================================================

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"   # absolute path to this script's folder
JUPYTER="/Applications/JupyterLab.app/Contents/Resources/jlab_server/bin/jupyter"
R_KERNEL="ir-vscode"                    # Jupyter kernel name for R notebooks
NB_TIMEOUT=120                          # seconds allowed per notebook cell

cd "$ROOT"

# -----------------------------------------------------------------------------
# render_notebook <path/to/notebook.ipynb>
#   Executes all cells and writes HTML next to the notebook.
# -----------------------------------------------------------------------------
render_notebook() {
  local nb="$1"
  local out_dir; out_dir="$(dirname "$nb")"
  local out_name; out_name="$(basename "${nb%.ipynb}.html")"
  echo "  [notebook] $nb -> $out_dir/$out_name"
  "$JUPYTER" nbconvert \
    --to html \
    --execute \
    --ExecutePreprocessor.kernel_name="$R_KERNEL" \
    --ExecutePreprocessor.timeout="$NB_TIMEOUT" \
    "$nb" \
    --output-dir "$out_dir" \
    --output "$out_name"
}

# -----------------------------------------------------------------------------
# render_rscript <path/to/script.R>
#   Runs all code and writes HTML next to the script. No intermediate
#   files are left behind (intermediates go to a temp directory).
# -----------------------------------------------------------------------------
render_rscript() {
  local script="$1"
  local out_dir; out_dir="$(dirname "$script")"
  local out_name; out_name="$(basename "${script%.R}.html")"
  echo "  [R script] $script -> $out_dir/$out_name"
  Rscript -e "
rmarkdown::render(
  '$script',
  output_format     = rmarkdown::html_document(
                        toc = TRUE, toc_float = TRUE,
                        theme = 'flatly', highlight = 'tango',
                        code_folding = 'show'),
  output_file       = '$out_name',
  output_dir        = '$out_dir',
  intermediates_dir = tempdir(),
  knit_root_dir     = '$ROOT',
  clean             = TRUE,
  quiet             = TRUE
)
"
}

# =============================================================================
# SCRIPTS TO RUN
# Add one line per file. Output HTML is always placed next to the source file.
# =============================================================================

if [ $# -gt 0 ]; then
  # Single-file mode: run only the file passed as argument
  for f in "$@"; do
    case "$f" in
      *.ipynb) render_notebook "$f" ;;
      *.R)     render_rscript  "$f" ;;
      *) echo "Unsupported file type: $f (use .ipynb or .R)" && exit 1 ;;
    esac
  done
else
  # Default: run all registered analyses
  echo "=== Running all analyses ==="
  render_notebook "Aim2_prediction/Aim2_prediction_adaptive_lasso.ipynb"
  # Add new scripts here — one line each:
  # render_notebook "NewProject/analysis.ipynb"
  # render_rscript  "NewProject/model.R"
fi

echo ""
echo "=== Done ==="
