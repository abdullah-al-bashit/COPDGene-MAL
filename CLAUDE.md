# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COPDGene multi-omics analysis studying Mechanically Affected Lung (MAL) and emphysema progression. All analyses use SomaScan blood proteomics (~4,979 proteins) from the COPDGene cohort.

**MAL** = Mechanically Affected Lung: CT-derived continuous score (0.01–0.40) measuring lung volume under destructive mechanical stress.  
**Outcome** = `Change_lung_density_vnb_P2_P3` (HU): Phase 3 − Phase 2 CT density. Negative = emphysema progressed.

## Data (never commit — PHI/HIPAA)

All raw data lives in `data/` which is fully gitignored.

| File | Contents |
|------|----------|
| `data/csv_exports/prot.csv` | SomaScan RFU protein abundance (~5k proteins × 5,670 subjects) |
| `data/csv_exports/pheno_MAL.csv` | Phenotypes, covariates, outcomes |
| `data/csv_exports/metadat5k.csv` | Protein annotations (seqID, EntrezGeneSymbol, Target, UniProt) |
| `data/Data/MAL.csv` | MAL scores (sid, meanmal) |

Key phenotype columns: `sid`, `Change_lung_density_vnb_P2_P3`, `lung_density_vnb_P2`, `meanmal`, `Age_P2`, `gender`, `race`, `ATS_PackYears_P2`, `SmokCigNow_P2`, `scanner_model_clean_P2`, `PC1`–`PC5`.

Protein columns follow the pattern `X{seqID}` (raw) → `log2_X{seqID}` (after log2 transformation). Join to metadata on `seqID = sub("^log2_X", "", predictor)`.

## Running Analyses

```bash
# Execute a single notebook and produce HTML report alongside it
bash run_analysis.sh Aim2_prediction/Aim2_prediction_adaptive_lasso.ipynb

# Execute all registered analyses
bash run_analysis.sh
```

To execute a notebook and save outputs back into the `.ipynb` (for GitHub rendering):
```bash
/Applications/JupyterLab.app/Contents/Resources/jlab_server/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=ir-vscode \
  --ExecutePreprocessor.timeout=3600 \
  path/to/notebook.ipynb
```

## R Kernel

Notebooks use the `ir-vscode` kernel (IRkernel registered via `IRkernel::installspec(name="ir-vscode", displayname="R VS Code")`). Always specify `--ExecutePreprocessor.kernel_name=ir-vscode` when running via nbconvert.

## Adding a New Analysis

1. Create a new directory under `Journal/` (e.g. `MAL_omics/`)
2. Build the analysis as a `.ipynb` notebook with the `ir-vscode` kernel
3. Register it in `run_analysis.sh` under the "SCRIPTS TO RUN" section:
   ```bash
   render_notebook "MAL_omics/analysis.ipynb"
   ```
4. Add generated output files (CSVs, HTML) to `.gitignore`

## Statistical Conventions

- **Log2-transform** all SomaScan RFU values before modelling (`log2(pmax(x, 1e-6))`)
- **Adaptive lasso**: ridge pre-step on residualised outcome → weights `wⱼ = 1/|β̂_ridge_j|` → cv.glmnet with `alpha=1, penalty.factor=wⱼ`
- **Covariates always unpenalized**: `penalty.factor=0` for `lung_density_vnb_P2`, `Age_P2`, `gender`, `race`, `ATS_PackYears_P2`, `SmokCigNow_P2`, `scanner_model_clean_P2`, `PC1`–`PC5`
- **Design matrix alignment**: when splitting data, use `ref_colnames` parameter in `build_design()` to zero-fill missing factor-level dummy columns in val/test splits
- **Post-selection OLS**: after lasso selects proteins, refit OLS on selected set for unbiased β, SE, CI, p-values
- **AIC comparison**: improvement of ≥ 2 units (lower) is considered meaningful
- **AUC**: use `pROC::roc(..., direction="auto")` — never hardcode direction

## Existing Analyses

### `MAL_omics/MAL_omics_proteomics.ipynb`
Full MAL omics paper pipeline. Analyses in order:
1. **Table 1** — `gtsummary::tbl_summary()`
2. **Differential expression** — `run_protein_lm()` loop over ~4,979 proteins; BH FDR
3. **GSEA** — `fgsea` + `msigdbr`; Hallmark, KEGG, Reactome; ranked by t-statistic
4. **STRING network** — `STRINGdb` on FDR < 0.05 proteins
5. **Adaptive LASSO → MAL risk score** — same framework as Aim2 prediction; outcome = `meanmal`; 70/30 split
6. **Risk score ~ emphysema progression** — 3-model AIC comparison (clinical / PRS / combined); improvement ≥ −2 is meaningful
7. **Causal mediation** — `medflex::neImpute()` + `neModel()`; MAL → PRS → progression; NIE/NDE/proportion mediated
8. **Figures** — volcano (2A), GSEA barplot (2B), STRING network (2C), quartile plot (3), supplement forest plots

Outputs gitignored (`MAL_omics/*.csv/pdf/png/html`)



### `Aim2_prediction/Aim2_prediction_adaptive_lasso.ipynb`
Predicts emphysema progression from proteomics. Two models: with and without MAL. Key finding: MAL does not improve prediction over proteomics alone (AIC favours Model 1); 287/316 proteins are shared between models; 29 proteins unique to Model 1 were capturing MAL-related signal.

Outputs (gitignored, regenerated on each run):
- `Aim2_model_comparison.csv` — AUC, MSE, AIC per model
- `Aim2_lasso_coefs_m1/m2.csv` — selected proteins with β, SE, CI, p
- `Aim2_proteomic_risk_score.csv` — per-subject PRS with split membership
- `Aim2_PRS_progression_association.csv` — lm() table for PRS → progression
