# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COPDGene multi-omics analysis studying Mechanically Affected Lung (MAL) and emphysema progression. All analyses use SomaScan blood proteomics (~4,979 proteins) from the COPDGene cohort.

**MAL** = Mechanically Affected Lung: CT-derived continuous score (0.01–0.40) measuring lung volume under destructive mechanical stress.  
**Outcome** = `Change_lung_density_vnb_P2_P3` (HU): Phase 3 − Phase 2 CT density. Negative = emphysema progressed.

## Directory Structure

```
Journal/
├── CLAUDE.md
├── MAL_omics_proteomics.ipynb    ← main analysis notebook (ir-vscode kernel)
├── MAL_omics_proteomics.html     ← rendered notebook report
├── report/                   ← MAL_omics_manuscript.tex/.pdf/.docx
├── data/                         ← PHI/HIPAA — gitignored
├── figures/                      ← generated PDF/PNG — gitignored
├── results/                      ← generated CSVs — gitignored
└── string_data/                  ← STRING database gz files — gitignored
```

## Data (never commit — PHI/HIPAA)

Lives in `data/` — gitignored.

| File | Contents |
|------|----------|
| `prot.csv` | SomaScan RFU (~5k proteins × 5,670 subjects) |
| `pheno_MAL.csv` | Phenotypes, covariates, outcomes |
| `metadat5k.csv` | Protein annotations (seqID, EntrezGeneSymbol, Target) |
| `MAL.csv` | MAL scores (sid, meanmal, pctprog, pcthighmal) |
| `CellData.csv` | BMI and comorbidity data (Phase 2) |

Key columns: `sid`, `Change_lung_density_vnb_P2_P3`, `lung_density_vnb_P2`, `meanmal`, `Age_P2`, `gender`, `race`, `ATS_PackYears_P2`, `SmokCigNow_P2`, `scanner_model_clean_P2`, `PC1`–`PC5`.

Protein columns: `X{seqID}` (raw RFU) → `log2_X{seqID}` (after log2 transform). Join metadata on `seqID = sub("^log2_X", "", predictor)`.

## Running the Notebook

From `Journal/` root (so `here::here()` resolves correctly):
```bash
/Applications/JupyterLab.app/Contents/Resources/jlab_server/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=ir-vscode \
  --ExecutePreprocessor.timeout=7200 \
  MAL_omics_proteomics.ipynb
```

R kernel: `ir-vscode` (registered via `IRkernel::installspec(name="ir-vscode", displayname="R VS Code")`).

## Notebook Path Variables (cell 2, setup)

```r
BASE       <- here::here()           # Journal/
DATA_DIR   <- file.path(BASE, "data")
FIG_DIR    <- file.path(BASE, "figures")
RES_DIR    <- file.path(BASE, "results")
STRING_DIR <- file.path(BASE, "string_data")
```

## Compiling the Manuscript

```bash
cd report
latexmk -pdf MAL_omics_manuscript.tex
```

Figures resolve via `\graphicspath{{../figures/}}` — always compile from `report/`.

Generate DOCX:
```bash
cd report
pandoc MAL_omics_manuscript.tex -o MAL_omics_manuscript.docx
```

## Statistical Conventions

- **Log2-transform** all SomaScan RFU values before modelling: `log2(pmax(x, 1e-6))`
- **Adaptive lasso**: ridge pre-step on residualised outcome → weights `wⱼ = 1/|β̂_ridge_j|` → cv.glmnet with `alpha=1, penalty.factor=wⱼ`
- **Covariates always unpenalized**: `penalty.factor=0` for `lung_density_vnb_P2`, `Age_P2`, `gender`, `race`, `ATS_PackYears_P2`, `SmokCigNow_P2`, `scanner_model_clean_P2`, `PC1`–`PC5`
- **Design matrix alignment**: use `ref_colnames` in `build_design()` to zero-fill missing factor-level dummy columns in val/test splits
- **Post-selection OLS**: refit OLS on lasso-selected proteins for unbiased β, SE, CI, p-values
- **AIC**: improvement of ≥ 2 units (lower) is meaningful
- **Seeds**: `set.seed(42)` before the 70/30 split AND before `fit_adaptive_lasso()`

## MAL Omics Analysis Pipeline

`MAL_omics_proteomics.ipynb` — 38 cells, n = 1,306 subjects:

1. **Table 1** — `gtsummary::tbl_summary()`
2. **Differential expression** — `run_protein_lm()` over 4,979 proteins; BH FDR; 14 significant
3. **Sensitivity** — DE re-run adjusting for CAD/CHF; all 14 proteins robust
4. **GSEA** — `fgsea` + `msigdbr`; Hallmark, KEGG, Reactome; 11 significant pathways
5. **STRING network** — `STRINGdb` v11.5 on 14 proteins; `input_directory = STRING_DIR`
6. **MCL clustering** — 2 clusters with n ≥ 3
7. **Adaptive LASSO → proRS** — outcome = `meanmal`; 572 proteins; test R² = 0.233; `set.seed(42)` before split AND lasso fit
8. **AIC comparison** — M1 clinical / M2 proRS+scanner / M3 clinical+proRS; M3 DAIC = -13.5
9. **Figure 3** — decile dot plot; D10/D1 = 3.1x, p < 0.001
10. **Causal mediation** — `medflex`; MAL → proRS → progression; NDE = -0.55 (p = 0.941), NIE = 13.93 (p = 0.002); complete mediation
11. **Per-protein mediation** — 572 proteins; 199 converged; 10 with NIE p < 0.05
12. **Save outputs** — CSV to `RES_DIR`, figures to `FIG_DIR`

Key result numbers: 572 proteins, R² = 0.233, M3 DAIC = -13.5, NIE = 13.93 HU (p = 0.002).
