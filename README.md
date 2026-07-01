# COPDGene MAL Omics: Proteomics Analysis

Plasma proteomics study of Mechanically Affected Lung (MAL) and emphysema progression in COPDGene.
SomaScan aptamer-based platform (~4,979 proteins), n = 1,306 subjects with complete Phase 2 proteomics, phenotype, and MAL data.

## Overview

**MAL** (Mechanically Affected Lung) is a CT-derived continuous score (0.01–0.40) measuring the fraction of lung volume under destructive mechanical stress. This analysis asks: does mechanical lung stress leave a detectable signal in blood proteomics, and can that signal predict who will develop worse emphysema?

**Outcome:** CT lung density change Phase 3 − Phase 2 (HU). Negative = emphysema progressed.

## Key Results

| Analysis | Result |
|---|---|
| Differential expression | 14 of 4,979 proteins associated with MAL (FDR < 0.05) |
| GSEA | 11 significant pathways; depleted metabolic, enriched inflammatory |
| Proteomic Risk Score (proRS) | 572-protein adaptive LASSO; test R² = 0.233 |
| AIC improvement (M3 vs M1) | ΔAIC = −13.5 |
| Causal mediation | NIE = 13.93 HU (p = 0.002); complete mediation |

## Repository Structure

```
Journal/
├── MAL_omics_proteomics.ipynb    ← main analysis notebook (R, ir-vscode kernel)
├── MAL_omics_proteomics.html     ← rendered notebook report
├── report/
│   ├── MAL_omics_manuscript.tex  ← LaTeX source
│   ├── MAL_omics_manuscript.pdf  ← compiled manuscript
│   └── MAL_omics_manuscript.docx ← Word export
├── data/                         ← PHI/HIPAA — gitignored
├── figures/                      ← generated figures — gitignored
├── results/                      ← generated CSVs — gitignored
└── string_data/                  ← STRING database cache — gitignored
```

## Running the Analysis

From the `Journal/` root:

```bash
/Applications/JupyterLab.app/Contents/Resources/jlab_server/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=ir-vscode \
  --ExecutePreprocessor.timeout=7200 \
  MAL_omics_proteomics.ipynb
```

R kernel registration (one-time):

```r
IRkernel::installspec(name = "ir-vscode", displayname = "R VS Code")
```

## Compiling the Manuscript

```bash
cd report
latexmk -pdf MAL_omics_manuscript.tex
pandoc MAL_omics_manuscript.tex -o MAL_omics_manuscript.docx
```

## Data

Patient-level data lives in `data/` (gitignored; PHI/HIPAA).

| File | Contents |
|---|---|
| `prot.csv` | SomaScan RFU (~4,979 proteins × 5,670 subjects) |
| `pheno_MAL.csv` | Phenotypes, covariates, outcomes |
| `metadat5k.csv` | Protein annotations (seqID, gene symbol, target name) |
| `MAL.csv` | MAL scores (sid, meanmal) |
| `CellData.csv` | BMI and comorbidity data (Phase 2) |
