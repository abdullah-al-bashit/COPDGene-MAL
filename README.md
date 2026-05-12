# COPD Journal — Analysis Project

## Project Structure

```
Journal/
├── run_analysis.sh                              # run all analyses and generate HTML reports
├── data/
│   ├── Data/MAL.csv                             # MAL scores (meanmal)
│   └── csv_exports/
│       ├── prot.csv                             # SomaScan protein abundance
│       ├── pheno_MAL.csv                        # phenotypes and covariates
│       └── metadat5k.csv                        # protein annotations
└── Aim2_prediction/
    ├── Aim2_prediction_adaptive_lasso.ipynb     # adaptive lasso notebook (main analysis)
    └── Aim2_prediction_adaptive_lasso.html      # generated report (after running)
```

---

## One-Time Setup

### 1. Install R

Download from https://cran.r-project.org/ then verify:

```bash
R --version
```

### 2. Install VS Code Extensions

```bash
code --install-extension ms-toolsai.jupyter
code --install-extension reditorsupport.r
```

### 3. Register the R Kernel for Notebooks

Open R and run:

```r
install.packages("IRkernel")
IRkernel::installspec(name = "ir-vscode", displayname = "R VS Code")
```

### 4. Install Required R Packages

```r
install.packages(c("glmnet", "pROC", "dplyr", "tibble", "readr", "here"))
```

---

## Running Analyses

All analyses are run through `run_analysis.sh` from the `Journal/` folder.

### Run everything

```bash
bash run_analysis.sh
```

Executes all registered notebooks and writes an HTML report next to each source file.

### Run a single notebook

```bash
bash run_analysis.sh Aim2_prediction/Aim2_prediction_adaptive_lasso.ipynb
```

### Add a new analysis

Open `run_analysis.sh` and add one line in the "SCRIPTS TO RUN" section:

```bash
render_notebook "NewProject/analysis.ipynb"   # for a notebook
render_rscript  "NewProject/model.R"           # for an R script
```

The HTML output is always placed next to the source file automatically.

---

## Working in VS Code

Open a notebook in VS Code, select the `R VS Code` kernel in the top-right picker, and run cells interactively. Use `Option+K` to open the Claude Code inline edit prompt inside a cell.
