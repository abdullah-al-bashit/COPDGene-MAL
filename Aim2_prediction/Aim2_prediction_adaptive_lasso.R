# =============================================================================
# ANALYSIS: Adaptive Lasso Prediction of Emphysema Progression
# Project:  COPDGene — Proteomics Aim 2 Prediction
# Author:   [Your Name]
# Date:     2026-05-12
# =============================================================================
#
# SECTION 0: BACKGROUND, METHODS, AND NOTATION
# =============================================================================
#
# CLINICAL BACKGROUND
# -------------------
# COPD (Chronic Obstructive Pulmonary Disease) is a progressive lung disease.
# Two structural features of the lung are central to this analysis:
#
#   MAL — Mechanically Affected Lung:
#     Regions of the lung parenchyma where mechanical stress from pressure
#     gradients causes structural deformation and tissue damage.
#     Quantified as a continuous score (meanmal) derived from CT imaging.
#     Higher score = greater mechanical injury = more affected lung volume.
#     Range: ~0.01 – 0.40 (unitless proportion of lung volume affected)
#
#   Emphysema — Alveolar Destruction:
#     The progressive destruction of the tiny air sacs (alveoli) in the lung.
#     CT measures this as lung density in Hounsfield Units (HU):
#       Healthy lung density: roughly -850 to -750 HU
#       Emphysematous lung:   roughly -950 to -1000 HU (more air = less dense)
#     OUTCOME: Change_lung_density_vnb_P2_P3 = CT density at Phase 3 − Phase 2
#       Negative value → lung became less dense → emphysema PROGRESSED
#       Positive value → lung became denser    → emphysema regressed / stable
#       Expected range: approximately −50 to +30 HU
#       Clinical threshold: change < 0 HU = any measurable progression
#
# PROTEOMICS BACKGROUND
# ---------------------
# SomaScan (SomaLogic) platform measures ~5,000 blood plasma proteins per subject.
# Each protein is measured as an RFU (Relative Fluorescence Unit) value:
#   RFU range: approximately 100 to 100,000 (right-skewed, log-normal distribution)
#
# We apply log2 transformation: X_log2 = log2(RFU)
#   WHY? Protein distributions are right-skewed; log2 makes them approximately normal,
#   satisfying linear model assumptions (normality, constant variance).
#   Biological meaning: a 1-unit increase on the log2 scale = a DOUBLING of protein level.
#   Example: RFU 512 → log2 = 9; RFU 1024 → log2 = 10 (twice the protein, 1-unit difference)
#
# STATISTICAL CHALLENGE
# ---------------------
# We have:
#   n ≈ 1,500 subjects (with complete Phase 2 + Phase 3 data)
#   p ≈ 4,980 proteins
# This is a high-dimensional problem (p >> n).
# Classical Ordinary Least Squares (OLS) regression:
#   β̂_OLS = (X'X)⁻¹ X'y
#   FAILS when p > n: the matrix (X'X) is singular (not invertible).
#   Even when p < n but close, OLS overfits badly and generalizes poorly.
# Solution: penalized (regularized) regression.
#
# ADAPTIVE LASSO: THEORY AND MOTIVATION
# --------------------------------------
# The Adaptive Lasso (Zou 2006, J. Am. Stat. Assoc.) solves:
#
#   argmin_β  (1/2n) ||y − Xβ||²   +   λ · Σⱼ wⱼ|βⱼ|
#             \_____________________/   \______________/
#               Data fit term             Penalty term
#
#   Notation:
#     y  ∈ ℝⁿ   : outcome vector (emphysema progression for each subject)
#     X  ∈ ℝⁿˣᵖ : predictor matrix (proteins + covariates)
#     β  ∈ ℝᵖ   : coefficient vector to estimate (one per predictor)
#     λ  ∈ (0,∞): penalty strength (tuned by cross-validation)
#     wⱼ ∈ (0,∞): protein-specific adaptive weight = 1 / |β̂ⱼ_ridge|
#
#   Data fit term: (1/2n)||y − Xβ||² = mean squared error on training data.
#     Minimizing this alone gives OLS (which overfits when p >> n).
#
#   Penalty term: λ Σⱼ wⱼ|βⱼ| — this is a WEIGHTED L1 norm.
#     The L1 penalty (|β|) has a geometric property: its minimizer tends to be
#     SPARSE (many coefficients exactly = 0). This performs variable selection.
#     The weight wⱼ = 1/|β̂ⱼ_ridge| gives proteins with strong ridge signal
#     (large |β̂ʳⁱᵈᵍᵉ|) a SMALL penalty → they survive selection.
#     Proteins with weak ridge signal get a LARGE penalty → forced to zero.
#
#   WHY ADAPTIVE over standard lasso?
#     Standard lasso (all wⱼ = 1) is inconsistent: it may retain false predictors.
#     Adaptive lasso satisfies the ORACLE PROPERTY: with correct λ, it asymptotically
#     selects exactly the true set of relevant proteins, with no false positives or negatives.
#
#   TWO-STEP PROCEDURE:
#     Step 1 — Ridge regression (L2 penalty):  fit β̂_ridge on proteins.
#              Ridge is stable even when p >> n (never produces singular solutions).
#              Use these estimates as the denominator of adaptive weights: wⱼ = 1/|β̂ⱼ_ridge|
#     Step 2 — Adaptive lasso with weights wⱼ from Step 1.
#
# TWO MODELS
# ----------
#   Model 1 (baseline): Proteins + baseline emphysema + standard covariates (NO MAL)
#   Model 2 (extended): Proteins + baseline emphysema + standard covariates + MAL
#   Comparison: does adding MAL improve emphysema progression prediction?
#
# EVALUATION METRICS
# ------------------
#   MSE (Mean Squared Error):
#     MSE = (1/n) Σᵢ (yᵢ − ŷᵢ)²
#     Unit: HU² (Hounsfield Units squared). Lower = better.
#     Example: MSE = 100 HU² → typical prediction error ≈ 10 HU
#
#   AUC (Area Under the ROC Curve):
#     AUC = P(ŷᵢ > ŷⱼ | yᵢ = 1, yⱼ = 0)
#     = probability the model assigns a higher predicted score to a true progressor
#       than to a non-progressor (random pair).
#     AUC = 0.5 → random chance; 0.7 → acceptable; 0.8 → good; 1.0 → perfect
#     Computed against binary outcome: progression_binary = I(change < 0 HU)
#
#   AIC (Akaike Information Criterion):
#     AIC = 2k − 2 ln(L̂)
#     k = number of model parameters; L̂ = maximized likelihood
#     Lower AIC = better balance of fit and parsimony.
#     Used to compare Model 1 vs Model 2 (on same dataset).
#
# PROTEOMIC RISK SCORE (PRS)
# --------------------------
#   After adaptive lasso selects k proteins, we define:
#   PRS_i = Σⱼ β̂_OLS_j · log₂(protein_{ij})    for selected proteins j = 1…k
#   Where β̂_OLS are UNBIASED post-selection OLS coefficients (see Section 8).
#   PRS_i is a single number per subject summarizing their proteomic "risk profile".
#   We standardize: PRS_scaled = (PRS − mean) / sd  → unit = 1 SD of proteomic risk.
#
# DATA SPLIT (60/20/20)
# ---------------------
#   Training (60%):    fit ridge (Step 1), fit adaptive lasso (Step 2), select proteins
#   Validation (20%):  choose lambda (λ); compare Model 1 vs Model 2
#   Test (20%):        report final AUC, MSE — never seen during fitting
#   With n ≈ 1,500:   train ≈ 900; val ≈ 300; test ≈ 300
#   set.seed(2024) ensures the same split on every run (reproducibility).
#
# =============================================================================


# =============================================================================
# SECTION 1: LOAD PACKAGES
# =============================================================================

library(glmnet)  # penalized regression: ridge (alpha=0) and lasso (alpha=1)
                 # uses coordinate descent; handles p >> n efficiently
library(pROC)    # ROC analysis and AUC estimation via DeLong's non-parametric method
library(dplyr)   # data manipulation: filter, mutate, select, inner_join
library(tibble)  # modern data frames with row names as a column
library(readr)   # fast CSV reading with correct type inference


# =============================================================================
# SECTION 2: DEFINE FILE PATHS
# =============================================================================

# All paths are relative to the project root: COPD/Journal/
# Adjust if running from a different working directory.
BASE      <- here::here()  # project root (COPD/Journal/)
PROT_PATH <- file.path(BASE, "data/csv_exports/prot.csv")        # protein RFU values
PHENO_PATH <- file.path(BASE, "data/csv_exports/pheno_MAL.csv")  # phenotypes + covariates
MAL_PATH  <- file.path(BASE, "data/Data/MAL.csv")                # MAL scores (meanmal)
META_PATH <- file.path(BASE, "data/csv_exports/metadat5k.csv")   # protein annotations
OUT_DIR   <- file.path(BASE, "Aim2_prediction")                  # output directory (this folder)

# Create output directory if it doesn't exist
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)


# =============================================================================
# SECTION 3: HELPER FUNCTIONS
# =============================================================================

# -------------------------------------------------------------------
# load_and_merge_data()
# Loads, merges, and preprocesses all data sources into a single
# analysis-ready data frame.
# Returns: list with $df (data frame) and $protein_cols (character vector of log2 protein names)
# -------------------------------------------------------------------
load_and_merge_data <- function(prot_path, pheno_path, mal_path, meta_path) {

  # Read protein abundance matrix
  # Columns: sid (subject ID), then ~4,980 proteins named X{SeqID}_{version}
  # Values: RFU (Relative Fluorescence Units), range ~100–100,000
  prot <- read_csv(prot_path, show_col_types = FALSE)

  # Read phenotype file
  # Contains: outcome (Change_lung_density_vnb_P2_P3), baseline emphysema,
  # clinical covariates, genetic PCs
  pheno <- read_csv(pheno_path, show_col_types = FALSE)

  # Read MAL scores
  # Contains: sid + meanmal (Mechanically Affected Lung score)
  mal <- read_csv(mal_path, show_col_types = FALSE) |>
    select(sid, meanmal)  # keep only what we need to avoid column collisions

  # Merge all three sources on subject ID (sid)
  # inner_join: keeps only subjects present in ALL three files
  # This is intentional — we need complete data for all sources
  df <- prot |>
    inner_join(pheno, by = "sid") |>
    inner_join(mal,   by = "sid")

  message("Rows after 3-way merge: ", nrow(df))  # expect ~3,000–5,000 before outcome filter

  # Identify protein columns: all columns starting with "X" (SomaScan naming convention)
  protein_cols <- grep("^X[0-9]", names(df), value = TRUE)
  message("Number of proteins: ", length(protein_cols))  # expect ~4,979

  # Log2-transform all protein columns
  # WHY: See SECTION 0 — RFU values are log-normal; log2 normalizes the distribution
  # and ensures a 1-unit change = doubling of protein (biologically interpretable)
  # Safety: pmax(., 1e-6) guards against log(0) if any RFU = 0 (rare but possible)
  df <- df |>
    mutate(across(all_of(protein_cols), ~ log2(pmax(., 1e-6))))

  # Rename protein columns to reflect log2 transformation
  protein_cols_log2 <- paste0("log2_", protein_cols)
  names(df)[names(df) %in% protein_cols] <- protein_cols_log2

  # Collapse rare scanner models to "Other"
  # WHY: When we split data into train/test, rare scanner models (<15 subjects)
  # may appear in test but not training → predict() fails on unseen factor levels
  scanner_counts <- table(df$scanner_model_clean_P2)
  rare_scanners  <- names(scanner_counts[scanner_counts < 15])
  df <- df |>
    mutate(scanner_model_clean_P2 = if_else(
      scanner_model_clean_P2 %in% rare_scanners, "Other", scanner_model_clean_P2
    ))

  # Define binary progression outcome for AUC computation
  # progression_binary = 1 if lung became LESS dense (= emphysema progressed)
  # progression_binary = 0 if lung stayed same or improved
  # Threshold = 0 HU: standard COPD progression definition in the literature
  df <- df |>
    mutate(progression_binary = as.integer(Change_lung_density_vnb_P2_P3 < 0))

  # Keep only complete cases for: outcome + all covariates
  # WHY: lm() and glmnet() cannot handle missing values in predictors or outcome
  # Assumption: data are Missing At Random (MAR) — the missingness is not related
  # to the values themselves after conditioning on observed variables
  required_cols <- c(
    "Change_lung_density_vnb_P2_P3", "lung_density_vnb_P2", "meanmal",
    "Age_P2", "gender", "race", "ATS_PackYears_P2", "SmokCigNow_P2",
    "scanner_model_clean_P2",
    paste0("PC", 1:5)
  )
  df <- df |> filter(if_all(all_of(required_cols), ~ !is.na(.)))

  message("Rows after complete-case filter: ", nrow(df))  # expect ~1,500–2,000

  list(df = df, protein_cols = protein_cols_log2)
}


# -------------------------------------------------------------------
# split_data()
# Randomly partitions the dataset into training (60%), validation (20%),
# and test (20%) sets using a reproducible seed.
# Returns: list with $train, $val, $test (data frames)
# -------------------------------------------------------------------
split_data <- function(df, seed = 2024, train_frac = 0.60, val_frac = 0.20) {
  # test_frac is implicit: 1 - train_frac - val_frac = 0.20

  set.seed(seed)  # fix random state → same split every run

  n       <- nrow(df)
  idx     <- sample(n)  # random permutation of row indices

  # Calculate cutpoints for 60 / 20 / 20 split
  n_train <- round(train_frac * n)            # ~60% of subjects
  n_val   <- round(val_frac   * n)            # ~20% of subjects
  # n_test  = n - n_train - n_val             # remaining ~20%

  list(
    train = df[idx[seq_len(n_train)], ],
    val   = df[idx[seq(n_train + 1, n_train + n_val)], ],
    test  = df[idx[seq(n_train + n_val + 1, n)], ]
  )
}


# -------------------------------------------------------------------
# build_design()
# Constructs the design matrix X and outcome vector y for glmnet.
# Unpenalized covariates get penalty.factor = 0 (always kept in model).
# Protein predictors get adaptive weights (or 1 if weights not yet computed).
#
# Args:
#   df               : data frame (training, validation, or test set)
#   protein_cols     : character vector of log2-protein column names
#   include_mal      : logical — TRUE adds meanmal as an unpenalized predictor (Model 2)
#   adaptive_weights : numeric vector of length = n_proteins, or NULL for uniform weights
#
# Returns: list with $X (matrix), $y (numeric), $y_bin (integer), $pf (penalty factor vector),
#          $colnames (character vector of X columns — pass as ref_colnames to val/test calls)
# -------------------------------------------------------------------
build_design <- function(df, protein_cols, include_mal = FALSE,
                         adaptive_weights = NULL, ref_colnames = NULL) {

  # Define the fixed (unpenalized) covariates
  # These are known confounders; they must always remain in the model.
  # penalty.factor = 0 means glmnet never shrinks these to zero regardless of λ.
  fixed_vars <- c(
    "lung_density_vnb_P2",    # baseline emphysema: we adjust for it to isolate PROGRESSION
    "Age_P2",                 # age: older patients progress faster
    "gender",                 # sex differences in lung physiology
    "race",                   # ancestry differences (also captured by PCs below)
    "ATS_PackYears_P2",       # cumulative smoking (pack-years): key COPD risk factor
    "SmokCigNow_P2",          # current smoking status (binary: 1=yes, 0=no)
    "scanner_model_clean_P2", # CT scanner model: different scanners produce different HU values
    paste0("PC", 1:5)         # genetic principal components 1–5: adjust for ancestry
  )

  if (include_mal) {
    # Model 2: add MAL as an unpenalized predictor
    # WHY unpenalized? MAL is the key biological variable of interest —
    # we don't want the lasso to remove it from the model by chance
    fixed_vars <- c(fixed_vars, "meanmal")
  }

  # Build formula for fixed covariates (factors will be dummy-coded automatically)
  fixed_formula <- as.formula(paste("~", paste(fixed_vars, collapse = " + "), "- 1"))

  # model.matrix() creates the numeric design matrix:
  #   - Factors → dummy variables (treatment coding; reference level dropped automatically)
  #   - "-1" removes the intercept column (glmnet handles intercept internally)
  X_fixed <- model.matrix(fixed_formula, data = df)
  n_fixed  <- ncol(X_fixed)  # number of fixed-covariate columns (including dummies)

  # Extract protein matrix (already log2-transformed)
  X_prot <- as.matrix(df[, protein_cols, drop = FALSE])

  # Combine: fixed covariates first, then proteins
  X <- cbind(X_fixed, X_prot)

  # Align columns to training reference when building val/test matrices.
  # WHY: model.matrix() expands factors based on levels present in each split.
  # If a rare factor level (e.g. a scanner model) is absent from val/test,
  # its dummy column is missing → column count mismatches → predict() fails.
  # Fix: add zero-filled columns for any level absent in this split,
  #      then reorder to exactly match the training column order.
  if (!is.null(ref_colnames)) {
    missing_cols <- setdiff(ref_colnames, colnames(X))
    if (length(missing_cols) > 0) {
      zero_mat <- matrix(0, nrow = nrow(X), ncol = length(missing_cols),
                         dimnames = list(NULL, missing_cols))
      X <- cbind(X, zero_mat)
    }
    X <- X[, ref_colnames, drop = FALSE]  # reorder to match training exactly
  }

  # Build penalty.factor vector (length = ncol(X))
  # Fixed covariates: penalty = 0 → always retained
  # Proteins: penalty = adaptive weight (or 1 if weights not yet computed)
  if (is.null(adaptive_weights)) {
    pf_prot <- rep(1, length(protein_cols))  # uniform weight = standard lasso
  } else {
    # adaptive_weights[j] = 1/|β̂_ridge_j|: higher = harsher penalty
    pf_prot <- adaptive_weights
  }
  pf <- c(rep(0, n_fixed), pf_prot)

  list(
    X        = X,
    y        = df$Change_lung_density_vnb_P2_P3,  # continuous outcome (HU)
    y_bin    = df$progression_binary,              # binary outcome (0/1) for AUC
    pf       = pf,
    n_fixed  = n_fixed,        # number of covariate columns (proteins start after this)
    colnames = colnames(X)     # return column names so val/test can align to these
  )
}


# -------------------------------------------------------------------
# compute_adaptive_weights()
# Step 1 of adaptive lasso: fits ridge regression on proteins only
# (with covariate signal removed) and returns adaptive penalty weights.
#
# WHY residualize first?
#   We want weights that reflect protein-specific predictive signal.
#   If we fit ridge on all predictors together, the weights conflate
#   covariate effects with protein effects. Residualizing removes
#   the covariate contribution from y before protein estimation.
#
# Args:
#   X_prot_train : matrix of log2-protein values (training set, n × p_prot)
#   y_resid_train: residuals from lm(y ~ fixed_covariates) on training set
#
# Returns: numeric vector of adaptive weights (length = p_prot)
# -------------------------------------------------------------------
compute_adaptive_weights <- function(X_prot_train, y_resid_train) {

  # Ridge regression (alpha = 0): L2 penalty Σβⱼ²
  # Does NOT perform variable selection (all coefficients remain non-zero)
  # Stable even when p >> n because it adds λI to (X'X), making it invertible
  # 10-fold cross-validation to select λ that minimizes out-of-fold MSE
  ridge_cv <- cv.glmnet(
    x       = X_prot_train,
    y       = y_resid_train,
    alpha   = 0,        # alpha=0 → pure ridge (L2 penalty)
    nfolds  = 10,       # 10-fold CV within training set
    standardize = TRUE  # standardize predictors internally (best practice for ridge)
  )

  # Extract coefficients at the lambda that minimized CV error
  # coef() returns a (p+1 × 1) sparse matrix: intercept + p protein coefficients
  # [-1] drops the intercept (we only need protein coefficients for weights)
  ridge_coef <- as.vector(coef(ridge_cv, s = "lambda.min"))[-1]

  # Adaptive weights: wⱼ = 1 / |β̂_ridge_j|
  # Strong predictors (large |β̂|) → small weight → lasso is lenient → keeps them
  # Weak predictors   (small |β̂|) → large weight → lasso is harsh  → zeros them
  # pmax(..., 1e-10): numerical floor prevents division-by-zero if any β̂ ≈ 0
  weights <- 1 / pmax(abs(ridge_coef), 1e-10)

  weights
}


# -------------------------------------------------------------------
# fit_adaptive_lasso()
# Step 2 of adaptive lasso: fits L1-penalized regression with
# per-predictor adaptive weights and selects λ via cross-validation.
#
# Args:
#   X              : full design matrix (fixed covariates + proteins)
#   y              : outcome vector
#   penalty_factor : penalty.factor vector (0 for covariates, wⱼ for proteins)
#   nfolds         : number of CV folds (default 10)
#
# Returns: cv.glmnet object (use coef(., s="lambda.min") to get coefficients)
# -------------------------------------------------------------------
fit_adaptive_lasso <- function(X, y, penalty_factor, nfolds = 10) {

  # cv.glmnet with alpha = 1: pure lasso (L1 penalty Σwⱼ|βⱼ|)
  # L1 geometry: the unit ball ||β||₁ ≤ t has CORNERS at the coordinate axes.
  # The optimal solution often falls at a corner → many βⱼ = exactly 0 (sparse).
  # This is why lasso performs VARIABLE SELECTION, while ridge (L2) does not.
  # penalty.factor: scales λ per predictor → adaptive penalty
  # standardize = TRUE: glmnet standardizes predictors internally for numerical stability
  #   (coefficients are returned on the ORIGINAL scale, not standardized scale)
  cv.glmnet(
    x              = X,
    y              = y,
    alpha          = 1,              # alpha=1 → lasso (L1 penalty)
    penalty.factor = penalty_factor, # per-predictor weights (0 = unpenalized)
    nfolds         = nfolds,
    standardize    = TRUE
  )
}


# -------------------------------------------------------------------
# evaluate_on_split()
# Computes MSE and AUC for a fitted glmnet model on a new data split.
# Used on validation set (to select λ and compare models) and test set (final metrics).
#
# Args:
#   model    : cv.glmnet object
#   X_new    : design matrix for the new split
#   y_new    : continuous outcome for the new split
#   y_bin    : binary outcome for AUC computation
#   lambda   : the λ value at which to evaluate (e.g., "lambda.min" or "lambda.1se")
#
# Returns: list with $mse (numeric) and $auc (numeric)
# -------------------------------------------------------------------
evaluate_on_split <- function(model, X_new, y_new, y_bin, lambda) {

  # Generate out-of-sample predictions
  # predict() applies the linear model β̂(λ) to new X: ŷ = Xβ̂
  y_pred <- as.vector(predict(model, newx = X_new, s = lambda))

  # MSE: measures average squared prediction error
  # Lower MSE → predictions are closer to true values on average
  mse <- mean((y_new - y_pred)^2)

  # AUC: measures discriminative ability for binary progression
  # pROC::roc() uses the trapezoidal rule to estimate the ROC curve area
  # direction = "<": higher predicted value → predicted to NOT progress (convention here)
  # We use quiet=TRUE to suppress the direction message
  roc_obj <- pROC::roc(y_bin, y_pred, direction = "<", quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))

  list(mse = mse, auc = auc_val)
}


# -------------------------------------------------------------------
# refit_ols()
# Post-selection OLS refit on the variables selected by adaptive lasso.
# Lasso coefficients are biased (shrunk toward zero by design).
# After lasso identifies WHICH predictors to include, we refit OLS
# on those predictors to get unbiased β̂, standard errors, CIs, and p-values.
#
# Post-selection inference caveat: the selected variables are correlated
# with the outcome by construction, so p-values are anti-conservative
# (inflated Type I error). Interpret as descriptive, not confirmatory.
#
# Args:
#   X_trainval   : design matrix (train + validation combined, n_tv × p)
#   y_trainval   : outcome vector (train + validation combined)
#   lasso_model  : cv.glmnet object
#   lambda       : chosen λ value
#   protein_cols : protein column names (to annotate output)
#   n_fixed      : number of fixed-covariate columns in X (to separate from proteins)
#
# Returns: data frame with columns: predictor, beta, se, ci_lower, ci_upper, p, AIC
# -------------------------------------------------------------------
refit_ols <- function(X_trainval, y_trainval, lasso_model, lambda,
                      protein_cols, n_fixed) {

  # Extract coefficient vector at chosen λ
  # coef() returns a sparse (p+1 × 1) matrix; drop intercept (row 1)
  lasso_coef <- as.vector(coef(lasso_model, s = lambda))[-1]

  # Identify which predictors have non-zero lasso coefficients (= "selected")
  # The first n_fixed coefficients are always non-zero (penalty = 0, unpenalized)
  # We only look at protein columns for selection
  protein_indices   <- seq(n_fixed + 1, length(lasso_coef))
  selected_proteins <- protein_indices[lasso_coef[protein_indices] != 0]

  # Combine: all fixed covariate indices + selected protein indices
  selected_all <- c(seq_len(n_fixed), selected_proteins)

  message("Proteins selected by lasso: ", length(selected_proteins))

  # Subset X to selected predictors only
  X_sel <- X_trainval[, selected_all, drop = FALSE]

  # Fit OLS on selected predictors
  # WHY OLS here? lm() produces Maximum Likelihood estimates: β̂_OLS = (X'X)⁻¹ X'y
  # These are UNBIASED (E[β̂] = β_true) and BLUE (Best Linear Unbiased Estimator, Gauss-Markov)
  ols_fit <- lm(y_trainval ~ X_sel)  # intercept is included by default (correct)

  # Extract coefficient summary: β̂, SE, t-statistic, p-value
  ols_summary <- summary(ols_fit)$coefficients  # matrix: rows = predictors, cols = Est, SE, t, p

  # Extract 95% confidence intervals
  # CI = β̂ ± t_{n-k, 0.975} × SE  (based on t-distribution with n-k degrees of freedom)
  ols_ci <- confint(ols_fit, level = 0.95)

  # AIC for model comparison
  # AIC = 2k − 2 ln(L̂): penalizes both poor fit and excessive parameters
  # Lower AIC → better model (balances fit vs. complexity)
  ols_aic <- AIC(ols_fit)

  # Build output table using rownames from ols_summary as predictor labels.
  # WHY rownames and not colnames(X_sel)? lm() silently drops rank-deficient
  # columns (perfect collinearity) and omits them from summary — so
  # nrow(ols_summary) may be less than ncol(X_sel)+1. Using rownames keeps
  # predictor, beta, CI, and p perfectly in sync.
  # Strip the "X_sel" prefix lm() automatically prepends to matrix column names
  # so that predictor names match the original protein_cols names downstream.
  clean_names <- sub("^X_sel", "", rownames(ols_summary))
  result <- tibble(
    predictor = clean_names,
    beta      = ols_summary[, "Estimate"],
    se        = ols_summary[, "Std. Error"],
    ci_lower  = ols_ci[rownames(ols_summary), 1],  # index by original rownames
    ci_upper  = ols_ci[rownames(ols_summary), 2],  # (before stripping prefix)
    p_value   = ols_summary[, "Pr(>|t|)"],
    AIC       = ols_aic   # model-level statistic, same value for all rows
  )

  result
}


# -------------------------------------------------------------------
# compute_prs()
# Computes the Proteomic Risk Score (PRS) for all subjects.
#
# PRS_i = Σⱼ β̂_OLS_j × log2(protein_{ij})   for selected proteins j
#
# Args:
#   df           : full merged data frame (all subjects)
#   ols_coefs    : output from refit_ols() — contains predictor names and β̂
#   protein_cols : log2-protein column names in df
#
# Returns: numeric vector of standardized PRS (PRS_scaled) for all subjects
# -------------------------------------------------------------------
compute_prs <- function(df, ols_coefs, protein_cols) {

  # Identify which predictors in the OLS coefficient table are proteins
  # (as opposed to fixed covariates like age, sex, etc.)
  protein_rows <- ols_coefs |>
    filter(predictor %in% protein_cols)  # keep only protein rows

  if (nrow(protein_rows) == 0) {
    stop("No protein coefficients found in OLS output. Check predictor names.")
  }

  # Extract the protein names and their OLS β̂ values
  selected_proteins <- protein_rows$predictor
  protein_betas     <- setNames(protein_rows$beta, selected_proteins)

  # Subset protein matrix to selected proteins
  X_prot_sel <- as.matrix(df[, selected_proteins, drop = FALSE])

  # PRS: weighted sum of log2 protein levels (matrix multiplication)
  # PRS_i = Σⱼ β̂ⱼ × log2(protein_ij)
  # %*% performs matrix multiplication: (n × k) %*% (k × 1) = (n × 1)
  prs_raw <- as.vector(X_prot_sel %*% protein_betas)

  # Standardize PRS: mean = 0, sd = 1
  # WHY? Makes the regression coefficient of PRS interpretable:
  # β_PRS = expected HU change per 1-SD increase in proteomic risk
  prs_scaled <- (prs_raw - mean(prs_raw, na.rm = TRUE)) / sd(prs_raw, na.rm = TRUE)

  prs_scaled
}


# =============================================================================
# SECTION 4: MAIN ANALYSIS PIPELINE
# =============================================================================

# --- 4.1: Load and preprocess all data ---
message("\n=== Loading and preprocessing data ===")
data_obj    <- load_and_merge_data(PROT_PATH, PHENO_PATH, MAL_PATH, META_PATH)
df_full     <- data_obj$df           # full analysis data frame
protein_cols <- data_obj$protein_cols # log2 protein column names (~4,979 columns)


# --- 4.2: Split into train / validation / test ---
message("\n=== Splitting data 60/20/20 ===")
splits <- split_data(df_full, seed = 2024, train_frac = 0.60, val_frac = 0.20)

message("Train n: ", nrow(splits$train))  # expect ~900
message("Val   n: ", nrow(splits$val))    # expect ~300
message("Test  n: ", nrow(splits$test))   # expect ~300


# --- 4.3: Build design matrices for training set ---
# First call uses uniform weights (adaptive_weights = NULL) to get the
# protein matrix needed for ridge regression in the next step.
message("\n=== Building training design matrices ===")

# Model 1: no MAL (include_mal = FALSE)
design_m1_train <- build_design(splits$train, protein_cols,
                                include_mal = FALSE, adaptive_weights = NULL)

# Model 2: with MAL (include_mal = TRUE)
design_m2_train <- build_design(splits$train, protein_cols,
                                include_mal = TRUE, adaptive_weights = NULL)


# --- 4.4: Compute adaptive weights from ridge regression ---
# Residualize y w.r.t. fixed covariates to isolate protein-specific signal
message("\n=== Computing adaptive weights via ridge regression ===")

# Extract the covariate-only portion of the design matrix (first n_fixed columns)
X_cov_train  <- design_m1_train$X[, seq_len(design_m1_train$n_fixed), drop = FALSE]
X_prot_train <- design_m1_train$X[, -seq_len(design_m1_train$n_fixed), drop = FALSE]
y_train      <- design_m1_train$y

# Residualize: remove covariate signal from y before ridge estimation
# lm.fit() is the fast internal fitting function (no formula parsing overhead)
cov_lm   <- lm.fit(X_cov_train, y_train)
y_resid  <- residuals(cov_lm)  # y with covariate effects removed

# Compute adaptive weights (returned as vector of length n_proteins)
adaptive_weights <- compute_adaptive_weights(X_prot_train, y_resid)

message("Adaptive weights: min = ", round(min(adaptive_weights), 4),
        "  max = ", round(max(adaptive_weights), 4))
message("Infinite weights (check): ", sum(is.infinite(adaptive_weights)))  # should be 0


# --- 4.5: Rebuild design matrices with adaptive weights ---
message("\n=== Rebuilding design matrices with adaptive weights ===")

# Training — build first to get reference column names
design_m1_train <- build_design(splits$train, protein_cols,
                                include_mal = FALSE, adaptive_weights = adaptive_weights)
design_m2_train <- build_design(splits$train, protein_cols,
                                include_mal = TRUE,  adaptive_weights = adaptive_weights)

# Validation — pass ref_colnames from training so columns always align
# This prevents predict() errors when a factor level (e.g. rare scanner) is
# absent from the validation or test split
design_m1_val <- build_design(splits$val, protein_cols,
                              include_mal = FALSE, adaptive_weights = adaptive_weights,
                              ref_colnames = design_m1_train$colnames)
design_m2_val <- build_design(splits$val, protein_cols,
                              include_mal = TRUE,  adaptive_weights = adaptive_weights,
                              ref_colnames = design_m2_train$colnames)

# Test — same column alignment as training
design_m1_test <- build_design(splits$test, protein_cols,
                               include_mal = FALSE, adaptive_weights = adaptive_weights,
                               ref_colnames = design_m1_train$colnames)
design_m2_test <- build_design(splits$test, protein_cols,
                               include_mal = TRUE,  adaptive_weights = adaptive_weights,
                               ref_colnames = design_m2_train$colnames)


# --- 4.6: Fit adaptive lasso models on training set ---
message("\n=== Fitting adaptive lasso — Model 1 (no MAL) ===")
lasso_m1 <- fit_adaptive_lasso(design_m1_train$X, design_m1_train$y,
                               design_m1_train$pf)

message("\n=== Fitting adaptive lasso — Model 2 (+MAL) ===")
lasso_m2 <- fit_adaptive_lasso(design_m2_train$X, design_m2_train$y,
                               design_m2_train$pf)


# --- 4.7: Validation — choose lambda and compare models ---
message("\n=== Validation: lambda selection and model comparison ===")

# Evaluate both lambda choices for each model on the validation set
# lambda.min: minimizes CV-MSE → more proteins selected, better in-sample fit
# lambda.1se: most parsimonious within 1 SE of min → fewer proteins, more conservative
val_m1_min  <- evaluate_on_split(lasso_m1, design_m1_val$X,
                                 design_m1_val$y, design_m1_val$y_bin, "lambda.min")
val_m1_1se  <- evaluate_on_split(lasso_m1, design_m1_val$X,
                                 design_m1_val$y, design_m1_val$y_bin, "lambda.1se")
val_m2_min  <- evaluate_on_split(lasso_m2, design_m2_val$X,
                                 design_m2_val$y, design_m2_val$y_bin, "lambda.min")
val_m2_1se  <- evaluate_on_split(lasso_m2, design_m2_val$X,
                                 design_m2_val$y, design_m2_val$y_bin, "lambda.1se")

# Print validation comparison
message("Model 1 lambda.min — MSE: ", round(val_m1_min$mse, 3),
        "  AUC: ", round(val_m1_min$auc, 3))
message("Model 1 lambda.1se — MSE: ", round(val_m1_1se$mse, 3),
        "  AUC: ", round(val_m1_1se$auc, 3))
message("Model 2 lambda.min — MSE: ", round(val_m2_min$mse, 3),
        "  AUC: ", round(val_m2_min$auc, 3))
message("Model 2 lambda.1se — MSE: ", round(val_m2_1se$mse, 3),
        "  AUC: ", round(val_m2_1se$auc, 3))

# Select lambda: choose lambda.min if its validation AUC > lambda.1se, else lambda.1se
# This is a data-driven choice based on the validation set (not the test set)
lambda_m1 <- if (val_m1_min$auc >= val_m1_1se$auc) "lambda.min" else "lambda.1se"
lambda_m2 <- if (val_m2_min$auc >= val_m2_1se$auc) "lambda.min" else "lambda.1se"

message("Selected lambda for Model 1: ", lambda_m1)
message("Selected lambda for Model 2: ", lambda_m2)


# --- 4.8: OLS refit on train + validation combined ---
# Using train+val (80% of data) before final test evaluation maximises the sample
# size for stable coefficient estimation and AIC comparison.
message("\n=== OLS refit on train + validation ===")

# Combine train and validation sets
df_trainval <- bind_rows(splits$train, splits$val)

design_m1_tv <- build_design(df_trainval, protein_cols,
                             include_mal = FALSE, adaptive_weights = adaptive_weights,
                             ref_colnames = design_m1_train$colnames)
design_m2_tv <- build_design(df_trainval, protein_cols,
                             include_mal = TRUE,  adaptive_weights = adaptive_weights,
                             ref_colnames = design_m2_train$colnames)

# Refit OLS using proteins selected by each model
ols_m1 <- refit_ols(design_m1_tv$X, design_m1_tv$y, lasso_m1, lambda_m1,
                    protein_cols, design_m1_tv$n_fixed)
ols_m2 <- refit_ols(design_m2_tv$X, design_m2_tv$y, lasso_m2, lambda_m2,
                    protein_cols, design_m2_tv$n_fixed)


# --- 4.9: Final test set evaluation (REPORTED METRICS) ---
message("\n=== Final test set evaluation ===")

test_m1 <- evaluate_on_split(lasso_m1, design_m1_test$X,
                             design_m1_test$y, design_m1_test$y_bin, lambda_m1)
test_m2 <- evaluate_on_split(lasso_m2, design_m2_test$X,
                             design_m2_test$y, design_m2_test$y_bin, lambda_m2)

# Count selected proteins per model (from OLS refit output, excluding covariate rows)
n_prot_m1 <- ols_m1 |>
  filter(predictor %in% protein_cols) |>  # rows that are proteins (not covariates)
  nrow()

n_prot_m2 <- ols_m2 |>
  filter(predictor %in% protein_cols) |>
  nrow()

# AIC values from OLS refit (model-level statistic, same for all rows)
aic_m1 <- unique(ols_m1$AIC)
aic_m2 <- unique(ols_m2$AIC)

# Build comparison table
comparison_table <- tibble(
  model              = c("Model 1 (no MAL)", "Model 2 (+MAL)"),
  n_proteins_selected = c(n_prot_m1, n_prot_m2),
  test_MSE           = c(test_m1$mse, test_m2$mse),
  test_AUC           = c(test_m1$auc, test_m2$auc),
  AIC_OLS_refit      = c(aic_m1, aic_m2),
  lambda_chosen      = c(lambda_m1, lambda_m2)
)

print(comparison_table)


# --- 4.10: Proteomic Risk Score ---
message("\n=== Computing Proteomic Risk Score (PRS) ===")

# PRS derived from Model 2 (the model that includes MAL)
# Applied to ALL subjects in the full dataset (not just test)
prs_scaled <- compute_prs(df_full, ols_m2, protein_cols)

# Verify PRS distribution (should be approximately Normal)
message("PRS: mean = ", round(mean(prs_scaled), 3),
        "  sd = ", round(sd(prs_scaled), 3),
        "  range: [", round(min(prs_scaled), 2), ", ", round(max(prs_scaled), 2), "]")

# Associate PRS with emphysema progression (full cohort)
# Covariates included to verify PRS association is independent of known confounders
prs_model <- lm(
  Change_lung_density_vnb_P2_P3 ~ prs_scaled + lung_density_vnb_P2 +
    Age_P2 + gender + race + ATS_PackYears_P2 + SmokCigNow_P2 +
    scanner_model_clean_P2 + PC1 + PC2 + PC3 + PC4 + PC5,
  data = df_full
)

# Extract PRS association results
prs_summary <- summary(prs_model)$coefficients
prs_ci      <- confint(prs_model)

# Build PRS association table
prs_assoc_table <- tibble(
  predictor = rownames(prs_summary),
  beta      = prs_summary[, "Estimate"],
  se        = prs_summary[, "Std. Error"],
  ci_lower  = prs_ci[, 1],
  ci_upper  = prs_ci[, 2],
  p_value   = prs_summary[, "Pr(>|t|)"]
)

# Report PRS row specifically
prs_row <- filter(prs_assoc_table, predictor == "prs_scaled")
message("PRS β = ", round(prs_row$beta, 3),
        "  95% CI: [", round(prs_row$ci_lower, 3), ", ", round(prs_row$ci_upper, 3), "]",
        "  p = ", signif(prs_row$p_value, 3))


# =============================================================================
# SECTION 5: SAVE OUTPUTS
# =============================================================================

# Model comparison table (primary result: AUC, MSE, AIC for Model 1 vs Model 2)
write_csv(comparison_table,
          file.path(OUT_DIR, "Aim2_model_comparison.csv"))
message("Saved: Aim2_model_comparison.csv")

# Model 1 OLS coefficients: selected proteins + β, SE, CI, p, AIC
write_csv(ols_m1,
          file.path(OUT_DIR, "Aim2_lasso_coefs_m1.csv"))
message("Saved: Aim2_lasso_coefs_m1.csv")

# Model 2 OLS coefficients: selected proteins + β, SE, CI, p, AIC (includes MAL row)
write_csv(ols_m2,
          file.path(OUT_DIR, "Aim2_lasso_coefs_m2.csv"))
message("Saved: Aim2_lasso_coefs_m2.csv")

# Per-subject proteomic risk score with outcome and split membership
prs_output <- tibble(
  sid                          = df_full$sid,
  prs_scaled                   = prs_scaled,
  Change_lung_density_vnb_P2_P3 = df_full$Change_lung_density_vnb_P2_P3,
  progression_binary           = df_full$progression_binary,
  split = case_when(
    df_full$sid %in% splits$train$sid ~ "train",
    df_full$sid %in% splits$val$sid   ~ "validation",
    df_full$sid %in% splits$test$sid  ~ "test"
  )
)
write_csv(prs_output,
          file.path(OUT_DIR, "Aim2_proteomic_risk_score.csv"))
message("Saved: Aim2_proteomic_risk_score.csv")

# PRS → emphysema progression association table (full lm() results)
write_csv(prs_assoc_table,
          file.path(OUT_DIR, "Aim2_PRS_progression_association.csv"))
message("Saved: Aim2_PRS_progression_association.csv")

message("\n=== Analysis complete ===")
