suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(readr); library(purrr)
  library(glmnet); library(broom); library(ggplot2); library(stringr)
})

OUT_DIR       <- "/Users/bashit.a/Documents/COPD/Journal/MAL_omics"
PROT_PATH     <- "/Users/bashit.a/Documents/COPD/Journal/data/csv_exports/prot.csv"
PHENO_PATH    <- "/Users/bashit.a/Documents/COPD/Journal/data/csv_exports/pheno_MAL.csv"
MAL_PATH      <- "/Users/bashit.a/Documents/COPD/Journal/data/Data/MAL.csv"
META_PATH     <- "/Users/bashit.a/Documents/COPD/Journal/data/csv_exports/metadat5k.csv"
CELLDATA_PATH <- "/Users/bashit.a/Documents/COPD/Journal/data/csv_exports/CellData.csv"

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading data...\n")
prot     <- read_csv(PROT_PATH,     show_col_types = FALSE)
pheno    <- read_csv(PHENO_PATH,    show_col_types = FALSE)
mal      <- read_csv(MAL_PATH,      show_col_types = FALSE)
bmi_p2   <- read_csv(CELLDATA_PATH, show_col_types = FALSE) |>
  filter(Phase_study == 2) |>
  select(sid, BMI_P2 = BMI)

df_raw <- prot |>
  inner_join(pheno, by = "sid") |>
  inner_join(mal,   by = "sid") |>
  left_join(bmi_p2, by = "sid")

# Log2 transform proteins
prot_raw_cols <- grep("^X[0-9]", names(df_raw), value = TRUE)
for (col in prot_raw_cols) {
  df_raw[[paste0("log2_", col)]] <- log2(pmax(df_raw[[col]], 1e-6))
}
protein_cols <- paste0("log2_", prot_raw_cols)

# Complete-case filter
required_cols <- c(
  "Change_lung_density_vnb_P2_P3", "lung_density_vnb_P2", "meanmal",
  "Age_P2", "gender", "race", "ATS_PackYears_P2", "SmokCigNow_P2",
  "BMI_P2", "scanner_model_clean_P2",
  "fev1_pp_gli_global_P2", "FEV1_FVC_post_P2",
  paste0("PC", 1:5)   # keeps n=1,306 consistent with notebook population
)
df_full <- df_raw |> filter(if_all(all_of(required_cols), ~!is.na(.)))
cat("n after complete-case:", nrow(df_full), "\n")

# ── Covariate sets ────────────────────────────────────────────────────────────
# Elastic net training: project out age/sex/race/smoking/pack-years/scanner.
# BMI excluded from training — the clinical comparison model carries it.
prog_fixed_vars <- c("Age_P2", "gender", "race", "ATS_PackYears_P2",
                     "SmokCigNow_P2", "scanner_model_clean_P2")

# Association model and Figure 3: per guideline spec
assoc_vars <- paste(c("Age_P2", "gender", "race", "ATS_PackYears_P2",
                      "SmokCigNow_P2", "BMI_P2", "scanner_model_clean_P2"),
                    collapse = " + ")

# AIC clinical model: per guideline spec (no scanner, no CT vars)
aic_clin_vars <- paste(c("Age_P2", "gender", "race", "ATS_PackYears_P2",
                          "SmokCigNow_P2", "BMI_P2"),
                       collapse = " + ")

# ── Helper functions ──────────────────────────────────────────────────────────
build_design <- function(df, protein_cols, fixed_vars,
                         adaptive_weights = NULL, ref_colnames = NULL) {
  X_fixed <- model.matrix(as.formula(paste("~", paste(fixed_vars, collapse="+"), "-1")),
                          data = df)
  n_fixed <- ncol(X_fixed)
  X_prot  <- as.matrix(df[, protein_cols, drop = FALSE])
  X       <- cbind(X_fixed, X_prot)
  if (!is.null(ref_colnames)) {
    miss <- setdiff(ref_colnames, colnames(X))
    if (length(miss))
      X <- cbind(X, matrix(0, nrow(X), length(miss), dimnames = list(NULL, miss)))
    X <- X[, ref_colnames, drop = FALSE]
  }
  pf <- c(rep(0, n_fixed),
          if (is.null(adaptive_weights)) rep(1, length(protein_cols))
          else adaptive_weights)
  list(X = X, pf = pf, n_fixed = n_fixed, colnames = colnames(X))
}

compute_adaptive_weights <- function(X_prot, y_resid) {
  rc <- cv.glmnet(X_prot, y_resid, alpha = 0, nfolds = 10, standardize = TRUE)
  1 / pmax(abs(as.vector(coef(rc, s = "lambda.min"))[-1]), 1e-10)
}

# Protein-only score: excludes clinical covariate effects from prediction.
# The design matrix X contains n_fixed clinical columns followed by protein columns.
# Only the protein coefficient × protein value terms are summed, so the score
# is independent of the clinical variables used as unpenalized covariates during
# training. This keeps the AIC comparison clean: Clinical + proRS tests whether
# protein signal adds beyond clinical factors.
prot_score <- function(fit, X, n_fixed, s = "lambda.min") {
  beta      <- as.vector(coef(fit, s = s))
  beta_prot <- beta[(n_fixed + 2L):length(beta)]   # skip intercept + clinical
  X_prot    <- X[, (n_fixed + 1L):ncol(X), drop = FALSE]
  as.vector(X_prot %*% beta_prot)
}

# ── 70/30 train/test split ────────────────────────────────────────────────────
set.seed(2024)
train_idx <- sample(nrow(df_full), round(0.70 * nrow(df_full)))
df_train  <- df_full[train_idx, ]
df_test   <- df_full[-train_idx, ]

# ── Elastic net proRS (progression-targeting) ───────────────────────────────────
cat("Fitting elastic net (alpha=0.5) for emphysema progression...\n")
y_tr <- df_train$Change_lung_density_vnb_P2_P3

d0   <- build_design(df_train, protein_cols, prog_fixed_vars)
aw   <- compute_adaptive_weights(
  d0$X[, -seq_len(d0$n_fixed), drop = FALSE],
  residuals(lm.fit(d0$X[, seq_len(d0$n_fixed), drop = FALSE], y_tr))
)
d_tr <- build_design(df_train, protein_cols, prog_fixed_vars, aw)
d_te <- build_design(df_test,  protein_cols, prog_fixed_vars, aw, d_tr$colnames)
d_fu <- build_design(df_full,  protein_cols, prog_fixed_vars, aw, d_tr$colnames)

enet <- cv.glmnet(d_tr$X, y_tr, alpha = 0.5,
                  penalty.factor = d_tr$pf, nfolds = 10, standardize = TRUE)

n_enet  <- sum(coef(enet, "lambda.min")[-1] != 0)
r2_enet <- cor(df_test$Change_lung_density_vnb_P2_P3,
               as.vector(predict(enet, d_te$X, s = "lambda.min")))^2
cat("Elastic net: proteins selected =", n_enet, "  test R2 =", round(r2_enet, 3), "\n")

prs_raw      <- prot_score(enet, d_fu$X, d_tr$n_fixed)
prs_tr_raw   <- prot_score(enet, d_tr$X, d_tr$n_fixed)
prs_tr_mean  <- mean(prs_tr_raw)
prs_tr_sd    <- sd(prs_tr_raw)
df_full$PRS_enet  <- (prs_raw - prs_tr_mean) / prs_tr_sd
df_train$PRS_enet <- df_full$PRS_enet[train_idx]
df_test$PRS_enet  <- df_full$PRS_enet[-train_idx]

# ── Prognostic elastic net: proteins only (no clinical covariates) ────────────
# The main elastic net above conditions on clinical covariates, so its protein
# coefficients capture residual variation and produce a weak standalone proRS.
# A protein-only elastic net gives a proRS with independent predictive power,
# enabling a clean AIC comparison: Clinical vs proRS vs Clinical + proRS.
# Clinical + proRS then tests whether protein signal adds BEYOND clinical factors.
cat("Fitting prognostic elastic net (proteins only, no clinical)...\n")
X_tr_prot <- as.matrix(df_train[, protein_cols])
X_te_prot <- as.matrix(df_test[,  protein_cols])
X_fu_prot <- as.matrix(df_full[,  protein_cols])

rc_prot  <- cv.glmnet(X_tr_prot, y_tr, alpha = 0, nfolds = 10, standardize = TRUE)
aw_prot  <- 1 / pmax(abs(as.vector(coef(rc_prot, "lambda.min"))[-1]), 1e-10)
enet_prs <- cv.glmnet(X_tr_prot, y_tr, alpha = 0.5,
                      penalty.factor = aw_prot, nfolds = 10, standardize = TRUE)

n_prs   <- sum(coef(enet_prs, "lambda.min")[-1] != 0)
r2_prs  <- cor(df_test$Change_lung_density_vnb_P2_P3,
               as.vector(predict(enet_prs, X_te_prot, "lambda.min")))^2
cat("Prognostic proRS: proteins =", n_prs, "  test R2 =", round(r2_prs, 3), "\n")

prs_prog_raw    <- as.vector(predict(enet_prs, X_fu_prot,  "lambda.min"))
prs_prog_tr_raw <- as.vector(predict(enet_prs, X_tr_prot,  "lambda.min"))
prs_prog_mean   <- mean(prs_prog_tr_raw)
prs_prog_sd     <- sd(prs_prog_tr_raw)
df_full$PRS_prog  <- (prs_prog_raw - prs_prog_mean) / prs_prog_sd
df_train$PRS_prog <- df_full$PRS_prog[train_idx]
df_test$PRS_prog  <- df_full$PRS_prog[-train_idx]

# ── 5-fold CV: OOS predictions for all n=1,306 subjects ──────────────────────
# Each subject is predicted exactly once by a model not trained on them.
# Protein-only elastic net (same approach as enet_prs above) is retrained per
# fold; adaptive weights are recomputed within each fold for correctness.
cat("5-fold cross-validation for full-cohort OOS predictions...\n")
set.seed(2025)
n_full     <- nrow(df_full)
cv_k       <- 5
fold_ids   <- sample(rep(seq_len(cv_k), length.out = n_full))
prs_cv_raw <- numeric(n_full)

for (fold in seq_len(cv_k)) {
  cat("  Fold", fold, "/", cv_k, "\n")
  idx_out  <- which(fold_ids == fold)
  idx_in   <- which(fold_ids != fold)
  X_cv_tr  <- as.matrix(df_full[idx_in,  protein_cols])
  X_cv_te  <- as.matrix(df_full[idx_out, protein_cols])
  y_cv_tr  <- df_full$Change_lung_density_vnb_P2_P3[idx_in]

  rc_cv   <- cv.glmnet(X_cv_tr, y_cv_tr, alpha = 0, nfolds = 5, standardize = TRUE)
  aw_cv   <- 1 / pmax(abs(as.vector(coef(rc_cv, "lambda.min"))[-1]), 1e-10)
  fit_cv  <- cv.glmnet(X_cv_tr, y_cv_tr, alpha = 0.5,
                       penalty.factor = aw_cv, nfolds = 5, standardize = TRUE)
  prs_cv_raw[idx_out] <- as.vector(predict(fit_cv, X_cv_te, s = "lambda.min"))
}

# All predictions are OOS; standardise on full distribution (no leakage)
df_full$PRS_cv  <- (prs_cv_raw - mean(prs_cv_raw)) / sd(prs_cv_raw)
cat("CV complete.\n")

# Top-protein table
meta      <- read_csv(META_PATH, show_col_types = FALSE)
enet_coef <- coef(enet, s = "lambda.min")
prot_sds  <- apply(df_full[, protein_cols], 2, sd, na.rm = TRUE)

enet_df <- tibble(
  term     = rownames(enet_coef)[-1],
  beta_raw = as.numeric(enet_coef[-1])
) |>
  filter(beta_raw != 0, term %in% protein_cols) |>
  mutate(
    seqID    = sub("^log2_X", "", term),
    beta_std = beta_raw * prot_sds[term]
  ) |>
  left_join(meta |> select(seqID, EntrezGeneSymbol, Target), by = "seqID") |>
  select(gene = EntrezGeneSymbol, target = Target, beta_std) |>
  arrange(desc(abs(beta_std)))

write_csv(enet_df, file.path(OUT_DIR, "enet_prs_coefs.csv"))
cat("Top 10 proteins (HU per 1-SD log2 protein):\n")
print(enet_df |> slice_head(n = 10) |> mutate(beta_std = round(beta_std, 3)))

# ── Figure 3: Quartile β vs Q1 (elastic net proRS) ─────────────────────────────
# Q1 = lowest risk (best outcome), Q4 = highest risk (worst outcome).
# Model: Change ~ factor(quartile) + assoc_vars (per guideline).
df_full$enet_risk_q <- ntile(-df_full$PRS_cv, 4)

fml_qdum <- as.formula(paste(
  "Change_lung_density_vnb_P2_P3 ~ factor(enet_risk_q) +", assoc_vars
))
mod_qdum <- lm(fml_qdum, data = df_full)
qdum_tbl <- broom::tidy(mod_qdum, conf.int = TRUE) |>
  filter(grepl("enet_risk_q", term)) |>
  mutate(
    quartile  = as.integer(sub(".*\\)", "", term)),
    estimate  = -estimate,
    tmp_lo    = conf.low,
    conf.low  = -conf.high,
    conf.high = -tmp_lo
  ) |>
  select(quartile, estimate, conf.low, conf.high, p.value)

qdum_tbl <- bind_rows(
  tibble(quartile = 1L, estimate = 0, conf.low = 0, conf.high = 0, p.value = NA_real_),
  qdum_tbl
) |> arrange(quartile)

cat("\nQuartile beta vs Q1 (positive = more emphysema worsening):\n")
print(qdum_tbl |> mutate(across(where(is.numeric), ~round(., 2))))

fig3 <- ggplot(qdum_tbl,
    aes(x = quartile, y = estimate, ymin = conf.low, ymax = conf.high)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_errorbar(width = 0.18, colour = "grey40", linewidth = 0.7) +
  geom_point(size = 4, colour = "#C0392B") +
  scale_x_continuous(
    breaks = 1:4,
    labels = c("Q1\n(lowest risk)", "Q2", "Q3", "Q4\n(highest risk)")
  ) +
  labs(
    x     = "Proteomic Risk Score (proRS) Quartile",
    y     = expression(beta ~ "(HU excess emphysema vs Q1, 95% CI)"),
    title = "Emphysema Progression by proRS Quartile"
  ) +
  theme_classic(base_size = 13)

ggsave(file.path(OUT_DIR, "Figure3_quartile_plot.pdf"), fig3, width = 6, height = 5)
ggsave(file.path(OUT_DIR, "Figure3_quartile_plot.png"), fig3, width = 6, height = 5, dpi = 300)
cat("Figure 3 saved.\n")

# ── Association model (full cohort, CV proRS) ───────────────────────────────────
# Change ~ PRS_cv + age + sex + race + smoking + pack-years + BMI + scanner
assoc_fml <- as.formula(paste(
  "Change_lung_density_vnb_P2_P3 ~ PRS_cv +", assoc_vars
))
mod_assoc <- lm(assoc_fml, data = df_full)
assoc_res <- broom::tidy(mod_assoc, conf.int = TRUE) |>
  filter(term == "PRS_cv") |>
  mutate(across(where(is.numeric), ~round(., 3)))
cat("\nAssociation: CV proRS with emphysema progression (adjusted, n =", nrow(df_full), "):\n")
print(assoc_res)
write_csv(assoc_res, file.path(OUT_DIR, "prs_association.csv"))

# ── AIC comparison (full cohort, CV proRS) ──────────────────────────────────────
cat("\n── AIC comparison (full cohort CV, n =", nrow(df_full), ") ──\n")

clin_fml     <- as.formula(paste("Change_lung_density_vnb_P2_P3 ~", aic_clin_vars))
prs_fml      <- as.formula("Change_lung_density_vnb_P2_P3 ~ PRS_cv")
clin_prs_fml <- as.formula(paste(
  "Change_lung_density_vnb_P2_P3 ~ PRS_cv +", aic_clin_vars
))

mod_clin     <- lm(clin_fml,     data = df_full)
mod_prs      <- lm(prs_fml,      data = df_full)
mod_clin_prs <- lm(clin_prs_fml, data = df_full)

aic_ref <- AIC(mod_clin)
aic_tbl <- tibble(
  model     = c("Clinical", "proRS", "Clinical + proRS"),
  aic       = c(AIC(mod_clin), AIC(mod_prs), AIC(mod_clin_prs)),
  delta_aic = aic - aic_ref
) |> mutate(across(where(is.numeric), ~round(., 1)))

cat("AIC reference (Clinical):", round(aic_ref, 1), "\n")
print(aic_tbl)

prs_clin_res <- broom::tidy(mod_clin_prs, conf.int = TRUE) |>
  filter(term == "PRS_cv") |>
  mutate(across(where(is.numeric), ~round(., 3)))
cat("\nproRS beta in Clinical + proRS model:\n")
print(prs_clin_res)

write_csv(aic_tbl,      file.path(OUT_DIR, "aic_comparison.csv"))
write_csv(prs_clin_res, file.path(OUT_DIR, "prs_aic_model_coef.csv"))
cat("AIC table saved.\n")
