# Step 08. Overall and recurrence-free survival by gCPS subtype (Figure 3, Supplementary Table S2a-d)
suppressPackageStartupMessages({
  library(survival); library(survminer); library(multcomp)
  library(dplyr); library(tidyr); library(readxl); library(stringr); library(broom); library(ggplot2)
})

root  <- Sys.getenv("GCPS_ROOT", getwd())
out   <- Sys.getenv("GCPS_OUT", file.path(root, "results"))
data  <- Sys.getenv("GCPS_DATA", file.path(root, "data"))
dir_out <- file.path(out, "08_survival")
dir.create(file.path(dir_out, "figures"), recursive = TRUE, showWarnings = FALSE)

palette <- c(gCPS1 = "#E69F00", gCPS2 = "#56B4E9", gCPS3 = "#009E73", gCPS4 = "#D55E00")
stages  <- c("Stage I", "Stage II", "Stage III", "Stage IV")
sid <- function(x) str_match(as.character(x), "((?:UM|U)[0-9]+)")[, 2]

labels <- read.csv(file.path(out, "05_subtypes", "gcps_labels.csv")) %>%
  mutate(sid = sid(sample_id), gCPS = factor(gCPS, levels = names(palette)))
stopifnot(identical(as.integer(table(labels$gCPS)), c(426L, 274L, 268L, 94L)))

cohort <- read_excel(file.path(data, "Supplementary_Table_01.xlsx")) %>%
  mutate(sid = sid(`Sample ID`)) %>% distinct(sid, .keep_all = TRUE) %>%
  transmute(sid,
            age = as.numeric(`Age at diagnosis`),
            sex = factor(ifelse(Sex == "Male", "Male", "Female"), levels = c("Female", "Male")),
            stage = factor(stages[match(`Tumour Stage`, stages)], levels = stages),
            msi = factor(ifelse(`MSI Status` == "MSI", "MSI-H", "MSS"), levels = c("MSS", "MSI-H")),
            pretreated = factor(ifelse(`Pre-Treated` == "Treated", "Treated", "Untreated"),
                                levels = c("Untreated", "Treated")),
            OS_event = ifelse(`Vital Status` == "Dead", 1, ifelse(`Vital Status` == "Alive", 0, NA)),
            OS_months = as.numeric(`Overall survival days`) / 30.44,
            RFS_event = ifelse(Recurrence == "Yes", 1, ifelse(Recurrence == "No", 0, NA)),
            RFS_months = as.numeric(`Recurrence free survival days`) / 30.44) %>%
  inner_join(labels %>% select(sid, gCPS), by = "sid")

pairwise <- function(fit, label, n, events) {
  g <- glht(fit, linfct = mcp(gCPS = "Tukey"))
  s <- summary(g, test = adjusted("none")); ci <- confint(g, calpha = univariate_calpha())
  parts <- str_split_fixed(names(s$test$coefficients), " - ", 2)
  tibble(model = label, group_a = parts[, 2], group_b = parts[, 1],
         HR = exp(s$test$coefficients), lower = exp(ci$confint[, "lwr"]), upper = exp(ci$confint[, "upr"]),
         P = s$test$pvalues, n = n, events = events) %>% mutate(P_adj = p.adjust(P, "BH"))
}

km <- function(d, time, event, title) {
  fit <- surv_fit(as.formula(sprintf("Surv(%s, %s) ~ gCPS", time, event)), data = d)
  p <- ggsurvplot(fit, data = d, palette = unname(palette), legend.labs = names(palette), legend.title = "",
                  xlab = "Time (months)", ylab = "Survival probability", title = title,
                  pval = TRUE, pval.method = FALSE, conf.int = TRUE, conf.int.alpha = 0.15,
                  risk.table = TRUE, risk.table.height = 0.3, censor.shape = "|", size = 1.4,
                  ggtheme = theme_classic(base_size = 14))
  p
}

save_km <- function(p, name) {
  base <- file.path(dir_out, "figures", name)
  png(paste0(base, ".png"), width = 8, height = 7, units = "in", res = 300); print(p); invisible(dev.off())
  pdf(paste0(base, ".pdf"), width = 8, height = 7); print(p); invisible(dev.off())
  svglite::svglite(paste0(base, ".svg"), width = 8, height = 7); print(p); invisible(dev.off())
}

fit_endpoint <- function(d, time, event, tag, title) {
  d <- d %>% filter(!is.na(.data[[time]]), !is.na(.data[[event]]), .data[[time]] >= 0, !is.na(stage)) %>% droplevels()
  form <- function(rhs) as.formula(sprintf("Surv(%s, %s) ~ %s", time, event, rhs))
  fit_adj <- coxph(form("gCPS + age + sex + msi + pretreated + strata(stage)"), data = d, ties = "efron")
  fit_raw <- coxph(form("gCPS"), data = d, ties = "efron")
  cat(sprintf("%s: n=%d, events=%d\n", tag, nrow(d), sum(d[[event]])))

  tab <- bind_rows(tidy(fit_raw, exponentiate = TRUE, conf.int = TRUE) %>% mutate(model = "unadjusted"),
                   tidy(fit_adj, exponentiate = TRUE, conf.int = TRUE) %>% mutate(model = "adjusted, stage-stratified")) %>%
    transmute(model, term, HR = estimate, lower = conf.low, upper = conf.high, P = p.value,
              n = nrow(d), events = sum(d[[event]]))
  write.csv(tab, file.path(dir_out, paste0(tag, "_cox.csv")), row.names = FALSE)

  pw <- pairwise(fit_adj, "adjusted", nrow(d), sum(d[[event]]))
  write.csv(pw, file.path(dir_out, paste0(tag, "_pairwise_adjusted.csv")), row.names = FALSE)

  ph <- as.data.frame(cox.zph(fit_adj)$table) %>% tibble::rownames_to_column("term")
  write.csv(ph, file.path(dir_out, paste0(tag, "_schoenfeld.csv")), row.names = FALSE)

  lr <- survdiff(form("gCPS"), data = d)
  cat(sprintf("  log-rank P = %.4f\n", 1 - pchisq(lr$chisq, length(lr$n) - 1)))
  save_km(km(d, time, event, title), paste0(tag, "_KM"))
  list(data = d, fit = fit_adj, pairwise = pw)
}

os  <- fit_endpoint(cohort, "OS_months", "OS_event", "OS", "Overall survival (all stages)")
rfs <- fit_endpoint(cohort %>% filter(stage %in% stages[1:3]), "RFS_months", "RFS_event", "RFS",
                    "Recurrence-free survival (stage I-III)")

# Forest plot: stage entered as an ordinary covariate so that ggforest can draw it
d <- os$data %>% rename(Subtype = gCPS, Age = age, Sex = sex, MSI = msi, Pretreatment = pretreated, Stage = stage)
fit_display <- coxph(Surv(OS_months, OS_event) ~ Subtype + Age + Sex + MSI + Pretreatment + Stage, data = d)
for (ext in c("png", "pdf", "svg"))
  ggsave(file.path(dir_out, "figures", paste0("OS_forest.", ext)),
         ggforest(fit_display, data = as.data.frame(d), fontsize = 1.0), width = 12, height = 10, dpi = 300)

writeLines(capture.output(sessionInfo()), file.path(dir_out, "session_info.txt"))
