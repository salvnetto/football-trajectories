# ==============================================================================
# COMPARAÇÃO DAS 3 VISÕES CONFORMAIS COM ARCHITECTURE_V2 EM MODO CROSS-CONFORMAL
# ==============================================================================
# Visão 1: old.qmd (Conformal Marginal Pontual Pré-Izbicki)
# Visão 2: architecture.R (Banda Simultânea Estática Global - Sugestões 1 a 3)
# Visão 3: architecture_v2.R (Cross-Conformal K-Fold + Escala Adaptativa + CRC)
# ==============================================================================
# Baseado estritamente nas recomendações de Rafael Izbicki (2026):
#   - Sugestão 4: Escala Adaptativa ŝ_t(x_i) condicionada ao contexto físico
#   - Sugestão 5: Cross-Conformal Predictor (Vovk 2015; Barber et al. 2021)
#   - Sugestão 6: Conformal Risk Control (Angelopoulos et al. 2024, ICLR)
# ==============================================================================

suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
  library(here)
  library(ggsoccer)
  library(ggforce)
})

source(here("notebooks", "architecture.R"))
source(here("notebooks", "architecture_v2.R"))

output_dir <- here("output")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. CARREGAMENTO DOS DADOS PREPARADOS
cat(">>> [1/4] Carregando dados de tracking e eventos...\n")
events <- read_csv(here('data', 'processed', 'events.csv'), show_col_types = FALSE) |>
  distinct() |>
  rename(team_with_poss = team_id) |>
  mutate(event_id = as.integer(as.factor(event_id)))

players_db <- read_csv(here('data', 'processed', 'players_database.csv'), show_col_types = FALSE) |>
  select(player_id, team_name) |>
  distinct()

tracking <- read_csv(here('data', 'processed', 'tracking.csv'), show_col_types = FALSE) |>
  distinct() |>
  left_join(players_db, by = "player_id")

events_prepared <- prepare_data(
  events = events,
  tracking = tracking,
  event_type_filter = "SHOT",
  event_subtype = NA,
  start_time = 30,
  end_time = 1,
  pred_time_event = 1
)

prep <- preprocess_trajectories(events_prepared, seed = 42)
graphs <- prep$graphs
scaler <- prep$scaler
folds  <- split_events_kfold(graphs, K = 5, seed = 42)

# Mapeamento evento -> fold out-of-fold
event_fold_map <- data.frame(
  event_id = integer(),
  fold_id = integer(),
  stringsAsFactors = FALSE
)
for (k in 1:5) {
  evts <- unique(sapply(graphs[folds[[k]]], function(g) g$event_id))
  event_fold_map <- bind_rows(event_fold_map, data.frame(event_id = evts, fold_id = k))
}

# 2. CARREGAMENTO DAS CALIBRAÇÕES E MODELOS SALVOS
cat(">>> [2/4] Carregando modelos das 5 dobras e calibrações conformes...\n")
fold_scale_models <- list()
for (k in 1:5) {
  fold_scale_models[[k]] <- readRDS(here("output", sprintf("fold_%d_scale_model.rds", k)))
}

calib_summary <- readRDS(here("output", "cross_conformal_calibration_summary.rds"))
r_pointwise_cross   <- calib_summary$r_pointwise
r_simultaneous_cross <- calib_summary$r_simultaneous
lambda_hat_cross    <- calib_summary$lambda_hat

# 3. FUNÇÕES DE INFERÊNCIA E PLOTAGEM
get_three_visions_pred <- function(evt, pid) {
  fid <- event_fold_map$fold_id[event_fold_map$event_id == evt]
  st <- torch_load(here("output", sprintf("fold_%d_weights.pt", fid)))
  f_model <- SocialLSTM$new(scaler = scaler)$to(device = "cpu")
  f_model$load_state_dict(st)
  f_model$eval()

  f_scale_model <- fold_scale_models[[fid]]

  g_idx <- which(sapply(graphs, function(g) g$event_id) == evt)
  ds <- TrajectoryDataset(graphs[g_idx])

  sub_old <- predict_with_regions(f_model, ds, scaler, r_pointwise_cross) |>
    filter(player_id == pid) |>
    mutate(metodo = "1. old.qmd\n(Conformal Marginal Pontual)")

  sub_a1 <- predict_with_regions(f_model, ds, scaler, r_simultaneous_cross) |>
    filter(player_id == pid) |>
    mutate(metodo = "2. architecture.R\n(Banda Simultânea Estática)")

  sub_v2 <- predict_with_adaptive_regions(f_model, ds, f_scale_model, lambda_hat_cross, scaler) |>
    filter(player_id == pid) |>
    mutate(metodo = "3. architecture_v2.R\n(Cross-Conformal Adaptativo + CRC)")

  list(old = sub_old, a1 = sub_a1, v2 = sub_v2)
}

plot_isolated_three_visions_cross <- function(target_event_id, target_player_id,
                                              events_prepared, sub_old, sub_a1, sub_v2) {
  comb_data <- bind_rows(sub_old, sub_a1, sub_v2) |>
    mutate(metodo = factor(metodo, levels = c(
      "1. old.qmd\n(Conformal Marginal Pontual)",
      "2. architecture.R\n(Banda Simultânea Estática)",
      "3. architecture_v2.R\n(Cross-Conformal Adaptativo + CRC)"
    )))

  hist_data <- events_prepared |>
    filter(event_id == target_event_id, player_id == target_player_id, time_sec <= 25) |>
    arrange(time_sec)

  all_x <- c(comb_data$pred_x, comb_data$true_x, hist_data$x)
  all_y <- c(comb_data$pred_y, comb_data$true_y, hist_data$y)

  min_x_circ <- min(comb_data$pred_x - comb_data$conf_radius)
  max_x_circ <- max(comb_data$pred_x + comb_data$conf_radius)
  min_y_circ <- min(comb_data$pred_y - comb_data$conf_radius)
  max_y_circ <- max(comb_data$pred_y + comb_data$conf_radius)

  min_x_box <- max(0, min(all_x, min_x_circ) - 3)
  max_x_box <- min(105, max(all_x, max_x_circ) + 3)
  min_y_box <- max(0, min(all_y, min_y_circ) - 3)
  max_y_box <- min(68, max(all_y, max_y_circ) + 3)

  span <- max(max_x_box - min_x_box, max_y_box - min_y_box)
  center_x <- (min_x_box + max_x_box) / 2
  center_y <- (min_y_box + max_y_box) / 2

  xlim_val <- c(max(0, center_x - span / 2), min(105, center_x + span / 2))
  ylim_val <- c(max(0, center_y - span / 2), min(68, center_y + span / 2))

  cov_old <- if (all(sub_old$covered)) "5/5 (100%)" else sprintf("%d/5 (Fuga)", sum(sub_old$covered))
  cov_a1  <- if (all(sub_a1$covered)) "5/5 (100%)" else sprintf("%d/5 (Fuga)", sum(sub_a1$covered))
  cov_v2  <- if (all(sub_v2$covered)) "5/5 (100%)" else sprintf("%d/5 (Fuga)", sum(sub_v2$covered))

  team_str <- if (sub_v2$team_code[1] == "Attack") "Atacante" else "Defensor"
  speed_val <- sub_v2$speed_obs[1]

  ggplot() +
    annotate_pitch(
      dimensions = pitch_international,
      fill = NA,
      colour = "grey60",
      linewidth = 0.5,
      limits = FALSE
    ) +
    theme_pitch() +
    theme(
      panel.background = element_rect(fill = "#f8fafc", colour = NA),
      plot.background = element_rect(fill = "#ffffff", colour = NA),
      panel.border = element_rect(colour = "grey75", fill = NA, linewidth = 0.6),
      legend.background = element_rect(fill = "#ffffff", colour = NA),
      legend.key = element_rect(fill = "#ffffff", colour = NA),
      legend.text = element_text(color = "grey20", size = 8.5),
      legend.title = element_text(color = "grey10", size = 9, face = "bold"),
      legend.position = "bottom",
      strip.background = element_rect(fill = "#e2e8f0", colour = "grey75", linewidth = 0.6),
      strip.text = element_text(color = "#0f172a", face = "bold", size = 10, margin = margin(t = 6, b = 6)),
      plot.title = element_text(color = "#0f172a", size = 13, face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "#334155", size = 9.5, hjust = 0, margin = margin(b = 6)),
      plot.caption = element_text(color = "#64748b", size = 8, hjust = 1),
      plot.margin = margin(12, 14, 10, 14)
    ) +
    geom_circle(
      data = comb_data,
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, color = factor(time_step)),
      fill = NA, linewidth = 0.8
    ) +
    geom_circle(
      data = comb_data,
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, fill = factor(time_step)),
      color = NA, alpha = 0.08
    ) +
    scale_color_viridis_d(name = "Horizonte Futuro (s)", option = "plasma", end = 0.9) +
    scale_fill_viridis_d(name = "Horizonte Futuro (s)", option = "plasma", end = 0.9) +
    geom_path(
      data = hist_data |> slice_tail(n = 5),
      aes(x = x, y = y),
      color = "grey40", linetype = "dashed", linewidth = 0.75
    ) +
    geom_point(
      data = hist_data |> slice_tail(n = 1),
      aes(x = x, y = y),
      color = "grey20", size = 2.4, shape = 21, fill = "white", stroke = 1.2
    ) +
    geom_path(data = comb_data, aes(x = pred_x, y = pred_y), color = "#00838f", linewidth = 1.1) +
    geom_point(data = comb_data, aes(x = pred_x, y = pred_y), color = "#00838f", size = 2.0) +
    geom_path(data = comb_data, aes(x = true_x, y = true_y), color = "#ea580c", linewidth = 1.1) +
    geom_point(data = comb_data, aes(x = true_x, y = true_y), color = "#ea580c", size = 2.0) +
    coord_fixed(xlim = xlim_val, ylim = ylim_val, expand = FALSE) +
    facet_wrap(~ metodo, ncol = 3) +
    labs(
      title = sprintf("Comparação de Trajetória Isolada: %s %s | Lance %s (Velocidade obs: %.2f m/s)",
                      team_str, target_player_id, target_event_id, speed_val),
      subtitle = sprintf("Verde-petróleo: Previsto | Laranja: Real | Cinza tracejado: Observado | Cobertura da Trajetória: old.qmd [%s] vs. arch.R [%s] vs. arch_v2.R [%s]",
                         cov_old, cov_a1, cov_v2),
      caption = "Mesmo modelo Social-LSTM (Alahi et al. 2016). Painel 1: old.qmd (Marginal). Painel 2: architecture.R (Simultâneo Estático). Painel 3: architecture_v2.R (Cross-Conformal CRC + Adaptativo)."
    )
}

# 4. GERAÇÃO DOS CASOS SELECIONADOS (CURVAS, SPRINTS LONGOS, TRANSIÇÕES)
cat(">>> [3/4] Renderizando novos gráficos comparativos com curvas e longas distâncias...\n")
cases_to_render <- list(
  list(evt = 7183,  pid = "DFL-OBJ-00008F", name = "13_lance7183_defensor_recuperacao_sprint_extremo"),
  list(evt = 3056,  pid = "DFL-OBJ-J01AQV", name = "14_lance3056_atacante_infiltracao_curva_diagonal"),
  list(evt = 6670,  pid = "DFL-OBJ-0028BZ", name = "15_lance6670_atacante_arco_infiltracao_area"),
  list(evt = 1325,  pid = "DFL-OBJ-00012X", name = "16_lance1325_defensor_corte_diagonal_transicao"),
  list(evt = 5730,  pid = "DFL-OBJ-002FXT", name = "17_lance5730_atacante_sprint_longo_profundidade"),
  list(evt = 7040,  pid = "DFL-OBJ-002GMO", name = "18_lance7040_defensor_perseguicao_curvada_lateral"),
  list(evt = 405,   pid = "DFL-OBJ-0027G0", name = "19_lance405_atacante_curva_alta_velocidade"),
  list(evt = 8229,  pid = "DFL-OBJ-002GM1", name = "20_lance8229_defensor_pivot_posicional_eficiente"),
  list(evt = 8583,  pid = "DFL-OBJ-0026V5", name = "21_lance8583_atacante_contra_ataque_sprint_longo"),
  list(evt = 10129, pid = "DFL-OBJ-0028GO", name = "22_lance10129_defensor_inversao_180deg_cobertura")
)

for (cs in cases_to_render) {
  preds <- get_three_visions_pred(cs$evt, cs$pid)
  p <- plot_isolated_three_visions_cross(cs$evt, cs$pid, events_prepared, preds$old, preds$a1, preds$v2)
  fpath <- here("output", sprintf("comparacao_3visoes_%s.png", cs$name))
  ggsave(fpath, plot = p, width = 15, height = 6.8, dpi = 300)
  cat(sprintf("Salvo: %s\n", basename(fpath)))
}

cat(">>> [4/4] Pipeline de renderização Cross-Conformal concluída com sucesso!\n")
