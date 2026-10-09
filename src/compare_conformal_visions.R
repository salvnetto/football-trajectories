# ==============================================================================
# COMPARAÇÃO DAS 3 VISÕES CONFORMAIS: old.qmd vs architecture.R vs architecture_v2.R
# ==============================================================================
# 1. Visão 1: old.qmd (Conformal Marginal Pontual Pré-Izbicki)
# 2. Visão 2: architecture.R (Banda Simultânea Estática Global - Sugestões 1 a 3)
# 3. Visão 3: architecture_v2.R (Escala Adaptativa ŝ_t(x_i) + CRC - Sugestões 4 a 6)
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

# 1. DADOS
cat(">>> [1/5] Carregando dados de tracking e eventos...\n")
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

# 2. MODELO
train_cfg <- list(seed = 42, lr = 0.003, max_epochs = 100, patience = 10, tf_rate = 1.0)
cat(">>> [2/5] Treinando / carregando modelo Social-LSTM e datasets...\n")
res <- run_experiment(events_prepared, train_cfg)

model <- res$model
test_dataset <- res$test_dataset
calib_dataset <- res$calib_dataset
val_dataset <- res$val_dataset
scaler <- res$scaler
alpha_level <- 0.10

# 3. CALIBRAÇÕES
cat(">>> [3/5] Calibrando os métodos (old.qmd, architecture.R, architecture_v2.R)...\n")

# A. Visão 1 & Visão 2 via architecture.R
conformal_calib <- calibrate_conformal(
  model = model,
  calib_dataset = calib_dataset,
  s_hat_dataset = val_dataset,
  scaler = scaler,
  alpha = alpha_level
)

# B. Visão 3 via architecture_v2.R (Escala Adaptativa + CRC)
calib_v2 <- calibrate_split_conformal_adaptive_crc(
  model = model,
  calib_dataset = calib_dataset,
  oof_scale_dataset = val_dataset,
  scaler = scaler,
  alpha = alpha_level,
  verbose = TRUE
)

# 4. PREDIÇÕES NO TESTE
cat(">>> [4/5] Gerando inferências no teste...\n")
df_old <- predict_with_regions(
  model = model,
  dataset = test_dataset,
  scaler = scaler,
  r_t = conformal_calib$r_pointwise
) |> mutate(metodo = "1. old.qmd\n(Conformal Marginal Pontual)")

df_arch1 <- predict_with_regions(
  model = model,
  dataset = test_dataset,
  scaler = scaler,
  r_t = conformal_calib$r_simultaneous
) |> mutate(metodo = "2. architecture.R\n(Banda Simultânea Estática)")

df_v2 <- predict_with_adaptive_regions(
  model = model,
  dataset = test_dataset,
  scale_model = calib_v2$scale_model,
  lambda_hat = calib_v2$lambda_hat,
  scaler = scaler
) |> mutate(metodo = "3. architecture_v2.R\n(Escala Adaptativa ŝ_t(x_i) + CRC)")

# 5. VISUALIZAÇÃO
plot_isolated_three_visions <- function(target_event_id, target_player_id,
                                         events_prepared, df_old, df_arch1, df_v2) {
  sub_old <- df_old |> filter(event_id == target_event_id, player_id == target_player_id)
  sub_a1  <- df_arch1 |> filter(event_id == target_event_id, player_id == target_player_id)
  sub_v2  <- df_v2 |> filter(event_id == target_event_id, player_id == target_player_id)

  comb_data <- bind_rows(sub_old, sub_a1, sub_v2) |>
    mutate(metodo = factor(metodo, levels = c(
      "1. old.qmd\n(Conformal Marginal Pontual)",
      "2. architecture.R\n(Banda Simultânea Estática)",
      "3. architecture_v2.R\n(Escala Adaptativa ŝ_t(x_i) + CRC)"
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
      caption = "Mesmo modelo Social-LSTM (Alahi et al. 2016). Painel 1: old.qmd (Marginal). Painel 2: architecture.R (Simultâneo Estático). Painel 3: architecture_v2.R (Adaptativo CRC)."
    )
}

cat(">>> [5/5] Renderizando gráficos comparativos das 3 visões (12 casos representativos)...\n")
cases <- list(
  list(evt = 4510, pid = "DFL-OBJ-00008K", name = "01_lance4510_atacante_velocidade"),
  list(evt = 3716, pid = "DFL-OBJ-0027VS", name = "02_lance3716_defensor_perseguicao"),
  list(evt = 8753, pid = "DFL-OBJ-J0157X", name = "03_lance8753_defensor_posicional_lento"),
  list(evt = 3716, pid = "DFL-OBJ-J014UG", name = "04_lance3716_atacante_ritmo_moderado"),
  list(evt = 8753, pid = "DFL-OBJ-0026RH", name = "05_lance8753_atacante_infiltracao_resgatado"),
  list(evt = 9724, pid = "DFL-OBJ-002FXT", name = "06_lance9724_atacante_finalizador_disputa_bola"),
  list(evt = 9724, pid = "DFL-OBJ-0028GO", name = "07_lance9724_defensor_bloqueio_direto"),
  list(evt = 8473, pid = "DFL-OBJ-J0157X", name = "08_lance8473_atacante_amplitude_alta_eficiencia"),
  list(evt = 6513, pid = "DFL-OBJ-002GM9", name = "09_lance6513_defensor_posicional_compacto"),
  list(evt = 1611, pid = "DFL-OBJ-0026ZI", name = "10_lance1611_defensor_transicao_fuga_auditoria"),
  list(evt = 4510, pid = "DFL-OBJ-002G5J", name = "11_lance4510_atacante_apoio_longo_resgatado"),
  list(evt = 7964, pid = "DFL-OBJ-002GM1", name = "12_lance7964_atacante_sprint_portador_bola")
)

for (cs in cases) {
  p <- plot_isolated_three_visions(cs$evt, cs$pid, events_prepared, df_old, df_arch1, df_v2)
  fpath <- here("output", sprintf("comparacao_3visoes_%s.png", cs$name))
  ggsave(fpath, plot = p, width = 15, height = 6.8, dpi = 300)
  cat(sprintf("Salvo: %s\n", fpath))
}

cat("Pipeline de comparação concluída com sucesso!\n")
