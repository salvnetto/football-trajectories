# ==========================================================================
# GERAÇÃO DE GRÁFICOS: TRAJETÓRIAS ISOLADAS
# Comparação: Conformal Antigo vs. Mudanças 1 a 3 (Mesmo Modelo Social-LSTM)
# ==========================================================================
# Gráficos focados em trajetórias individuais de jogadores para visualização
# desobstruída e direta das regiões de predição conforme no campo de futebol.
# ==========================================================================

suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
  library(here)
  library(ggsoccer)
  library(ggforce)
})

source(here("notebooks", "architecture.R"))
source(here("notebooks", "conformal_ellipses.R"))

output_dir <- here("output")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. CARREGAMENTO DOS DADOS
cat(">>> [1/5] Carregando tracking e eventos...\n")

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

# 2. MODELAGEM (Social-LSTM, semente fixa 42)
train_cfg <- list(
  seed = 42,
  lr = 0.003,
  max_epochs = 100,
  patience = 10,
  tf_rate = 1.0
)

cat(">>> [2/5] Treinando modelo e preparando partições...\n")
res <- run_experiment(events_prepared, train_cfg)

model <- res$model
test_dataset <- res$test_dataset
calib_dataset <- res$calib_dataset
val_dataset <- res$val_dataset
scaler <- res$scaler

# 3. CALIBRAÇÃO CONFORME
cat(">>> [3/5] Calibrando regiões conformes (Conformal Antigo vs Mudanças 1 a 3)...\n")
alpha_level <- 0.10

# A. Conformal Circular (Antigo = r_pointwise, Novo = r_simultaneous)
conformal_calib <- calibrate_conformal(
  model = model,
  calib_dataset = calib_dataset,
  s_hat_dataset = val_dataset,
  scaler = scaler,
  alpha = alpha_level
)

# B. Conformal Elíptico de Mahalanobis
calib_ellipses <- calibrate_conformal_ellipses(
  model = model,
  calib_dataset = calib_dataset,
  shape_dataset = val_dataset,
  scaler = scaler,
  alpha = alpha_level
)

# 4. PREDIÇÕES NO CONJUNTO DE TESTE COM O MESMO MODELO
cat(">>> [4/5] Gerando inferências conformes no conjunto de teste...\n")

test_pred_antigo <- predict_with_regions(
  model = model,
  dataset = test_dataset,
  scaler = scaler,
  r_t = conformal_calib$r_pointwise
) |> mutate(metodo = "Conformal Antigo\n(Quantis Marginais Pontuais)")

test_pred_novo <- predict_with_regions(
  model = model,
  dataset = test_dataset,
  scaler = scaler,
  r_t = conformal_calib$r_simultaneous
) |> mutate(metodo = "Mudanças 1 a 3\n(Banda Conforme Simultânea)")

pred_ellipses_all <- predict_with_ellipses(
  model = model,
  dataset = test_dataset,
  scaler = scaler,
  ellipse_pars = calib_ellipses$ellipse_pars,
  Sigma_t = calib_ellipses$Sigma_t,
  q_hat = calib_ellipses$q_hat
) |> mutate(metodo = "Mudanças 1 a 3\n(Elipses de Mahalanobis)")

# 5. FUNÇÕES DE PLOTAGEM DE TRAJETÓRIAS ISOLADAS
cat(">>> [5/5] Renderizando gráficos de trajetórias isoladas em output/ ...\n")

# A. Função para Comparação em 2 Painéis (Conformal Antigo vs Mudanças 1 a 3)
plot_isolated_comparison <- function(target_event_id, target_player_id,
                                     events_prepared, test_pred_antigo, test_pred_novo) {
  
  sub_antigo <- test_pred_antigo |>
    filter(event_id == target_event_id, player_id == target_player_id)
  
  sub_novo <- test_pred_novo |>
    filter(event_id == target_event_id, player_id == target_player_id)
  
  comb_data <- bind_rows(sub_antigo, sub_novo)
  
  hist_data <- events_prepared |>
    filter(event_id == target_event_id, player_id == target_player_id, time_sec <= 25) |>
    arrange(time_sec)
  
  # Delimitação segura contendo todos os círculos e caminhos
  all_x <- c(comb_data$pred_x, comb_data$true_x, hist_data$x)
  all_y <- c(comb_data$pred_y, comb_data$true_y, hist_data$y)
  
  min_x_circ <- min(comb_data$pred_x - comb_data$conf_radius)
  max_x_circ <- max(comb_data$pred_x + comb_data$conf_radius)
  min_y_circ <- min(comb_data$pred_y - comb_data$conf_radius)
  max_y_circ <- max(comb_data$pred_y + comb_data$conf_radius)
  
  min_x_box <- min(all_x, min_x_circ) - 2.5
  max_x_box <- max(all_x, max_x_circ) + 2.5
  min_y_box <- min(all_y, min_y_circ) - 2.5
  max_y_box <- max(all_y, max_y_circ) + 2.5
  
  span <- max(max_x_box - min_x_box, max_y_box - min_y_box)
  center_x <- (min_x_box + max_x_box) / 2
  center_y <- (min_y_box + max_y_box) / 2
  
  xlim_val <- c(center_x - span / 2, center_x + span / 2)
  ylim_val <- c(center_y - span / 2, center_y + span / 2)
  
  cov_antigo <- if (all(sub_antigo$covered)) "100% (5/5)" else sprintf("%d/5 passos (Fuga)", sum(sub_antigo$covered))
  cov_novo   <- if (all(sub_novo$covered)) "100% (5/5)" else sprintf("%d/5 passos", sum(sub_novo$covered))
  
  team_str <- if (comb_data$team_code[1] == "Attack") "Atacante" else "Defensor"
  ade_val <- mean(comb_data$distance[1:5])
  fde_val <- comb_data$distance[5]
  
  p <- ggplot() +
    annotate_pitch(
      dimensions = pitch_international,
      fill = NA,
      colour = "grey50",
      linewidth = 0.55,
      limits = FALSE
    ) +
    theme_pitch() +
    theme(
      panel.background = element_rect(fill = "#fafafa", colour = NA),
      plot.background = element_rect(fill = "#ffffff", colour = NA),
      panel.border = element_rect(colour = "grey70", fill = NA, linewidth = 0.6),
      legend.background = element_rect(fill = "#ffffff", colour = NA),
      legend.key = element_rect(fill = "#ffffff", colour = NA),
      legend.text = element_text(color = "grey20", size = 9),
      legend.title = element_text(color = "grey10", size = 9.5, face = "bold"),
      legend.position = "right",
      strip.background = element_rect(fill = "#edf2f7", colour = "grey70", linewidth = 0.6),
      strip.text = element_text(color = "#0f172a", face = "bold", size = 11, margin = margin(t = 7, b = 7)),
      plot.title = element_text(color = "#0f172a", size = 13, face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "#334155", size = 9.5, hjust = 0, margin = margin(b = 6)),
      plot.caption = element_text(color = "#64748b", size = 8.5, hjust = 1),
      plot.margin = margin(12, 14, 10, 14)
    ) +
    # Regiões Conformes Circulares
    geom_circle(
      data = comb_data,
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, color = factor(time_step)),
      fill = NA,
      linewidth = 0.85
    ) +
    geom_circle(
      data = comb_data,
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, fill = factor(time_step)),
      color = NA,
      alpha = 0.08
    ) +
    scale_color_viridis_d(name = "Horizonte (s)", option = "plasma", end = 0.9) +
    scale_fill_viridis_d(name = "Horizonte (s)", option = "plasma", end = 0.9) +
    
    # Histórico Observado (t <= 25s)
    geom_path(
      data = hist_data |> slice_tail(n = 5),
      aes(x = x, y = y),
      color = "grey45", linetype = "dashed", linewidth = 0.75
    ) +
    geom_point(
      data = hist_data |> slice_tail(n = 1),
      aes(x = x, y = y),
      color = "grey20", size = 2.4, shape = 21, fill = "white", stroke = 1.2
    ) +
    
    # Trajetória Prevista pelo Modelo
    geom_path(
      data = comb_data,
      aes(x = pred_x, y = pred_y),
      color = "#00838f", linewidth = 1.15
    ) +
    geom_point(
      data = comb_data,
      aes(x = pred_x, y = pred_y),
      color = "#00838f", size = 2.2
    ) +
    
    # Trajetória Real (Ground Truth)
    geom_path(
      data = comb_data,
      aes(x = true_x, y = true_y),
      color = "#ea580c", linewidth = 1.15
    ) +
    geom_point(
      data = comb_data,
      aes(x = true_x, y = true_y),
      color = "#ea580c", size = 2.2
    ) +
    
    coord_fixed(
      xlim = xlim_val,
      ylim = ylim_val,
      expand = FALSE
    ) +
    facet_wrap(~ metodo, ncol = 2) +
    labs(
      title = sprintf("Trajetória Isolada: %s %s | Lance %s (ADE: %.2f m | FDE: %.2f m)",
                      team_str, target_player_id, target_event_id, ade_val, fde_val),
      subtitle = sprintf("Verde-petróleo: Previsto | Laranja: Real | Cinza tracejado: Observado | Cobertura: %s (Esquerda) vs. %s (Direita)",
                         cov_antigo, cov_novo),
      caption = "Mesmo modelo Social-LSTM. Esquerda: Conformal Antigo (Marginal). Direita: Mudanças 1 a 3 (Simultâneo)."
    )
  
  return(p)
}

# B. Função para Comparação em 3 Painéis (Antigo vs Mudanças 1 a 3 Banda vs Mudanças 1 a 3 Elipses)
plot_isolated_triplet <- function(target_event_id, target_player_id,
                                  events_prepared, test_pred_antigo, test_pred_novo,
                                  pred_ellipses_all) {
  
  sub_antigo <- test_pred_antigo |>
    filter(event_id == target_event_id, player_id == target_player_id) |>
    mutate(metodo = "1. Conformal Antigo\n(Quantis Marginais)")
  
  sub_novo <- test_pred_novo |>
    filter(event_id == target_event_id, player_id == target_player_id) |>
    mutate(metodo = "2. Mudanças 1 a 3\n(Banda Simultânea)")
  
  comb_circles <- bind_rows(sub_antigo, sub_novo)
  
  sub_ellipses <- pred_ellipses_all |>
    filter(event_id == target_event_id, player_id == target_player_id) |>
    mutate(metodo = "3. Mudanças 1 a 3\n(Elipses de Mahalanobis)")
  
  ellipse_poly <- sub_ellipses |>
    rowwise() |>
    mutate(pts = list(ellipse_points(pred_x, pred_y, a, b, angle, n = 90))) |>
    unnest(pts) |>
    ungroup()
  
  hist_data <- events_prepared |>
    filter(event_id == target_event_id, player_id == target_player_id, time_sec <= 25) |>
    arrange(time_sec)
  
  all_x <- c(comb_circles$pred_x, comb_circles$true_x, hist_data$x, ellipse_poly$x)
  all_y <- c(comb_circles$pred_y, comb_circles$true_y, hist_data$y, ellipse_poly$y)
  
  min_x_circ <- min(comb_circles$pred_x - comb_circles$conf_radius)
  max_x_circ <- max(comb_circles$pred_x + comb_circles$conf_radius)
  min_y_circ <- min(comb_circles$pred_y - comb_circles$conf_radius)
  max_y_circ <- max(comb_circles$pred_y + comb_circles$conf_radius)
  
  min_x_box <- min(all_x, min_x_circ) - 2.5
  max_x_box <- max(all_x, max_x_circ) + 2.5
  min_y_box <- min(all_y, min_y_circ) - 2.5
  max_y_box <- max(all_y, max_y_circ) + 2.5
  
  span <- max(max_x_box - min_x_box, max_y_box - min_y_box)
  center_x <- (min_x_box + max_x_box) / 2
  center_y <- (min_y_box + max_y_box) / 2
  
  xlim_val <- c(center_x - span / 2, center_x + span / 2)
  ylim_val <- c(center_y - span / 2, center_y + span / 2)
  
  trajs_all <- bind_rows(
    comb_circles |> select(metodo, time_step, pred_x, pred_y, true_x, true_y, covered),
    sub_ellipses |> select(metodo, time_step, pred_x, pred_y, true_x, true_y, covered)
  )
  
  team_str <- if (comb_circles$team_code[1] == "Attack") "Atacante" else "Defensor"
  ade_val <- mean(comb_circles$distance[1:5])
  fde_val <- comb_circles$distance[5]
  
  cov1 <- sprintf("%d/5", sum(sub_antigo$covered))
  cov2 <- sprintf("%d/5", sum(sub_novo$covered))
  cov3 <- sprintf("%d/5", sum(sub_ellipses$covered))
  
  p <- ggplot() +
    annotate_pitch(
      dimensions = pitch_international,
      fill = NA,
      colour = "grey50",
      linewidth = 0.5,
      limits = FALSE
    ) +
    theme_pitch() +
    theme(
      panel.background = element_rect(fill = "#fafafa", colour = NA),
      plot.background = element_rect(fill = "#ffffff", colour = NA),
      panel.border = element_rect(colour = "grey70", fill = NA, linewidth = 0.6),
      legend.background = element_rect(fill = "#ffffff", colour = NA),
      legend.key = element_rect(fill = "#ffffff", colour = NA),
      legend.text = element_text(color = "grey20", size = 8.5),
      legend.title = element_text(color = "grey10", size = 9, face = "bold"),
      legend.position = "right",
      strip.background = element_rect(fill = "#edf2f7", colour = "grey70", linewidth = 0.6),
      strip.text = element_text(color = "#0f172a", face = "bold", size = 10, margin = margin(t = 6, b = 6)),
      plot.title = element_text(color = "#0f172a", size = 12.5, face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "#334155", size = 9, hjust = 0, margin = margin(b = 6)),
      plot.caption = element_text(color = "#64748b", size = 8, hjust = 1),
      plot.margin = margin(10, 12, 8, 12)
    ) +
    geom_circle(
      data = comb_circles,
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, color = factor(time_step)),
      fill = NA,
      linewidth = 0.8
    ) +
    geom_circle(
      data = comb_circles,
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, fill = factor(time_step)),
      color = NA,
      alpha = 0.07
    ) +
    geom_polygon(
      data = ellipse_poly,
      aes(x = x, y = y, group = factor(time_step), fill = factor(time_step)),
      color = NA,
      alpha = 0.07
    ) +
    geom_path(
      data = ellipse_poly,
      aes(x = x, y = y, group = factor(time_step), color = factor(time_step)),
      linewidth = 0.8
    ) +
    scale_color_viridis_d(name = "Horizonte (s)", option = "plasma", end = 0.9) +
    scale_fill_viridis_d(name = "Horizonte (s)", option = "plasma", end = 0.9) +
    
    geom_path(
      data = hist_data |> slice_tail(n = 5),
      aes(x = x, y = y),
      color = "grey45", linetype = "dashed", linewidth = 0.7
    ) +
    geom_point(
      data = hist_data |> slice_tail(n = 1),
      aes(x = x, y = y),
      color = "grey20", size = 2.0, shape = 21, fill = "white", stroke = 1.1
    ) +
    
    geom_path(data = trajs_all, aes(x = pred_x, y = pred_y), color = "#00838f", linewidth = 1.0) +
    geom_point(data = trajs_all, aes(x = pred_x, y = pred_y), color = "#00838f", size = 1.8) +
    
    geom_path(data = trajs_all, aes(x = true_x, y = true_y), color = "#ea580c", linewidth = 1.0) +
    geom_point(data = trajs_all, aes(x = true_x, y = true_y), color = "#ea580c", size = 1.8) +
    
    coord_fixed(xlim = xlim_val, ylim = ylim_val, expand = FALSE) +
    facet_wrap(~ metodo, ncol = 3) +
    labs(
      title = sprintf("Trajetória Isolada: %s %s | Lance %s (ADE: %.2f m | FDE: %.2f m)",
                      team_str, target_player_id, target_event_id, ade_val, fde_val),
      subtitle = sprintf("Verde-petróleo: Previsto | Laranja: Real | Cobertura empírica: Painel 1 (%s), Painel 2 (%s), Painel 3 (%s)",
                         cov1, cov2, cov3),
      caption = "Mesmo modelo Social-LSTM. Comparação direta entre o Conformal Antigo e as formulações de Mudanças 1 a 3."
    )
  
  return(p)
}

# 6. SELEÇÃO DE TRAJETÓRIAS E GERAÇÃO DA GALERIA
# Lista de 12 trajetórias representativas para a comparação em 2 painéis
selected_pairs_2p <- list(
  # Casos onde o Conformal Antigo sofre fuga temporal e Mudanças 1 a 3 recupera cobertura 100%:
  list(evt = 1611,  pid = "DFL-OBJ-0002AU", tag = "01_lance1611_atacante_arrancada"),
  list(evt = 3716,  pid = "DFL-OBJ-0000F8", tag = "02_lance3716_atacante_diagonal"),
  list(evt = 3338,  pid = "DFL-OBJ-0000F8", tag = "03_lance3338_atacante_infiltracao"),
  list(evt = 10226, pid = "DFL-OBJ-0002DT", tag = "04_lance10226_atacante_velocidade"),
  list(evt = 548,   pid = "DFL-OBJ-0000IA", tag = "05_lance548_defensor_recuperacao"),
  list(evt = 4510,  pid = "DFL-OBJ-J01L1F", tag = "06_lance4510_atacante_penetracao"),
  list(evt = 6513,  pid = "DFL-OBJ-0000F8", tag = "07_lance6513_defensor_bloqueio"),
  list(evt = 8753,  pid = "DFL-OBJ-002GN4", tag = "08_lance8753_defensor_cobertura"),
  
  # Casos de alta acurácia (ADE estreito), ilustrando a escala relativa da região conforme:
  list(evt = 9724,  pid = "DFL-OBJ-0026RH", tag = "09_lance9724_atacante_finalizacao_precisa"),
  list(evt = 6513,  pid = "DFL-OBJ-002GMO", tag = "10_lance6513_defensor_posicionamento"),
  list(evt = 1611,  pid = "DFL-OBJ-0027G0", tag = "11_lance1611_atacante_apoio_ofensivo"),
  list(evt = 1611,  pid = "DFL-OBJ-J01D1W", tag = "12_lance1611_defensor_recomposicao")
)

# Salvar as 12 comparações em 2 painéis
for (item in selected_pairs_2p) {
  p <- plot_isolated_comparison(
    target_event_id = item$evt,
    target_player_id = item$pid,
    events_prepared = events_prepared,
    test_pred_antigo = test_pred_antigo,
    test_pred_novo = test_pred_novo
  )
  fname <- sprintf("trajetoria_isolada_%s.png", item$tag)
  fpath <- file.path(output_dir, fname)
  ggsave(fpath, plot = p, width = 12, height = 6.5, dpi = 300)
  cat(sprintf("  [+] Salvo: %s\n", fname))
}

# Lista de 4 trajetórias para a comparação tripla (3 painéis)
selected_pairs_3p <- list(
  list(evt = 3716, pid = "DFL-OBJ-0000F8", tag = "13_tripla_lance3716_atacante_diagonal"),
  list(evt = 1611, pid = "DFL-OBJ-0002AU", tag = "14_tripla_lance1611_atacante_arrancada"),
  list(evt = 4510, pid = "DFL-OBJ-J01L1F", tag = "15_tripla_lance4510_atacante_penetracao"),
  list(evt = 6007, pid = "DFL-OBJ-J01H9X", tag = "16_tripla_lance6007_defensor_perseguicao")
)

# Salvar as 4 comparações em 3 painéis
for (item in selected_pairs_3p) {
  p <- plot_isolated_triplet(
    target_event_id = item$evt,
    target_player_id = item$pid,
    events_prepared = events_prepared,
    test_pred_antigo = test_pred_antigo,
    test_pred_novo = test_pred_novo,
    pred_ellipses_all = pred_ellipses_all
  )
  fname <- sprintf("trajetoria_isolada_%s.png", item$tag)
  fpath <- file.path(output_dir, fname)
  ggsave(fpath, plot = p, width = 15, height = 5.8, dpi = 300)
  cat(sprintf("  [+] Salvo: %s\n", fname))
}

cat("\n=== SUCESSO: 16 GRÁFICOS DE TRAJETÓRIAS ISOLADAS SALVOS EM output/ ===\n")
