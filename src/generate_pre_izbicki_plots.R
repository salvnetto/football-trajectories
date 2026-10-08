# ==========================================================================
# GERAÇÃO DE GRÁFICOS: REGIÕES CONFORMES PRÉ-IZBICKI (STATUS QUO PONTUAL)
# ==========================================================================
# Este script reproduz e visualiza o método conformal utilizado antes das
# mudanças propostas por Rafael Izbicki (2026, docs/conformal_trajetórias-1.pdf).
#
# Características do método pré-Izbicki:
# 1. Escore de não-conformidade pontual marginal e_{i,t} = ||y_{i,t} - ŷ_{i,t}||_2
# 2. Quantil calibrado independentemente para cada horizonte t in {1, ..., 5}:
#    r_t = quantile(e_{i,t}, probs = ceiling((n+1)(1-alpha))/n)
# 3. Regiões circulares isotrópicas C_t = { p : ||p - ŷ_{i,t}||_2 <= r_t }
# 4. Raio idêntico para todos os jogadores no mesmo instante t (sem adaptatividade).
# ==========================================================================

suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
  library(here)
  library(ggsoccer)
  library(ggforce)
})

source(here("notebooks", "architecture.R"))

# Diretório de saída
output_dir <- here("output")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 1. CARREGAMENTO DOS DADOS E PREPARAÇÃO
cat("\n>>> [1/5] Carregando e preparando dados de tracking e eventos...\n")

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

# 2. TREINAMENTO / OBTENÇÃO DO MODELO
train_cfg <- list(
  seed = 42,
  lr = 0.003,
  max_epochs = 100,
  patience = 10,
  tf_rate = 1.0
)

cat("\n>>> [2/5] Executando experimento com sementes reprodutíveis...\n")
res <- run_experiment(events_prepared, train_cfg)

model <- res$model
test_dataset <- res$test_dataset
calib_dataset <- res$calib_dataset
val_dataset <- res$val_dataset
scaler <- res$scaler

# 3. CALIBRAÇÃO CONFORME PONTUAL MARGINAL (STATUS QUO PRÉ-IZBICKI)
cat("\n>>> [3/5] Calculando quantis conformes pontuais marginais (Pré-Izbicki)...\n")

alpha_level <- 0.10 # Cobertura nominal de 90%

# Calibração conforme rigorosa
conformal_calib <- calibrate_conformal(
  model = model,
  calib_dataset = calib_dataset,
  s_hat_dataset = val_dataset,
  scaler = scaler,
  alpha = alpha_level
)

# O raio pontual marginal histórico (Status Quo descrito na Seção 1 do PDF de Izbicki):
r_pointwise <- conformal_calib$r_pointwise
names(r_pointwise) <- paste0("t=", 1:5)

cat("\nRaios Conformes Pontuais Marginais (metros):\n")
print(round(r_pointwise, 3))

# 4. PREDIÇÃO CONFORME NO CONJUNTO DE TESTE
cat("\n>>> [4/5] Gerando predições e regiões no conjunto de teste...\n")
test_predictions <- predict_with_regions(
  model = model,
  dataset = test_dataset,
  scaler = scaler,
  r_t = r_pointwise
)

# Métricas de cobertura
coverage_marginal <- test_predictions |>
  group_by(time_step) |>
  summarize(
    empirical_coverage = mean(covered),
    mean_radius = mean(conf_radius),
    .groups = "drop"
  )

coverage_trajectory <- test_predictions |>
  group_by(event_id, player_id) |>
  summarize(trajectory_covered = all(covered), .groups = "drop") |>
  summarize(
    total_trajectories = n(),
    empirical_coverage = mean(trajectory_covered)
  )

cat("\n--- Cobertura Empírica Marginal por Horizonte ---\n")
print(coverage_marginal)

cat("\n--- Cobertura Empírica Simultânea da Trajetória Inteira (Limitação Pré-Izbicki) ---\n")
print(coverage_trajectory)

# 5. GERAÇÃO DOS GRÁFICOS (> 10 GRÁFICOS)
cat("\n>>> [5/5] Renderizando e salvando gráficos em output/ ...\n")

# Função de plotagem do lance completo adaptada para o método pré-Izbicki
plot_pre_izbicki_event <- function(target_event_id, events_prepared, test_predictions,
                                   r_radii, alpha = 0.10) {
  pred_data <- test_predictions |>
    filter(event_id == target_event_id)

  valid_pids <- unique(pred_data$player_id)

  hist_data <- events_prepared |>
    filter(event_id == target_event_id, time_sec <= 25, player_id %in% valid_pids) |>
    arrange(player_id, time_sec)

  r_str <- paste(sprintf("r%d=%.2fm", 1:5, r_radii), collapse = ", ")

  ggplot() +
    annotate_pitch(
      dimensions = pitch_international,
      fill = NA,
      colour = "white",
      limits = FALSE
    ) +
    theme_pitch() +
    theme(
      panel.background = element_rect(fill = "#1a1e29", colour = NA),
      plot.background = element_rect(fill = "#1a1e29", colour = NA),
      legend.background = element_rect(fill = "#1a1e29", colour = NA),
      legend.key = element_rect(fill = "#1a1e29", colour = NA),
      legend.text = element_text(color = "white", size = 9),
      legend.title = element_text(color = "white", size = 10, face = "bold"),
      plot.title = element_text(color = "white", size = 13, face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "#b0b8c4", size = 9, hjust = 0),
      plot.caption = element_text(color = "#7b8794", size = 8, hjust = 1),
      plot.margin = margin(12, 14, 12, 14)
    ) +
    # A. Regiões Conformes Circulares Marginais (Isotrópicas / Raio idêntico)
    geom_circle(
      data = pred_data,
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, fill = as.factor(time_step), group = node_id),
      color = NA,
      alpha = 0.18
    ) +
    scale_fill_viridis_d(name = "Horizonte (s)", option = "viridis") +

    # B. Histórico Observado (t <= 25s)
    geom_path(
      data = hist_data |> group_by(player_id) |> slice_tail(n = 5),
      aes(x = x, y = y, group = player_id),
      color = "white", alpha = 0.55, linewidth = 0.5, linetype = "dashed"
    ) +
    geom_point(
      data = hist_data |> group_by(player_id) |> slice_tail(n = 1),
      aes(x = x, y = y),
      color = "white", size = 1.6, shape = 21, fill = "black"
    ) +

    # C. Trajetória Prevista pelo Modelo
    geom_path(
      data = pred_data,
      aes(x = pred_x, y = pred_y, group = node_id),
      color = "#00f5d4", linewidth = 0.85
    ) +
    geom_point(
      data = pred_data,
      aes(x = pred_x, y = pred_y, group = node_id),
      color = "#00f5d4", size = 1.4
    ) +

    # D. Trajetória Real (Ground Truth)
    geom_path(
      data = pred_data,
      aes(x = true_x, y = true_y, group = node_id),
      color = "#ffb703", linewidth = 0.85
    ) +
    geom_point(
      data = pred_data,
      aes(x = true_x, y = true_y, group = node_id),
      color = "#ffb703", size = 1.4
    ) +

    coord_fixed(
      xlim = c(0, 105),
      ylim = c(0, 68),
      expand = FALSE
    ) +
    labs(
      title = sprintf("Regiões Conformes Pontuais (Status Quo Pré-Izbicki, 1 - \u03b1 = %.0f%%) | Lance %s",
                      (1 - alpha) * 100, target_event_id),
      subtitle = sprintf("Quantis pontuais marginais (%s) | Ciano: Previsto | Âmbar: Real | Branco: Observado", r_str),
      caption = "Metodologia: Círculos isotrópicos com raio marginal homogêneo por horizonte (sem garantia de trajetória completa)."
    )
}

test_events <- unique(test_predictions$event_id)
cat(sprintf("Total de lances disponíveis no teste: %d\n", length(test_events)))

# Selecionar até 12 lances completos
n_full_events <- min(12, length(test_events))
selected_events <- test_events[1:n_full_events]

saved_files <- character()

# Salvar os lances de campo inteiro
for (i in seq_along(selected_events)) {
  evt <- selected_events[i]
  p <- plot_pre_izbicki_event(evt, events_prepared, test_predictions, r_pointwise, alpha_level)
  fname <- sprintf("0%d_pre_izbicki_lance_%s.png", i, evt)
  if (i >= 10) fname <- sprintf("%d_pre_izbicki_lance_%s.png", i, evt)
  fpath <- file.path(output_dir, fname)
  ggsave(fpath, plot = p, width = 11, height = 7, dpi = 300)
  saved_files <- c(saved_files, fname)
  cat(sprintf("  [+] Salvo: %s\n", fname))
}

# --- ESTUDOS DE CASO FOCADOS (Ilustrando as limitações pontuadas por Rafael Izbicki) ---

# Caso 13: Corrida Linear / Alta Velocidade (mostra como o círculo isotrópico não adapta a direção)
# Busca jogador com maior deslocamento linear na janela
fast_players <- test_predictions |>
  group_by(event_id, player_id) |>
  summarize(
    disp = sqrt((last(true_x) - first(true_x))^2 + (last(true_y) - first(true_y))^2),
    team_code = first(team_code),
    .groups = "drop"
  ) |>
  arrange(desc(disp))

best_sprinter <- fast_players |> slice_head(n = 1)

p_sprint <- {
  target_event <- best_sprinter$event_id
  target_player <- best_sprinter$player_id

  pred_sub <- test_predictions |> filter(event_id == target_event, player_id == target_player)
  hist_sub <- events_prepared |> filter(event_id == target_event, player_id == target_player, time_sec <= 25)

  # Delimitar janela focal ao redor do jogador
  x_mid <- mean(c(pred_sub$pred_x, pred_sub$true_x))
  y_mid <- mean(c(pred_sub$pred_y, pred_sub$true_y))
  pad <- 14

  ggplot() +
    annotate_pitch(dimensions = pitch_international, fill = NA, colour = "white", limits = FALSE) +
    theme_pitch() +
    theme(
      panel.background = element_rect(fill = "#1a1e29", colour = NA),
      plot.background = element_rect(fill = "#1a1e29", colour = NA),
      legend.background = element_rect(fill = "#1a1e29", colour = NA),
      legend.key = element_rect(fill = "#1a1e29", colour = NA),
      legend.text = element_text(color = "white", size = 9),
      legend.title = element_text(color = "white", size = 10, face = "bold"),
      plot.title = element_text(color = "white", size = 12, face = "bold"),
      plot.subtitle = element_text(color = "#b0b8c4", size = 9),
      plot.caption = element_text(color = "#7b8794", size = 8)
    ) +
    geom_circle(data = pred_sub,
                aes(x0 = pred_x, y0 = pred_y, r = conf_radius, fill = as.factor(time_step)),
                color = "white", linewidth = 0.3, alpha = 0.22) +
    scale_fill_viridis_d(name = "Horizonte (s)") +
    geom_path(data = hist_sub |> slice_tail(n = 5), aes(x = x, y = y),
              color = "white", linetype = "dashed", linewidth = 0.7) +
    geom_point(data = hist_sub |> slice_tail(n = 1), aes(x = x, y = y),
               color = "white", size = 2.5, shape = 21, fill = "black") +
    geom_path(data = pred_sub, aes(x = pred_x, y = pred_y), color = "#00f5d4", linewidth = 1.1) +
    geom_point(data = pred_sub, aes(x = pred_x, y = pred_y), color = "#00f5d4", size = 2) +
    geom_path(data = pred_sub, aes(x = true_x, y = true_y), color = "#ffb703", linewidth = 1.1) +
    geom_point(data = pred_sub, aes(x = true_x, y = true_y), color = "#ffb703", size = 2) +
    coord_fixed(xlim = c(max(0, x_mid - pad), min(105, x_mid + pad)),
                ylim = c(max(0, y_mid - pad), min(68, y_mid + pad)),
                expand = FALSE) +
    labs(
      title = "Estudo de Caso 1: Jogador em Velocidade Linear (Pré-Izbicki)",
      subtitle = "Limitação: Círculo isotrópico não reflete a incerteza alongada na direção do movimento",
      caption = "Izbicki §1: 'a largura não distingue um jogador correndo em linha reta de um jogador disputando a bola.'"
    )
}
fname_sprint <- "13_pre_izbicki_caso_corrida_linear.png"
ggsave(file.path(output_dir, fname_sprint), plot = p_sprint, width = 9, height = 7, dpi = 300)
saved_files <- c(saved_files, fname_sprint)
cat(sprintf("  [+] Salvo: %s\n", fname_sprint))

# Caso 14: Disputa / Aglomeração na Área (mostra que jogador disputando a bola recebe o mesmo raio)
# Encontrar lance com jogadores próximos à bola
crowded_player <- test_predictions |>
  filter(pred_x > 80, pred_y > 20, pred_y < 48) |>
  group_by(event_id) |>
  filter(n_distinct(player_id) >= 4) |>
  ungroup() |>
  slice_head(n = 1)

p_crowd <- {
  target_event <- crowded_player$event_id
  pred_sub <- test_predictions |>
    filter(event_id == target_event, pred_x > 75, pred_y > 18, pred_y < 50)
  pids <- unique(pred_sub$player_id)
  hist_sub <- events_prepared |>
    filter(event_id == target_event, player_id %in% pids, time_sec <= 25)

  ggplot() +
    annotate_pitch(dimensions = pitch_international, fill = NA, colour = "white", limits = FALSE) +
    theme_pitch() +
    theme(
      panel.background = element_rect(fill = "#1a1e29", colour = NA),
      plot.background = element_rect(fill = "#1a1e29", colour = NA),
      legend.background = element_rect(fill = "#1a1e29", colour = NA),
      legend.key = element_rect(fill = "#1a1e29", colour = NA),
      legend.text = element_text(color = "white", size = 9),
      legend.title = element_text(color = "white", size = 10, face = "bold"),
      plot.title = element_text(color = "white", size = 12, face = "bold"),
      plot.subtitle = element_text(color = "#b0b8c4", size = 9),
      plot.caption = element_text(color = "#7b8794", size = 8)
    ) +
    geom_circle(data = pred_sub,
                aes(x0 = pred_x, y0 = pred_y, r = conf_radius, fill = as.factor(time_step), group = node_id),
                color = NA, alpha = 0.15) +
    scale_fill_viridis_d(name = "Horizonte (s)") +
    geom_path(data = hist_sub |> group_by(player_id) |> slice_tail(n = 5),
              aes(x = x, y = y, group = player_id), color = "white", linetype = "dashed", linewidth = 0.5) +
    geom_path(data = pred_sub, aes(x = pred_x, y = pred_y, group = node_id), color = "#00f5d4", linewidth = 0.9) +
    geom_point(data = pred_sub, aes(x = pred_x, y = pred_y, group = node_id), color = "#00f5d4", size = 1.5) +
    geom_path(data = pred_sub, aes(x = true_x, y = true_y, group = node_id), color = "#ffb703", linewidth = 0.9) +
    geom_point(data = pred_sub, aes(x = true_x, y = true_y, group = node_id), color = "#ffb703", size = 1.5) +
    coord_fixed(xlim = c(70, 105), ylim = c(15, 53), expand = FALSE) +
    labs(
      title = "Estudo de Caso 2: Aglomeração na Área Final (Pré-Izbicki)",
      subtitle = "Limitação: Todos os jogadores recebem o mesmo raio homogêneo, ignorando o caos da disputa",
      caption = "Izbicki §4: 'Um jogador em movimento retilíneo e isolado deveria receber uma região menor; na área disputada, maior.'"
    )
}
fname_crowd <- "14_pre_izbicki_caso_aglomeracao_area.png"
ggsave(file.path(output_dir, fname_crowd), plot = p_crowd, width = 9, height = 7, dpi = 300)
saved_files <- c(saved_files, fname_crowd)
cat(sprintf("  [+] Salvo: %s\n", fname_crowd))

# Caso 15: Quebra de Cobertura Simultânea (onde passos individuais cobrem, mas a trajetória falha)
uncovered_trajs <- test_predictions |>
  group_by(event_id, player_id) |>
  summarize(
    covered_steps = sum(covered),
    total_steps = n(),
    trajectory_covered = all(covered),
    .groups = "drop"
  ) |>
  filter(covered_steps >= 3, !trajectory_covered) |>
  arrange(desc(covered_steps))

leak_player <- uncovered_trajs |> slice_head(n = 1)

p_leak <- {
  target_event <- leak_player$event_id
  target_player <- leak_player$player_id

  pred_sub <- test_predictions |> filter(event_id == target_event, player_id == target_player)
  hist_sub <- events_prepared |> filter(event_id == target_event, player_id == target_player, time_sec <= 25)

  x_mid <- mean(c(pred_sub$pred_x, pred_sub$true_x))
  y_mid <- mean(c(pred_sub$pred_y, pred_sub$true_y))
  pad <- 12

  ggplot() +
    annotate_pitch(dimensions = pitch_international, fill = NA, colour = "white", limits = FALSE) +
    theme_pitch() +
    theme(
      panel.background = element_rect(fill = "#1a1e29", colour = NA),
      plot.background = element_rect(fill = "#1a1e29", colour = NA),
      legend.background = element_rect(fill = "#1a1e29", colour = NA),
      legend.key = element_rect(fill = "#1a1e29", colour = NA),
      legend.text = element_text(color = "white", size = 9),
      legend.title = element_text(color = "white", size = 10, face = "bold"),
      plot.title = element_text(color = "white", size = 12, face = "bold"),
      plot.subtitle = element_text(color = "#b0b8c4", size = 9),
      plot.caption = element_text(color = "#7b8794", size = 8)
    ) +
    geom_circle(data = pred_sub,
                aes(x0 = pred_x, y0 = pred_y, r = conf_radius, fill = as.factor(time_step)),
                color = "white", linewidth = 0.3, alpha = 0.22) +
    scale_fill_viridis_d(name = "Horizonte (s)") +
    geom_path(data = hist_sub |> slice_tail(n = 5), aes(x = x, y = y),
              color = "white", linetype = "dashed", linewidth = 0.6) +
    geom_point(data = hist_sub |> slice_tail(n = 1), aes(x = x, y = y),
               color = "white", size = 2.2, shape = 21, fill = "black") +
    geom_path(data = pred_sub, aes(x = pred_x, y = pred_y), color = "#00f5d4", linewidth = 1) +
    geom_point(data = pred_sub, aes(x = pred_x, y = pred_y), color = "#00f5d4", size = 2) +
    geom_path(data = pred_sub, aes(x = true_x, y = true_y), color = "#ffb703", linewidth = 1) +
    geom_point(data = pred_sub, aes(x = true_x, y = true_y), color = "#ffb703", size = 2) +
    coord_fixed(xlim = c(max(0, x_mid - pad), min(105, x_mid + pad)),
                ylim = c(max(0, y_mid - pad), min(68, y_mid + pad)),
                expand = FALSE) +
    labs(
      title = "Estudo de Caso 3: Fuga Temporal da Trajetória (Pré-Izbicki)",
      subtitle = "Passos isolados cobrem o alvo, mas a trajetória completa escapa da região conformal",
      caption = "Izbicki §1: 'Não há uma garantia sobre a cobertura da trajetória inteira sob quantis marginais.'"
    )
}
fname_leak <- "15_pre_izbicki_caso_fuga_trajetoria.png"
ggsave(file.path(output_dir, fname_leak), plot = p_leak, width = 9, height = 7, dpi = 300)
saved_files <- c(saved_files, fname_leak)
cat(sprintf("  [+] Salvo: %s\n", fname_leak))

# Caso 16: Gráfico Diagnóstico Comparativo de Cobertura Marginal vs Simultânea
diag_df <- bind_rows(
  coverage_marginal |>
    transmute(
      tipo = "Marginal por Passo",
      etiqueta = paste0("t = ", time_step, "s"),
      cobertura = empirical_coverage,
      raio = mean_radius
    ),
  tibble(
    tipo = "Simultânea (Trajetória Completa)",
    etiqueta = "Trajetória (t = 1..5s)",
    cobertura = coverage_trajectory$empirical_coverage,
    raio = NA_real_
  )
)

p_diag <- ggplot(diag_df, aes(x = etiqueta, y = cobertura, fill = tipo)) +
  geom_col(width = 0.55, alpha = 0.85) +
  geom_hline(yintercept = 0.90, linetype = "dashed", color = "#ff4d6d", linewidth = 0.8) +
  annotate("text", x = 2.5, y = 0.92, label = "Meta Nominal (1 - \u03b1 = 90%)",
           color = "#ff4d6d", fontface = "bold", size = 3.8) +
  geom_text(aes(label = sprintf("%.1f%%", cobertura * 100)),
            vjust = -0.6, color = "white", fontface = "bold", size = 3.5) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1.05)) +
  scale_fill_manual(values = c("Marginal por Passo" = "#00f5d4",
                               "Simultânea (Trajetória Completa)" = "#ffb703")) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background = element_rect(fill = "#1a1e29", colour = NA),
    panel.background = element_rect(fill = "#1a1e29", colour = NA),
    panel.grid.major = element_line(color = "#2c3345", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    axis.text = element_text(color = "white", size = 10),
    axis.title = element_text(color = "white", size = 11, face = "bold"),
    legend.position = "top",
    legend.text = element_text(color = "white", size = 9),
    legend.title = element_blank(),
    plot.title = element_text(color = "white", size = 13, face = "bold"),
    plot.subtitle = element_text(color = "#b0b8c4", size = 9.5),
    plot.caption = element_text(color = "#7b8794", size = 8)
  ) +
  labs(
    title = "Diagnóstico Empírico da Abordagem Pré-Izbicki: Cobertura Marginal vs Simultânea",
    subtitle = "Quantis marginais preservam ~90% em cada instante, mas a cobertura da trajetória inteira degrada",
    x = "Horizonte Temporal",
    y = "Taxa de Cobertura Empírica",
    caption = "Evidência empírica da necessidade da formulação de supremo temporal proposta por Rafael Izbicki."
  )

fname_diag <- "16_pre_izbicki_diagnostico_cobertura.png"
ggsave(file.path(output_dir, fname_diag), plot = p_diag, width = 10, height = 6.5, dpi = 300)
saved_files <- c(saved_files, fname_diag)
cat(sprintf("  [+] Salvo: %s\n", fname_diag))

cat(sprintf("\n=== SUCESSO: %d GRÁFICOS GERADOS E SALVOS EM %s ===\n", length(saved_files), output_dir))
