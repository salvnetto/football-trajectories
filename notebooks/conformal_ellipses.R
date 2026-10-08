# ==========================================================================
# CONFORMAL PREDICTION ELÍPTICA — TRAJETÓRIAS 2D
# ==========================================================================
# Escore de não-conformidade de Mahalanobis (região = elipse):
#
#   R_i = max_{t in H} sqrt( e_{i,t}' Σ̂_t^{-1} e_{i,t} ),
#   C_t  = { p : (p - p̂_t)' Σ̂_t^{-1} (p - p̂_t) <= q̂^2 }
#
# Σ̂_t (2x2) é a covariância empírica do resíduo 2D e_{i,t} = y_{i,t} - ŷ_{i,t}
# no horizonte t, estimada em resíduos OUT-OF-FOLD (shape_dataset, nunca usado
# no treino NEM na calibração). Com Σ̂_t fixada antes da calibração, os escores
# de calibração permanecem permutáveis e vale a garantia split-conformal
# (Vovk, Gammerman & Shafer 2005; Lei, Rinaldo & Wasserman 2018) ao nível de
# trajetória de jogador — com a mesma ressalva do Alvo B naive (PDF §3):
# o n efetivo é o nº de jogadas, não de trajetórias.
#
# A elipse em t é o nível q̂ do quadrático definido por Σ̂_t: semi-eixos
# a_t = q̂·sqrt(λ1_t), b_t = q̂·sqrt(λ2_t), orientação dada pelo autovetor de λ1_t.
# ==========================================================================

# Resíduos 2D por (jogada, jogador, horizonte): devolve lista de arrays
# [n_jogadores_jogada, 5, 2] por jogada e o array empilhado [total, 5, 2].
compute_error_vectors <- function(model, dataset, scaler) {
  model$eval()
  device <- model$parameters[[1]]$device
  ts <- mu_sd_tensors(scaler, device)

  loader <- dataloader(dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)
  err_by_event <- list()

  with_no_grad({
    coro::loop(for (batch in loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat <- batch$x_cat$to(device = device)
      batch$y <- batch$y$to(device = device)

      preds <- model(batch)

      preds_real <- preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu

      # Vetor de resíduo 2D em metros: e = (e_x, e_y)
      err_vec <- as.array((preds_real - y_true_real)$cpu()) # [num_nodes, 5, 2]
      err_by_event[[length(err_by_event) + 1]] <- err_vec
    })
  })

  # Empilha por jogada (primeira dimensão = trajetórias, ordem jogada-major)
  err_array <- do.call(abind::abind, c(err_by_event, list(along = 1)))

  list(err_by_event = err_by_event, err_array = err_array)
}

# Covariância empírica do resíduo 2D por horizonte t (ridge numérico mínimo
# para garantir invertibilidade).
estimate_error_covariances <- function(err_array, ridge = 1e-6) {
  # err_array: [n_trajetorias, 5, 2]
  lapply(1:5, function(t) {
    e <- matrix(err_array[, t, ], ncol = 2) # [n, 2]
    cov(e) + diag(ridge, 2)
  })
}

# Escore simultâneo por trajetória: R_i = max_t sqrt(e' Σ̂_t^{-1} e)
mahalanobis_scores <- function(err_array, Sigma_t) {
  R <- matrix(NA_real_, nrow(err_array), 5)
  for (t in 1:5) {
    e <- matrix(err_array[, t, ], ncol = 2) # [n, 2]
    Sinv <- solve(Sigma_t[[t]])
    R[, t] <- sqrt(rowSums((e %*% Sinv) * e))
  }
  apply(R, 1, max)
}

# Calibração split-conformal com escore de Mahalanobis.
# Devolve q̂ e os parâmetros das elipses (a_t, b_t, angle_t) por horizonte.
calibrate_conformal_ellipses <- function(model, calib_dataset, shape_dataset,
                                         scaler, alpha = 0.10) {
  # ---- Passo 1: forma da região (Σ̂_t) em resíduos out-of-fold ----
  shape_res <- compute_error_vectors(model, shape_dataset, scaler)
  Sigma_t <- estimate_error_covariances(shape_res$err_array)

  # ---- Passo 2: escores de Mahalanobis no conjunto de calibração ----
  calib_res <- compute_error_vectors(model, calib_dataset, scaler)
  R_player <- mahalanobis_scores(calib_res$err_array, Sigma_t)

  n_players <- length(R_player)
  n_events <- length(calib_res$err_by_event)

  # ---- Passo 3: quantil com correção finita ⌈(n+1)(1-α)⌉/n ----
  q_level <- min(1.0, ceiling((n_players + 1) * (1 - alpha)) / n_players)
  q_hat <- quantile(R_player, probs = q_level, names = FALSE)

  # ---- Passo 4: semi-eixos e orientação das elipses (autodecomposição) ----
  ellipse_pars <- do.call(rbind, lapply(1:5, function(t) {
    eig <- eigen(Sigma_t[[t]], symmetric = TRUE)
    ord <- order(eig$values, decreasing = TRUE)
    lam <- eig$values[ord]
    v1 <- eig$vectors[, ord[1]]
    data.frame(
      time_step = t,
      a = q_hat * sqrt(max(lam[1], 0)),
      b = q_hat * sqrt(max(lam[2], 0)),
      angle = atan2(v1[2], v1[1]),
      area = pi * q_hat^2 * sqrt(max(lam[1], 0) * max(lam[2], 0))
    )
  }))

  cat(sprintf("\n=== CALIBRAÇÃO CONFORME ELÍPTICA (alpha = %.2f) ===\n", alpha))
  cat(sprintf("Σ̂_t estimada em %d jogadas out-of-fold (validação)\n",
              length(shape_res$err_by_event)))
  cat(sprintf("Calibração: %d jogadas independentes (%d trajetórias)\n",
              n_events, n_players))
  cat(sprintf("q̂ (escore de Mahalanobis simultâneo): %.3f\n", q_hat))
  cat("Semi-eixos a_t, b_t (metros):\n")
  print(round(ellipse_pars[, c("a", "b")], 2))

  list(
    q_hat = q_hat,
    R_player = R_player,
    Sigma_t = Sigma_t,
    ellipse_pars = ellipse_pars,
    alpha = alpha,
    n_players = n_players,
    n_events = n_events
  )
}

# Predições de teste com regiões elípticas: uma linha por (jogador, horizonte)
# com semi-eixos (a, b), ângulo, distância de Mahalanobis e cobertura.
predict_with_ellipses <- function(model, dataset, scaler, ellipse_pars, Sigma_t, q_hat) {
  model$eval()
  device <- model$parameters[[1]]$device
  ts <- mu_sd_tensors(scaler, device)

  loader <- dataloader(dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)
  results_df <- list()

  with_no_grad({
    coro::loop(for (batch in loader) {
      evt_id <- batch$event_id
      player_ids <- batch$player_ids
      team_codes <- batch$team_codes
      team_names <- batch$team_names

      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat <- batch$x_cat$to(device = device)
      batch$y <- batch$y$to(device = device)

      preds <- model(batch)

      preds_real <- preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu

      preds_arr <- as.array(preds_real$cpu())
      y_true_arr <- as.array(y_true_real$cpu())

      num_nodes <- dim(preds_arr)[1]
      seq_len <- dim(preds_arr)[2]

      for (node in 1:num_nodes) {
        pid <- player_ids[node]
        tcode <- team_codes[node]
        tname <- team_names[node]

        for (t in 1:seq_len) {
          px <- preds_arr[node, t, 1]
          py <- preds_arr[node, t, 2]
          tx <- y_true_arr[node, t, 1]
          ty <- y_true_arr[node, t, 2]

          e <- c(px - tx, py - ty)
          Sinv <- solve(Sigma_t[[t]])
          md <- sqrt(sum((e %*% Sinv) * e))

          results_df[[length(results_df) + 1]] <- data.frame(
            event_id = evt_id,
            node_id = node,
            player_id = pid,
            team_code = tcode,
            team_name = tname,
            time_step = t,
            pred_x = px,
            pred_y = py,
            true_x = tx,
            true_y = ty,
            a = ellipse_pars$a[t],
            b = ellipse_pars$b[t],
            angle = ellipse_pars$angle[t],
            mahalanobis = md,
            covered = md <= q_hat,
            stringsAsFactors = FALSE
          )
        }
      }
    })
  })

  bind_rows(results_df)
}

# Pontos do contorno da elipse { p : (p - mu)' Σ^{-1} (p - mu) = q^2 },
# na parametrização padrão rotacionada pelo autovetor principal.
ellipse_points <- function(x0, y0, a, b, angle, n = 120) {
  theta <- seq(0, 2 * pi, length.out = n)
  x <- a * cos(theta)
  y <- b * sin(theta)
  data.frame(
    x = x0 + x * cos(angle) - y * sin(angle),
    y = y0 + x * sin(angle) + y * cos(angle)
  )
}

# ==========================================================================
# VISUALIZAÇÃO — Regiões conformes elípticas no campo (105 x 68 m)
# ==========================================================================
plot_conformal_event_ellipses <- function(target_event_id, events_prepared,
                                          test_predictions, alpha = 0.10) {
  pred_data <- test_predictions |>
    filter(event_id == target_event_id)

  if (nrow(pred_data) == 0) {
    stop(sprintf("Lance event_id %s não encontrado em test_predictions.", target_event_id))
  }

  valid_pids <- unique(pred_data$player_id)

  # Trajetória histórica (últimos 5 frames observados t <= 25s) dos jogadores previstos
  hist_data <- events_prepared |>
    filter(event_id == target_event_id, time_sec <= 25, player_id %in% valid_pids) |>
    arrange(player_id, time_sec)

  # Contornos elípticos: um polígono por (jogador, horizonte)
  ellipse_data <- pred_data |>
    rowwise() |>
    mutate(pts = list(ellipse_points(pred_x, pred_y, a, b, angle))) |>
    unnest(pts) |>
    ungroup()

  p <- ggplot() +
    # Geometria oficial do campo (105 x 68 m)
    annotate_pitch(
      dimensions = pitch_international,
      fill = NA,
      colour = "white",
      limits = FALSE
    ) +
    theme_pitch() +
    theme(
      panel.background = element_rect(fill = "#1e222d", colour = NA),
      plot.background = element_rect(fill = "#1e222d", colour = NA),
      legend.background = element_rect(fill = "#1e222d", colour = NA),
      legend.key = element_rect(fill = "#1e222d", colour = NA),
      legend.text = element_text(color = "white", size = 9),
      legend.title = element_text(color = "white", size = 10, face = "bold"),
      plot.title = element_text(color = "white", size = 13, face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "#cccccc", size = 9, hjust = 0),
      plot.margin = margin(12, 12, 12, 12)
    ) +
    # A. Regiões Conformes Elípticas (escore de Mahalanobis)
    geom_polygon(
      data = ellipse_data,
      aes(x = x, y = y, fill = factor(time_step),
          group = interaction(node_id, time_step)),
      color = NA,
      alpha = 0.18
    ) +
    scale_fill_viridis_d(name = "Horizonte (s)") +

    # B. Histórico Observado
    geom_path(
      data = hist_data |> group_by(player_id) |> slice_tail(n = 5),
      aes(x = x, y = y, group = player_id),
      color = "white", alpha = 0.6, linewidth = 0.5, linetype = "dashed"
    ) +
    geom_point(
      data = hist_data |> group_by(player_id) |> slice_tail(n = 1),
      aes(x = x, y = y),
      color = "white", size = 1.8, shape = 21, fill = "black"
    ) +

    # C. Trajetória Prevista pelo Modelo
    geom_path(
      data = pred_data,
      aes(x = pred_x, y = pred_y, group = node_id),
      color = "#00f5d4", linewidth = 0.9
    ) +
    geom_point(
      data = pred_data,
      aes(x = pred_x, y = pred_y, group = node_id),
      color = "#00f5d4", size = 1.6
    ) +

    # D. Trajetória Real (Ground Truth)
    geom_path(
      data = pred_data,
      aes(x = true_x, y = true_y, group = node_id),
      color = "#ffb703", linewidth = 0.9
    ) +
    geom_point(
      data = pred_data,
      aes(x = true_x, y = true_y, group = node_id),
      color = "#ffb703", size = 1.6
    ) +

    # Fixação de Proporção Cartesiana Real (1:1)
    coord_fixed(
      xlim = c(0, 105),
      ylim = c(0, 68),
      expand = FALSE
    ) +
    labs(
      title = sprintf("Regiões Conformes Elípticas (1 - \u03b1 = %.0f%%) | Lance %s", (1 - alpha) * 100, target_event_id),
      subtitle = "Azul: Previsto | Amarelo: Real | Branco: Observado (t \u2264 25s) | Elipses: regi\u00f5es conformes de Mahalanobis"
    )

  return(p)
}
