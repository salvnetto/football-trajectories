# ==============================================================================
# CONFORMAL_BASELINE.R — PREDIÇÃO CONFORME SIMULTÂNEA BASELINE
# ==============================================================================
# Implementação das rotinas numéricas de calibração split-conformal
# (Izbicki 2026; Diquigiovanni et al. 2022; Lei et al. 2018).
#
# Suporta alternância explícita entre regiões:
#  - region_type = "circular": bandas simultâneas com raio r_t = q̂ · ŝ_t
#  - region_type = "elliptic": elipses simultâneas de Mahalanobis com Σ̂_t
#
# Este módulo não contém funções de plotagem gráfica.
# ==============================================================================

suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
})

# ==============================================================================
# 1. EXTRAÇÃO DE ERROS E RESÍDUOS
# ==============================================================================

#' Extrai resíduos escalares e vetoriais 2D em metros sobre um dataset
#' @param model Modelo treinado
#' @param dataset TrajectoryDataset
#' @param scaler Parâmetros de escalonamento para conversão física
#' @return Lista contendo err_matrix, err_by_event, err_array e err_vec_by_event
compute_baseline_errors <- function(model, dataset, scaler) {
  model$eval()
  device <- model$parameters[[1]]$device
  ts <- mu_sd_tensors(scaler, device)

  loader <- dataloader(dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  err_by_event <- list()
  err_vec_by_event <- list()

  with_no_grad({
    coro::loop(for (batch in loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat  <- batch$x_cat$to(device = device)
      batch$y      <- batch$y$to(device = device)

      preds <- model(batch)

      # Desnormalização para coordenadas físicas reais do campo (metros)
      preds_real  <- preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu

      # 1. Distância Euclidiana pontual (metros)
      diff_sq <- (preds_real - y_true_real)$pow(2)
      distances <- torch_sqrt(diff_sq$sum(dim = 3) + 1e-8)
      dist_mat <- as.matrix(distances$cpu())
      err_by_event[[length(err_by_event) + 1]] <- dist_mat

      # 2. Vetor de resíduo 2D (dx, dy) em metros
      diff_vec <- preds_real - y_true_real
      vec_arr <- as.array(diff_vec$cpu())
      err_vec_by_event[[length(err_vec_by_event) + 1]] <- vec_arr
    })
  })

  err_matrix <- do.call(rbind, err_by_event)
  err_array  <- do.call(abind::abind, c(err_vec_by_event, list(along = 1)))

  list(
    err_by_event = err_by_event,
    err_matrix = err_matrix,
    err_vec_by_event = err_vec_by_event,
    err_array = err_array
  )
}


# ==============================================================================
# 2. CALIBRAÇÃO CONFORME BASELINE (CIRCULAR & ELÍPTICA)
# ==============================================================================

#' Calibração split-conformal com seleção de geometria da região
#' @param model Modelo preditivo treinado
#' @param calib_dataset Dataset independente de calibração
#' @param scaler Parâmetros de escalonamento
#' @param shape_dataset Dataset opcional out-of-fold para forma da região (se NULL, usa calib_dataset)
#' @param alpha Nível de significância (cobertura nominal 1 - alpha)
#' @param region_type Tipo da região conformal: "circular" ou "elliptic"
#' @param ridge Regularização de Tikhonov para matrizes de covariância
#' @param verbose Lógico: se TRUE, imprime resumo da calibração
#' @return Lista com parâmetros calibrados (raios circulares ou elipses de Mahalanobis)
calibrate_conformal_baseline <- function(model,
                                         calib_dataset,
                                         scaler,
                                         shape_dataset = NULL,
                                         alpha = 0.10,
                                         region_type = c("circular", "elliptic"),
                                         ridge = 1e-6,
                                         verbose = TRUE) {
  region_type <- match.arg(region_type)

  # Dados de erro para estimar a forma da região (shape)
  shape_source <- if (is.null(shape_dataset)) calib_dataset else shape_dataset
  shape_res <- compute_baseline_errors(model, shape_source, scaler)

  # Dados de erro para cálculo dos escores e quantis na calibração
  calib_res <- if (is.null(shape_dataset)) shape_res else compute_baseline_errors(model, calib_dataset, scaler)

  n_players <- nrow(calib_res$err_matrix)
  n_events  <- length(calib_res$err_by_event)
  q_level   <- min(1.0, ceiling((n_players + 1) * (1 - alpha)) / n_players)
  q_level_evt <- min(1.0, ceiling((n_events + 1) * (1 - alpha)) / n_events)

  if (region_type == "circular") {
    # ---- MODALIDADE CIRCULAR ----
    # 1. Forma temporal típica: mediana dos erros em metros
    s_hat <- apply(shape_res$err_matrix, 2, median)
    s_hat[s_hat < 1e-4] <- 1e-4

    # 2. Escore supremo normalizado por jogador: R_i = max_t (e_{i,t} / s_hat_t)
    R_player <- apply(calib_res$err_matrix, 1, function(row) max(row / s_hat))
    q_hat_player <- quantile(R_player, probs = q_level, names = FALSE)

    # Raios simultâneos r_t = q̂ · ŝ_t
    r_simultaneous <- q_hat_player * s_hat

    # Alvo A: Jogada completa (todas as trajetórias simultaneamente contidas)
    R_event <- sapply(calib_res$err_by_event, function(mat) {
      max(apply(mat, 1, function(row) max(row / s_hat)))
    })
    q_hat_event <- quantile(R_event, probs = q_level_evt, names = FALSE)
    r_event <- q_hat_event * s_hat

    # Quantis marginais pontuais por segundo (contraste metodológico)
    q_pointwise <- numeric(5)
    for (t in 1:5) {
      q_pointwise[t] <- quantile(calib_res$err_matrix[, t], probs = q_level, names = FALSE)
    }

    if (verbose) {
      cat(sprintf("\n=== CALIBRAÇÃO CONFORME BASELINE [CIRCULAR] (1 - alpha = %.0f%%) ===\n", (1 - alpha) * 100))
      cat(sprintf("Amostra: %d jogadas independentes (%d trajetórias)\n", n_events, n_players))
      cat("Forma da banda s_hat (m):", round(s_hat, 2), "\n")
      cat(sprintf("q̂ simultâneo (Alvo B): %.3f\n", q_hat_player))
      cat("Raios simultâneos r_t (m):", round(r_simultaneous, 2), "\n")
      cat(sprintf("q̂ jogada completa (Alvo A): %.3f\n", q_hat_event))
      cat("Raios Alvo A (m):", round(r_event, 2), "\n")
      cat("Raios pontuais marginais (m):", round(q_pointwise, 2), "\n\n")
    }

    list(
      region_type = "circular",
      s_hat = s_hat,
      q_hat_player = q_hat_player,
      r_simultaneous = r_simultaneous,
      q_hat_event = q_hat_event,
      r_event = r_event,
      r_pointwise = q_pointwise,
      alpha = alpha,
      n_players = n_players,
      n_events = n_events
    )

  } else {
    # ---- MODALIDADE ELÍPTICA (MAHALANOBIS) ----
    # 1. Matrizes de covariância espacial do erro 2D por horizonte
    Sigma_t <- estimate_error_covariances(shape_res$err_array, ridge = ridge)

    # 2. Escores de Mahalanobis na calibração: R_i = max_t sqrt(e' Sigma_t^{-1} e)
    R_player <- mahalanobis_scores(calib_res$err_array, Sigma_t)
    q_hat_player <- quantile(R_player, probs = q_level, names = FALSE)

    # Escore Alvo A (max por jogada)
    # Reorganiza escores de Mahalanobis por jogada
    cum_idx <- 0
    R_event <- numeric(n_events)
    for (k in 1:n_events) {
      n_k <- nrow(calib_res$err_by_event[[k]])
      idx_k <- (cum_idx + 1):(cum_idx + n_k)
      R_event[k] <- max(R_player[idx_k])
      cum_idx <- cum_idx + n_k
    }
    q_hat_event <- quantile(R_event, probs = q_level_evt, names = FALSE)

    # 3. Semi-eixos e orientação das elipses via decomposição espectral
    ellipse_pars <- do.call(rbind, lapply(1:5, function(t) {
      eig <- eigen(Sigma_t[[t]], symmetric = TRUE)
      ord <- order(eig$values, decreasing = TRUE)
      lam <- eig$values[ord]
      v1  <- eig$vectors[, ord[1]]
      data.frame(
        time_step = t,
        a = q_hat_player * sqrt(max(lam[1], 0)),
        b = q_hat_player * sqrt(max(lam[2], 0)),
        angle = atan2(v1[2], v1[1]),
        area = pi * (q_hat_player^2) * sqrt(max(lam[1], 0) * max(lam[2], 0))
      )
    }))

    ellipse_pars_event <- do.call(rbind, lapply(1:5, function(t) {
      eig <- eigen(Sigma_t[[t]], symmetric = TRUE)
      ord <- order(eig$values, decreasing = TRUE)
      lam <- eig$values[ord]
      v1  <- eig$vectors[, ord[1]]
      data.frame(
        time_step = t,
        a = q_hat_event * sqrt(max(lam[1], 0)),
        b = q_hat_event * sqrt(max(lam[2], 0)),
        angle = atan2(v1[2], v1[1]),
        area = pi * (q_hat_event^2) * sqrt(max(lam[1], 0) * max(lam[2], 0))
      )
    }))

    if (verbose) {
      cat(sprintf("\n=== CALIBRAÇÃO CONFORME BASELINE [ELÍPTICA] (1 - alpha = %.0f%%) ===\n", (1 - alpha) * 100))
      cat(sprintf("Amostra: %d jogadas independentes (%d trajetórias)\n", n_events, n_players))
      cat(sprintf("q̂ Mahalanobis (Alvo B): %.3f\n", q_hat_player))
      cat("Semi-eixos a_t, b_t (m):\n")
      print(round(ellipse_pars[, c("time_step", "a", "b", "area")], 2))
      cat(sprintf("q̂ Mahalanobis jogada completa (Alvo A): %.3f\n\n", q_hat_event))
    }

    list(
      region_type = "elliptic",
      Sigma_t = Sigma_t,
      q_hat_player = q_hat_player,
      ellipse_pars = ellipse_pars,
      q_hat_event = q_hat_event,
      ellipse_pars_event = ellipse_pars_event,
      alpha = alpha,
      n_players = n_players,
      n_events = n_events
    )
  }
}


# ==============================================================================
# 3. PREDIÇÃO CONFORME & TESTE DE COBERTURA
# ==============================================================================

#' Executa inferência no dataset de teste com regiões conformes calibradas
#' @param model Modelo preditivo
#' @param dataset TrajectoryDataset de teste
#' @param scaler Parâmetros de escalonamento
#' @param calib_res Resultado retornado por calibrate_conformal_baseline
#' @param region_type "circular" ou "elliptic" (se omitido, usa o do calib_res)
#' @return Data frame tabular com previsões, geometria das regiões e indicador de cobertura
predict_conformal_baseline <- function(model,
                                       dataset,
                                       scaler,
                                       calib_res,
                                       region_type = c("circular", "elliptic")) {
  if (missing(region_type)) {
    region_type <- calib_res$region_type
  } else {
    region_type <- match.arg(region_type)
  }

  model$eval()
  device <- model$parameters[[1]]$device
  ts <- mu_sd_tensors(scaler, device)

  loader <- dataloader(dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)
  results_df <- list()

  with_no_grad({
    coro::loop(for (batch in loader) {
      evt_id      <- batch$event_id
      player_ids  <- batch$player_ids
      team_codes  <- batch$team_codes
      team_names  <- batch$team_names

      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat  <- batch$x_cat$to(device = device)
      batch$y      <- batch$y$to(device = device)

      preds <- model(batch)

      preds_real  <- preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu

      preds_arr  <- as.array(preds_real$cpu())
      y_true_arr <- as.array(y_true_real$cpu())

      num_nodes <- dim(preds_arr)[1]
      seq_len   <- dim(preds_arr)[2]

      for (node in 1:num_nodes) {
        pid   <- player_ids[node]
        tcode <- team_codes[node]
        tname <- team_names[node]

        for (t in 1:seq_len) {
          px <- preds_arr[node, t, 1]
          py <- preds_arr[node, t, 2]
          tx <- y_true_arr[node, t, 1]
          ty <- y_true_arr[node, t, 2]

          dist <- sqrt((px - tx)^2 + (py - ty)^2)

          if (region_type == "circular") {
            rad <- calib_res$r_simultaneous[t]
            is_covered <- dist <= rad

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
              conf_radius = rad,
              distance = dist,
              covered = is_covered,
              stringsAsFactors = FALSE
            )
          } else {
            e <- c(px - tx, py - ty)
            Sinv <- solve(calib_res$Sigma_t[[t]])
            md <- sqrt(sum((e %*% Sinv) * e))
            is_covered <- md <= calib_res$q_hat_player

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
              a = calib_res$ellipse_pars$a[t],
              b = calib_res$ellipse_pars$b[t],
              angle = calib_res$ellipse_pars$angle[t],
              distance = dist,
              mahalanobis = md,
              covered = is_covered,
              stringsAsFactors = FALSE
            )
          }
        }
      }
    })
  })

  bind_rows(results_df)
}

#' Avalia estatisticamente as predições conformes no conjunto de teste
#' @param test_predictions Data frame retornado por predict_conformal_baseline
#' @param alpha Nível de significância
#' @return Lista com resumos de cobertura simultânea, pontual e erros por time
evaluate_conformal_summary <- function(test_predictions, alpha = 0.10) {
  # 1. Cobertura Simultânea da Trajetória (Alvo >= 1 - alpha)
  simultaneous <- test_predictions |>
    group_by(event_id, player_id) |>
    summarize(trajectory_covered = all(covered), .groups = "drop") |>
    summarize(
      total_trajectories = n(),
      empirical_coverage = mean(trajectory_covered),
      target_coverage = 1 - alpha
    )

  # 2. Cobertura Pontual por Horizonte
  pointwise <- test_predictions |>
    group_by(time_step) |>
    summarize(
      coverage = mean(covered),
      mean_distance = mean(distance),
      .groups = "drop"
    )

  # 3. Desempenho por Papel Tático
  by_role <- test_predictions |>
    group_by(team_code) |>
    summarize(
      ADE = mean(distance),
      FDE = mean(distance[time_step == max(time_step)]),
      coverage = mean(covered),
      .groups = "drop"
    )

  list(
    simultaneous_coverage = simultaneous,
    pointwise_coverage = pointwise,
    role_metrics = by_role
  )
}
