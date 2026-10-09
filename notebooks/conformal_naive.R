# ==============================================================================
# CONFORMAL_NAIVE.R — CONFORMAL NAIVE COM FORMA TEMPORAL OUT-OF-FOLD
# ==============================================================================
# Implementação fiel da calibração split-conformal (Izbicki 2026; Diquigiovanni
# et al. 2022) com separação metodológica estrita:
#  - Forma da região (ŝ_t ou Σ̂_t) estimada em resíduos out-of-fold (validação)
#  - Calibração de escores realizada em jogadas independentes (calibração)
#  - Alvo B naive (por jogador) vs Alvo A (jogada completa)
#  - Variante exata de aferição por sorteio de 1 jogador por jogada
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
# 1. EXTRAÇÃO DE ERROS E MATRIZES RESIDUAIS
# ==============================================================================

#' Extrai resíduos escalares e vetoriais por jogada e empilhados
#' @param model Modelo treinado
#' @param dataset TrajectoryDataset
#' @param scaler Parâmetros do scaler para conversão física
#' @return Lista com err_matrix, err_by_event, err_array e err_vec_by_event
compute_error_matrices <- function(model, dataset, scaler) {
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

      # Desnormalização para metros reais
      preds_real  <- preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu

      # Erro escalar (distância Euclidiana)
      diff_sq <- (preds_real - y_true_real)$pow(2)
      distances <- torch_sqrt(diff_sq$sum(dim = 3) + 1e-8)
      dist_mat <- as.matrix(distances$cpu())
      err_by_event[[length(err_by_event) + 1]] <- dist_mat

      # Erro vetorial 2D (dx, dy)
      vec_arr <- as.array((preds_real - y_true_real)$cpu())
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

#' Estima a forma da banda circular (mediana dos erros) com piso numérico
#' @param err_matrix Matriz de distâncias [n_trajetorias, 5]
#' @param floor_eps Piso numérico mínimo para evitar divisão por zero
#' @return Vetor s_hat com 5 medianas
estimate_band_shape <- function(err_matrix, floor_eps = 1e-4) {
  s_hat <- apply(err_matrix, 2, median)
  s_hat[s_hat < floor_eps] <- floor_eps
  s_hat
}


# ==============================================================================
# 2. CALIBRAÇÃO CONFORME NAIVE (CIRCULAR & ELÍPTICA)
# ==============================================================================

#' Calibração split-conformal naive com suporte a regiões circulares e elípticas
#' @param model Modelo treinado
#' @param calib_dataset Dataset independente de calibração
#' @param s_hat_dataset Dataset out-of-fold para forma da região (validação)
#' @param scaler Parâmetros de escalonamento
#' @param alpha Nível de significância (cobertura nominal 1 - alpha)
#' @param region_type Tipo da região: "circular" ou "elliptic"
#' @param n_exact_draws Número de repetições Monte Carlo para a variante exata
#' @param exact_seed Semente aleatória para os sorteios exatos
#' @param ridge Regularização de Tikhonov para matrizes de covariância
#' @param verbose Lógico: se TRUE, imprime resumo da calibração
#' @return Lista contendo parâmetros calibrados e estatísticas de ordem
calibrate_conformal_naive <- function(model,
                                      calib_dataset,
                                      s_hat_dataset,
                                      scaler,
                                      alpha = 0.10,
                                      region_type = c("circular", "elliptic"),
                                      n_exact_draws = 200,
                                      exact_seed = 7301,
                                      ridge = 1e-6,
                                      verbose = TRUE) {
  region_type <- match.arg(region_type)

  # ---- Passo 1: Forma da região em resíduos out-of-fold (validação) ----
  shape_res <- compute_error_matrices(model, s_hat_dataset, scaler)
  n_events_s_hat <- length(shape_res$err_by_event)

  # ---- Passo 2: Escores no conjunto de calibração independente ----
  calib_res <- compute_error_matrices(model, calib_dataset, scaler)
  err_by_event <- calib_res$err_by_event
  err_matrix   <- calib_res$err_matrix
  err_array    <- calib_res$err_array

  n_players <- nrow(err_matrix)
  n_events  <- length(err_by_event)

  q_level_player <- min(1.0, ceiling((n_players + 1) * (1 - alpha)) / n_players)
  q_level_event  <- min(1.0, ceiling((n_events + 1) * (1 - alpha)) / n_events)

  if (region_type == "circular") {
    # ---- MODALIDADE CIRCULAR ----
    s_hat <- estimate_band_shape(shape_res$err_matrix)

    # (Alvo B) R_{k,i} = max_{t in H} e_{k,i,t} / s_hat_t
    R_player <- apply(err_matrix, 1, function(row) max(row / s_hat))
    q_hat_player <- quantile(R_player, probs = q_level_player, names = FALSE)
    r_simultaneous <- q_hat_player * s_hat

    # (Alvo A) R_k = max_i max_{t in H} e_{k,i,t} / s_hat_t
    R_event <- sapply(err_by_event, function(mat) {
      max(apply(mat, 1, function(row) max(row / s_hat)))
    })
    q_hat_event <- quantile(R_event, probs = q_level_event, names = FALSE)
    r_event <- q_hat_event * s_hat

    # Variante exata do Alvo B (aferição): 1 jogador sorteado por jogada
    set.seed(exact_seed)
    q_hat_exact_draws <- replicate(n_exact_draws, {
      R_exact <- sapply(err_by_event, function(mat) {
        max(mat[sample.int(nrow(mat), size = 1), ] / s_hat)
      })
      quantile(R_exact, probs = q_level_event, names = FALSE)
    })
    q_hat_exact <- mean(q_hat_exact_draws)
    q_hat_exact_sd <- sd(q_hat_exact_draws)

    # Quantis marginais pontuais
    q_pointwise <- numeric(5)
    for (t in 1:5) {
      q_pointwise[t] <- quantile(err_matrix[, t], probs = q_level_player, names = FALSE)
    }

    if (verbose) {
      cat(sprintf("\n=== CALIBRAÇÃO CONFORME NAIVE [CIRCULAR] (1 - alpha = %.0f%%) ===\n", (1 - alpha) * 100))
      cat(sprintf("ŝ_t estimada em %d jogadas out-of-fold (validação)\n", n_events_s_hat))
      cat("Forma da banda ŝ_t (m):", round(s_hat, 2), "\n")
      cat(sprintf("Calibração: %d jogadas independentes (%d trajetórias)\n", n_events, n_players))
      cat(sprintf("q̂ Alvo B naive: %.3f\n", q_hat_player))
      cat("Raios conformes simultâneos r_t (m):", round(r_simultaneous, 2), "\n")
      cat(sprintf("q̂ variante exata Alvo B (%d sorteios): %.3f ± %.3f\n",
                  n_exact_draws, q_hat_exact, q_hat_exact_sd))
      cat(sprintf("q̂ Alvo A (jogada completa): %.3f\n", q_hat_event))
      cat("Raios Alvo A (m):", round(r_event, 2), "\n\n")
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
      q_hat_exact = q_hat_exact,
      q_hat_exact_sd = q_hat_exact_sd,
      q_hat_exact_draws = q_hat_exact_draws,
      n_players = n_players,
      n_events = n_events,
      n_events_s_hat = n_events_s_hat,
      s_hat_source = "validacao_out_of_fold"
    )

  } else {
    # ---- MODALIDADE ELÍPTICA (MAHALANOBIS) ----
    Sigma_t <- estimate_error_covariances(shape_res$err_array, ridge = ridge)

    # Escore de Mahalanobis na calibração
    R_player <- mahalanobis_scores(err_array, Sigma_t)
    q_hat_player <- quantile(R_player, probs = q_level_player, names = FALSE)

    # Reagrupa por jogada para Alvo A e variante exata
    cum_idx <- 0
    R_by_play <- list()
    for (k in 1:n_events) {
      n_k <- nrow(err_by_event[[k]])
      idx_k <- (cum_idx + 1):(cum_idx + n_k)
      R_by_play[[k]] <- R_player[idx_k]
      cum_idx <- cum_idx + n_k
    }

    R_event <- sapply(R_by_play, max)
    q_hat_event <- quantile(R_event, probs = q_level_event, names = FALSE)

    # Variante exata elíptica (1 jogador sorteado por jogada)
    set.seed(exact_seed)
    q_hat_exact_draws <- replicate(n_exact_draws, {
      R_exact <- sapply(R_by_play, function(v) v[sample.int(length(v), size = 1)])
      quantile(R_exact, probs = q_level_event, names = FALSE)
    })
    q_hat_exact <- mean(q_hat_exact_draws)
    q_hat_exact_sd <- sd(q_hat_exact_draws)

    # Parâmetros das elipses
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
      cat(sprintf("\n=== CALIBRAÇÃO CONFORME NAIVE [ELÍPTICA] (1 - alpha = %.0f%%) ===\n", (1 - alpha) * 100))
      cat(sprintf("Σ̂_t estimada em %d jogadas out-of-fold (validação)\n", n_events_s_hat))
      cat(sprintf("Calibração: %d jogadas independentes (%d trajetórias)\n", n_events, n_players))
      cat(sprintf("q̂ Alvo B naive (Mahalanobis): %.3f\n", q_hat_player))
      cat("Semi-eixos a_t, b_t (m):\n")
      print(round(ellipse_pars[, c("time_step", "a", "b", "area")], 2))
      cat(sprintf("q̂ variante exata Alvo B (%d sorteios): %.3f ± %.3f\n",
                  n_exact_draws, q_hat_exact, q_hat_exact_sd))
      cat(sprintf("q̂ Alvo A (jogada completa): %.3f\n\n", q_hat_event))
    }

    list(
      region_type = "elliptic",
      Sigma_t = Sigma_t,
      q_hat_player = q_hat_player,
      ellipse_pars = ellipse_pars,
      q_hat_event = q_hat_event,
      ellipse_pars_event = ellipse_pars_event,
      alpha = alpha,
      q_hat_exact = q_hat_exact,
      q_hat_exact_sd = q_hat_exact_sd,
      q_hat_exact_draws = q_hat_exact_draws,
      n_players = n_players,
      n_events = n_events,
      n_events_s_hat = n_events_s_hat,
      shape_source = "validacao_out_of_fold"
    )
  }
}


# ==============================================================================
# 3. PREDIÇÃO & INFERÊNCIA DE TESTE
# ==============================================================================

#' Produz predições e checagem de cobertura no teste
#' @param model Modelo preditivo
#' @param dataset TrajectoryDataset de teste
#' @param scaler Parâmetros de escalonamento
#' @param calib_res Resultado de calibrate_conformal_naive
#' @param region_type "circular" ou "elliptic"
#' @return Data frame tabular com cobertura e métricas
predict_conformal_naive <- function(model,
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

#' Cobertura empírica medida por repetições de sorteio ao nível da jogada (Alvo B)
#' @param test_predictions Data frame gerado por predict_conformal_naive
#' @param B Número de repetições Monte Carlo
#' @param seed Semente aleatória
#' @return Tibble com média e desvio padrão da cobertura ao nível da jogada
coverage_by_play_draws <- function(test_predictions, B = 200, seed = 7302) {
  set.seed(seed)

  trajectories <- test_predictions |>
    group_by(event_id, player_id) |>
    summarize(trajectory_covered = all(covered), .groups = "drop")

  covs <- replicate(B, {
    trajectories |>
      group_by(event_id) |>
      slice_sample(n = 1) |>
      pull(trajectory_covered) |>
      mean()
  })

  tibble(
    n_test_events = n_distinct(test_predictions$event_id),
    n_draws = B,
    mean_coverage = mean(covs),
    sd_coverage = sd(covs)
  )
}
