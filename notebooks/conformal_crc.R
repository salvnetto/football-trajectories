# ==============================================================================
# CONFORMAL_CRC.R — CONTROLE DE RISCO CONFORME (CRC) & ESCALA ADAPTATIVA
# ==============================================================================
# Implementação fiel das Sugestões 4, 5 e 6 de Rafael Izbicki (2026),
# baseadas em Angelopoulos, Bates, Fisch, Lei & Schuster (ICLR 2024):
#
# 1. Sugestão 4 (Escala Adaptativa ŝ_t(x_i)):
#    Modelo condicional log(e_{i,t} + eps) = g(x_i, t) ajustado em resíduos
#    out-of-fold com covariáveis físicas e cinemáticas do jogador em T_obs = 25.
#
# 2. Sugestão 6 (Controle de Risco Conforme - CRC):
#    A unidade permutável é a JOGADA ofensiva k. Perda:
#      L_k(lambda) = (1 / N_k) * sum_{i=1}^{N_k} 1{ M_{k,i} > lambda }
#    com garantia finita: E[ L_nova(lambda_hat) ] <= alpha.
#
# 3. Geometria da Região Conforme (Parâmetro Obrigatório):
#    - region_type = "circular": raio adaptativo r_{i,t} = lambda_hat * ŝ_t(x_i)
#    - region_type = "elliptic": elipses adaptativas com covariância direcional
#      Γ̂_t dos resíduos normalizados por ŝ_t(x_i).
#
# Este módulo não contém funções de plotagem gráfica.
# ==============================================================================

suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
})

# ==============================================================================
# 1. EXTRAÇÃO DE COVARIÁVEIS DO JOGADOR EM T_obs = 25 (SUGESTÃO 4)
# ==============================================================================

#' Extrai características cinemáticas e contextuais dos nós no instante t = 25
#' @param batch Lote da jogada
#' @param scaler Parâmetros do scaler para conversão em unidades físicas
#' @return Tibble com uma linha por jogador da jogada
extract_player_covariates <- function(batch, scaler) {
  num_nodes <- batch$num_nodes
  x_cont_cpu <- as.array(batch$x_cont$cpu()) # [num_nodes, 25, 8]
  x_cat_cpu  <- as.array(batch$x_cat$cpu())  # [num_nodes, 2] -> (team_id_idx, role_idx)

  # Frame 25 (último frame observado)
  f25 <- x_cont_cpu[, 25, ]

  # Desnormalização para coordenadas físicas reais
  cx <- scaler$center["x"];           sx <- scaler$scale["x"]
  cy <- scaler$center["y"];           sy <- scaler$scale["y"]
  cbs <- scaler$center["ball_speed"]; sbs <- scaler$scale["ball_speed"]
  cdist <- scaler$center["dist_to_ball"]; sdist <- scaler$scale["dist_to_ball"]
  cvx <- scaler$center["vel_x"];      svx <- scaler$scale["vel_x"]
  cvy <- scaler$center["vel_y"];      svy <- scaler$scale["vel_y"]

  x_m     <- f25[, 1] * sx + cx
  y_m     <- f25[, 2] * sy + cy
  bs_m    <- f25[, 3] * sbs + cbs
  dist_m  <- f25[, 6] * sdist + cdist
  vel_x_m <- f25[, 7] * svx + cvx
  vel_y_m <- f25[, 8] * svy + cvy
  speed_m <- sqrt(vel_x_m^2 + vel_y_m^2)

  role_idx <- x_cat_cpu[, 2]
  is_attack <- as.numeric(role_idx == 2L)

  tibble(
    event_id = batch$event_id,
    node_id = 1:num_nodes,
    player_id = batch$player_ids,
    team_code = batch$team_codes,
    x = x_m,
    y = y_m,
    speed = speed_m,
    vel_x = vel_x_m,
    vel_y = vel_y_m,
    dist_to_ball = dist_m,
    ball_speed = bs_m,
    is_attack = is_attack
  )
}


# ==============================================================================
# 2. MODELO DE ESCALA ADAPTATIVA CONDICIONAL ŝ_t(x_i) (SUGESTÃO 4)
# ==============================================================================

#' Extrai resíduos escalares e vetoriais out-of-fold pareados com covariáveis
#' @param model Modelo preditivo treinado
#' @param dataset TrajectoryDataset out-of-fold (validação)
#' @param scaler Parâmetros de escalonamento
#' @return Data frame com resíduos pareados por jogador e horizonte
extract_oof_residual_dataset <- function(model, dataset, scaler) {
  model$eval()
  device <- model$parameters[[1]]$device
  ts <- mu_sd_tensors(scaler, device)
  loader <- dataloader(dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  oof_rows <- list()

  with_no_grad({
    coro::loop(for (batch in loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat  <- batch$x_cat$to(device = device)
      batch$y      <- batch$y$to(device = device)

      preds <- model(batch)

      # Erro físico em metros
      preds_real  <- preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu
      diff_real   <- preds_real - y_true_real

      diff_sq   <- diff_real$pow(2)
      distances <- torch_sqrt(diff_sq$sum(dim = 3) + 1e-8)
      err_mat   <- as.matrix(distances$cpu())
      diff_arr  <- as.array(diff_real$cpu())

      covs <- extract_player_covariates(batch, scaler)

      # Expande para cada horizonte t = 1..5
      for (t in 1:5) {
        step_df <- covs |>
          mutate(
            time_step = t,
            error_m = err_mat[, t],
            error_x = diff_arr[, t, 1],
            error_y = diff_arr[, t, 2]
          )
        oof_rows[[length(oof_rows) + 1]] <- step_df
      }
    })
  })

  bind_rows(oof_rows)
}

#' Ajusta modelo de regressão log-linear para prever resíduo condicional g(x_i, t)
#' @param oof_df Data frame out-of-fold gerado por extract_oof_residual_dataset
#' @param eps Piso numérico de segurança para log(e + eps)
#' @return Lista contendo modelo lm ajustado e estatísticas de padronização
fit_adaptive_scale_model <- function(oof_df, eps = 1e-4) {
  safe_sd <- function(v) {
    s <- sd(v, na.rm = TRUE)
    if (is.na(s) || s < 1e-5) 1.0 else s
  }

  mean_speed <- mean(oof_df$speed, na.rm = TRUE)
  sd_speed   <- safe_sd(oof_df$speed)
  mean_dist  <- mean(oof_df$dist_to_ball, na.rm = TRUE)
  sd_dist    <- safe_sd(oof_df$dist_to_ball)
  mean_bs    <- mean(oof_df$ball_speed, na.rm = TRUE)
  sd_bs      <- safe_sd(oof_df$ball_speed)

  df <- oof_df |>
    mutate(
      log_err = log(error_m + eps),
      t = as.numeric(time_step),
      t2 = t^2,
      speed_s = (speed - mean_speed) / sd_speed,
      dist_s = (dist_to_ball - mean_dist) / sd_dist,
      ball_speed_s = (ball_speed - mean_bs) / sd_bs
    )

  fit <- lm(
    log_err ~ t + t2 + speed_s + dist_s + ball_speed_s + is_attack +
      speed_s:t + dist_s:t,
    data = df
  )

  list(
    model = fit,
    eps = eps,
    scalers = list(
      mean_speed = mean_speed, sd_speed = sd_speed,
      mean_dist = mean_dist, sd_dist = sd_dist,
      mean_bs = mean_bs, sd_bs = sd_bs
    )
  )
}

#' Prediz ŝ_t(x_i) = exp( ĝ(x_i, t) ) para novas observações
#' @param scale_model Modelo retornado por fit_adaptive_scale_model
#' @param player_covs Covariáveis extraídas por extract_player_covariates
#' @param time_step Passo temporal (1..5)
#' @param floor_eps Limiar inferior mínimo para a escala
#' @return Vetor com as escalas adaptativas individuais
predict_adaptive_scale <- function(scale_model, player_covs, time_step, floor_eps = 1e-3) {
  sc <- scale_model$scalers
  t_val <- as.numeric(time_step)

  df_pred <- player_covs |>
    mutate(
      t = t_val,
      t2 = t_val^2,
      speed_s = (speed - sc$mean_speed) / sc$sd_speed,
      dist_s = (dist_to_ball - sc$mean_dist) / sc$sd_dist,
      ball_speed_s = (ball_speed - sc$mean_bs) / sc$sd_bs
    )

  pred_log <- predict(scale_model$model, newdata = df_pred)
  pmax(exp(pred_log), floor_eps)
}

#' Estima a matriz de covariância 2x2 dos resíduos normalizados pela escala adaptativa
#' @param oof_df Data frame de resíduos out-of-fold com error_x e error_y
#' @param scale_model Modelo de escala adaptativa
#' @param ridge Regularização de Tikhonov
#' @return Lista de 5 matrizes 2x2 Gamma_t
estimate_adaptive_covariance <- function(oof_df, scale_model, ridge = 1e-6) {
  Gamma_t <- list()
  for (t in 1:5) {
    sub_df <- oof_df |> filter(time_step == t)
    s_hat <- predict_adaptive_scale(scale_model, sub_df, time_step = t)
    norm_ex <- sub_df$error_x / s_hat
    norm_ey <- sub_df$error_y / s_hat
    mat <- cbind(norm_ex, norm_ey)
    Gamma_t[[t]] <- cov(mat) + diag(ridge, 2)
  }
  Gamma_t
}


# ==============================================================================
# 3. CONTROLE DE RISCO CONFORME — CRC (SUGESTÃO 6)
# ==============================================================================

#' Computa os escores normalizados adaptativos M_{k,i} por jogada
#' @param model Modelo preditivo treinado
#' @param dataset TrajectoryDataset
#' @param scale_model Modelo de escala adaptativa
#' @param scaler Parâmetros do scaler
#' @param region_type "circular" ou "elliptic"
#' @param Gamma_t Lista de covariâncias normalizadas (necessário para "elliptic")
#' @return Lista de avaliações por jogada
compute_play_conformal_evals <- function(model, dataset, scale_model, scaler,
                                        region_type = c("circular", "elliptic"),
                                        Gamma_t = NULL) {
  region_type <- match.arg(region_type)

  model$eval()
  device <- model$parameters[[1]]$device
  ts <- mu_sd_tensors(scaler, device)
  loader <- dataloader(dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  play_evals <- list()

  with_no_grad({
    coro::loop(for (batch in loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat  <- batch$x_cat$to(device = device)
      batch$y      <- batch$y$to(device = device)

      preds <- model(batch)
      preds_real  <- preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu
      diff_real   <- preds_real - y_true_real

      diff_sq   <- diff_real$pow(2)
      distances <- torch_sqrt(diff_sq$sum(dim = 3) + 1e-8)
      err_mat   <- as.matrix(distances$cpu()) # [num_nodes, 5]
      diff_arr  <- as.array(diff_real$cpu())  # [num_nodes, 5, 2]

      covs <- extract_player_covariates(batch, scaler)
      num_nodes <- batch$num_nodes
      seq_len <- dim(err_mat)[2]

      # Matriz de escalas adaptativas ŝ_t(x_i) [num_nodes, 5]
      s_hat_mat <- matrix(NA_real_, nrow = num_nodes, ncol = seq_len)
      for (t in 1:seq_len) {
        s_hat_mat[, t] <- predict_adaptive_scale(scale_model, covs, time_step = t)
      }

      if (region_type == "circular") {
        # M_{k,i} = max_t [ e_{k,i,t} / ŝ_t(x_{k,i}) ]
        norm_err_mat <- err_mat / s_hat_mat
        M_ki <- apply(norm_err_mat, 1, max)
      } else {
        # Escore de Mahalanobis adaptativo:
        # d_M = (1 / ŝ_t) * sqrt(e' Gamma_t^{-1} e)
        M_mat <- matrix(NA_real_, nrow = num_nodes, ncol = seq_len)
        for (t in 1:seq_len) {
          Ginv <- solve(Gamma_t[[t]])
          for (i in 1:num_nodes) {
            e <- diff_arr[i, t, ]
            M_mat[i, t] <- sqrt(sum((e %*% Ginv) * e)) / s_hat_mat[i, t]
          }
        }
        M_ki <- apply(M_mat, 1, max)
      }

      play_evals[[length(play_evals) + 1]] <- list(
        event_id = batch$event_id,
        num_nodes = num_nodes,
        err_mat = err_mat,
        diff_arr = diff_arr,
        s_hat_mat = s_hat_mat,
        M_ki = M_ki,
        covs = covs
      )
    })
  })

  play_evals
}

#' Calibra lambda_hat via Conformal Risk Control (Angelopoulos et al. 2024; Izbicki 2026 §6)
#' @param play_evals Lista de avaliações por jogada gerada por compute_play_conformal_evals
#' @param alpha Nível de risco admissível (esperança de fração de trajetórias não cobertas)
#' @return Lista contendo lambda_hat, empirical_risk e bound finito
calibrate_conformal_risk_control <- function(play_evals, alpha = 0.10) {
  n <- length(play_evals)

  all_M <- unlist(lapply(play_evals, function(p) p$M_ki))
  candidate_lambdas <- sort(unique(c(0, all_M, all_M + 1e-5)))

  # Perda média da jogada L_k(lambda) = mean(M_{k,i} > lambda)
  loss_at_lambda <- function(lam) {
    mean(sapply(play_evals, function(p) {
      mean(p$M_ki > lam)
    }))
  }

  # Critério CRC exato: [n / (n + 1)] * L_bar(lambda) + [1 / (n + 1)] <= alpha
  target_loss_bound <- (alpha * (n + 1) - 1) / n
  finite_sample_valid <- target_loss_bound > 0

  if (!finite_sample_valid) {
    warning(sprintf(
      "Aviso Teórico CRC: n = %d é insuficiente para a garantia exata 1/(n+1) <= alpha (exige n >= %d). Utilizando calibração plug-in L_bar(lambda) <= alpha.",
      n, ceiling(1 / alpha) - 1
    ))
    criterion_func <- function(lam) loss_at_lambda(lam) <= alpha
  } else {
    criterion_func <- function(lam) {
      (n / (n + 1)) * loss_at_lambda(lam) + (1 / (n + 1)) <= alpha
    }
  }

  lambda_hat <- Inf
  for (lam in candidate_lambdas) {
    if (criterion_func(lam)) {
      lambda_hat <- lam
      break
    }
  }

  if (is.infinite(lambda_hat)) {
    lambda_hat <- max(candidate_lambdas)
  }

  emp_loss <- loss_at_lambda(lambda_hat)
  upper_bound <- (n / (n + 1)) * emp_loss + (1 / (n + 1))

  list(
    lambda_hat = lambda_hat,
    alpha = alpha,
    n_plays = n,
    total_trajectories = length(all_M),
    empirical_risk = emp_loss,
    crc_upper_bound = upper_bound,
    finite_sample_valid = finite_sample_valid
  )
}

#' Calibração split-conformal com escala adaptativa e CRC
#' @param model Modelo preditivo treinado
#' @param calib_dataset Dataset independente de calibração
#' @param oof_scale_dataset Dataset out-of-fold para ajuste de ŝ_t(x_i) (validação)
#' @param scaler Parâmetros de escalonamento
#' @param alpha Nível de risco nominal
#' @param region_type "circular" ou "elliptic"
#' @param ridge Regularização de Tikhonov
#' @param verbose Lógico: se TRUE, imprime resumo da calibração
#' @return Lista com parâmetros calibrados (lambda_hat, scale_model, Gamma_t, crc_calib)
calibrate_split_conformal_adaptive_crc <- function(model,
                                                  calib_dataset,
                                                  oof_scale_dataset,
                                                  scaler,
                                                  alpha = 0.10,
                                                  region_type = c("circular", "elliptic"),
                                                  ridge = 1e-6,
                                                  verbose = TRUE) {
  region_type <- match.arg(region_type)

  # 1. Ajusta o modelo de escala adaptativa ŝ_t(x_i) nos resíduos out-of-fold da validação
  oof_df <- extract_oof_residual_dataset(model, oof_scale_dataset, scaler)
  scale_model <- fit_adaptive_scale_model(oof_df)

  # 2. Se elíptico, estima covariância direcional normalizada Gamma_t
  Gamma_t <- NULL
  if (region_type == "elliptic") {
    Gamma_t <- estimate_adaptive_covariance(oof_df, scale_model, ridge = ridge)
  }

  # 3. Avalia as jogadas de calibração
  calib_evals <- compute_play_conformal_evals(
    model = model,
    dataset = calib_dataset,
    scale_model = scale_model,
    scaler = scaler,
    region_type = region_type,
    Gamma_t = Gamma_t
  )

  # 4. Calibra lambda_hat via CRC
  crc_calib <- calibrate_conformal_risk_control(calib_evals, alpha = alpha)

  if (verbose) {
    cat(sprintf("\n=== CALIBRAÇÃO SPLIT-CONFORMAL ADAPTATIVA CRC [%s] (alpha = %.2f) ===\n",
                toupper(region_type), alpha))
    cat(sprintf("Jogadas de calibração: %d (%d trajetórias)\n",
                crc_calib$n_plays, crc_calib$total_trajectories))
    cat(sprintf("lambda_hat calibrado: %.4f\n", crc_calib$lambda_hat))
    cat(sprintf("Risco empírico na calibração E[L_nova]: %.3f (Alvo: <= %.2f)\n\n",
                crc_calib$empirical_risk, alpha))
  }

  list(
    region_type = region_type,
    lambda_hat = crc_calib$lambda_hat,
    scale_model = scale_model,
    Gamma_t = Gamma_t,
    crc_calib = crc_calib,
    calib_evals = calib_evals
  )
}

#' Executa K-Fold Cross-Conformal com CRC (Sugestão 5)
#' @param graphs Lista completa de grafos
#' @param scaler Parâmetros de escalonamento
#' @param cfg Configuração de treino
#' @param K Número de dobras
#' @param alpha Nível de risco nominal
#' @param region_type "circular" ou "elliptic"
#' @param seed Semente aleatória
#' @param verbose Lógico: se TRUE, imprime progresso das dobras
#' @param model_trainer Função opcional para treino (se NULL, usa train_social_lstm)
#' @return Lista com lambda_hat e calibração agregada
run_cross_conformal <- function(graphs, scaler, cfg, K = 5, alpha = 0.10,
                               region_type = c("circular", "elliptic"),
                               seed = 42, verbose = TRUE, model_trainer = NULL) {
  region_type <- match.arg(region_type)
  set_seed(seed)
  folds <- split_events_kfold(graphs, K = K, seed = seed)
  actual_K <- length(folds)

  if (verbose) {
    cat(sprintf("\n=== INICIANDO CROSS-CONFORMAL CRC [%s] (%d Dobras) ===\n", toupper(region_type), actual_K))
  }

  all_oof_play_evals <- list()
  fold_models <- list()
  fold_scale_models <- list()
  fold_Gammas <- list()

  for (k in seq_along(folds)) {
    test_idx <- folds[[k]]
    train_pool_idx <- unlist(folds[-k])

    n_pool <- length(train_pool_idx)
    n_internal_val <- max(1L, floor(0.20 * n_pool))
    shuffled_pool <- sample(train_pool_idx)
    internal_val_idx <- shuffled_pool[1:n_internal_val]
    internal_train_idx <- shuffled_pool[(n_internal_val + 1):n_pool]

    mirrored_train <- lapply(graphs[internal_train_idx], mirror_graph, scaler = scaler)
    train_graphs <- c(graphs[internal_train_idx], mirrored_train)

    train_ds <- TrajectoryDataset(train_graphs)
    val_ds   <- TrajectoryDataset(graphs[internal_val_idx])
    test_ds  <- TrajectoryDataset(graphs[test_idx])

    # Treinamento da rede na dobra
    trainer_fn <- if (!is.null(model_trainer)) model_trainer else train_social_lstm
    fit_res <- trainer_fn(
      train_dataset = train_ds,
      val_dataset = val_ds,
      scaler = scaler,
      cfg = cfg,
      seed = seed + k * 100,
      verbose = FALSE
    )
    fold_model <- fit_res$model

    # Resíduos out-of-fold na validação interna para ajustar ŝ_t(x_i)
    internal_oof_df <- extract_oof_residual_dataset(fold_model, val_ds, scaler)
    fold_scale_model <- fit_adaptive_scale_model(internal_oof_df)

    fold_Gamma <- NULL
    if (region_type == "elliptic") {
      fold_Gamma <- estimate_adaptive_covariance(internal_oof_df, fold_scale_model)
    }

    # Avalia na dobra k (estritamente não vista)
    fold_play_evals <- compute_play_conformal_evals(
      model = fold_model,
      dataset = test_ds,
      scale_model = fold_scale_model,
      scaler = scaler,
      region_type = region_type,
      Gamma_t = fold_Gamma
    )

    all_oof_play_evals <- c(all_oof_play_evals, fold_play_evals)
    fold_models[[k]] <- fold_model
    fold_scale_models[[k]] <- fold_scale_model
    fold_Gammas[[k]] <- fold_Gamma
  }

  crc_calib <- calibrate_conformal_risk_control(all_oof_play_evals, alpha = alpha)

  if (verbose) {
    cat(sprintf("\n=== CALIBRAÇÃO CROSS-CONFORMAL CRC CONCLUÍDA [%s] ===\n", toupper(region_type)))
    cat(sprintf("Jogadas avaliadas: %d (%d trajetórias)\n", crc_calib$n_plays, crc_calib$total_trajectories))
    cat(sprintf("lambda_hat calibrado: %.4f | Risco empírico: %.3f\n\n", crc_calib$lambda_hat, crc_calib$empirical_risk))
  }

  list(
    region_type = region_type,
    lambda_hat = crc_calib$lambda_hat,
    crc_calib = crc_calib,
    all_oof_play_evals = all_oof_play_evals,
    fold_models = fold_models,
    fold_scale_models = fold_scale_models,
    fold_Gammas = fold_Gammas,
    scaler = scaler,
    K = actual_K
  )
}


# ==============================================================================
# 4. INFERÊNCIA & AVALIAÇÃO DE REGIÕES ADAPTATIVAS
# ==============================================================================

#' Produz regiões conformes adaptativas (circulares ou elípticas) no conjunto de teste
#' @param model Modelo preditivo
#' @param dataset TrajectoryDataset de teste
#' @param scale_model Modelo de escala adaptativa
#' @param lambda_hat Fator conformal calibrado
#' @param scaler Parâmetros do scaler
#' @param region_type "circular" ou "elliptic"
#' @param Gamma_t Lista de matrizes 2x2 para "elliptic"
#' @return Data frame tabular com predições e geometrias individuais
predict_with_adaptive_regions <- function(model, dataset, scale_model, lambda_hat, scaler,
                                          region_type = c("circular", "elliptic"),
                                          Gamma_t = NULL) {
  region_type <- match.arg(region_type)

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
      diff_real   <- preds_real - y_true_real

      preds_arr  <- as.array(preds_real$cpu())
      y_true_arr <- as.array(y_true_real$cpu())
      diff_arr   <- as.array(diff_real$cpu())

      covs <- extract_player_covariates(batch, scaler)
      num_nodes <- dim(preds_arr)[1]
      seq_len   <- dim(preds_arr)[2]

      # Escalas adaptativas ŝ_t(x_i)
      s_hat_mat <- matrix(NA_real_, nrow = num_nodes, ncol = seq_len)
      for (t in 1:seq_len) {
        s_hat_mat[, t] <- predict_adaptive_scale(scale_model, covs, time_step = t)
      }

      for (node in 1:num_nodes) {
        pid   <- player_ids[node]
        tcode <- team_codes[node]
        tname <- team_names[node]
        p_speed <- covs$speed[node]
        p_dist  <- covs$dist_to_ball[node]

        for (t in 1:seq_len) {
          px <- preds_arr[node, t, 1]
          py <- preds_arr[node, t, 2]
          tx <- y_true_arr[node, t, 1]
          ty <- y_true_arr[node, t, 2]

          s_hat_it <- s_hat_mat[node, t]
          dist <- sqrt((px - tx)^2 + (py - ty)^2)

          if (region_type == "circular") {
            rad_it <- lambda_hat * s_hat_it
            is_covered <- dist <= rad_it

            results_df[[length(results_df) + 1]] <- data.frame(
              event_id = evt_id,
              node_id = node,
              player_id = pid,
              team_code = tcode,
              team_name = tname,
              speed_obs = p_speed,
              dist_to_ball_obs = p_dist,
              time_step = t,
              pred_x = px,
              pred_y = py,
              true_x = tx,
              true_y = ty,
              s_hat = s_hat_it,
              conf_radius = rad_it,
              distance = dist,
              covered = is_covered,
              stringsAsFactors = FALSE
            )
          } else {
            # Elipse adaptativa: semi-eixos escalonados por lambda_hat * ŝ_t(x_i)
            eig <- eigen(Gamma_t[[t]], symmetric = TRUE)
            ord <- order(eig$values, decreasing = TRUE)
            lam <- eig$values[ord]
            v1  <- eig$vectors[, ord[1]]

            a_it <- lambda_hat * s_hat_it * sqrt(max(lam[1], 0))
            b_it <- lambda_hat * s_hat_it * sqrt(max(lam[2], 0))
            angle_t <- atan2(v1[2], v1[1])

            e <- diff_arr[node, t, ]
            Ginv <- solve(Gamma_t[[t]])
            md <- sqrt(sum((e %*% Ginv) * e)) / s_hat_it
            is_covered <- md <= lambda_hat

            results_df[[length(results_df) + 1]] <- data.frame(
              event_id = evt_id,
              node_id = node,
              player_id = pid,
              team_code = tcode,
              team_name = tname,
              speed_obs = p_speed,
              dist_to_ball_obs = p_dist,
              time_step = t,
              pred_x = px,
              pred_y = py,
              true_x = tx,
              true_y = ty,
              s_hat = s_hat_it,
              a = a_it,
              b = b_it,
              angle = angle_t,
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

#' Avaliação estatística completa das propriedades das regiões conformes
#' @param pred_df Data frame retornado por predict_with_adaptive_regions
#' @param alpha Nível de significância nominal
#' @return Lista de métricas de cobertura, perda por jogada e eficiência
evaluate_conformal_regions <- function(pred_df, alpha = 0.10) {
  # 1. Cobertura da trajetória completa por jogador
  has_conf_rad <- "conf_radius" %in% names(pred_df)
  rad_col <- if (has_conf_rad) sym("conf_radius") else sym("a")

  traj_summary <- pred_df |>
    group_by(event_id, player_id) |>
    summarize(
      trajectory_covered = all(covered),
      mean_radius = mean(!!rad_col),
      .groups = "drop"
    )

  # 2. Perda por jogada L_k(lambda_hat)
  play_summary <- traj_summary |>
    group_by(event_id) |>
    summarize(
      n_players = n(),
      n_escaped = sum(!trajectory_covered),
      play_loss = mean(!trajectory_covered),
      .groups = "drop"
    )

  # 3. Métricas globais
  overall_trajectory_coverage <- mean(traj_summary$trajectory_covered)
  mean_play_loss <- mean(play_summary$play_loss)

  # 4. Cobertura pontual por horizonte
  pointwise_cov <- pred_df |>
    group_by(time_step) |>
    summarize(
      coverage = mean(covered),
      mean_radius = mean(!!rad_col),
      min_radius = min(!!rad_col),
      max_radius = max(!!rad_col),
      .groups = "drop"
    )

  list(
    mean_play_loss = mean_play_loss,
    overall_trajectory_coverage = overall_trajectory_coverage,
    target_coverage = 1 - alpha,
    pointwise = pointwise_cov,
    play_summary = play_summary,
    traj_summary = traj_summary
  )
}
