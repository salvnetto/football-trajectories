# ==============================================================================
# ARCHITECTURE_SOCIAL.R — REDE ESPAÇOTEMPORAL Social-LSTM (Alahi et al. 2016)
# ==============================================================================
# Implementação fiel de Alahi, Goel, Ramanathan, Robicquet, Fei-Fei & Savarese
# (2016), "Social LSTM: Human Trajectory Prediction in Crowded Spaces", CVPR 2016.
#
# Mecânica exata:
#  - Eq. (1): Tensor social H_i^t (grade espacial N_o x N_o x D) via scatter-add
#  - Eq. (2): Embeddings phi(pos) e phi(H) com sum-pooling 8x8; célula LSTM
#  - Eqs. (3)-(4): Cabeça gaussiana bivariada 5D [mu_x, mu_y, log sigma_x, log sigma_y, atanh rho]
#  - Perda: NLL da distribuição normal bivariada correlacionada
#
# Este módulo é puramente numérico e isolado de previsões conformes e plots.
# ==============================================================================

suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
  library(R6)
})

# ==============================================================================
# 1. DEFINIÇÃO DA REDE NEURAL SocialLSTM
# ==============================================================================

SocialLSTM <- nn_module(
  "SocialLSTM",

  initialize = function(scaler, pos_embed_dim = 64L, social_embed_dim = 64L,
                        hidden_dim = 128L, num_cells = 32L, pool_size = 8L,
                        window_m = 10, dropout_rate = 0.5) {
    self$hidden_dim <- hidden_dim
    self$num_cells <- num_cells
    self$pool_size <- pool_size
    self$window_m <- window_m

    # Embeddings (Eq. 2 do artigo): coordenadas e tensor social
    self$emb_pos <- nn_sequential(
      nn_linear(2L, pos_embed_dim),
      nn_relu()
    )

    # H (32x32x128) -> sum-pooling 8x8 -> (4x4x128) achatado -> a_i^t
    self$emb_social <- nn_sequential(
      nn_linear(as.integer((num_cells %/% pool_size)^2) * hidden_dim, social_embed_dim),
      nn_relu()
    )

    # Um LSTM por trajetória, pesos compartilhados (Eq. 2)
    self$cell <- nn_lstm(pos_embed_dim + social_embed_dim, hidden_dim, batch_first = TRUE)

    # Cabeça de saída (Eq. 4): [mu_x, mu_y, log sigma_x, log sigma_y, atanh rho]
    self$out <- nn_linear(hidden_dim, 5L)

    # Dropout nos embeddings
    self$dropout_emb <- nn_dropout(dropout_rate)

    # Buffers para converter posições z-scoreadas -> metros (grade de pooling)
    self$mu_pos <- nn_buffer(torch_tensor(scaler$center[c("x", "y")])$view(c(1, 2)))
    self$sd_pos <- nn_buffer(torch_tensor(scaler$scale[c("x", "y")])$view(c(1, 2)))
  },

  # Tensor social H_i^t + embedding a_i^t (Eqs. 1-2 do artigo)
  social_context = function(pos_m, h_prev) {
    num_nodes <- pos_m$shape[1]
    nc <- self$num_cells
    cell_size <- self$window_m / nc
    D <- self$hidden_dim

    # rel[i, j, :] = pos_j - pos_i
    rel <- pos_m$unsqueeze(1) - pos_m$unsqueeze(2) # [n, n, 2]

    # Célula (m, n) de cada par na grade centrada no jogador i
    cell_xy <- torch_floor((rel + self$window_m / 2) / cell_size)$clamp(0, nc - 1L)
    inside <- (rel$abs() < self$window_m / 2)$all(dim = 3) # [n, n] bool

    # Exclui o próprio jogador (Eq. 1 soma sobre vizinhos j != i)
    mask <- inside * (torch_eye(num_nodes, device = rel$device) == 0)

    # Célula linear do par (i, j): linha m = y, coluna n = x
    idx <- (cell_xy[, , 2] * nc + cell_xy[, , 1])$to(dtype = torch_long())

    # Scatter-add vetorizado: acumula h_j^{t-1} na grade do jogador i
    flat_sel <- torch_nonzero(mask)
    cell_ij <- idx$masked_select(mask)

    H <- torch_zeros(c(num_nodes * nc * nc, D), device = pos_m$device)
    if (flat_sel$numel() > 0) {
      src <- h_prev$index_select(1, flat_sel[, 2])
      flat <- (flat_sel[, 1] - 1L) * (nc * nc) + cell_ij + 1L # 1-based R Torch
      H$index_add_(1, flat, src)
    }
    H <- H$view(c(num_nodes, nc, nc, D))

    # Sum-pooling 8x8 sem sobreposição (Sec. 3.2 do artigo): 32x32 -> 4x4
    H <- H$view(c(num_nodes, as.integer(nc / self$pool_size), self$pool_size,
                  as.integer(nc / self$pool_size), self$pool_size, D))
    H <- H$sum(dim = 3)$sum(dim = 4)

    self$emb_social(H$flatten(start_dim = 2))
  },

  # Executa o codificador até t = 25 e retorna o estado oculto latente [num_nodes, hidden_dim]
  extract_latent = function(batch) {
    pos <- batch$x_cont[, , 1:2]
    num_nodes <- pos$shape[1]
    device <- pos$device

    h_prev <- torch_zeros(c(1, num_nodes, self$hidden_dim), device = device)
    c_prev <- torch_zeros(c(1, num_nodes, self$hidden_dim), device = device)

    for (t in 1:25) {
      pos_t <- pos[, t, ]
      a_t <- self$social_context(pos_t * self$sd_pos + self$mu_pos, h_prev[1, , ])
      e_t <- self$dropout_emb(self$emb_pos(pos_t))
      a_t <- self$dropout_emb(a_t)
      step_in <- torch_cat(list(e_t, a_t), dim = 2)$unsqueeze(2)
      hc <- self$cell(step_in, list(h_prev, c_prev))
      h_prev <- hc[[2]][[1]]
      c_prev <- hc[[2]][[2]]
    }

    h_prev[1, , ]
  },

  forward = function(batch, teacher_forcing = FALSE, tf_rate = 1.0, return_params = FALSE) {
    pos <- batch$x_cont[, , 1:2] # [num_nodes, 25, 2]
    target <- batch$y            # [num_nodes, 5, 2]
    num_nodes <- pos$shape[1]
    device <- pos$device

    h_prev <- torch_zeros(c(1, num_nodes, self$hidden_dim), device = device)
    c_prev <- torch_zeros(c(1, num_nodes, self$hidden_dim), device = device)

    # ---- Encoder: t = 1..25 ----
    for (t in 1:25) {
      pos_t <- pos[, t, ]
      a_t <- self$social_context(pos_t * self$sd_pos + self$mu_pos, h_prev[1, , ])
      e_t <- self$dropout_emb(self$emb_pos(pos_t))
      a_t <- self$dropout_emb(a_t)
      step_in <- torch_cat(list(e_t, a_t), dim = 2)$unsqueeze(2)
      hc <- self$cell(step_in, list(h_prev, c_prev))
      h_prev <- hc[[2]][[1]]
      c_prev <- hc[[2]][[2]]
    }

    # ---- Decodificador Autoregressivo: t = 26..30 ----
    input_pos <- pos[, 25, ]
    input_seq_l <- list()
    preds <- list()
    mu_xy_l <- list()
    sig_xy_l <- list()
    rho_l <- list()

    for (k in 1:5) {
      if (teacher_forcing && k > 1) {
        use_gt <- torch_rand(1, device = device) < tf_rate
        input_pos <- torch_where(use_gt, target[, k - 1, ], input_pos)
      }
      input_seq_l[[k]] <- input_pos$unsqueeze(2)

      a_k <- self$social_context(input_pos * self$sd_pos + self$mu_pos, h_prev[1, , ])
      e_k <- self$dropout_emb(self$emb_pos(input_pos))
      a_k <- self$dropout_emb(a_k)
      step_in <- torch_cat(list(e_k, a_k), dim = 2)$unsqueeze(2)
      hc <- self$cell(step_in, list(h_prev, c_prev))
      h_prev <- hc[[2]][[1]]
      c_prev <- hc[[2]][[2]]

      out5 <- self$out(h_prev[1, , ])
      mu_xy <- out5[, 1:2]             # Deslocamento médio
      sig_xy <- torch_exp(out5[, 3:4]) # Desvio padrão
      rho <- torch_tanh(out5[, 5])     # Coeficiente de correlação

      pred_pos <- input_pos + mu_xy
      preds[[k]] <- pred_pos$unsqueeze(2)
      mu_xy_l[[k]] <- mu_xy$unsqueeze(2)
      sig_xy_l[[k]] <- sig_xy$unsqueeze(2)
      rho_l[[k]] <- rho$unsqueeze(2)
      input_pos <- pred_pos
    }

    preds_tensor <- torch_cat(preds, dim = 2) # [num_nodes, 5, 2]

    if (return_params) {
      list(
        preds = preds_tensor,
        mu_xy = torch_cat(mu_xy_l, dim = 2),
        sig_xy = torch_cat(sig_xy_l, dim = 2),
        rho = torch_cat(rho_l, dim = 2),
        input_seq = torch_cat(input_seq_l, dim = 2)
      )
    } else {
      preds_tensor
    }
  }
)

#' Fábrica para o modelo Social-LSTM
#' @param scaler Objeto de escalonamento com center e scale
#' @param num_teams Ignorado (mantido para assinatura polimórfica com GNN)
#' @return Instância de SocialLSTM
social_lstm_factory <- function(scaler, num_teams = NULL) {
  SocialLSTM$new(scaler = scaler)
}


# ==============================================================================
# 2. FUNÇÕES DE PERDA ESTATÍSTICA (GAUSSIANA BIVARIADA)
# ==============================================================================

#' Log-Likelihood Negativo (NLL) da Gaussiana Bivariada com correlação rho
#' @param dx Deslocamento em x
#' @param dy Deslocamento em y
#' @param mu_x Média predita de dx
#' @param mu_y Média predita de dy
#' @param sig_x Desvio padrão predito de dx
#' @param sig_y Desvio padrão predito de dy
#' @param rho Correlação bivariada predita
#' @param eps Regularização numérica
#' @return Escalar Torch com a perda NLL média
bivariate_gaussian_nll <- function(dx, dy, mu_x, mu_y, sig_x, sig_y, rho, eps = 1e-4) {
  sig_x <- sig_x$clamp(min = eps)
  sig_y <- sig_y$clamp(min = eps)
  rho2 <- rho^2

  z <- (dx - mu_x)^2 / sig_x^2 + (dy - mu_y)^2 / sig_y^2 -
    2 * rho * (dx - mu_x) * (dy - mu_y) / (sig_x * sig_y)

  torch_mean(
    torch_log(sig_x) + torch_log(sig_y) +
      0.5 * torch_log(1 - rho2$clamp(max = 1 - eps)) +
      z / (2 * (1 - rho2$clamp(min = eps)))
  )
}

#' Computa a NLL total para uma saída com parâmetros bivariados
#' @param model_out Saída do modelo com lista de parâmetros (return_params = TRUE)
#' @param batch Lote contendo o tensor alvo batch$y
#' @return Perda NLL
compute_social_nll <- function(model_out, batch) {
  target <- batch$y
  dx <- target - model_out$input_seq
  bivariate_gaussian_nll(
    dx[, , 1], dx[, , 2],
    model_out$mu_xy[, , 1], model_out$mu_xy[, , 2],
    model_out$sig_xy[, , 1], model_out$sig_xy[, , 2],
    model_out$rho
  )
}


# ==============================================================================
# 3. ROTINA DE TREINAMENTO E OTIMIZAÇÃO (Social-LSTM)
# ==============================================================================

#' Treina o modelo Social-LSTM com otimizador RMS-prop e Teacher Forcing
#' @param train_dataset TrajectoryDataset de treino
#' @param val_dataset TrajectoryDataset de validação
#' @param scaler Parâmetros de escalonamento
#' @param cfg Configuração de treino (lr, patience, max_epochs, tf_rate, stop_metric)
#' @param seed Semente aleatória
#' @param verbose Lógico: se TRUE, imprime progresso das épocas
#' @return Lista contendo model, history, n_params, val_nll_best, val_ade_best, val_fde_best
train_social_lstm <- function(train_dataset, val_dataset, scaler,
                              cfg = list(lr = 0.003, max_epochs = 100, patience = 10,
                                         tf_rate = 1.0, stop_metric = "ade"),
                              seed = 42, verbose = TRUE) {
  set_seed(seed)
  device <- select_device()

  train_loader <- dataloader(train_dataset, batch_size = 1, shuffle = TRUE, collate_fn = custom_collate)
  val_loader   <- dataloader(val_dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  model <- social_lstm_factory(scaler = scaler)$to(device = device)
  n_params <- sum(sapply(model$parameters, function(p) p$numel()))

  lr_val <- if (!is.null(cfg$lr)) cfg$lr else 0.003
  max_epochs <- if (!is.null(cfg$max_epochs)) cfg$max_epochs else 100
  patience <- if (!is.null(cfg$patience)) cfg$patience else 10
  tf_rate <- if (!is.null(cfg$tf_rate)) cfg$tf_rate else 1.0
  stop_metric <- if (!is.null(cfg$stop_metric) && cfg$stop_metric == "ade") "ade" else "nll"

  optimizer <- optim_rmsprop(model$parameters, lr = lr_val)
  early_stopping <- EarlyStopping$new(patience = patience)

  history <- data.frame(
    epoch = integer(),
    train_nll = numeric(), train_ade = numeric(), train_fde = numeric(),
    val_nll = numeric(), val_ade = numeric(), val_fde = numeric()
  )

  if (verbose) {
    cat(sprintf("Treinando Social-LSTM (%d parâmetros) no dispositivo: %s | Critério Early Stopping: %s\n",
                n_params, device$type, toupper(stop_metric)))
  }

  for (epoch in 1:max_epochs) {
    model$train()
    train_nll <- 0
    train_ade <- 0
    train_fde <- 0
    total_nodes_train <- 0

    coro::loop(for (batch in train_loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$y <- batch$y$to(device = device)

      optimizer$zero_grad()
      out <- model(batch, teacher_forcing = TRUE, tf_rate = tf_rate, return_params = TRUE)

      loss <- compute_social_nll(out, batch)
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      optimizer$step()

      metrics <- compute_ade_loss_and_fde(out$preds, batch$y, scaler)

      num_nodes <- batch$num_nodes
      train_nll <- train_nll + loss$item() * num_nodes
      train_ade <- train_ade + metrics$ade_loss$item() * num_nodes
      train_fde <- train_fde + metrics$fde * num_nodes
      total_nodes_train <- total_nodes_train + num_nodes
    })

    train_nll <- train_nll / total_nodes_train
    train_ade <- train_ade / total_nodes_train
    train_fde <- train_fde / total_nodes_train

    # Validação
    model$eval()
    val_nll <- 0
    val_ade <- 0
    val_fde <- 0
    total_nodes_val <- 0

    with_no_grad({
      coro::loop(for (batch in val_loader) {
        batch$x_cont <- batch$x_cont$to(device = device)
        batch$y <- batch$y$to(device = device)

        out <- model(batch, teacher_forcing = FALSE, return_params = TRUE)
        nll <- compute_social_nll(out, batch)
        metrics <- compute_ade_loss_and_fde(out$preds, batch$y, scaler)

        num_nodes <- batch$num_nodes
        val_nll <- val_nll + nll$item() * num_nodes
        val_ade <- val_ade + metrics$ade_loss$item() * num_nodes
        val_fde <- val_fde + metrics$fde * num_nodes
        total_nodes_val <- total_nodes_val + num_nodes
      })
    })

    val_nll <- val_nll / total_nodes_val
    val_ade <- val_ade / total_nodes_val
    val_fde <- val_fde / total_nodes_val

    history <- rbind(history, data.frame(
      epoch = epoch,
      train_nll = train_nll, train_ade = train_ade, train_fde = train_fde,
      val_nll = val_nll, val_ade = val_ade, val_fde = val_fde
    ))

    if (verbose && (epoch %% 5 == 0 || epoch == 1)) {
      cat(sprintf("Época %03d | Treino NLL: %.3f, ADE: %.2f m | Val NLL: %.3f, ADE: %.2f m, FDE: %.2f m\n",
                  epoch, train_nll, train_ade, val_nll, val_ade, val_fde))
    }

    early_stopping$step(if (stop_metric == "ade") val_ade else val_nll, model)
    if (early_stopping$early_stop) {
      if (verbose) cat(sprintf("Early stopping atingido na época %d.\n", epoch))
      break
    }
  }

  # Restaurar os melhores pesos selecionados pela validação
  with_no_grad({
    for (i in seq_along(model$parameters)) {
      model$parameters[[i]]$copy_(early_stopping$best_model_wts[[i]])
    }
  })

  list(
    model = model,
    history = history,
    n_params = n_params,
    val_nll_best = min(history$val_nll),
    val_ade_best = min(history$val_ade),
    val_fde_best = min(history$val_fde)
  )
}
