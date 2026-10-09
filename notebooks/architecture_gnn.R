# ==============================================================================
# ARCHITECTURE_GNN.R — REDE ESPAÇOTEMPORAL GRAPH NEURAL NETWORK (Seq2Seq + GNN)
# ==============================================================================
# Implementação e rotina de treino da arquitetura de referência baseada em:
#  - Atenção multi-cabeça espacial entre nós a cada frame (Graph Multi-Head Attention)
#  - Codificador temporal LSTM sobre o horizonte observado (t = 1..25)
#  - Pooling de contexto social global invariante à permutação
#  - Decodificador temporal autoregressivo LSTM para o horizonte alvo (t = 26..30)
#
# Este módulo é puramente numérico e isolado de previsões conformes e plots.
# ==============================================================================

suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
  library(R6)
})

# ==============================================================================
# 1. DEFINIÇÃO DA REDE NEURAL TrajectorySeq2SeqGNN
# ==============================================================================

TrajectorySeq2SeqGNN <- nn_module(
  "TrajectorySeq2SeqGNN",

  initialize = function(num_teams, num_roles = 2, cont_dim = 8,
                        hidden_dim = 64, dropout_rate = 0.2, scaler = NULL) {
    # 1. Embeddings Categóricos (Indexação base 1)
    self$team_emb <- nn_embedding(num_embeddings = num_teams + 2L, embedding_dim = 4L)
    self$role_emb <- nn_embedding(num_embeddings = num_roles + 2L, embedding_dim = 4L)

    input_dim <- cont_dim + 4L + 4L
    self$feat_proj <- nn_linear(input_dim, hidden_dim)

    # 2. Camada Espacial (Graph Multi-Head Attention entre os nós a cada frame)
    self$spatial_attention <- nn_multihead_attention(embed_dim = hidden_dim, num_heads = 4L, batch_first = TRUE)
    self$norm1 <- nn_layer_norm(hidden_dim)
    self$dropout1 <- nn_dropout(dropout_rate)

    # 3. Codificador Temporal (LSTM ao longo dos 25 frames históricos)
    self$temporal_encoder <- nn_lstm(input_size = hidden_dim, hidden_size = hidden_dim, batch_first = TRUE)

    # 4. Decodificador Temporal Autoregressivo
    # Entrada: [posição_atual (2), velocidade_atual (2), contexto_global (hidden_dim)]
    self$dec_input_proj <- nn_linear(2L + 2L + hidden_dim, hidden_dim)
    self$temporal_decoder <- nn_lstm(input_size = hidden_dim, hidden_size = hidden_dim, batch_first = TRUE)

    # 5. Camada de Saída: prediz o deslocamento incremental no espaço padronizado
    self$out <- nn_sequential(
      nn_linear(hidden_dim, hidden_dim),
      nn_gelu(),
      nn_linear(hidden_dim, 2L)
    )

    # Buffers para harmonização cinemática física entre deslocamento e escala de velocidade
    if (!is.null(scaler)) {
      self$sd_pos <- nn_buffer(torch_tensor(scaler$scale[c("x", "y")])$view(c(1, 2)))
      self$sd_vel <- nn_buffer(torch_tensor(scaler$scale[c("vel_x", "vel_y")])$view(c(1, 2)))
      self$mu_vel <- nn_buffer(torch_tensor(scaler$center[c("vel_x", "vel_y")])$view(c(1, 2)))
    } else {
      self$sd_pos <- nn_buffer(torch_ones(1, 2))
      self$sd_vel <- nn_buffer(torch_ones(1, 2))
      self$mu_vel <- nn_buffer(torch_zeros(1, 2))
    }
  },

  forward = function(batch) {
    x_cont <- batch$x_cont       # [num_nodes, seq_len = 25, cont_dim = 8]
    team_idx <- batch$x_cat[, 1] # [num_nodes]
    role_idx <- batch$x_cat[, 2] # [num_nodes]

    num_nodes <- x_cont$shape[1]
    seq_len <- x_cont$shape[2]

    # Cinemática no último frame observado (t = 25)
    last_known_pos <- x_cont[, seq_len, 1:2]
    last_known_vel <- x_cont[, seq_len, 7:8]

    # Embeddings
    team_feat <- self$team_emb(team_idx)$unsqueeze(2)$expand(c(-1, seq_len, -1))
    role_feat <- self$role_emb(role_idx)$unsqueeze(2)$expand(c(-1, seq_len, -1))

    x_in <- torch_cat(list(x_cont, team_feat, role_feat), dim = 3)
    x_proj <- self$feat_proj(x_in)

    # Camada Espacial: Atenção entre jogadores a cada instante
    x_spatial_in <- x_proj$transpose(1, 2) # [seq_len, num_nodes, hidden_dim]
    spatial_out <- self$spatial_attention(x_spatial_in, x_spatial_in, x_spatial_in)[[1]]
    x_spatio_temporal <- self$norm1(spatial_out$transpose(1, 2) + x_proj)
    x_spatio_temporal <- self$dropout1(x_spatio_temporal)

    # Codificador Temporal
    enc_out <- self$temporal_encoder(x_spatio_temporal)
    enc_states <- enc_out[[1]]
    last_hidden <- enc_states[, seq_len, ] # [num_nodes, hidden_dim]

    # Contexto Global (Social Max-Pooling invariante à permutação)
    global_ctx <- last_hidden$max(dim = 1, keepdim = TRUE)[[1]] # [1, hidden_dim]
    global_ctx_exp <- global_ctx$expand(c(num_nodes, -1))

    # Decodificador Temporal Autoregressivo (5 passos no futuro)
    h_t <- last_hidden$unsqueeze(1) # [1, num_nodes, hidden_dim]
    c_t <- torch_zeros_like(h_t)
    hx <- list(h_t, c_t)

    current_pos <- last_known_pos
    current_vel <- last_known_vel

    preds <- list()

    for (t in 1:5) {
      step_in <- torch_cat(list(current_pos, current_vel, global_ctx_exp), dim = 2)
      step_in <- self$dec_input_proj(step_in)$unsqueeze(2)

      dec_out <- self$temporal_decoder(step_in, hx)
      dec_hidden <- dec_out[[1]]
      hx <- dec_out[[2]]

      pred_disp <- self$out(dec_hidden$squeeze(2)) # [num_nodes, 2]

      # Atualização da posição no espaço escalonado
      current_pos <- current_pos + pred_disp

      # Atualização fisicamente consistente da velocidade
      current_vel <- (pred_disp * self$sd_pos - self$mu_vel) / self$sd_vel

      preds[[t]] <- current_pos$unsqueeze(2)
    }

    preds_tensor <- torch_cat(preds, dim = 2) # [num_nodes, 5, 2]
    return(preds_tensor)
  }
)

#' Fábrica para instanciação de TrajectorySeq2SeqGNN
#' @param scaler Parâmetros de escalonamento com center e scale
#' @param encoders Lista contendo num_teams e num_roles
#' @param hidden_dim Dimensão da representação oculta
#' @param dropout_rate Taxa de dropout
#' @return Instância de TrajectorySeq2SeqGNN
gnn_model_factory <- function(scaler, encoders, hidden_dim = 64, dropout_rate = 0.2) {
  TrajectorySeq2SeqGNN(
    num_teams = encoders$num_teams,
    num_roles = encoders$num_roles,
    cont_dim = 8L,
    hidden_dim = hidden_dim,
    dropout_rate = dropout_rate,
    scaler = scaler
  )
}


# ==============================================================================
# 2. ROTINA DE TREINAMENTO E OTIMIZAÇÃO (GNN)
# ==============================================================================

#' Treina o modelo TrajectorySeq2SeqGNN com otimizador Adam e scheduler
#' @param train_dataset TrajectoryDataset de treino
#' @param val_dataset TrajectoryDataset de validação
#' @param scaler Parâmetros do scaler para cálculo de perdas físicas
#' @param encoders Metadados categóricos (num_teams, num_roles)
#' @param cfg Lista de configuração de hiperparâmetros (lr, max_epochs, patience)
#' @param seed Semente aleatória
#' @param verbose Lógico: se TRUE, imprime progresso das épocas
#' @return Lista contendo model, history, n_params, val_ade_best, val_fde_best
train_gnn <- function(train_dataset, val_dataset, scaler, encoders,
                      cfg = list(lr = 0.001, max_epochs = 100, patience = 10),
                      seed = 42, verbose = TRUE) {
  set_seed(seed)
  device <- select_device()

  train_loader <- dataloader(train_dataset, batch_size = 1, shuffle = TRUE, collate_fn = custom_collate)
  val_loader   <- dataloader(val_dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  model <- gnn_model_factory(scaler = scaler, encoders = encoders)$to(device = device)
  n_params <- sum(sapply(model$parameters, function(p) p$numel()))

  lr_val <- if (!is.null(cfg$lr)) cfg$lr else 0.001
  max_epochs <- if (!is.null(cfg$max_epochs)) cfg$max_epochs else 100
  patience <- if (!is.null(cfg$patience)) cfg$patience else 10

  optimizer <- optim_adam(model$parameters, lr = lr_val, weight_decay = 1e-5)
  scheduler <- lr_reduce_on_plateau(optimizer, mode = "min", factor = 0.5, patience = 2)
  early_stopping <- EarlyStopping$new(patience = patience)

  history <- data.frame(
    epoch = integer(),
    train_ade = numeric(), train_fde = numeric(),
    val_ade = numeric(), val_fde = numeric()
  )

  if (verbose) {
    cat(sprintf("Treinando GNN (%d parâmetros) no dispositivo: %s\n", n_params, device$type))
  }

  for (epoch in 1:max_epochs) {
    model$train()
    train_ade <- 0
    train_fde <- 0
    total_nodes_train <- 0

    coro::loop(for (batch in train_loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat <- batch$x_cat$to(device = device)
      batch$y <- batch$y$to(device = device)

      optimizer$zero_grad()
      predictions <- model(batch)
      metrics <- compute_ade_loss_and_fde(predictions, batch$y, scaler)

      loss <- metrics$ade_loss + metrics$fde_loss
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      optimizer$step()

      num_nodes <- batch$num_nodes
      train_ade <- train_ade + metrics$ade_loss$item() * num_nodes
      train_fde <- train_fde + metrics$fde * num_nodes
      total_nodes_train <- total_nodes_train + num_nodes
    })

    train_ade <- train_ade / total_nodes_train
    train_fde <- train_fde / total_nodes_train

    # Validação
    model$eval()
    val_ade <- 0
    val_fde <- 0
    total_nodes_val <- 0

    with_no_grad({
      coro::loop(for (batch in val_loader) {
        batch$x_cont <- batch$x_cont$to(device = device)
        batch$x_cat <- batch$x_cat$to(device = device)
        batch$y <- batch$y$to(device = device)

        predictions <- model(batch)
        metrics <- compute_ade_loss_and_fde(predictions, batch$y, scaler)

        num_nodes <- batch$num_nodes
        val_ade <- val_ade + metrics$ade_loss$item() * num_nodes
        val_fde <- val_fde + metrics$fde * num_nodes
        total_nodes_val <- total_nodes_val + num_nodes
      })
    })

    val_ade <- val_ade / total_nodes_val
    val_fde <- val_fde / total_nodes_val

    scheduler$step(val_fde)

    history <- rbind(history, data.frame(
      epoch = epoch,
      train_ade = train_ade, train_fde = train_fde,
      val_ade = val_ade, val_fde = val_fde
    ))

    if (verbose && (epoch %% 5 == 0 || epoch == 1)) {
      cat(sprintf("Época %03d | Treino ADE: %.2f m, FDE: %.2f m | Val ADE: %.2f m, FDE: %.2f m\n",
                  epoch, train_ade, train_fde, val_ade, val_fde))
    }

    early_stopping$step(val_fde, model)
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
    val_ade_best = min(history$val_ade),
    val_fde_best = min(history$val_fde)
  )
}
