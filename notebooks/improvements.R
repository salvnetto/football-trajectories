# ==========================================================================
#   5. Execução: carregamento dos dados & janelas
#   6. Experimento E3_mirror
#   7. Calibração conforme & predições no teste
#   8. Plot da jogada de teste
#   9. Avaliação & tabelas de resultados
# ==========================================================================


# ==========================================================================
# 0. SETUP
# ==========================================================================
library(torch)
library(tidyverse)
library(ggsoccer)
library(R6)
library(here)
library(ggforce)

source(here("src", "modeling.R"))   # prepare_data()


# ==========================================================================
# 1. PRÉ-PROCESSAMENTO & CONSTRUÇÃO DO DATASET
# ==========================================================================

# Colação customizada para lotes de tamanho 1 (preserva tipos e metadados R)
custom_collate <- function(batch) {
  batch[[1]]
}

# Espelhamento lateral exato y -> 68 - y no espaço z-scoreado:
# y'_s = (68 - 2*mu_y)/sd_y - y_s  (a média/desvio do scaler NÃO se alteram)
mirror_graph <- function(g, scaler) {
  a_y     <- (68 - 2 * scaler$center["y"])     / scaler$scale["y"]
  a_yball <- (68 - 2 * scaler$center["y_ball"]) / scaler$scale["y_ball"]
  a_vely  <- (0  - 2 * scaler$center["vel_y"])  / scaler$scale["vel_y"]

  g2 <- g
  g2$x_cont <- g$x_cont$clone()
  g2$x_cont[, , 2] <- a_y     - g2$x_cont[, , 2]   # y
  g2$x_cont[, , 5] <- a_yball - g2$x_cont[, , 5]   # y_ball
  g2$x_cont[, , 8] <- a_vely  - g2$x_cont[, , 8]   # vel_y
  g2$y <- g$y$clone()
  g2$y[, , 2] <- a_y - g2$y[, , 2]
  g2
}

preprocess_and_build_dataset <- function(df) {
  set.seed(42)
  df <- as.data.frame(df)

  # 1. Codificação categórica (1 = Defense, 2 = Attack)
  df <- df |>
    mutate(
      role_idx = if_else(team_code == "Attack", 2L, 1L),
      team_id_idx = as.integer(as.factor(team_id))
    ) |> 
    group_by(player_id, event_id) |>
    arrange(time_sec) |>
    mutate(
      dist_to_ball = sqrt((x - x_ball)^2 + (y - y_ball)^2),
      vel_x = x - lag(x, default = first(x)),
      vel_y = y - lag(y, default = first(y))
    ) |>
    ungroup()

  num_teams <- max(df$team_id_idx, na.rm = TRUE)

  # 2. Particionamento por jogada independente (sem vazamento entre frames
  #    da mesma jogada): 60% treino, 15% validação, 15% calibração, 10% teste
  unique_events <- unique(df$event_id)
  n_events <- length(unique_events)
  shuffled_events <- sample(unique_events)

  n_train <- floor(0.60 * n_events)
  n_val <- floor(0.15 * n_events)
  n_calib <- floor(0.15 * n_events)

  train_events <- shuffled_events[1:n_train]
  val_events <- shuffled_events[(n_train + 1):(n_train + n_val)]
  calib_events <- shuffled_events[(n_train + n_val + 1):(n_train + n_val + n_calib)]
  test_events <- shuffled_events[(n_train + n_val + n_calib + 1):n_events]

  # 3. Escalonamento z-score ajustado APENAS no treino (t <= 25)
  cont_cols <- c('x', 'y', 'ball_speed', 'x_ball', 'y_ball', 'dist_to_ball', 'vel_x', 'vel_y')
  train_obs <- df |> filter(event_id %in% train_events, time_sec <= 25)

  scaler_center <- colMeans(train_obs[, cont_cols], na.rm = TRUE)
  scaler_scale <- apply(train_obs[, cont_cols], 2, sd, na.rm = TRUE)
  scaler_scale[scaler_scale == 0] <- 1

  df[, cont_cols] <- scale(df[, cont_cols], center = scaler_center, scale = scaler_scale)

  # 4. Construção dos grafos por jogada
  event_list <- df |> group_split(event_id)
  graphs <- list()

  for (event_df in event_list) {
    evt_id <- event_df$event_id[1]

    obs_df <- event_df |> filter(time_sec <= 25)
    target_df <- event_df |> filter(time_sec >= 26 & time_sec <= 30)

    obs_counts <- table(obs_df$player_id)
    target_counts <- table(target_df$player_id)

    # Jogadores presentes em todos os 25 frames observados e 5 alvos
    valid_players <- names(obs_counts)[obs_counts == 25]
    valid_players <- valid_players[valid_players %in% names(target_counts)]
    valid_players <- valid_players[target_counts[valid_players] == 5]
    valid_players <- sort(valid_players)

    if (length(valid_players) < 2) next

    node_features_seq <- list()
    node_cat_features <- list()
    node_targets_seq <- list()
    node_player_ids <- character()
    node_team_codes <- character()
    node_team_names <- character()

    for (pid in valid_players) {
      p_obs <- obs_df |> filter(player_id == pid) |> arrange(time_sec)
      p_tgt <- target_df |> filter(player_id == pid) |> arrange(time_sec)

      node_features_seq[[length(node_features_seq) + 1]] <- as.matrix(p_obs[, cont_cols])
      node_cat_features[[length(node_cat_features) + 1]] <- as.integer(p_obs[1, c('team_id_idx', 'role_idx')])
      node_targets_seq[[length(node_targets_seq) + 1]] <- as.matrix(p_tgt[, c('x', 'y')])

      node_player_ids <- c(node_player_ids, pid)
      node_team_codes <- c(node_team_codes, p_obs$team_code[1])
      node_team_names <- c(node_team_names, p_obs$team_name[1])
    }

    x_cont <- torch_tensor(abind::abind(node_features_seq, along = 0), dtype = torch_float())
    x_cat <- torch_tensor(do.call(rbind, node_cat_features), dtype = torch_long())
    y_tgt <- torch_tensor(abind::abind(node_targets_seq, along = 0), dtype = torch_float())

    graphs[[length(graphs) + 1]] <- list(
      x_cont = x_cont, 
      x_cat = x_cat, 
      y = y_tgt, 
      num_nodes = length(valid_players),
      event_id = evt_id,
      player_ids = node_player_ids,
      team_codes = node_team_codes,
      team_names = node_team_names
    )
  }

  # Mapear eventos para índices de grafos
  graph_event_ids <- sapply(graphs, function(g) g$event_id)
  train_indices <- which(graph_event_ids %in% train_events)
  val_indices <- which(graph_event_ids %in% val_events)
  calib_indices <- which(graph_event_ids %in% calib_events)
  test_indices <- which(graph_event_ids %in% test_events)

  # 5. Aumento por espelhamento lateral (sempre ativo em E3)
  scaler_ls <- list(center = scaler_center, scale = scaler_scale)
  mirrored <- lapply(graphs[train_indices], mirror_graph, scaler = scaler_ls)
  n_orig <- length(graphs)
  graphs <- c(graphs, mirrored)
  train_indices <- c(train_indices, (n_orig + 1):(n_orig + length(mirrored)))

  list(
    graphs = graphs,
    splits = list(
      train = train_indices,
      val = val_indices,
      calib = calib_indices,
      test = test_indices
    ),
    num_teams = num_teams,
    scaler = list(center = scaler_center, scale = scaler_scale)
  )
}

TrajectoryDataset <- dataset(
  name = "TrajectoryDataset",
  initialize = function(graphs) { self$graphs <- graphs },
  .getitem = function(i) { self$graphs[[i]] },
  .length = function() { length(self$graphs) }
)


# ==========================================================================
# 2. DEFINIÇÃO DO MODELO
# ==========================================================================
TrajectorySeq2SeqGNNv2 <- nn_module(
  "TrajectorySeq2SeqGNNv2",

  initialize = function(num_teams, scaler) {
    cont_dim <- 8
    hidden_dim <- 64
    dropout_rate <- 0.2

    # ---- Encoder espaçotemporal ----
    self$team_emb <- nn_embedding(num_embeddings = num_teams + 2, embedding_dim = 4)
    self$role_emb <- nn_embedding(num_embeddings = 4, embedding_dim = 4) # num_roles + 2

    self$feat_proj <- nn_linear(cont_dim + 4 + 4, hidden_dim)

    self$spatial_attention <- nn_multihead_attention(embed_dim = hidden_dim, num_heads = 4, batch_first = TRUE)
    self$norm1 <- nn_layer_norm(hidden_dim)
    self$dropout1 <- nn_dropout(dropout_rate)

    self$temporal_encoder <- nn_lstm(input_size = hidden_dim, hidden_size = hidden_dim, batch_first = TRUE)

    # ---- Decodificador com interação social dinâmica (Social-BiGAT / STGAT) ----
    self$dec_input_proj <- nn_linear(2 + 2 + hidden_dim, hidden_dim)
    self$temporal_decoder <- nn_lstm(input_size = hidden_dim, hidden_size = hidden_dim, batch_first = TRUE)

    # Atenção de grafo recomputada a cada passo futuro
    self$social_attention <- nn_multihead_attention(embed_dim = hidden_dim, num_heads = 4, batch_first = TRUE)
    self$norm_social <- nn_layer_norm(hidden_dim)
    self$dropout_social <- nn_dropout(dropout_rate)

    # Cabeça de saída: prevê o resíduo em relação ao prior cinemático
    self$out <- nn_sequential(
      nn_linear(hidden_dim, hidden_dim),
      nn_gelu(),
      nn_linear(hidden_dim, 2)
    )

    # Buffers para harmonização cinemática entre deslocamento e escala de velocidade
    self$sd_pos <- nn_buffer(torch_tensor(scaler$scale[c("x", "y")])$view(c(1, 2)))
    self$sd_vel <- nn_buffer(torch_tensor(scaler$scale[c("vel_x", "vel_y")])$view(c(1, 2)))
    self$mu_vel <- nn_buffer(torch_tensor(scaler$center[c("vel_x", "vel_y")])$view(c(1, 2)))
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

    # Camada Espacial: atenção entre jogadores a cada instante (histórico)
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

      z_t <- dec_hidden$squeeze(2) # [num_nodes, hidden_dim]

      # Troca de mensagens entre jogadores no passo t
      social_in <- z_t$unsqueeze(1) # [1, num_nodes, hidden_dim]
      attn_out <- self$social_attention(social_in, social_in, social_in)[[1]]
      z_t <- self$norm_social(z_t + attn_out$squeeze(1))
      z_t <- self$dropout_social(z_t)

      pred_disp <- self$out(z_t) # [num_nodes, 2] — resíduo em relação ao prior

      # Prior cinemático de velocidade constante
      v_pos <- (current_vel * self$sd_vel + self$mu_vel) / self$sd_pos
      prior_disp <- t * v_pos
      disp_total <- prior_disp + pred_disp

      # Atualização da posição no espaço escalonado
      current_pos <- current_pos + disp_total

      # Atualização fisicamente e dimensionalmente consistente da velocidade
      current_vel <- (disp_total * self$sd_pos - self$mu_vel) / self$sd_vel

      preds[[t]] <- current_pos$unsqueeze(2)
    }

    preds_tensor <- torch_cat(preds, dim = 2) # [num_nodes, 5, 2]
    return(preds_tensor)
  }
)


# ==========================================================================
# 3. TREINO
# ==========================================================================
compute_ade_loss_and_fde <- function(preds, targets, scaler) {
  mu <- c(scaler$center["x"], scaler$center["y"])
  sd <- c(scaler$scale["x"], scaler$scale["y"])
  device <- preds$device

  mu_tensor <- torch_tensor(mu, device = device)$view(c(1, 1, 2))
  sd_tensor <- torch_tensor(sd, device = device)$view(c(1, 1, 2))

  preds_real <- preds * sd_tensor + mu_tensor
  targets_real <- targets * sd_tensor + mu_tensor

  diff_sq <- (preds_real - targets_real)$pow(2)
  distances <- torch_sqrt(diff_sq$sum(dim = 3) + 1e-8)

  ade_loss <- distances$mean()
  seq_len <- distances$shape[2]

  fde_loss <- distances[, seq_len]$mean()
  fde <- fde_loss$item()

  return(list(ade_loss = ade_loss, fde_loss = fde_loss, fde = fde))
}

EarlyStopping <- R6Class("EarlyStopping",
  public = list(
    patience = NULL, delta = NULL, counter = 0, best_loss = NULL,
    early_stop = FALSE, best_model_wts = NULL,
    initialize = function(patience = 10, delta = 0.0001) {
      self$patience <- patience
      self$delta <- delta
    },
    step = function(val_loss, model) {
      if (is.null(self$best_loss)) {
        self$best_loss <- val_loss
        self$best_model_wts <- lapply(model$parameters, function(x) x$clone())
      } else if (val_loss > self$best_loss - self$delta) {
        self$counter <- self$counter + 1
        if (self$counter >= self$patience) self$early_stop <- TRUE
      } else {
        self$best_loss <- val_loss
        self$best_model_wts <- lapply(model$parameters, function(x) x$clone())
        self$counter <- 0
      }
    }
  )
)

run_experiment <- function(df_raw) {
  # Reproduzibilidade estrita (seed 42)
  set.seed(42)
  torch_manual_seed(42)

  device <- if (cuda_is_available()) {
    torch_device("cuda")
  } else if (backends_mps_is_available()) {
    torch_device("mps")
  } else {
    torch_device("cpu")
  }

  prep_data <- preprocess_and_build_dataset(df_raw)
  graphs <- prep_data$graphs
  splits <- prep_data$splits
  scaler <- prep_data$scaler

  train_dataset <- TrajectoryDataset(graphs[splits$train])
  val_dataset <- TrajectoryDataset(graphs[splits$val])
  calib_dataset <- TrajectoryDataset(graphs[splits$calib])
  test_dataset <- TrajectoryDataset(graphs[splits$test])

  train_loader <- dataloader(train_dataset, batch_size = 1, shuffle = TRUE, collate_fn = custom_collate)
  val_loader <- dataloader(val_dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)
  calib_loader <- dataloader(calib_dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  model <- TrajectorySeq2SeqGNNv2$new(
    num_teams = prep_data$num_teams,
    scaler = scaler
  )$to(device = device)

  n_params <- sum(sapply(model$parameters, function(p) p$numel()))

  # Configuração de treino do E3_mirror (inalterada em relação ao notebook)
  lr <- 0.001
  max_epochs <- 100
  patience <- 10

  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = 1e-5)
  scheduler <- lr_reduce_on_plateau(optimizer, mode = "min", factor = 0.5, patience = 2)
  early_stopping <- EarlyStopping$new(patience = patience)

  history <- data.frame(epoch = integer(), train_ade = numeric(), train_fde = numeric(),
                        val_ade = numeric(), val_fde = numeric())

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

    history <- rbind(history, data.frame(
      epoch = epoch, train_ade = train_ade, train_fde = train_fde,
      val_ade = val_ade, val_fde = val_fde
    ))

    scheduler$step(val_fde)

    if (epoch %% 5 == 0 || epoch == 1) {
      cat(sprintf("Época %03d | Treino ADE: %.2f m, FDE: %.2f m | Val ADE: %.2f m, FDE: %.2f m\n",
                  epoch, train_ade, train_fde, val_ade, val_fde))
    }

    early_stopping$step(val_fde, model)
    if (early_stopping$early_stop) {
      cat(sprintf("Early stopping atingido na época %d.\n", epoch))
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
    test_dataset = test_dataset,
    calib_loader = calib_loader,
    scaler = scaler,
    history = history,
    n_params = n_params,
    val_ade_best = min(history$val_ade),
    val_fde_best = min(history$val_fde)
  )
}


# ==========================================================================
# 4. PREDIÇÃO CONFORME: CALIBRAÇÃO & INFERÊNCIA
# ==========================================================================

# Quantil conforme com correção exata de amostra finita:
# ceil((n + 1) * (1 - alpha)) / n
finite_sample_quantile_level <- function(n, alpha) {
  min(1.0, ceiling((n + 1) * (1 - alpha)) / n)
}

# Forma temporal típica do erro s_hat (mediana dos erros a cada horizonte, piso 1e-4)
band_shape <- function(err_matrix) {
  s_hat <- apply(err_matrix, 2, median)
  s_hat[s_hat < 1e-4] <- 1e-4
  s_hat
}

# Escore por trajetória de jogador: R_i = max_{t in 1:5} (e_{i,t} / s_hat_t)
score_supremum_trajectory <- function(err_matrix, s_hat) {
  apply(err_matrix, 1, function(row) max(row / s_hat))
}

# Coleta dos erros de calibração (laço estável: não muda ao trocar escores)
compute_calibration_errors <- function(model, calib_loader, scaler) {
  model$eval()
  device <- model$parameters[[1]]$device

  mu <- c(scaler$center["x"], scaler$center["y"])
  sd <- c(scaler$scale["x"], scaler$scale["y"])
  mu_tensor <- torch_tensor(mu, device = device)$view(c(1, 1, 2))
  sd_tensor <- torch_tensor(sd, device = device)$view(c(1, 1, 2))

  all_player_errors <- list()

  with_no_grad({
    coro::loop(for (batch in calib_loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat <- batch$x_cat$to(device = device)
      batch$y <- batch$y$to(device = device)

      preds <- model(batch)

      preds_real <- preds * sd_tensor + mu_tensor
      y_true_real <- batch$y * sd_tensor + mu_tensor

      diff_sq <- (preds_real - y_true_real)$pow(2)
      distances <- torch_sqrt(diff_sq$sum(dim = 3) + 1e-8) # [num_nodes, 5]
      dist_mat <- as.matrix(distances$cpu())

      all_player_errors[[length(all_player_errors) + 1]] <- dist_mat
    })
  })

  list(
    err_matrix = do.call(rbind, all_player_errors), # [total_trajetórias, 5]
    all_player_errors = all_player_errors
  )
}

# Calibração: devolve s_hat, quantil e raios simultâneos r_t
calibrate_conformal <- function(model, calib_loader, scaler, alpha = 0.10) {
  errs <- compute_calibration_errors(model, calib_loader, scaler)
  err_matrix <- errs$err_matrix
  all_player_errors <- errs$all_player_errors

  n_players <- nrow(err_matrix)
  n_events <- length(all_player_errors)

  # Passo 1: Forma temporal típica do erro s_hat
  s_hat <- band_shape(err_matrix)

  # Passo 2: Escore de não-conformidade por trajetória
  R_player <- score_supremum_trajectory(err_matrix, s_hat)

  # Passo 3: Quantil conforme com correção exata de amostra finita
  q_hat_player <- quantile(R_player, probs = finite_sample_quantile_level(n_players, alpha), names = FALSE)
  r_simultaneous <- q_hat_player * s_hat

  cat(sprintf("\n=== CALIBRAÇÃO CONFORME RIGOROSA (alpha = %.2f) ===\n", alpha))
  cat(sprintf("Amostra de Calibração: %d jogadas independentes (%d trajetórias)\n", n_events, n_players))
  cat("Forma da banda s_hat (medianas em metros):", round(s_hat, 2), "\n")
  cat(sprintf("Quantil Conforme Simultâneo: %.3f\n", q_hat_player))
  cat("Raios Conformes Simultâneos (r_t em metros):", round(r_simultaneous, 2), "\n\n")

  list(
    s_hat = s_hat,
    q_hat_player = q_hat_player,
    r_simultaneous = r_simultaneous,
    alpha = alpha
  )
}

predict_with_regions <- function(model, dataset, scaler, r_t) {
  model$eval()
  device <- model$parameters[[1]]$device

  mu <- c(scaler$center["x"], scaler$center["y"])
  sd <- c(scaler$scale["x"], scaler$scale["y"])
  mu_tensor <- torch_tensor(mu, device = device)$view(c(1, 1, 2))
  sd_tensor <- torch_tensor(sd, device = device)$view(c(1, 1, 2))

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

      preds_real <- preds * sd_tensor + mu_tensor
      y_true_real <- batch$y * sd_tensor + mu_tensor

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
          rad <- r_t[t]
          dist <- sqrt((px - tx)^2 + (py - ty)^2)

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
            covered = dist <= rad,
            stringsAsFactors = FALSE
          )
        }
      }
    })
  })

  bind_rows(results_df)
}


# ==========================================================================
# 5. EXECUÇÃO
# ==========================================================================
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

# Janelas de 30 s centradas na finalização (25 observados + 5 alvo)
events_prepared <- prepare_data(
  events = events, 
  tracking = tracking, 
  event_type_filter = "SHOT", 
  event_subtype = NA, 
  start_time = 30, 
  end_time = 1, 
  pred_time_event = 1
)


res <- run_experiment(events_prepared)

model <- res$model
test_dataset <- res$test_dataset
calib_loader <- res$calib_loader
scaler <- res$scaler

cat(sprintf("\nValidação: ADE %.2f m | FDE %.2f m | %d parâmetros\n",
            res$val_ade_best, res$val_fde_best, res$n_params))


# ==========================================================================
# 6. CALIBRAÇÃO CONFORME & PREDIÇÕES NO TESTE
# ==========================================================================
# 1. Calibração Conforme no conjunto independente (metodologia inalterada)
alpha_level <- 0.10
conformal_calib <- calibrate_conformal(
  model = model, 
  calib_loader = calib_loader, 
  scaler = scaler, 
  alpha = alpha_level
)

# 2. Inferência Conforme no Conjunto de Teste
test_predictions <- predict_with_regions(
  model = model, 
  dataset = test_dataset, 
  scaler = scaler, 
  r_t = conformal_calib$r_simultaneous
)

head(test_predictions)


# Regiões conformes simultâneas de uma jogada no campo
plot_conformal_event <- function(target_event_id, events_prepared, test_predictions, alpha = 0.10) {
  pred_data <- test_predictions |>
    filter(event_id == target_event_id)

  if (nrow(pred_data) == 0) {
    stop(sprintf("Lance event_id %s não encontrado em test_predictions.", target_event_id))
  }

  valid_pids <- unique(pred_data$player_id)

  hist_data <- events_prepared |>
    filter(event_id == target_event_id, time_sec <= 25, player_id %in% valid_pids) |>
    arrange(player_id, time_sec)

  p <- ggplot() +
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
    geom_circle(
      data = pred_data,
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, fill = as.factor(time_step), group = node_id),
      color = NA,
      alpha = 0.18
    ) +
    scale_fill_manual(name = "Horizonte (s)", values = viridisLite::viridis(5)) +

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

    coord_fixed(
      xlim = c(0, 105), 
      ylim = c(0, 68), 
      expand = FALSE 
    ) +
    labs(
      title = sprintf("Regiões Conformes Simultâneas (1 - \u03b1 = %.0f%%) | Lance %s", (1 - alpha) * 100, target_event_id),
      subtitle = "Azul: Previsto | Amarelo: Real | Branco: Observado (t \u2264 25s) | Bandas: Regiões Conformes"
    )

  return(p)
}

# Jogada de teste do modelo final
test_events_available <- unique(test_predictions$event_id)
cat("Lances disponíveis no Teste:", test_events_available, "\n")

sample_event_id <- test_events_available[6]

conf <- plot_conformal_event(
  target_event_id = sample_event_id, 
  events_prepared = events_prepared,
  test_predictions = test_predictions,
  alpha = alpha_level
)

conf


final_overall <- test_predictions |>
  summarize(
    ADE = mean(distance, na.rm = TRUE),
    FDE = mean(distance[time_step == max(time_step)], na.rm = TRUE),
    .groups = "drop"
  )
cat(sprintf("MODELO FINAL  — Teste: ADE = %.2f m | FDE = %.2f m\n", final_overall$ADE, final_overall$FDE))


