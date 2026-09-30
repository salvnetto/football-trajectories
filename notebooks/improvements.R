# ==========================================================================
#   0. SETUP
#   1. PRÉ-PROCESSAMENTO & CONSTRUÇÃO DO DATASET
#   2. DEFINIÇÃO DO MODELO
#   3. TREINO
#   4. INFRAESTRUTURA CONFORME
#   5. CALIBRAÇÃO: RISK CONTROL & CROSS-CONFORMAL
#   6. EXECUÇÃO: DADOS & MODELO SPLIT-CONFORMAL
#   7. RESÍDUOS OOF DO TREINO: MODELO DE ESCALA
#   8. CALIBRAÇÃO & PREDIÇÕES NO TESTE
#   9. CROSS-CONFORMAL
#   10. AVALIAÇÃO & PLOT DA JOGADA
#
#   Implementa os itens 1-6 de docs/conformal_trajetórias-1.pdf:
#     1. Um único score para a trajetória completa (em vez de um quantil por horizonte);
#     2. r_t = q_hat * s_hat_t, com s_hat_t estimado por resíduos out-of-fold do treino;
#     3. Unidade de calibração = (jogada, jogador) — Alvo B obtido APENAS via risk
#        control (item 6): a Seção 6 substitui a Seção 3 do pdf (o ingênuo não
#        tem garantia de amostra finita e a variante exata descarta 19/20 dos dados);
#     4. Escala adaptativa s_hat_t(x_i) = exp(g(x_i, t)) sobre o estado latente da rede;
#     5. Cross-conformal (Vovk 2015; Barber et al. 2021) com dobras por jogada;
#     6. Conformal risk control (Angelopoulos et al. 2024): E[L_nova(lambda_hat)] <= alpha.
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

  # Encoder espaçotemporal: devolve o estado latente h_i(T_obs) por jogador
  # e a cinemática do último frame observado. h_i(T_obs) é a representação
  # congelada da rede usada como preditor do modelo de escala (item 4).
  encode = function(batch) {
    x_cont <- batch$x_cont       # [num_nodes, seq_len = 25, cont_dim = 8]
    team_idx <- batch$x_cat[, 1] # [num_nodes]
    role_idx <- batch$x_cat[, 2] # [num_nodes]

    num_nodes <- x_cont$shape[1]
    seq_len <- x_cont$shape[2]

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
    enc_states <- self$temporal_encoder(x_spatio_temporal)[[1]]

    list(
      last_hidden = enc_states[, seq_len, ],   # [num_nodes, hidden_dim]
      last_known_pos = x_cont[, seq_len, 1:2], # [num_nodes, 2]
      last_known_vel = x_cont[, seq_len, 7:8], # [num_nodes, 2]
      num_nodes = num_nodes,
      seq_len = seq_len
    )
  },

  forward = function(batch, return_latents = FALSE) {
    enc <- self$encode(batch)

    num_nodes <- enc$num_nodes
    seq_len <- enc$seq_len
    last_hidden <- enc$last_hidden
    last_known_pos <- enc$last_known_pos
    last_known_vel <- enc$last_known_vel

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

    if (return_latents) {
      return(list(preds = preds_tensor, last_hidden = last_hidden))
    }
    preds_tensor
  }
)


# ==========================================================================
# 3. TREINO
# ==========================================================================

# Tensores de média/desvio para desescalonar posições (x, y)
mu_sd_tensors <- function(scaler, device) {
  mu <- c(scaler$center["x"], scaler$center["y"])
  sd <- c(scaler$scale["x"], scaler$scale["y"])
  list(
    mu = torch_tensor(mu, device = device)$view(c(1, 1, 2)),
    sd = torch_tensor(sd, device = device)$view(c(1, 1, 2))
  )
}

compute_ade_loss_and_fde <- function(preds, targets, scaler) {
  ts <- mu_sd_tensors(scaler, preds$device)

  preds_real <- preds * ts$sd + ts$mu
  targets_real <- targets * ts$sd + ts$mu

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

select_device <- function() {
  if (cuda_is_available()) {
    torch_device("cuda")
  } else if (backends_mps_is_available()) {
    torch_device("mps")
  } else {
    torch_device("cpu")
  }
}

# Treino de uma rede a partir de datasets prontos; devolve o modelo com os
# melhores pesos da validação restaurados. Semente própria por chamada.
train_model <- function(train_dataset, val_dataset, num_teams, scaler, cfg, seed, verbose = TRUE) {
  set.seed(seed)
  torch_manual_seed(seed)

  device <- select_device()

  train_loader <- dataloader(train_dataset, batch_size = 1, shuffle = TRUE, collate_fn = custom_collate)
  val_loader <- dataloader(val_dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  model <- TrajectorySeq2SeqGNNv2$new(
    num_teams = num_teams,
    scaler = scaler
  )$to(device = device)

  n_params <- sum(sapply(model$parameters, function(p) p$numel()))

  # Configuração de treino do E3_mirror (inalterada em relação ao notebook)
  optimizer <- optim_adam(model$parameters, lr = cfg$lr, weight_decay = cfg$weight_decay)
  scheduler <- lr_reduce_on_plateau(optimizer, mode = "min", factor = 0.5, patience = 2)
  early_stopping <- EarlyStopping$new(patience = cfg$patience)

  history <- data.frame(epoch = integer(), train_ade = numeric(), train_fde = numeric(),
                        val_ade = numeric(), val_fde = numeric())

  for (epoch in 1:cfg$max_epochs) {
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

# Modelo split-conformal: treina apenas nas jogadas de treino (a calibração
# permanece intacta) e devolve também os datasets dos demais conjuntos.
run_experiment <- function(df_raw, cfg) {
  prep_data <- preprocess_and_build_dataset(df_raw)
  graphs <- prep_data$graphs
  splits <- prep_data$splits
  scaler <- prep_data$scaler

  train_dataset <- TrajectoryDataset(graphs[splits$train])
  val_dataset <- TrajectoryDataset(graphs[splits$val])

  fit <- train_model(
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    num_teams = prep_data$num_teams,
    scaler = scaler,
    cfg = cfg,
    seed = cfg$seed
  )

  c(
    fit,
    list(
      prep_data = prep_data,
      scaler = scaler,
      train_dataset = train_dataset,
      val_dataset = val_dataset,
      calib_dataset = TrajectoryDataset(graphs[splits$calib]),
      test_dataset = TrajectoryDataset(graphs[splits$test])
    )
  )
}


# ==========================================================================
# 4. INFRAESTRUTURA CONFORME
# ==========================================================================

# Predições, erros e estados latentes de um dataset numa única passada.
# Uma linha por (jogador, horizonte); a coluna `latent` guarda h_i(T_obs).
collect_predictions <- function(model, dataset, scaler) {
  model$eval()
  device <- model$parameters[[1]]$device
  ts <- mu_sd_tensors(scaler, device)

  loader <- dataloader(dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)
  rows <- list()

  with_no_grad({
    coro::loop(for (batch in loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat <- batch$x_cat$to(device = device)
      batch$y <- batch$y$to(device = device)

      out <- model(batch, return_latents = TRUE)

      preds_real <- out$preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu

      preds_arr <- as.array(preds_real$cpu())   # [num_nodes, 5, 2]
      y_true_arr <- as.array(y_true_real$cpu())
      lat_arr <- as.matrix(out$last_hidden$cpu()) # [num_nodes, hidden_dim]

      num_nodes <- dim(preds_arr)[1]
      n_steps <- dim(preds_arr)[2]

      for (node in seq_len(num_nodes)) {
        for (t in seq_len(n_steps)) {
          px <- preds_arr[node, t, 1]
          py <- preds_arr[node, t, 2]
          tx <- y_true_arr[node, t, 1]
          ty <- y_true_arr[node, t, 2]

          rows[[length(rows) + 1]] <- tibble(
            event_id = batch$event_id,
            player_id = batch$player_ids[node],
            node_id = node,
            team_code = batch$team_codes[node],
            team_name = batch$team_names[node],
            time_step = t,
            pred_x = px,
            pred_y = py,
            true_x = tx,
            true_y = ty,
            distance = sqrt((px - tx)^2 + (py - ty)^2),
            latent = list(lat_arr[node, ])
          )
        }
      }
    })
  })

  bind_rows(rows)
}

# Escore de não-conformidade por trajetória (unidade = (jogada, jogador);
# Alvo B obtido via risk control, item 6): R = max_t e_{i,t} / s_hat_{i,t}
trajectory_scores <- function(pred_df) {
  pred_df |>
    group_by(event_id, player_id) |>
    summarise(score = max(distance / s_hat), .groups = "drop")
}

# Quantil conforme = estatística de ordem ceil((n + 1) * (1 - alpha)) da
# amostra ordenada de scores (item 2, Passo 2)
conformal_quantile <- function(scores, alpha) {
  n <- length(scores)
  k <- min(n, ceiling((n + 1) * (1 - alpha)))
  sort(scores)[k]
}

# Modelo de escala adaptativa (item 4): regressão ridge em log(e + eps) com
# preditores (h_i(T_obs), t) — o estado latente congelado da rede e o
# horizonte num único modelo. Ajustado SÓ com resíduos out-of-fold do treino.
fit_scale_model <- function(train_preds, lambda = 0.1, eps = 1e-6) {
  lat_mat <- do.call(rbind, train_preds$latent) # [n_traj*5, hidden_dim]
  lat_center <- colMeans(lat_mat)
  lat_scale <- apply(lat_mat, 2, sd)
  lat_scale[lat_scale == 0] <- 1

  Z <- sweep(sweep(lat_mat, 2, lat_center, "-"), 2, lat_scale, "/")

  levels_t <- sort(unique(train_preds$time_step))
  t_onehot <- model.matrix(~ 0 + factor(time_step), data = train_preds)

  X <- cbind(Z, t_onehot)
  y <- log(train_preds$distance + eps)

  p <- ncol(X)
  beta <- solve(crossprod(X) + lambda * diag(p), crossprod(X, y))

  list(beta = beta, lat_center = lat_center, lat_scale = lat_scale, levels = levels_t)
}

# s_hat_t(x_i) = exp(g(x_i, t)) para novas trajetórias (item 4)
predict_scale <- function(fit, pred_df) {
  lat_mat <- do.call(rbind, pred_df$latent)
  Z <- sweep(sweep(lat_mat, 2, fit$lat_center, "-"), 2, fit$lat_scale, "/")
  t_onehot <- model.matrix(~ 0 + factor(time_step, levels = fit$levels), data = pred_df)

  X <- cbind(Z, t_onehot)
  as.numeric(exp(X %*% fit$beta))
}

# Regiões conformes: círculo C_t = {p : ||p - p_hat(t)||_2 <= q_hat * s_hat_t}
# no horizonte t (item 2, Passo 3)
add_conformal_regions <- function(pred_df, q_hat) {
  pred_df |>
    mutate(
      conf_radius = q_hat * s_hat,
      covered = distance <= conf_radius
    )
}

# Cobertura empírica ao nível da trajetória, área média da região (eficiência)
# e fração média de trajetórias fora da banda por jogada (alvo do item 6)
coverage_summary <- function(pred_df) {
  cov_traj <- pred_df |>
    group_by(event_id, player_id) |>
    summarise(covered_all = all(covered), .groups = "drop")

  list(
    coverage = mean(cov_traj$covered_all),
    mean_area = pi * mean(pred_df$conf_radius^2),
    frac_outside_play = mean(
      pred_df |>
        group_by(event_id) |>
        summarise(frac_out = mean(!covered), .groups = "drop") |>
        pull(frac_out)
    )
  )
}


# ==========================================================================
# 5. CALIBRAÇÃO: RISK CONTROL & CROSS-CONFORMAL
# ==========================================================================

# Conformal risk control (item 6; Angelopoulos et al. 2024): perda por jogada
# L_k(lambda) = fração de trajetórias com score > lambda (não crescente em
# lambda). lambda_hat = inf{lambda : (sum_k L_k(lambda) + 1) / (n + 1) <= alpha}
# garante E[L_nova(lambda_hat)] <= alpha, média sobre jogadas novas.
risk_control_lambda <- function(scores_df, alpha) {
  n_plays <- n_distinct(scores_df$event_id)
  scores_by_play <- split(scores_df$score, scores_df$event_id)

  loss_sum <- function(lambda) {
    sum(sapply(scores_by_play, function(s) mean(s > lambda)))
  }

  threshold <- alpha * (n_plays + 1) - 1
  candidates <- sort(unique(scores_df$score))

  lambda_hat <- NA_real_
  for (lambda in candidates) {
    if (loss_sum(lambda) <= threshold) {
      lambda_hat <- lambda
      break
    }
  }
  if (is.na(lambda_hat)) lambda_hat <- max(candidates)
  lambda_hat
}

# Índices dos grafos cujas jogadas estão em play_ids (inclui os espelhos)
graph_indices_for_plays <- function(graphs, play_ids) {
  which(sapply(graphs, function(g) g$event_id) %in% play_ids)
}

# Apenas a primeira ocorrência de cada jogada (descarta os espelhos), usada na
# coleta de scores e de resíduos out-of-fold para que cada jogada contribua
# uma única vez
first_graph_indices_for_plays <- function(graphs, play_ids) {
  idx <- graph_indices_for_plays(graphs, play_ids)
  idx[!duplicated(sapply(graphs[idx], function(g) g$event_id))]
}

# Jogadas (únicas) dos grafos nos índices dados
play_ids_for_indices <- function(graphs, idx) {
  unique(sapply(graphs[idx], function(g) g$event_id))
}

# Resíduos out-of-fold do treino (item 4): K dobras internas das jogadas de
# treino; cada jogada recebe predição de uma rede treinada sem ela e contribui
# UMA ÚNICA vez (o espelho não entra nos resíduos — seria a mesma jogada duas
# vezes). Os resíduos alimentam o modelo de escala.
compute_oof_train_errors <- function(prep, train_cfg, k = 5) {
  graphs <- prep$graphs
  val_dataset <- TrajectoryDataset(graphs[prep$splits$val])
  train_plays <- play_ids_for_indices(graphs, prep$splits$train)

  set.seed(train_cfg$seed)
  folds <- split(train_plays, sample(rep(seq_len(k), length.out = length(train_plays))))

  fold_preds <- list()

  for (fold in seq_len(k)) {
    cat(sprintf("Resíduos OOF do treino: dobra %d/%d\n", fold, k))
    fit_plays <- setdiff(train_plays, folds[[fold]])

    net <- train_model(
      train_dataset = TrajectoryDataset(graphs[graph_indices_for_plays(graphs, fit_plays)]),
      val_dataset = val_dataset,
      num_teams = prep$num_teams,
      scaler = prep$scaler,
      cfg = train_cfg,
      seed = train_cfg$seed + fold,
      verbose = FALSE
    )$model

    fold_preds[[fold]] <- collect_predictions(
      net,
      TrajectoryDataset(graphs[first_graph_indices_for_plays(graphs, folds[[fold]])]),
      prep$scaler
    )
  }

  bind_rows(fold_preds)
}

# Cross-conformal (item 5; Vovk 2015; Barber et al. 2021): K dobras por jogada
# completa (a jogada e seu espelho nunca são divididos entre dobras; jogadores
# e instantes de uma mesma jogada nunca mudam de dobra). Para cada dobra k:
# (i) split interno das K-1 dobras (80/20) gera resíduos OOF que ajustam o
# modelo de escala da dobra; (ii) a rede treinada nas K-1 dobras completas
# calcula os scores da dobra k; (iii) os scores das K dobras são reunidos e
# q_hat é obtido como no item 2. A rede final treina em todos os dados
# (treino + calibração) e a jogada nova usa a média das K previsões dos
# modelos de escala das dobras (Passo 4); a validação fica reservada ao early
# stopping.
cross_conformal <- function(prep, train_cfg, cf_cfg, k = 5) {
  graphs <- prep$graphs
  val_dataset <- TrajectoryDataset(graphs[prep$splits$val])

  cc_plays <- unique(c(
    play_ids_for_indices(graphs, prep$splits$train),
    play_ids_for_indices(graphs, prep$splits$calib)
  ))

  set.seed(train_cfg$seed)
  folds <- split(cc_plays, sample(rep(seq_len(k), length.out = length(cc_plays))))
  fold_scores <- list()
  scale_models <- list()

  for (fold in seq_len(k)) {
    cat(sprintf("Cross-conformal: dobra %d/%d\n", fold, k))
    fold_plays <- folds[[fold]]
    compl_plays <- setdiff(cc_plays, fold_plays)

    # (i) split interno das K-1 dobras para os resíduos do modelo de escala
    set.seed(train_cfg$seed + 100 * fold)
    scale_fit_plays <- sample(compl_plays, floor(0.8 * length(compl_plays)))
    scale_resid_plays <- setdiff(compl_plays, scale_fit_plays)

    scale_net <- train_model(
      train_dataset = TrajectoryDataset(graphs[graph_indices_for_plays(graphs, scale_fit_plays)]),
      val_dataset = val_dataset,
      num_teams = prep$num_teams,
      scaler = prep$scaler,
      cfg = train_cfg,
      seed = train_cfg$seed + 100 * fold + 1,
      verbose = FALSE
    )$model
    scale_resid_preds <- collect_predictions(
      scale_net,
      TrajectoryDataset(graphs[first_graph_indices_for_plays(graphs, scale_resid_plays)]),
      prep$scaler
    )
    scale_model_k <- fit_scale_model(scale_resid_preds, lambda = cf_cfg$scale_lambda, eps = cf_cfg$scale_eps)
    scale_models[[fold]] <- scale_model_k

    # (ii) rede completa das K-1 dobras -> scores da dobra k
    fold_net <- train_model(
      train_dataset = TrajectoryDataset(graphs[graph_indices_for_plays(graphs, compl_plays)]),
      val_dataset = val_dataset,
      num_teams = prep$num_teams,
      scaler = prep$scaler,
      cfg = train_cfg,
      seed = train_cfg$seed + 100 * fold + 2,
      verbose = FALSE
    )$model

    fold_preds <- collect_predictions(
      fold_net,
      TrajectoryDataset(graphs[first_graph_indices_for_plays(graphs, fold_plays)]),
      prep$scaler
    )
    fold_preds$s_hat <- predict_scale(scale_model_k, fold_preds)

    fold_scores[[fold]] <- trajectory_scores(fold_preds) |> mutate(fold = fold)
  }

  scores <- bind_rows(fold_scores)
  q_hat <- conformal_quantile(scores$score, cf_cfg$alpha)

  # (iii) rede final: treino em todos os dados (treino + calibração)
  final_fit <- train_model(
    train_dataset = TrajectoryDataset(graphs[graph_indices_for_plays(graphs, cc_plays)]),
    val_dataset = val_dataset,
    num_teams = prep$num_teams,
    scaler = prep$scaler,
    cfg = train_cfg,
    seed = train_cfg$seed + 1000
  )

  list(q_hat = q_hat, scores = scores, final_model = final_fit$model, scale_models = scale_models)
}


# ==========================================================================
# 6. EXECUÇÃO: DADOS & MODELO SPLIT-CONFORMAL
# ==========================================================================

# Configurações do experimento (sementes explícitas em todo treino)
train_cfg <- list(
  seed = 42,
  lr = 0.001,
  max_epochs = 100,
  patience = 10,
  weight_decay = 1e-5
)

conformal_cfg <- list(
  alpha = 0.10,
  k_folds = 5,
  scale_lambda = 0.1,
  scale_eps = 1e-6
)

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


res <- run_experiment(events_prepared, train_cfg)

model <- res$model
prep <- res$prep_data
test_dataset <- res$test_dataset
calib_dataset <- res$calib_dataset
scaler <- res$scaler

cat(sprintf("\nValidação: ADE %.2f m | FDE %.2f m | %d parâmetros\n",
            res$val_ade_best, res$val_fde_best, res$n_params))


# ==========================================================================
# 7. RESÍDUOS OOF DO TREINO: MODELO DE ESCALA (item 4)
# ==========================================================================
# s_hat_t(x_i) é estimado exclusivamente com resíduos out-of-fold do treino
# (o modelo de escala do item 4 usa apenas dados de treino — nunca calibração).
oof_train_preds <- compute_oof_train_errors(prep, train_cfg, k = conformal_cfg$k_folds)

scale_model <- fit_scale_model(
  oof_train_preds,
  lambda = conformal_cfg$scale_lambda,
  eps = conformal_cfg$scale_eps
)


# ==========================================================================
# 8. CALIBRAÇÃO & PREDIÇÕES NO TESTE (itens 2, 4 e 6)
# ==========================================================================
alpha_level <- conformal_cfg$alpha

calib_preds <- collect_predictions(model, calib_dataset, scaler)
test_preds <- collect_predictions(model, test_dataset, scaler)

# Escala adaptativa — item 4: s_hat_t(x_i) = exp(g(x_i, t))
calib_preds$s_hat <- predict_scale(scale_model, calib_preds)
calib_scores_adapt <- trajectory_scores(calib_preds)

test_preds$s_hat <- predict_scale(scale_model, test_preds)

# Alvo B via conformal risk control — item 6 (a Seção 6 do pdf substitui a
# Seção 3): L_k(lambda) = fração de jogadores da jogada k com score > lambda;
# lambda_hat = inf{lambda : (sum_k L_k(lambda) + 1) / (n + 1) <= alpha} garante
# E[L_nova(lambda_hat)] <= alpha, com a jogada como unidade permutável.
lambda_hat <- risk_control_lambda(calib_scores_adapt, alpha_level)
test_rc <- add_conformal_regions(test_preds, lambda_hat)

cat(sprintf("\n=== SPLIT-CONFORMAL — ALVO B VIA RISK CONTROL (alpha = %.2f) ===\n", alpha_level))
cat(sprintf("lambda_hat (risk control): %.3f\n", lambda_hat))


# ==========================================================================
# 9. CROSS-CONFORMAL (item 5)
# ==========================================================================

# Passo 4 do item 5: rede final treinada em todos os dados; s_hat_t(x_i) da
# jogada nova = média das K previsões dos modelos de escala das dobras.
cc <- cross_conformal(prep, train_cfg, conformal_cfg, k = conformal_cfg$k_folds)

test_preds_cc <- collect_predictions(cc$final_model, test_dataset, scaler)
scale_preds_cc <- lapply(cc$scale_models, predict_scale, pred_df = test_preds_cc)
test_preds_cc$s_hat <- rowMeans(do.call(cbind, scale_preds_cc))
test_cc <- add_conformal_regions(test_preds_cc, cc$q_hat)


# ==========================================================================
# 10. AVALIAÇÃO & PLOT DA JOGADA
# ==========================================================================
eval_rc <- coverage_summary(test_rc)
eval_cc <- coverage_summary(test_cc)

results_table <- tibble(
  metodo = c(
    "Risk control",
    "Cross-conformal"
  ),
  quantil = c(lambda_hat, cc$q_hat),
  cobertura = c(eval_rc$coverage, eval_cc$coverage),
  area_media_m2 = c(eval_rc$mean_area, eval_cc$mean_area),
  frac_fora_media_jogada = c(eval_rc$frac_outside_play, eval_cc$frac_outside_play)
)

results_table


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

# Plot com as regiões do risk control (item 6): raio = lambda_hat * s_hat_t(x_i)
test_predictions <- test_rc

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


overall_ade_fde <- function(pred_df) {
  pred_df |>
    summarize(
      ADE = mean(distance, na.rm = TRUE),
      FDE = mean(distance[time_step == max(time_step)], na.rm = TRUE),
      .groups = "drop"
    )
}

final_overall <- overall_ade_fde(test_predictions)
cat(sprintf("MODELO SPLIT-CONFORMAL — Teste: ADE = %.2f m | FDE = %.2f m\n",
            final_overall$ADE, final_overall$FDE))

final_overall_cc <- overall_ade_fde(test_preds_cc)
cat(sprintf("MODELO FINAL CROSS-CONFORMAL — Teste: ADE = %.2f m | FDE = %.2f m\n",
            final_overall_cc$ADE, final_overall_cc$FDE))
# Plot com as regiões do risk control (item 6): raio = lambda_hat * s_hat_t(x_i)
test_predictions <- test_rc