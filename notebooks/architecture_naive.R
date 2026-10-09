# ==========================================================================
# 0. SETUP
# ==========================================================================
library(torch)
library(tidyverse)
library(R6)
library(here)
library(ggsoccer)
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
  #    Apenas o treino é espelhado: a calibração permanece intacta.
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
# 2. DEFINIÇÃO DO MODELO — Social-LSTM (Alahi et al., CVPR 2016)
# ==========================================================================
# Implementação fiel de Alahi, Goel, Ramanathan, Robicquet, Fei-Fei &
# Savarese (2016), "Social LSTM: Human Trajectory Prediction in Crowded
# Spaces", CVPR 2016.
#
# Mecânica exata do artigo:
#  - Eq. (1): tensor social H_i^t (grade N_o x N_o x D) — os estados ocultos
#    h_j^{t-1} dos vizinhos são somados nas células da grade relativa à
#    posição do jogador i no instante t.
#  - Eq. (2): e_i^t = phi(x_i^t, y_i^t; W_e)  [embedding 64-d com ReLU];
#    a_i^t = phi(H_i^t; W_a) [MLP com ReLU sobre H após sum-pooling 8x8
#    sem overlap]; h_i^t = LSTM(h_i^{t-1}, [e_i^t, a_i^t]; W_l) — um LSTM por
#    trajetória com pesos compartilhados entre jogadores (D = 128).
#  - Eqs. (3)-(4): [mu, sigma, rho]_i^{t+1} = W_p h_i^t (cabeça linear 5 x D),
#    gaussiana bivariada sobre o deslocamento; sigma via exp e rho via tanh.
#  - Perda: NLL da gaussiana bivariada com correlação rho, somada sobre os
#    passos previstos (Eq. 4). Treino com teacher forcing (posições reais);
#    inferência autoregressiva (Sec. 3.1 do artigo).
#
# Adaptações declaradas (não alteram a arquitetura):
#  - Janela 25 observados / 5 alvos (dado a 1 Hz; o artigo usa 8/12 a 2,5 Hz).
#  - Coordenadas z-scoreadas (transformação linear; o pooling usa metros).
#  - Grade N_o = 32 células (artigo) numa janela de 10 m ao redor do jogador:
#    o artigo não fixa a janela física (implementações de referência usam 4 m
#    para pedestres); 10 m é a escala de interação a 1 Hz no futebol.
#  - Dropout de 0.5 nos embeddings [e_i^t | a_i^t]: o artigo não usa dropout,
#    mas as implementações de referência o aplicam nos embeddings e o regime
#    de amostra pequena (15% de validação ~ 40 jogadas) sem ele leva a
#    sobrequadro (NLL de validação cresce enquanto a de treino cai).
#  - Scheduled sampling (Bengio et al. 2015): disponível via cfg$tf_rate < 1.
#    Testado com tf_rate = 0.5 para mitigar exposure bias; não melhorou o ADE
#    de validação neste dataset (5.27 m vs 4.99 m com TF puro). O default
#    tf_rate = 1.0 é o procedimento exato do artigo.

SocialLSTM <- nn_module(
  "SocialLSTM",

  initialize = function(scaler, pos_embed_dim = 64L, social_embed_dim = 64L,
                        hidden_dim = 128L, num_cells = 32L, pool_size = 8L,
                        window_m = 10, dropout_rate = 0.5) {
    self$hidden_dim <- hidden_dim
    self$num_cells <- num_cells
    self$pool_size <- pool_size
    self$window_m <- window_m

    # Embeddings (Eq. 2 do artigo): phi das coordenadas e phi do tensor social
    self$emb_pos <- nn_sequential(
      nn_linear(2, pos_embed_dim),
      nn_relu()
    )
    # H (32x32x128) -> sum-pooling 8x8 -> (4x4x128) achatado -> a_i^t
    self$emb_social <- nn_sequential(
      nn_linear(as.integer((num_cells %/% pool_size)^2) * hidden_dim, social_embed_dim),
      nn_relu()
    )

    # Um LSTM por trajetória, pesos compartilhados (Eq. 2); em R torch o
    # LSTM cell é o nn_lstm avançado passo a passo com seq_len = 1.
    self$cell <- nn_lstm(pos_embed_dim + social_embed_dim, hidden_dim, batch_first = TRUE)

    # Cabeça de saída (Eq. 4): [mu_x, mu_y, log sigma_x, log sigma_y, atanh rho]
    self$out <- nn_linear(hidden_dim, 5)

    # Dropout nos embeddings (implementações de referência; ver adaptações acima)
    self$dropout_emb <- nn_dropout(dropout_rate)

    # Buffers para converter posições z-scoreadas -> metros (grade de pooling)
    self$mu_pos <- nn_buffer(torch_tensor(scaler$center[c("x", "y")])$view(c(1, 2)))
    self$sd_pos <- nn_buffer(torch_tensor(scaler$scale[c("x", "y")])$view(c(1, 2)))
  },

  # Tensor social H_i^t + embedding a_i^t (Eqs. 1-2 do artigo).
  # pos_m: [num_nodes, 2] em metros; h_prev: [num_nodes, hidden_dim].
  social_context = function(pos_m, h_prev) {
    num_nodes <- pos_m$shape[1]
    nc <- self$num_cells
    cell_size <- self$window_m / nc
    D <- self$hidden_dim

    # rel[i, j, :] = pos_j - pos_i
    rel <- pos_m$unsqueeze(1) - pos_m$unsqueeze(2)            # [n, n, 2]

    # Célula (m, n) de cada par na grade centrada no jogador i
    cell_xy <- torch_floor((rel + self$window_m / 2) / cell_size)$clamp(0, nc - 1L)
    inside <- (rel$abs() < self$window_m / 2)$all(dim = 3)    # [n, n] bool

    # Exclui o próprio jogador (Eq. 1 soma sobre vizinhos j != i)
    mask <- inside * (torch_eye(num_nodes, device = rel$device) == 0)

    # Célula linear do par (i, j): linha m = y, coluna n = x
    idx <- (cell_xy[, , 2] * nc + cell_xy[, , 1])$to(dtype = torch_long())  # [n, n]

    # Scatter-add vetorizado: para cada par (i, j) na janela, soma h_j^{t-1}
    # na célula correspondente da grade do jogador i (Eq. 1 do artigo).
    # Convenções do R torch: torch_nonzero, index_select e index_add_ são
    # todos 1-based; o índice linear achatado é (i-1)*nc^2 + cell + 1.
    flat_sel <- torch_nonzero(mask)                           # [P, 2]
    cell_ij <- idx$masked_select(mask)                        # [P]

    H <- torch_zeros(c(num_nodes * nc * nc, D), device = pos_m$device)
    if (flat_sel$numel() > 0) {
      src <- h_prev$index_select(1, flat_sel[, 2])            # [P, D]
      flat <- (flat_sel[, 1] - 1L) * (nc * nc) + cell_ij + 1L # [P] 1-based
      H$index_add_(1, flat, src)                              # acumula duplicatas
    }
    H <- H$view(c(num_nodes, nc, nc, D))                      # [n, nc, nc, D]

    # Sum-pooling 8x8 sem overlap (artigo, Sec. 3.2): 32x32 -> 4x4
    H <- H$view(c(num_nodes, as.integer(nc / self$pool_size), self$pool_size,
                  as.integer(nc / self$pool_size), self$pool_size, D))
    H <- H$sum(dim = 3)$sum(dim = 4)                          # [n, 4, 4, D]

    self$emb_social(H$flatten(start_dim = 2))                 # [n, social_embed_dim]
  },

  forward = function(batch, teacher_forcing = FALSE, tf_rate = 1.0, return_params = FALSE) {
    pos <- batch$x_cont[, , 1:2]  # [num_nodes, 25, 2] — apenas posições (artigo)
    target <- batch$y             # [num_nodes, 5, 2]
    num_nodes <- pos$shape[1]
    device <- pos$device

    h_prev <- torch_zeros(c(1, num_nodes, self$hidden_dim), device = device)
    c_prev <- torch_zeros(c(1, num_nodes, self$hidden_dim), device = device)

    # ---- Encoder: t = 1..25 (Eq. 2; o tensor social usa h^{t-1}, Eq. 1) ----
    for (t in 1:25) {
      pos_t <- pos[, t, ]
      a_t <- self$social_context(pos_t * self$sd_pos + self$mu_pos, h_prev[1, , ])
      e_t <- self$dropout_emb(self$emb_pos(pos_t))
      a_t <- self$dropout_emb(a_t)
      step_in <- torch_cat(list(e_t, a_t), dim = 2)$unsqueeze(2)  # [n, 1, in_dim]
      hc <- self$cell(step_in, list(h_prev, c_prev))
      h_prev <- hc[[2]][[1]]
      c_prev <- hc[[2]][[2]]
    }

    # ---- Decodificador: t = 26..30 (Eqs. 3-4) ----
    # Teacher forcing (artigo, Sec. 3.1) com scheduled sampling (Bengio et al.
    # 2015): com probabilidade 1 - tf_rate a entrada do passo k é a posição
    # prevista no passo k-1, mitigando o exposure bias do horizonte de 5 s a
    # 1 Hz (desvio de treino documentado; a arquitetura não muda).
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

      out5 <- self$out(h_prev[1, , ])          # [n, 5]
      mu_xy <- out5[, 1:2]                     # deslocamento médio (z-scoreado)
      sig_xy <- torch_exp(out5[, 3:4])         # sigma via exp (artigo)
      rho <- torch_tanh(out5[, 5])             # correlação via tanh (artigo)

      pred_pos <- input_pos + mu_xy
      preds[[k]] <- pred_pos$unsqueeze(2)
      mu_xy_l[[k]] <- mu_xy$unsqueeze(2)
      sig_xy_l[[k]] <- sig_xy$unsqueeze(2)
      rho_l[[k]] <- rho$unsqueeze(2)
      input_pos <- pred_pos
    }

    preds_tensor <- torch_cat(preds, dim = 2)  # [num_nodes, 5, 2]

    if (return_params) {
      list(
        preds = preds_tensor,
        mu_xy = torch_cat(mu_xy_l, dim = 2),   # [num_nodes, 5, 2]
        sig_xy = torch_cat(sig_xy_l, dim = 2),
        rho = torch_cat(rho_l, dim = 2),       # [num_nodes, 5, 1]
        input_seq = torch_cat(input_seq_l, dim = 2)  # [num_nodes, 5, 2]
      )
    } else {
      preds_tensor
    }
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

# NLL da gaussiana bivariada (perda do artigo, Eqs. 3-4). dx, dy: deslocamentos-
# alvo; mu_x, mu_y, sig_x, sig_y, rho: parâmetros previstos. Constantes
# aditivas (log 2*pi) omitidas — irrelevantes para a otimização.
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

# NLL total de um lote: os deslocamentos-alvo são medidos em relação à
# sequência de posições efetivamente usadas pelo decodificador (GT no teacher
# forcing, previstas na inferência/scheduled sampling) — devolvida pelo modelo.
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

# Factory da arquitetura principal: Social-LSTM (Alahi et al., CVPR 2016).
social_lstm_factory <- function(scaler, num_teams) {
  SocialLSTM$new(scaler = scaler)
}

# Treino de uma rede a partir de datasets prontos; devolve o modelo com os
# melhores pesos da validação restaurados. Semente própria por chamada.
# model_factory(scaler, num_teams) instancia a arquitetura desejada.
train_model <- function(train_dataset, val_dataset, scaler, cfg, seed,
                        model_factory, num_teams, verbose = TRUE) {
  set.seed(seed)
  torch_manual_seed(seed)

  device <- select_device()

  train_loader <- dataloader(train_dataset, batch_size = 1, shuffle = TRUE, collate_fn = custom_collate)
  val_loader <- dataloader(val_dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  model <- model_factory(scaler = scaler, num_teams = num_teams)$to(device = device)

  n_params <- sum(sapply(model$parameters, function(p) p$numel()))

  # RMS-prop com lr = 0.003 (artigo, Sec. 3.2); clip de gradiente como salvaguarda
  optimizer <- optim_rmsprop(model$parameters, lr = cfg$lr)
  early_stopping <- EarlyStopping$new(patience = cfg$patience)

  # Critério de seleção/early stopping: NLL (objetivo do artigo) por default;
  # cfg$stop_metric = "ade" usa o ADE de validação — o artigo treina com
  # número fixo de épocas, sem seleção por época; com ~40 jogadas de validação
  # a NLL autoregressiva é ruidosa e o ADE é o critério estável da comparação.
  stop_metric <- if (!is.null(cfg$stop_metric) && cfg$stop_metric == "ade") "ade" else "nll"

  history <- data.frame(epoch = integer(), train_nll = numeric(),
                        train_ade = numeric(), train_fde = numeric(),
                        val_nll = numeric(), val_ade = numeric(), val_fde = numeric())

  for (epoch in 1:cfg$max_epochs) {
    model$train()
    train_nll <- 0
    train_ade <- 0
    train_fde <- 0
    total_nodes_train <- 0

    coro::loop(for (batch in train_loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$y <- batch$y$to(device = device)

      optimizer$zero_grad()
      # Teacher forcing + scheduled sampling (tf_rate no cfg)
      out <- model(batch, teacher_forcing = TRUE, tf_rate = cfg$tf_rate, return_params = TRUE)

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
      epoch = epoch, train_nll = train_nll,
      train_ade = train_ade, train_fde = train_fde,
      val_nll = val_nll, val_ade = val_ade, val_fde = val_fde
    ))

    if (verbose && (epoch %% 5 == 0 || epoch == 1)) {
      cat(sprintf("Época %03d | Treino NLL: %.3f, ADE: %.2f m, FDE: %.2f m | Val NLL: %.3f, ADE: %.2f m, FDE: %.2f m\n",
                  epoch, train_nll, train_ade, train_fde, val_nll, val_ade, val_fde))
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

# Executa um experimento completo: preprocessamento, treino e datasets.
run_experiment <- function(df_raw, cfg, model_factory = social_lstm_factory) {
  prep_data <- preprocess_and_build_dataset(df_raw)
  graphs <- prep_data$graphs
  splits <- prep_data$splits
  scaler <- prep_data$scaler

  train_dataset <- TrajectoryDataset(graphs[splits$train])
  val_dataset <- TrajectoryDataset(graphs[splits$val])

  fit <- train_model(
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    scaler = scaler,
    cfg = cfg,
    seed = cfg$seed,
    model_factory = model_factory,
    num_teams = prep_data$num_teams
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
# 4. PREDIÇÃO CONFORME
# ==========================================================================
# Calibração split-conformal (Izbicki 2026; Diquigiovanni et al. 2022).
#
# (PDF Seção 1) Notação: jogada k, jogador i, horizonte t ∈ H = {26,...,30}
# (passos 1:5 no código); e_{i,t} = ||p_i(t) - p̂_i(t)||_2 em metros.
# (PDF Seção 2, Passo 1) ŝ_t = median_i e_{i,t}, estimada com resíduos
# out-of-fold do treino (aqui: split de validação, nunca usado no treino).
# (PDF Seção 2, Passo 2) R = max_{t∈H} e_t/ŝ_t; q̂ = estatística de ordem
# ⌈(n+1)(1-α)⌉ da amostra ordenada dos R's.
# (PDF Seção 2, Passo 3) C_t = {p : ||p - p̂(t)||_2 ≤ q̂·ŝ_t} (leitura simultânea).
# (PDF Seção 3) Alvo B = par (jogada, jogador): conformal naive, sem garantia
# exata nessa forma direta (trajetórias da mesma jogada não são permutáveis);
# a variante exata (um jogador sorteado por jogada de calibração) serve de
# aferição. Alvo A = jogada completa: R_k = max_i max_t e/ŝ_t, com garantia
# finita válida (jogadas são permutáveis).

# Resíduos e_{i,t} por (jogada, jogador, horizonte) em metros (PDF Seção 1).
# Devolve uma matriz [n_jogadores, 5] por jogada e a matriz empilhada
# [total_trajetórias, 5].
compute_error_matrices <- function(model, dataset, scaler) {
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

      # Desnormalização para coordenadas físicas reais do campo (metros)
      preds_real <- preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu

      diff_sq <- (preds_real - y_true_real)$pow(2)
      distances <- torch_sqrt(diff_sq$sum(dim = 3) + 1e-8) # [num_nodes, 5]
      dist_mat <- as.matrix(distances$cpu())

      err_by_event[[length(err_by_event) + 1]] <- dist_mat
    })
  })

  list(
    err_by_event = err_by_event,
    err_matrix = do.call(rbind, err_by_event) # [total_trajetórias, 5]
  )
}

# (PDF Seção 2, Passo 1) Forma temporal da banda: ŝ_t = median_i e_{i,t}.
# O piso evita a explosão dos escores normalizados e_t/ŝ_t.
estimate_band_shape <- function(err_matrix, floor_eps = 1e-4) {
  s_hat <- apply(err_matrix, 2, median)
  s_hat[s_hat < floor_eps] <- floor_eps
  s_hat
}

# Calibração conforme: ŝ_t vem de resíduos out-of-fold (s_hat_dataset, PDF
# Seção 2, Passo 1) e o conjunto de calibração é formado por jogadas
# independentes (nunca usadas no treino nem na estimação de ŝ_t).
#
# Alvo B (PDF Seção 3): conformal naive por trajetória de jogador. A garantia
# em amostra finita NÃO vale nessa forma direta (n efetivo = nº de jogadas,
# não de trajetórias); a cobertura é reportada como resultado empírico. A
# variante exata (um jogador sorteado por jogada de calibração) serve de
# aferição: q̂ próximos indicam que a dependência intra-jogada não atrapalha.
calibrate_conformal <- function(model, calib_dataset, s_hat_dataset, scaler,
                                alpha = 0.10, n_exact_draws = 200,
                                exact_seed = 7301) {
  # ---- (PDF Seção 2, Passo 1) ŝ_t com resíduos out-of-fold do treino ----
  s_hat_res <- compute_error_matrices(model, s_hat_dataset, scaler)
  s_hat <- estimate_band_shape(s_hat_res$err_matrix)
  n_events_s_hat <- length(s_hat_res$err_by_event)

  # ---- (PDF Seção 2, Passo 2) escores no conjunto de calibração ----
  calib_res <- compute_error_matrices(model, calib_dataset, scaler)
  err_by_event <- calib_res$err_by_event
  err_matrix <- calib_res$err_matrix

  n_players <- nrow(err_matrix)
  n_events <- length(err_by_event)

  # (PDF Seção 3, Alvo B) R_{k,i} = max_{t∈H} e_{k,i,t} / ŝ_t
  R_player <- apply(err_matrix, 1, function(row) max(row / s_hat))

  # (PDF Seção 2, Passo 2) q̂ = estatística de ordem ⌈(n+1)(1-α)⌉ da amostra
  # ordenada dos R's (quantil empírico com correção finita). No Alvo B naive,
  # n = nº de trajetórias; o n efetivo é o nº de jogadas (PDF Seção 3).
  q_level_player <- min(1.0, ceiling((n_players + 1) * (1 - alpha)) / n_players)
  q_hat_player <- quantile(R_player, probs = q_level_player, names = FALSE)

  # (PDF Seção 2, Passo 3) raios simultâneos r_t = q̂ · ŝ_t
  r_simultaneous <- q_hat_player * s_hat

  # (PDF Seção 3, Alvo A) R_k = max_i max_{t∈H} e_{k,i,t} / ŝ_t por jogada
  R_event <- sapply(err_by_event, function(mat) {
    max(apply(mat, 1, function(row) max(row / s_hat)))
  })
  q_level_event <- min(1.0, ceiling((n_events + 1) * (1 - alpha)) / n_events)
  q_hat_event <- quantile(R_event, probs = q_level_event, names = FALSE)
  r_event <- q_hat_event * s_hat

  # (PDF Seção 3, "Como proceder") Variante exata do Alvo B (aferição):
  # sorteando um jogador por jogada de calibração, as unidades voltam a ser
  # permutáveis com a unidade de teste (um jogador sorteado de uma jogada
  # nova), e a garantia é recuperada ao custo de 19/20 trajetórias.
  q_level_exact <- min(1.0, ceiling((n_events + 1) * (1 - alpha)) / n_events)
  set.seed(exact_seed)
  q_hat_exact_draws <- replicate(n_exact_draws, {
    R_exact <- sapply(err_by_event, function(mat) {
      max(mat[sample.int(nrow(mat), size = 1), ] / s_hat)
    })
    quantile(R_exact, probs = q_level_exact, names = FALSE)
  })
  q_hat_exact <- mean(q_hat_exact_draws)
  q_hat_exact_sd <- sd(q_hat_exact_draws)

  # (PDF Seção 1) Quantis pontuais marginais por horizonte — o status quo que
  # a banda simultânea substitui; mantido apenas como contraste metodológico.
  q_pointwise <- numeric(5)
  for (t in 1:5) {
    q_level_pw <- min(1.0, ceiling((n_players + 1) * (1 - alpha)) / n_players)
    q_pointwise[t] <- quantile(err_matrix[, t], probs = q_level_pw, names = FALSE)
  }

  cat(sprintf("\n=== CALIBRAÇÃO CONFORME (alpha = %.2f) ===\n", alpha))
  cat(sprintf("ŝ_t estimada em %d jogadas out-of-fold (validação) — PDF §2, Passo 1\n",
              n_events_s_hat))
  cat("Forma da banda ŝ_t (medianas em metros):", round(s_hat, 2), "\n")
  cat(sprintf("Calibração: %d jogadas independentes (%d trajetórias) — PDF §3\n",
              n_events, n_players))
  cat(sprintf("q̂ Alvo B naive (n = %d trajetórias; n efetivo ≈ %d jogadas): %.3f — PDF §3\n",
              n_players, n_events, q_hat_player))
  cat("Raios conformes simultâneos r_t = q̂·ŝ_t (metros):", round(r_simultaneous, 2), "— PDF §2, Passo 3\n")
  cat(sprintf("q̂ variante exata Alvo B (%d sorteios, 1 jogador/jogada): %.3f ± %.3f — PDF §3, Como proceder\n",
              n_exact_draws, q_hat_exact, q_hat_exact_sd))
  cat(sprintf("q̂ Alvo A (jogada completa): %.3f — bandas largas por construção (PDF §3)\n",
              q_hat_event))
  cat("Raios Alvo A (metros):", round(r_event, 2), "\n")
  cat("Quantis marginais pontuais (contraste PDF §1, em metros):", round(q_pointwise, 2), "\n\n")

  list(
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
}

# Predições com regiões conformes no teste: uma linha por (jogador, horizonte)
# com o raio r_t da banda simultânea e a cobertura do alvo verdadeiro.
predict_with_regions <- function(model, dataset, scaler, r_t) {
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

# (PDF Seção 3, "Como proceder") Cobertura empírica do Alvo B medida por
# repetições ao nível da jogada: a cada repetição sorteia-se uma trajetória
# por jogada de teste e calcula-se a fração coberta. A média pondera jogadas
# igualmente (n efetivo = nº de jogadas); o DP quantifica a flutuação do
# estimador entre repetições.
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


# ==========================================================================
# 5. VISUALIZAÇÃO
# ==========================================================================
# Regiões conformes simultâneas de uma jogada no campo (105 x 68 m)
plot_conformal_event <- function(target_event_id, events_prepared, test_predictions, alpha = 0.10) {
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
    # A. Regiões Conformes de Predição (Tubos de Incerteza)
    geom_circle(
      data = pred_data,
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, fill = as.factor(time_step), group = node_id),
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
      title = sprintf("Regiões Conformes Simultâneas (1 - \u03b1 = %.0f%%) | Lance %s", (1 - alpha) * 100, target_event_id),
      subtitle = "Azul: Previsto | Amarelo: Real | Branco: Observado (t \u2264 25s) | Bandas: Regiões Conformes"
    )

  return(p)
}
