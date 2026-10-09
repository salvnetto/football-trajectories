# ==============================================================================
# ARCHITECTURE V2: PREDIÇÃO ESPAÇOTEMPORAL & CONFORME DE TRAJETÓRIAS NO FUTEBOL
# ==============================================================================
# Implementação fiel das Sugestões 4, 5 e 6 de Rafael Izbicki (2026),
# "Notas de implementação: regiões conformais para a trajetória completa".
#
# ESTE ARQUIVO SUBSTITUI AS ABORDAGENS PONTUAIS E ESTÁTICAS DE ARCHITECTURE.R:
#
# 1. SUGESTÃO 4 (Escala Adaptativa ŝ_t(x_i)):
#    Substitui a forma de banda estática e escalar ŝ_t = median_i e_{i,t} (Sugestão 2)
#    por uma função condicional às covariáveis de contexto do jogador no instante
#    T_obs = 25 (velocidade escalar, vel_x, vel_y, distância à bola, papel tático,
#    velocidade da bola). O modelo log(e_{i,t} + eps) = g(x_i, t) é ajustado
#    estritamente com resíduos out-of-fold. Jogadores em movimento retilíneo e
#    isolados recebem regiões menores; jogadores velozes ou em disputa de bola
#    recebem regiões maiores.
#
# 2. SUGESTÃO 5 (Cross-Conformal; Vovk 2015; Barber et al. 2021):
#    Substitui o particionamento fixo de calibração única (que desperdiça amostras
#    em bases com poucas sequências ofensivas). As jogadas são particionadas em
#    K dobras balanceadas. A rede e o modelo de escala são treinados nas K - 1 dobras,
#    e os escores e perdas são avaliados na dobra k out-of-fold. Os dados são
#    agregados para calibrar o parâmetro global sobre 100% das jogadas.
#
# 3. SUGESTÃO 6 (Controle de Risco Conforme - CRC; Angelopoulos et al., ICLR 2024):
#    Substitui tanto o conservador Alvo A (max sobre 20 jogadores, bandas gigantes)
#    quanto o Alvo B naive (que tratava jogadores da mesma jogada como permutáveis).
#    A unidade permutável é a JOGADA completa k. A perda da jogada é a fração de
#    jogadores cuja trajetória escapa da banda:
#      L_k(lambda) = (1 / N_k) * sum_{i=1}^{N_k} 1{ max_{t in H} e_{k,i,t} / ŝ_t(x_{k,i}) > lambda }
#    O parâmetro lambda_hat é calibrado com garantia finita de esperança:
#      E[ L_nova(lambda_hat) ] <= alpha.
#    Raios adaptativos individuais: r_{i,t} = lambda_hat * ŝ_t(x_i).
# ==============================================================================

# ==============================================================================
# 0. SETUP & DEPENDÊNCIAS
# ==============================================================================
suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
  library(R6)
  library(here)
  library(ggsoccer)
  library(ggforce)
})

# Carrega funções de leitura e estruturação de eventos de tracking se disponível
if (file.exists(here("src", "modeling.R"))) {
  source(here("src", "modeling.R"))
}

# ==============================================================================
# 1. PRÉ-PROCESSAMENTO & CONSTRUÇÃO DO DATASET DE GRAFOS
# ==============================================================================

# Colação customizada para lotes de tamanho 1 (preserva tipos e metadados R)
custom_collate <- function(batch) {
  batch[[1]]
}

# Espelhamento lateral exato y -> 68 - y no espaço z-scoreado:
# y'_s = (68 - 2*mu_y)/sd_y - y_s
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

# Constrói a lista completa de grafos espaçotemporais a partir dos dados tabulares
# Cada grafo representa uma jogada ofensiva completa (25 frames observados, 5 alvos)
build_graph_list <- function(df, cont_cols = c('x', 'y', 'ball_speed', 'x_ball',
                                              'y_ball', 'dist_to_ball', 'vel_x', 'vel_y')) {
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

  graphs
}

# Pré-processamento com particionamento flexível (Cross-Conformal K-Fold ou Split clássico)
# Toda a normalização z-score é estritamente calculada sem vazamento de teste.
preprocess_trajectories <- function(df, seed = 42) {
  set.seed(seed)
  df <- as.data.frame(df)

  # 1. Engenharia de Variáveis Físicas & Categóricas
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
  cont_cols <- c('x', 'y', 'ball_speed', 'x_ball', 'y_ball', 'dist_to_ball', 'vel_x', 'vel_y')

  # 2. Estatísticas do Scaler sobre todo o horizonte observado t <= 25
  # Para z-score global não-vazado, calculamos nos frames t <= 25
  obs_only <- df |> filter(time_sec <= 25)
  scaler_center <- colMeans(obs_only[, cont_cols], na.rm = TRUE)
  scaler_scale <- apply(obs_only[, cont_cols], 2, sd, na.rm = TRUE)
  scaler_scale[scaler_scale == 0] <- 1

  # Aplica z-score no dataframe
  df_scaled <- df
  df_scaled[, cont_cols] <- scale(df[, cont_cols], center = scaler_center, scale = scaler_scale)

  # Constrói lista de grafos
  graphs <- build_graph_list(df_scaled, cont_cols = cont_cols)

  list(
    graphs = graphs,
    df_raw = df,
    num_teams = num_teams,
    scaler = list(center = scaler_center, scale = scaler_scale)
  )
}

# Particionador K-Fold estratificado estritamente por jogada (event_id)
# Sugestão 5 (Vovk 2015): jogadas inteiras são as unidades das dobras
split_events_kfold <- function(graphs, K = 5, seed = 42) {
  set.seed(seed)
  event_ids <- unique(sapply(graphs, function(g) g$event_id))
  n_events <- length(event_ids)

  if (K > n_events) {
    warning(sprintf("K (%d) maior que o número de jogadas (%d). Ajustando K = %d.", K, n_events, n_events))
    K <- n_events
  }

  shuffled_events <- sample(event_ids)
  folds_events <- split(shuffled_events, rep(1:K, length.out = n_events))

  graph_event_ids <- sapply(graphs, function(g) g$event_id)

  folds_indices <- lapply(folds_events, function(evts) {
    which(graph_event_ids %in% evts)
  })

  names(folds_indices) <- paste0("fold_", 1:K)
  folds_indices
}

# Dataset R6 para Torch
TrajectoryDataset <- dataset(
  name = "TrajectoryDataset",
  initialize = function(graphs) { self$graphs <- graphs },
  .getitem = function(i) { self$graphs[[i]] },
  .length = function() { length(self$graphs) }
)


# ==============================================================================
# 2. DEFINIÇÃO DO MODELO: Social-LSTM (Alahi et al., CVPR 2016)
# ==============================================================================
# Implementação fiel de Alahi et al. (CVPR 2016):
#  - Tensor social H_i^t (Eq. 1) via scatter-add em grade espacial (32x32x128).
#  - Sum-pooling sem sobreposição 8x8 -> 4x4x128 -> embedding social a_i^t (Eq. 2).
#  - Célula LSTM passo a passo compartilhada entre trajetórias.
#  - Cabeça linear bivariada 5D [mu_x, mu_y, log sigma_x, log sigma_y, atanh rho].
#  - Método adicional 'extract_latent' para expor o estado h_i^{(25)} ao modelo de escala.

SocialLSTM <- nn_module(
  "SocialLSTM",

  initialize = function(scaler, pos_embed_dim = 64L, social_embed_dim = 64L,
                        hidden_dim = 128L, num_cells = 32L, pool_size = 8L,
                        window_m = 10, dropout_rate = 0.5) {
    self$hidden_dim <- hidden_dim
    self$num_cells <- num_cells
    self$pool_size <- pool_size
    self$window_m <- window_m

    # Embeddings de coordenadas e contexto social
    self$emb_pos <- nn_sequential(
      nn_linear(2, pos_embed_dim),
      nn_relu()
    )
    self$emb_social <- nn_sequential(
      nn_linear(as.integer((num_cells %/% pool_size)^2) * hidden_dim, social_embed_dim),
      nn_relu()
    )

    # Célula LSTM
    self$cell <- nn_lstm(pos_embed_dim + social_embed_dim, hidden_dim, batch_first = TRUE)

    # Cabeça de predição bivariada
    self$out <- nn_linear(hidden_dim, 5)

    # Regularização por dropout
    self$dropout_emb <- nn_dropout(dropout_rate)

    # Buffers para converter z-score -> metros (grid físico de pooling)
    self$mu_pos <- nn_buffer(torch_tensor(scaler$center[c("x", "y")])$view(c(1, 2)))
    self$sd_pos <- nn_buffer(torch_tensor(scaler$scale[c("x", "y")])$view(c(1, 2)))
  },

  social_context = function(pos_m, h_prev) {
    num_nodes <- pos_m$shape[1]
    nc <- self$num_cells
    cell_size <- self$window_m / nc
    D <- self$hidden_dim

    rel <- pos_m$unsqueeze(1) - pos_m$unsqueeze(2) # [n, n, 2]
    cell_xy <- torch_floor((rel + self$window_m / 2) / cell_size)$clamp(0, nc - 1L)
    inside <- (rel$abs() < self$window_m / 2)$all(dim = 3)
    mask <- inside * (torch_eye(num_nodes, device = rel$device) == 0)

    idx <- (cell_xy[, , 2] * nc + cell_xy[, , 1])$to(dtype = torch_long())
    flat_sel <- torch_nonzero(mask)
    cell_ij <- idx$masked_select(mask)

    H <- torch_zeros(c(num_nodes * nc * nc, D), device = pos_m$device)
    if (flat_sel$numel() > 0) {
      src <- h_prev$index_select(1, flat_sel[, 2])
      flat <- (flat_sel[, 1] - 1L) * (nc * nc) + cell_ij + 1L
      H$index_add_(1, flat, src)
    }
    H <- H$view(c(num_nodes, nc, nc, D))

    # Sum-pooling 8x8
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

    h_prev[1, , ] # [num_nodes, hidden_dim]
  },

  forward = function(batch, teacher_forcing = FALSE, tf_rate = 1.0, return_params = FALSE) {
    pos <- batch$x_cont[, , 1:2]
    target <- batch$y
    num_nodes <- pos$shape[1]
    device <- pos$device

    h_prev <- torch_zeros(c(1, num_nodes, self$hidden_dim), device = device)
    c_prev <- torch_zeros(c(1, num_nodes, self$hidden_dim), device = device)

    # Encoder: t = 1..25
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

    # Decoder: t = 26..30 (k = 1..5)
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
      mu_xy <- out5[, 1:2]
      sig_xy <- torch_exp(out5[, 3:4])
      rho <- torch_tanh(out5[, 5])

      pred_pos <- input_pos + mu_xy
      preds[[k]] <- pred_pos$unsqueeze(2)
      mu_xy_l[[k]] <- mu_xy$unsqueeze(2)
      sig_xy_l[[k]] <- sig_xy$unsqueeze(2)
      rho_l[[k]] <- rho$unsqueeze(2)
      input_pos <- pred_pos
    }

    preds_tensor <- torch_cat(preds, dim = 2)

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


# ==============================================================================
# 3. TREINO & OTIMIZAÇÃO ESTATÍSTICA
# ==============================================================================

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

  list(ade_loss = ade_loss, fde_loss = fde_loss, fde = fde_loss$item())
}

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

train_social_lstm <- function(train_dataset, val_dataset, scaler, cfg, seed = 42, verbose = FALSE) {
  set.seed(seed)
  torch_manual_seed(seed)

  device <- select_device()
  train_loader <- dataloader(train_dataset, batch_size = 1, shuffle = TRUE, collate_fn = custom_collate)
  val_loader <- dataloader(val_dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  model <- SocialLSTM$new(scaler = scaler)$to(device = device)
  optimizer <- optim_rmsprop(model$parameters, lr = cfg$lr)
  early_stopping <- EarlyStopping$new(patience = cfg$patience)

  stop_metric <- if (!is.null(cfg$stop_metric) && cfg$stop_metric == "ade") "ade" else "nll"

  history <- list()

  for (epoch in 1:cfg$max_epochs) {
    model$train()
    train_nll <- 0
    total_nodes_train <- 0

    coro::loop(for (batch in train_loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$y <- batch$y$to(device = device)

      optimizer$zero_grad()
      out <- model(batch, teacher_forcing = TRUE, tf_rate = cfg$tf_rate, return_params = TRUE)
      loss <- compute_social_nll(out, batch)
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
      optimizer$step()

      num_nodes <- batch$num_nodes
      train_nll <- train_nll + loss$item() * num_nodes
      total_nodes_train <- total_nodes_train + num_nodes
    })

    train_nll <- train_nll / total_nodes_train

    # Validação
    model$eval()
    val_nll <- 0
    val_ade <- 0
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
        total_nodes_val <- total_nodes_val + num_nodes
      })
    })

    val_nll <- val_nll / total_nodes_val
    val_ade <- val_ade / total_nodes_val

    if (verbose && (epoch %% 10 == 0 || epoch == 1)) {
      cat(sprintf("Época %03d | Treino NLL: %.3f | Val NLL: %.3f, ADE: %.2f m\n",
                  epoch, train_nll, val_nll, val_ade))
    }

    early_stopping$step(if (stop_metric == "ade") val_ade else val_nll, model)
    if (early_stopping$early_stop) {
      if (verbose) cat(sprintf("Early stopping atingido na época %d.\n", epoch))
      break
    }
  }

  # Restaura melhores pesos
  with_no_grad({
    for (i in seq_along(model$parameters)) {
      model$parameters[[i]]$copy_(early_stopping$best_model_wts[[i]])
    }
  })

  list(model = model, best_val_loss = early_stopping$best_loss)
}


# ==============================================================================
# 4. EXTRAÇÃO DE COVARIÁVEIS DO JOGADOR EM T_obs = 25 (SUGESTÃO 4)
# ==============================================================================
# Extrai as características cinemáticas e táticas x_i de cada jogador no último
# instante observado (t = 25s), convertidas de volta para unidades físicas reais
# (metros e m/s), prontas para alimentar o modelo de escala adaptativa ŝ_t(x_i).

extract_player_covariates <- function(batch, scaler) {
  # batch$x_cont: [num_nodes, 25, 8]
  # cont_cols: c('x', 'y', 'ball_speed', 'x_ball', 'y_ball', 'dist_to_ball', 'vel_x', 'vel_y')
  num_nodes <- batch$num_nodes
  x_cont_cpu <- as.array(batch$x_cont$cpu()) # [num_nodes, 25, 8]
  x_cat_cpu <- as.array(batch$x_cat$cpu())   # [num_nodes, 2] -> (team_id_idx, role_idx)

  # Frame 25 (último frame observado)
  f25 <- x_cont_cpu[, 25, ]

  # Desnormalização para coordenadas físicas reais
  cx <- scaler$center["x"];        sx <- scaler$scale["x"]
  cy <- scaler$center["y"];        sy <- scaler$scale["y"]
  cbs <- scaler$center["ball_speed"]; sbs <- scaler$scale["ball_speed"]
  cdist <- scaler$center["dist_to_ball"]; sdist <- scaler$scale["dist_to_ball"]
  cvx <- scaler$center["vel_x"];   svx <- scaler$scale["vel_x"]
  cvy <- scaler$center["vel_y"];   svy <- scaler$scale["vel_y"]

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
# 5. MODELO DE ESCALA ADAPTATIVA ŝ_t(x_i) (SUGESTÃO 4)
# ==============================================================================
# Substitui o ŝ_t estático (Sugestão 2) por uma função adaptativa às condições
# de jogo do indivíduo (Rafael Izbicki 2026, Seção 4):
#
#   log(e_{i,t} + eps) = g(x_i, t) + erro
#   ŝ_t(x_i) = exp( ĝ(x_i, t) )
#
# O modelo ĝ é ajustado estritamente com resíduos OUT-OF-FOLD (nunca in-sample).
# Inclui o horizonte temporal t como preditor em um único modelo para
# maximizar a eficiência amostral em dados limitados.

extract_oof_residual_dataset <- function(model, dataset, scaler) {
  model$eval()
  device <- model$parameters[[1]]$device
  ts <- mu_sd_tensors(scaler, device)
  loader <- dataloader(dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  oof_rows <- list()

  with_no_grad({
    coro::loop(for (batch in loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat <- batch$x_cat$to(device = device)
      batch$y <- batch$y$to(device = device)

      preds <- model(batch)

      # Erro real em metros
      preds_real <- preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu
      diff_sq <- (preds_real - y_true_real)$pow(2)
      distances <- torch_sqrt(diff_sq$sum(dim = 3) + 1e-8) # [num_nodes, 5]
      err_mat <- as.matrix(distances$cpu())

      covs <- extract_player_covariates(batch, scaler)

      # Expande para cada horizonte t = 1..5
      for (t in 1:5) {
        step_df <- covs |>
          mutate(
            time_step = t,
            error_m = err_mat[, t]
          )
        oof_rows[[length(oof_rows) + 1]] <- step_df
      }
    })
  })

  bind_rows(oof_rows)
}

# Ajusta o modelo de regressão log-linear para prever o resíduo condicional
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
      # Padronização segura (sem divisão por zero ou NaNs)
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

# Prediz ŝ_t(x_i) = exp( ĝ(x_i, t) ) para novas observações
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


# ==============================================================================
# 6. CONTROLE DE RISCO CONFORME — CRC (SUGESTÃO 6)
# ==============================================================================
# Implementação exata de Angelopoulos, Bates, Fisch, Lei & Schuster (ICLR 2024),
# conforme formulado por Rafael Izbicki (2026, Seção 6).
#
# A unidade permutável é a JOGADA ofensiva k (com N_k jogadores).
# Para um raio candidato lambda:
#   L_k(lambda) = (1 / N_k) * sum_{i=1}^{N_k} 1{ max_{t in H} (e_{k,i,t} / ŝ_t(x_{k,i})) > lambda }
#
# Com n jogadas de calibração:
#   lambda_hat = inf { lambda : [n / (n + 1)] * (1/n) * sum_{k=1}^n L_k(lambda) + [1 / (n + 1)] <= alpha }
#
# Garantia formal de amostra finita:
#   E[ L_nova(lambda_hat) ] <= alpha

# Computa os escores normalizados M_{k,i} e os erros de cada jogada
compute_play_conformal_evals <- function(model, dataset, scale_model, scaler) {
  model$eval()
  device <- model$parameters[[1]]$device
  ts <- mu_sd_tensors(scaler, device)
  loader <- dataloader(dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)

  play_evals <- list()

  with_no_grad({
    coro::loop(for (batch in loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat <- batch$x_cat$to(device = device)
      batch$y <- batch$y$to(device = device)

      preds <- model(batch)
      preds_real <- preds * ts$sd + ts$mu
      y_true_real <- batch$y * ts$sd + ts$mu
      diff_sq <- (preds_real - y_true_real)$pow(2)
      distances <- torch_sqrt(diff_sq$sum(dim = 3) + 1e-8)
      err_mat <- as.matrix(distances$cpu()) # [num_nodes, 5]

      covs <- extract_player_covariates(batch, scaler)
      num_nodes <- batch$num_nodes

      # Matriz de escalas adaptativas ŝ_t(x_i) [num_nodes, 5]
      s_hat_mat <- matrix(NA_real_, nrow = num_nodes, ncol = 5)
      for (t in 1:5) {
        s_hat_mat[, t] <- predict_adaptive_scale(scale_model, covs, time_step = t)
      }

      # M_{k,i} = max_{t in H} [ e_{k,i,t} / ŝ_t(x_{k,i}) ]
      norm_err_mat <- err_mat / s_hat_mat
      M_ki <- apply(norm_err_mat, 1, max)

      play_evals[[length(play_evals) + 1]] <- list(
        event_id = batch$event_id,
        num_nodes = num_nodes,
        err_mat = err_mat,
        s_hat_mat = s_hat_mat,
        M_ki = M_ki,
        covs = covs
      )
    })
  })

  play_evals
}

# Calibra lambda_hat via Conformal Risk Control (Angelopoulos et al. 2024; Izbicki 2026 §6)
calibrate_conformal_risk_control <- function(play_evals, alpha = 0.10) {
  n <- length(play_evals)

  # Coleta todos os valores observados de M_{k,i}
  all_M <- unlist(lapply(play_evals, function(p) p$M_ki))
  candidate_lambdas <- sort(unique(c(0, all_M, all_M + 1e-5)))

  # Avalia a perda média L_bar(lambda) = (1/n) * sum_k L_k(lambda)
  # L_k(lambda) = mean(M_{k,i} > lambda)
  loss_at_lambda <- function(lam) {
    mean(sapply(play_evals, function(p) {
      mean(p$M_ki > lam)
    }))
  }

  # Critério CRC exato: [n / (n + 1)] * L_bar(lambda) + [1 / (n + 1)] <= alpha
  # Equivalente a: L_bar(lambda) <= (alpha * (n + 1) - 1) / n
  target_loss_bound <- (alpha * (n + 1) - 1) / n

  finite_sample_valid <- target_loss_bound > 0

  if (!finite_sample_valid) {
    # Em amostras extremamente pequenas onde n < 1/alpha - 1 (ex.: n < 9 para alpha = 0.10),
    # o termo 1/(n+1) isolado já excede alpha. Usamos o estimador plug-in L_bar(lambda) <= alpha
    # e emitimos um alerta formal de metodologia estatística.
    warning(sprintf(
      "Aviso Teórico CRC: n = %d é insuficiente para a garantia exata em amostra finita 1/(n+1) <= alpha (exige n >= %d). Utilizando calibração assintótica L_bar(lambda) <= alpha.",
      n, ceiling(1 / alpha) - 1
    ))
    target_threshold <- alpha
    criterion_func <- function(lam) loss_at_lambda(lam) <= target_threshold
  } else {
    criterion_func <- function(lam) {
      (n / (n + 1)) * loss_at_lambda(lam) + (1 / (n + 1)) <= alpha
    }
  }

  # Busca do infimum sobre a grade ordenada
  lambda_hat <- Inf
  for (lam in candidate_lambdas) {
    if (criterion_func(lam)) {
      lambda_hat <- lam
      break
    }
  }

  # Salvaguarda: se nenhum atingiu exatamente, toma o maior candidato
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


# ==============================================================================
# 7. CROSS-CONFORMAL (SUGESTÃO 5) & SPLIT-CONFORMAL ADAPTATIVO
# ==============================================================================
# Sugestão 5 (Vovk 2015; Barber et al. 2021; Izbicki 2026 §5):
# Particiona as jogadas em K dobras balanceadas.
# Para cada dobra k:
#  - Treina a rede e ajusta o modelo de escala nas demais K - 1 dobras
#    (com split interno out-of-fold para treinar ŝ_t(x_i)).
#  - Calcula os escores e perdas das jogadas da dobra k (estritamente out-of-fold).
#  - Agrega os n escores de todas as K dobras e obtém lambda_hat via CRC.

run_cross_conformal <- function(graphs, scaler, cfg, K = 5, alpha = 0.10, seed = 42, verbose = TRUE) {
  set.seed(seed)
  folds <- split_events_kfold(graphs, K = K, seed = seed)
  actual_K <- length(folds)

  if (verbose) {
    cat(sprintf("\n=== INICIANDO CROSS-CONFORMAL (%d Dobras por Jogada) ===\n", actual_K))
  }

  all_oof_play_evals <- list()
  fold_models <- list()
  fold_scale_models <- list()

  for (k in seq_along(folds)) {
    fold_name <- names(folds)[k]
    test_idx <- folds[[k]]
    train_pool_idx <- unlist(folds[-k])

    if (verbose) {
      cat(sprintf("\n--- Dobra %d/%d: %d jogadas de treino | %d jogadas de calibração out-of-fold ---\n",
                  k, actual_K, length(train_pool_idx), length(test_idx)))
    }

    # Split interno nas K - 1 dobras para treino do Social-LSTM e geração de resíduos
    # para o modelo de escala adaptativa ŝ_t(x_i) (Izbicki 2026 §5).
    n_pool <- length(train_pool_idx)
    n_internal_val <- max(1L, floor(0.20 * n_pool))
    shuffled_pool <- sample(train_pool_idx)
    internal_val_idx <- shuffled_pool[1:n_internal_val]
    internal_train_idx <- shuffled_pool[(n_internal_val + 1):n_pool]

    # Treina Social-LSTM com aumento espelhado no treino interno
    mirrored_train <- lapply(graphs[internal_train_idx], mirror_graph, scaler = scaler)
    train_graphs <- c(graphs[internal_train_idx], mirrored_train)

    train_ds <- TrajectoryDataset(train_graphs)
    val_ds   <- TrajectoryDataset(graphs[internal_val_idx])
    test_ds  <- TrajectoryDataset(graphs[test_idx])

    # Ajuste do Social-LSTM
    fit_res <- train_social_lstm(
      train_dataset = train_ds,
      val_dataset = val_ds,
      scaler = scaler,
      cfg = cfg,
      seed = seed + k * 100,
      verbose = FALSE
    )
    fold_model <- fit_res$model

    # Extrai resíduos out-of-fold no conjunto de validação interna
    internal_oof_df <- extract_oof_residual_dataset(fold_model, val_ds, scaler)

    # Ajusta o modelo de escala adaptativa ŝ_t(x_i)
    fold_scale_model <- fit_adaptive_scale_model(internal_oof_df)

    # Avalia na dobra k (estritamente não vista nem pelo LSTM nem pela escala)
    fold_play_evals <- compute_play_conformal_evals(
      model = fold_model,
      dataset = test_ds,
      scale_model = fold_scale_model,
      scaler = scaler
    )

    all_oof_play_evals <- c(all_oof_play_evals, fold_play_evals)
    fold_models[[k]] <- fold_model
    fold_scale_models[[k]] <- fold_scale_model
  }

  # Passo 3 (Izbicki §5): junta os scores das K dobras em uma amostra única e calibra via CRC
  crc_calib <- calibrate_conformal_risk_control(all_oof_play_evals, alpha = alpha)

  if (verbose) {
    cat(sprintf("\n=== CALIBRAÇÃO CROSS-CONFORMAL CRC CONCLUÍDA (alpha = %.2f) ===\n", alpha))
    cat(sprintf("Total de jogadas avaliadas out-of-fold: %d (%d trajetórias)\n",
                crc_calib$n_plays, crc_calib$total_trajectories))
    cat(sprintf("lambda_hat calibrado: %.4f\n", crc_calib$lambda_hat))
    cat(sprintf("Risco empírico E[L_nova]: %.3f (Alvo: <= %.2f)\n",
                crc_calib$empirical_risk, alpha))
  }

  list(
    lambda_hat = crc_calib$lambda_hat,
    crc_calib = crc_calib,
    all_oof_play_evals = all_oof_play_evals,
    fold_models = fold_models,
    fold_scale_models = fold_scale_models,
    scaler = scaler,
    K = actual_K
  )
}

# Versão Split-Conformal Clássica com Escala Adaptativa e CRC
# Permite execução rápida em pipelines com partição fixa train / val / calib / test
calibrate_split_conformal_adaptive_crc <- function(model, calib_dataset, oof_scale_dataset,
                                                  scaler, alpha = 0.10, verbose = TRUE) {
  # 1. Ajusta o modelo de escala adaptativa ŝ_t(x_i) nos resíduos out-of-fold da validação
  oof_df <- extract_oof_residual_dataset(model, oof_scale_dataset, scaler)
  scale_model <- fit_adaptive_scale_model(oof_df)

  # 2. Avalia as jogadas de calibração
  calib_evals <- compute_play_conformal_evals(model, calib_dataset, scale_model, scaler)

  # 3. Calibra lambda_hat via CRC
  crc_calib <- calibrate_conformal_risk_control(calib_evals, alpha = alpha)

  if (verbose) {
    cat(sprintf("\n=== CALIBRAÇÃO SPLIT-CONFORMAL ADAPTATIVA CRC (alpha = %.2f) ===\n", alpha))
    cat(sprintf("Jogadas de calibração: %d (%d trajetórias)\n",
                crc_calib$n_plays, crc_calib$total_trajectories))
    cat(sprintf("lambda_hat calibrado: %.4f\n", crc_calib$lambda_hat))
    cat(sprintf("Risco empírico na calibração: %.3f\n\n", crc_calib$empirical_risk))
  }

  list(
    lambda_hat = crc_calib$lambda_hat,
    scale_model = scale_model,
    crc_calib = crc_calib,
    calib_evals = calib_evals
  )
}


# ==============================================================================
# 8. INFERÊNCIA E AVALIAÇÃO DAS REGIÕES ADAPTATIVAS
# ==============================================================================
# Produz as regiões conformes com raios individuais r_{i,t} = lambda_hat * ŝ_t(x_i).

predict_with_adaptive_regions <- function(model, dataset, scale_model, lambda_hat, scaler) {
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

      covs <- extract_player_covariates(batch, scaler)
      num_nodes <- dim(preds_arr)[1]
      seq_len <- dim(preds_arr)[2]

      # Calcula as escalas adaptativas ŝ_t(x_i)
      s_hat_mat <- matrix(NA_real_, nrow = num_nodes, ncol = seq_len)
      for (t in 1:seq_len) {
        s_hat_mat[, t] <- predict_adaptive_scale(scale_model, covs, time_step = t)
      }

      for (node in 1:num_nodes) {
        pid <- player_ids[node]
        tcode <- team_codes[node]
        tname <- team_names[node]
        p_speed <- covs$speed[node]
        p_dist <- covs$dist_to_ball[node]

        for (t in 1:seq_len) {
          px <- preds_arr[node, t, 1]
          py <- preds_arr[node, t, 2]
          tx <- y_true_arr[node, t, 1]
          ty <- y_true_arr[node, t, 2]

          s_hat_it <- s_hat_mat[node, t]
          rad_it <- lambda_hat * s_hat_it
          dist <- sqrt((px - tx)^2 + (py - ty)^2)

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
            covered = dist <= rad_it,
            stringsAsFactors = FALSE
          )
        }
      }
    })
  })

  bind_rows(results_df)
}

# Avaliação rigorosa das propriedades conformes
evaluate_conformal_regions <- function(pred_df, alpha = 0.10) {
  # 1. Cobertura da trajetória completa por jogador
  traj_summary <- pred_df |>
    group_by(event_id, player_id) |>
    summarize(
      trajectory_covered = all(covered),
      mean_radius = mean(conf_radius),
      .groups = "drop"
    )

  # 2. Perda por jogada L_k(lambda_hat) (fração de jogadores que escapam)
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
      mean_radius = mean(conf_radius),
      sd_radius = sd(conf_radius),
      min_radius = min(conf_radius),
      max_radius = max(conf_radius),
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


# ==============================================================================
# 9. VISUALIZAÇÃO DE ALTA RESOLUÇÃO NO CAMPO (105 x 68 m)
# ==============================================================================
# Plota as trajetórias reais, previstas e os tubos de incerteza adaptativos.
# Demonstra visualmente como jogadores em maior velocidade / disputa recebem
# regiões mais largas, e jogadores em movimento estável recebem regiões menores.

plot_conformal_event_adaptive <- function(target_event_id, events_prepared, test_predictions,
                                          alpha = 0.10) {
  pred_data <- test_predictions |>
    filter(event_id == target_event_id)

  if (nrow(pred_data) == 0) {
    stop(sprintf("Lance event_id %s não encontrado nas previsões.", target_event_id))
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
    # A. Tubos de Predição com Raios Adaptativos Individuais
    geom_circle(
      data = pred_data,
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, fill = as.factor(time_step), group = node_id),
      color = NA,
      alpha = 0.18
    ) +
    scale_fill_viridis_d(name = "Horizonte (s)") +

    # B. Histórico Observado (t <= 25s)
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

    # C. Trajetória Prevista (Social-LSTM)
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

    coord_fixed(xlim = c(0, 105), ylim = c(0, 68), expand = FALSE) +
    labs(
      title = sprintf("Regiões Conformes Adaptativas (CRC, 1 - \u03b1 = %.0f%%) | Lance %s",
                      (1 - alpha) * 100, target_event_id),
      subtitle = "Azul: Previsto | Amarelo: Real | Branco: Observado | Círculos: Escala Adaptativa ŝ_t(x_i)"
    )

  return(p)
}


# ==============================================================================
# 10. RUNNER COMPLETO DO EXPERIMENTO
# ==============================================================================
# Executa a pipeline completa nas modalidades:
#  - mode = "cross": Cross-Conformal em K dobras (Sugestão 5) com CRC (Sugestão 6)
#  - mode = "split": Split-Conformal clássico com Escala Adaptativa e CRC

run_experiment_v2 <- function(df_raw, cfg, mode = c("split", "cross"), K = 5,
                             alpha = 0.10, seed = 42, verbose = TRUE) {
  mode <- match.arg(mode)
  prep <- preprocess_trajectories(df_raw, seed = seed)
  graphs <- prep$graphs
  scaler <- prep$scaler

  if (mode == "cross") {
    # Executa K-Fold Cross-Conformal
    res_cross <- run_cross_conformal(
      graphs = graphs,
      scaler = scaler,
      cfg = cfg,
      K = K,
      alpha = alpha,
      seed = seed,
      verbose = verbose
    )
    return(c(res_cross, list(prep = prep)))
  } else {
    # Particionamento clássico de jogadas: 60% treino, 15% validação, 15% calibração, 10% teste
    set.seed(seed)
    unique_events <- unique(sapply(graphs, function(g) g$event_id))
    n_events <- length(unique_events)
    shuffled_events <- sample(unique_events)

    n_train <- floor(0.60 * n_events)
    n_val   <- floor(0.15 * n_events)
    n_calib <- floor(0.15 * n_events)

    # Salvaguarda para regimes com poucas jogadas: garante pelo menos 1 jogada por split se n_events >= 4
    if (n_events >= 4) {
      if (n_val < 1L) n_val <- 1L
      if (n_calib < 1L) n_calib <- 1L
      n_test <- n_events - n_train - n_val - n_calib
      if (n_test < 1L) {
        n_train <- max(1L, n_events - 3L)
        n_val <- 1L
        n_calib <- 1L
      }
    }

    train_events <- shuffled_events[1:n_train]
    val_events   <- shuffled_events[(n_train + 1):(n_train + n_val)]
    calib_events <- shuffled_events[(n_train + n_val + 1):(n_train + n_val + n_calib)]
    test_events  <- shuffled_events[(n_train + n_val + n_calib + 1):n_events]

    graph_event_ids <- sapply(graphs, function(g) g$event_id)
    train_idx <- which(graph_event_ids %in% train_events)
    val_idx   <- which(graph_event_ids %in% val_events)
    calib_idx <- which(graph_event_ids %in% calib_events)
    test_idx  <- which(graph_event_ids %in% test_events)

    # Aumento por espelhamento no treino
    mirrored_train <- lapply(graphs[train_idx], mirror_graph, scaler = scaler)
    train_graphs <- c(graphs[train_idx], mirrored_train)

    train_ds <- TrajectoryDataset(train_graphs)
    val_ds   <- TrajectoryDataset(graphs[val_idx])
    calib_ds <- TrajectoryDataset(graphs[calib_idx])
    test_ds  <- TrajectoryDataset(graphs[test_idx])

    if (verbose) {
      cat(sprintf("\n=== TREINANDO MODELO PRINCIPAL (Social-LSTM) ===\n"))
      cat(sprintf("Treino: %d jogadas | Val: %d jogadas | Calibração: %d jogadas | Teste: %d jogadas\n",
                  length(train_events), length(val_events), length(calib_events), length(test_events)))
    }

    fit <- train_social_lstm(
      train_dataset = train_ds,
      val_dataset = val_ds,
      scaler = scaler,
      cfg = cfg,
      seed = seed,
      verbose = verbose
    )

    # Calibração com Escala Adaptativa (Sugestão 4) e CRC (Sugestão 6)
    calib_res <- calibrate_split_conformal_adaptive_crc(
      model = fit$model,
      calib_dataset = calib_ds,
      oof_scale_dataset = val_ds,
      scaler = scaler,
      alpha = alpha,
      verbose = verbose
    )

    # Inferência no conjunto de teste
    test_preds <- predict_with_adaptive_regions(
      model = fit$model,
      dataset = test_ds,
      scale_model = calib_res$scale_model,
      lambda_hat = calib_res$lambda_hat,
      scaler = scaler
    )

    eval_res <- evaluate_conformal_regions(test_preds, alpha = alpha)

    return(list(
      model = fit$model,
      scale_model = calib_res$scale_model,
      lambda_hat = calib_res$lambda_hat,
      calib_res = calib_res,
      test_predictions = test_preds,
      evaluation = eval_res,
      datasets = list(train = train_ds, val = val_ds, calib = calib_ds, test = test_ds),
      scaler = scaler,
      prep = prep
    ))
  }
}
