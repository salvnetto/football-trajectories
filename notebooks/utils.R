# ==============================================================================
# UTILS.R — FUNÇÕES AUXILIARES COMPARTILHADAS (ESTATÍSTICA & DEEP LEARNING)
# ==============================================================================
# Módulo contendo utilitários compartilhados entre arquiteturas de rede neural
# e métodos de predição conforme para trajetórias de futebol:
#  - Reprodutibilidade e alocação de dispositivos
#  - Estruturação de tensores e datasets Torch
#  - Pré-processamento, aumento de dados e escalonamento não-vazado
#  - Métricas cinemáticas de erro (ADE, FDE) e Early Stopping
#  - Funções de covariância e distâncias espaciais (Mahalanobis)
# ==============================================================================

suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
  library(R6)
})

# ==============================================================================
# 1. REPRODUCIBILIDADE & DISPOSITIVO COMPUTACIONAL
# ==============================================================================

#' Fixar sementes pseudoaleatórias para reprodutibilidade estrita
#' @param seed Inteiro com a semente aleatória
set_seed <- function(seed = 42) {
  set.seed(seed)
  torch_manual_seed(seed)
}

#' Selecionar dispositivo Torch disponível (CUDA, MPS ou CPU)
#' @return Objeto torch_device
select_device <- function() {
  if (cuda_is_available()) {
    torch_device("cuda")
  } else if (backends_mps_is_available()) {
    torch_device("mps")
  } else {
    torch_device("cpu")
  }
}

get_device <- select_device


# ==============================================================================
# 2. TORCH DATASET & COLLATE
# ==============================================================================

#' Colação customizada para lotes de tamanho 1 (preserva tipos e metadados R)
#' @param batch Lista de elementos do dataloader
#' @return Primeiro elemento do lote
custom_collate <- function(batch) {
  batch[[1]]
}

#' Dataset R6 para gerenciamento de sequências de grafos espaçotemporais
TrajectoryDataset <- dataset(
  name = "TrajectoryDataset",
  initialize = function(graphs) {
    self$graphs <- graphs
  },
  .getitem = function(i) {
    self$graphs[[i]]
  },
  .length = function() {
    length(self$graphs)
  }
)


# ==============================================================================
# 3. TRANSFORMAÇÕES DE COORDENADAS & CINEMÁTICA
# ==============================================================================

#' Cria tensores de média e desvio padrão para desnormalização de posições (x, y)
#' @param scaler Lista contendo 'center' e 'scale'
#' @param device Dispositivo torch onde os tensores serão alocados
#' @return Lista com tensores 'mu' e 'sd' com shape [1, 1, 2]
mu_sd_tensors <- function(scaler, device) {
  mu <- c(scaler$center["x"], scaler$center["y"])
  sd <- c(scaler$scale["x"], scaler$scale["y"])
  list(
    mu = torch_tensor(mu, device = device)$view(c(1, 1, 2)),
    sd = torch_tensor(sd, device = device)$view(c(1, 1, 2))
  )
}

#' Espelhamento lateral exato y -> 68 - y no espaço z-scoreado
#' y'_s = (68 - 2*mu_y)/sd_y - y_s (mantém média e desvio do scaler intactos)
#' @param g Grafo da jogada com tensores x_cont e y
#' @param scaler Lista com parâmetros de escalonamento
#' @return Grafo com coordenadas y invertidas simetricamente
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


# ==============================================================================
# 4. CONSTRUÇÃO DE GRAFOS & PRÉ-PROCESSAMENTO
# ==============================================================================

#' Constrói a lista de grafos espaçotemporais a partir de dados tabulares
#' @param df Data frame tabular contendo posições de tracking
#' @param cont_cols Vetor com nomes das colunas contínuas
#' @return Lista de grafos estruturados para entrada nos modelos
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

#' Pré-processa trajetórias com estratificação por jogada independente
#' @param df Data frame tabular bruto
#' @param train_prop Proporção de jogadas para treino
#' @param val_prop Proporção de jogadas para validação
#' @param calib_prop Proporção de jogadas para calibração conformal
#' @param seed Semente aleatória para o sorteio das jogadas
#' @param augment_train Lógico: se TRUE, duplica dados de treino com espelhamento lateral
#' @return Lista contendo grafos, índices de splits, encoders e scaler
preprocess_and_build_dataset <- function(df,
                                         train_prop = 0.60,
                                         val_prop = 0.15,
                                         calib_prop = 0.15,
                                         seed = 42,
                                         augment_train = TRUE) {
  set_seed(seed)
  df <- as.data.frame(df)

  # 1. Codificação Categórica (1 = Defense, 2 = Attack) e Cinemática
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
  encoders <- list(num_teams = num_teams, num_roles = 2L)

  # 2. Particionamento por jogada independente (sem vazamento entre frames)
  unique_events <- unique(df$event_id)
  n_events <- length(unique_events)
  shuffled_events <- sample(unique_events)

  n_train <- floor(train_prop * n_events)
  n_val   <- floor(val_prop * n_events)
  n_calib <- floor(calib_prop * n_events)

  # Salvaguarda para bases pequenas: garante ao menos 1 jogada por partição quando n >= 4
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

  # 3. Escalonamento z-score ajustado APENAS no treino observado (t <= 25)
  cont_cols <- c('x', 'y', 'ball_speed', 'x_ball', 'y_ball', 'dist_to_ball', 'vel_x', 'vel_y')
  train_obs <- df |> filter(event_id %in% train_events, time_sec <= 25)

  scaler_center <- colMeans(train_obs[, cont_cols], na.rm = TRUE)
  scaler_scale  <- apply(train_obs[, cont_cols], 2, sd, na.rm = TRUE)
  scaler_scale[scaler_scale == 0] <- 1

  df[, cont_cols] <- scale(df[, cont_cols], center = scaler_center, scale = scaler_scale)

  # 4. Construção dos grafos por jogada
  graphs <- build_graph_list(df, cont_cols = cont_cols)

  # Mapear eventos para índices de grafos
  graph_event_ids <- sapply(graphs, function(g) g$event_id)
  train_indices <- which(graph_event_ids %in% train_events)
  val_indices   <- which(graph_event_ids %in% val_events)
  calib_indices <- which(graph_event_ids %in% calib_events)
  test_indices  <- which(graph_event_ids %in% test_events)

  # 5. Aumento por espelhamento lateral (apenas no treino)
  scaler_ls <- list(center = scaler_center, scale = scaler_scale)
  if (augment_train && length(train_indices) > 0) {
    mirrored <- lapply(graphs[train_indices], mirror_graph, scaler = scaler_ls)
    n_orig <- length(graphs)
    graphs <- c(graphs, mirrored)
    train_indices <- c(train_indices, (n_orig + 1):(n_orig + length(mirrored)))
  }

  list(
    graphs = graphs,
    splits = list(
      train = train_indices,
      val = val_indices,
      calib = calib_indices,
      test = test_indices
    ),
    num_teams = num_teams,
    encoders = encoders,
    scaler = scaler_ls,
    df_raw = df
  )
}

#' Particionador K-Fold estratificado estritamente por jogada (event_id)
#' @param graphs Lista de grafos
#' @param K Número de dobras
#' @param seed Semente aleatória
#' @return Lista de vetores de índices por dobra
split_events_kfold <- function(graphs, K = 5, seed = 42) {
  set_seed(seed)
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


# ==============================================================================
# 5. MÉTRICAS E AVALIAÇÃO DE AJUSTE
# ==============================================================================

#' Calcula erro de deslocamento médio (ADE) e erro de deslocamento final (FDE)
#' @param preds Tensor predito no espaço padronizado [num_nodes, seq_len, 2]
#' @param targets Tensor alvo real padronizado [num_nodes, seq_len, 2]
#' @param scaler Parâmetros do scaler para conversão física em metros
#' @return Lista com ade_loss (tensor), fde_loss (tensor) e fde (numérico)
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

  list(ade_loss = ade_loss, fde_loss = fde_loss, fde = fde)
}

#' Avalia ADE e FDE agregados em um dataset completo (ponderado por nós)
#' @param model Modelo Torch treinado
#' @param dataset TrajectoryDataset
#' @param scaler Parâmetros do scaler
#' @return Lista com ADE e FDE em metros
evaluate_dataset <- function(model, dataset, scaler) {
  model$eval()
  device <- model$parameters[[1]]$device

  loader <- dataloader(dataset, batch_size = 1, shuffle = FALSE, collate_fn = custom_collate)
  ade_sum <- 0
  fde_sum <- 0
  n_nodes_total <- 0

  with_no_grad({
    coro::loop(for (batch in loader) {
      batch$x_cont <- batch$x_cont$to(device = device)
      batch$x_cat <- batch$x_cat$to(device = device)
      batch$y <- batch$y$to(device = device)

      metrics <- compute_ade_loss_and_fde(model(batch), batch$y, scaler)

      n_nodes <- batch$num_nodes
      ade_sum <- ade_sum + metrics$ade_loss$item() * n_nodes
      fde_sum <- fde_sum + metrics$fde * n_nodes
      n_nodes_total <- n_nodes_total + n_nodes
    })
  })

  list(ADE = ade_sum / n_nodes_total, FDE = fde_sum / n_nodes_total)
}

#' Classe R6 para Early Stopping com restauração dos melhores pesos
EarlyStopping <- R6Class(
  "EarlyStopping",
  public = list(
    patience = NULL,
    delta = NULL,
    counter = 0,
    best_loss = NULL,
    early_stop = FALSE,
    best_model_wts = NULL,

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


# ==============================================================================
# 6. ÁLGEBRA LINEAR ESPACIAL & DISTÂNCIAS DE MAHALANOBIS
# ==============================================================================

#' Estima as matrizes de covariância 2x2 do erro espacial por horizonte temporal
#' @param err_array Array 3D com dimensões [n_trajetorias, 5, 2] em metros
#' @param ridge Regularização de Tikhonov na diagonal para garantir invertibilidade
#' @return Lista de 5 matrizes 2x2
estimate_error_covariances <- function(err_array, ridge = 1e-6) {
  lapply(1:5, function(t) {
    e <- matrix(err_array[, t, ], ncol = 2)
    cov(e) + diag(ridge, 2)
  })
}

#' Computa o escore supremo de Mahalanobis ao longo dos horizontes temporais
#' @param err_array Array 3D [n_trajetorias, 5, 2] com erros físicos em metros
#' @param Sigma_t Lista de 5 matrizes de covariância 2x2
#' @return Vetor numérico com R_i = max_t sqrt(e' Sigma_t^{-1} e)
mahalanobis_scores <- function(err_array, Sigma_t) {
  n_traj <- dim(err_array)[1]
  R <- matrix(NA_real_, nrow = n_traj, ncol = 5)
  for (t in 1:5) {
    e <- matrix(err_array[, t, ], ncol = 2)
    Sinv <- solve(Sigma_t[[t]])
    R[, t] <- sqrt(rowSums((e %*% Sinv) * e))
  }
  apply(R, 1, max)
}
