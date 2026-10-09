# ==============================================================================
# PIPELINE.R — ORQUESTRADOR PRINCIPAL DE MODELAGEM E PREDIÇÃO CONFORME
# ==============================================================================
# Script central de experimentação que orquestra o pipeline ponta a ponta:
#  1. Carregamento de dependências e módulos desacoplados
#  2. Configuração via seletores explícitos de Arquitetura e Método Conformal
#  3. Preparação e escalonamento não-vazado dos dados de tracking
#  4. Treinamento da arquitetura selecionada (GNN ou Social-LSTM)
#  5. Avaliação preditiva out-of-sample (ADE e FDE no teste)
#  6. Calibração da região conformal selecionada (Baseline, Naive ou CRC)
#     com suporte à geometria configurada (Circular ou Elíptica)
#  7. Inferência conformal e relatório estatístico de cobertura
# ==============================================================================

# ==============================================================================
# 0. SETUP & CARREGAMENTO DE DEPENDÊNCIAS
# ==============================================================================
suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
  library(R6)
  library(here)
})

# Carregamento dos módulos modulares de notebooks/
source(here("notebooks", "utils.R"))
source(here("notebooks", "architecture_gnn.R"))
source(here("notebooks", "architecture_social.R"))
source(here("notebooks", "conformal_baseline.R"))
source(here("notebooks", "conformal_naive.R"))
source(here("notebooks", "conformal_crc.R"))

# Carregamento do pipeline de janelamento de tracking se disponível
if (file.exists(here("src", "modeling.R"))) {
  source(here("src", "modeling.R"))
}

# ==============================================================================
# 1. PAINEL DE CONFIGURAÇÃO & SELETORES DO EXPERIMENTO
# ==============================================================================

# SELETOR DE ARQUITETURA:
#  - "social" : Social-LSTM (Alahi et al. 2016, CVPR)
#  - "gnn"    : TrajectorySeq2SeqGNN (Graph Attention + LSTM)
ARCH_CHOICE <- "social"

# SELETOR DE MÉTODO CONFORME:
#  - "crc"      : Conformal Risk Control adaptativo (Sugestões 4-6, Izbicki 2026)
#  - "naive"    : Conformal Naive com forma out-of-fold e variantes exatas
#  - "baseline" : Conformal Simultâneo Baseline (Alvo A / Alvo B)
CONFORMAL_CHOICE <- "crc"

# GEOMETRIA DA REGIÃO CONFORME:
#  - "circular" : Bandas circulares simultâneas (raio r_t em metros)
#  - "elliptic" : Elipses de incerteza direcional (Mahalanobis)
REGION_TYPE <- "circular"

# PARÂMETROS ESTATÍSTICOS E DE TREINO:
ALPHA <- 0.10  # Nível de significância (cobertura nominal 1 - alpha = 90%)
SEED  <- 42    # Semente pseudoaleatória para reprodutibilidade

TRAIN_CFG <- list(
  seed = SEED,
  lr = if (ARCH_CHOICE == "gnn") 0.001 else 0.003,
  max_epochs = 100,
  patience = 10,
  tf_rate = 1.0,        # Scheduled sampling / teacher forcing
  stop_metric = "ade"   # Métrica de early stopping ("ade" ou "nll")
)


# ==============================================================================
# 2. FUNÇÃO PRINCIPAL DE EXECUÇÃO DO PIPELINE
# ==============================================================================

run_pipeline <- function(arch_choice = ARCH_CHOICE,
                         conformal_choice = CONFORMAL_CHOICE,
                         region_type = REGION_TYPE,
                         alpha = ALPHA,
                         seed = SEED,
                         train_cfg = TRAIN_CFG,
                         verbose = TRUE) {

  set_seed(seed)
  arch_choice      <- match.arg(arch_choice, c("social", "gnn"))
  conformal_choice <- match.arg(conformal_choice, c("crc", "naive", "baseline"))
  region_type      <- match.arg(region_type, c("circular", "elliptic"))

  if (verbose) {
    cat("==============================================================================\n")
    cat("               PIPELINE DE MODELAGEM & PREDIÇÃO CONFORME                     \n")
    cat("==============================================================================\n")
    cat(sprintf(" Arquitetura Selecionada: %s\n", toupper(arch_choice)))
    cat(sprintf(" Método Conformal:       %s\n", toupper(conformal_choice)))
    cat(sprintf(" Geometria da Região:    %s\n", toupper(region_type)))
    cat(sprintf(" Cobertura Nominal:      %.1f%% (alpha = %.2f)\n", (1 - alpha) * 100, alpha))
    cat(sprintf(" Semente Aleatória:      %d\n", seed))
    cat("==============================================================================\n\n")
  }

  # ----------------------------------------------------------------------------
  # ETAPA 1: CARREGAMENTO DE DADOS E CONSTRUÇÃO DAS JANELAS
  # ----------------------------------------------------------------------------
  events_path   <- here("data", "processed", "events.csv")
  tracking_path <- here("data", "processed", "tracking.csv")
  players_path  <- here("data", "processed", "players_database.csv")

  if (!file.exists(events_path) || !file.exists(tracking_path)) {
    stop("Arquivos de dados processados não encontrados em data/processed/.")
  }

  if (verbose) cat(">> Carregando dados brutos de tracking e eventos...\n")

  events <- read_csv(events_path, show_col_types = FALSE) |>
    distinct() |>
    rename(team_with_poss = team_id) |>
    mutate(event_id = as.integer(as.factor(event_id)))

  players_db <- if (file.exists(players_path)) {
    read_csv(players_path, show_col_types = FALSE) |>
      select(player_id, team_name) |>
      distinct()
  } else {
    tibble(player_id = character(), team_name = character())
  }

  tracking <- read_csv(tracking_path, show_col_types = FALSE) |>
    distinct()

  if (nrow(players_db) > 0) {
    tracking <- tracking |> left_join(players_db, by = "player_id")
  }

  if (verbose) cat(">> Estruturando janelas de 30s centradas nas finalizações...\n")
  events_prepared <- prepare_data(
    events = events,
    tracking = tracking,
    event_type_filter = "SHOT",
    event_subtype = NA,
    start_time = 30,
    end_time = 1,
    pred_time_event = 1
  )

  # ----------------------------------------------------------------------------
  # ETAPA 2: PRÉ-PROCESSAMENTO & ESTRATIFICAÇÃO NÃO-VAZADA
  # ----------------------------------------------------------------------------
  if (verbose) cat(">> Pré-processando coordenadas e construindo datasets de grafos...\n")
  prep_data <- preprocess_and_build_dataset(
    df = events_prepared,
    train_prop = 0.60,
    val_prop = 0.15,
    calib_prop = 0.15,
    seed = seed,
    augment_train = TRUE
  )

  graphs   <- prep_data$graphs
  splits   <- prep_data$splits
  scaler   <- prep_data$scaler
  encoders <- prep_data$encoders

  train_ds <- TrajectoryDataset(graphs[splits$train])
  val_ds   <- TrajectoryDataset(graphs[splits$val])
  calib_ds <- TrajectoryDataset(graphs[splits$calib])
  test_ds  <- TrajectoryDataset(graphs[splits$test])

  if (verbose) {
    cat(sprintf(" Splits: Treino: %d grafos | Val: %d | Calib: %d | Teste: %d\n\n",
                length(splits$train), length(splits$val), length(splits$calib), length(splits$test)))
  }

  # ----------------------------------------------------------------------------
  # ETAPA 3: TREINAMENTO DA ARQUITETURA ESCOLHIDA
  # ----------------------------------------------------------------------------
  if (verbose) cat(sprintf(">> Iniciando treinamento da arquitetura [%s]...\n", toupper(arch_choice)))

  fit_result <- if (arch_choice == "gnn") {
    train_gnn(
      train_dataset = train_ds,
      val_dataset   = val_ds,
      scaler        = scaler,
      encoders      = encoders,
      cfg           = train_cfg,
      seed          = seed,
      verbose       = verbose
    )
  } else {
    train_social_lstm(
      train_dataset = train_ds,
      val_dataset   = val_ds,
      scaler        = scaler,
      cfg           = train_cfg,
      seed          = seed,
      verbose       = verbose
    )
  }

  model <- fit_result$model

  # ----------------------------------------------------------------------------
  # ETAPA 4: AVALIAÇÃO DO AJUSTE NO CONJUNTO DE TESTE
  # ----------------------------------------------------------------------------
  if (verbose) cat("\n>> Avaliando desempenho de deslocamento no conjunto de teste...\n")
  test_metrics <- evaluate_dataset(model, test_ds, scaler)

  if (verbose) {
    cat(sprintf(" Desempenho no Teste: ADE = %.2f m | FDE = %.2f m\n\n",
                test_metrics$ADE, test_metrics$FDE))
  }

  # ----------------------------------------------------------------------------
  # ETAPA 5: CALIBRAÇÃO CONFORME & INFERÊNCIA NO TESTE
  # ----------------------------------------------------------------------------
  if (verbose) {
    cat(sprintf(">> Executando calibração conformal [%s | %s]...\n",
                toupper(conformal_choice), toupper(region_type)))
  }

  calib_output <- list()
  test_predictions <- NULL
  eval_results <- NULL

  if (conformal_choice == "baseline") {
    calib_output <- calibrate_conformal_baseline(
      model         = model,
      calib_dataset = calib_ds,
      scaler        = scaler,
      shape_dataset = val_ds,
      alpha         = alpha,
      region_type   = region_type,
      verbose       = verbose
    )

    test_predictions <- predict_conformal_baseline(
      model       = model,
      dataset     = test_ds,
      scaler      = scaler,
      calib_res   = calib_output,
      region_type = region_type
    )

    eval_results <- evaluate_conformal_summary(test_predictions, alpha = alpha)

  } else if (conformal_choice == "naive") {
    calib_output <- calibrate_conformal_naive(
      model         = model,
      calib_dataset = calib_ds,
      s_hat_dataset = val_ds,
      scaler        = scaler,
      alpha         = alpha,
      region_type   = region_type,
      verbose       = verbose
    )

    test_predictions <- predict_conformal_naive(
      model       = model,
      dataset     = test_ds,
      scaler      = scaler,
      calib_res   = calib_output,
      region_type = region_type
    )

    eval_results <- list(
      summary = evaluate_conformal_summary(test_predictions, alpha = alpha),
      by_play = coverage_by_play_draws(test_predictions, B = 200, seed = seed)
    )

  } else if (conformal_choice == "crc") {
    calib_output <- calibrate_split_conformal_adaptive_crc(
      model             = model,
      calib_dataset     = calib_ds,
      oof_scale_dataset = val_ds,
      scaler            = scaler,
      alpha             = alpha,
      region_type       = region_type,
      verbose           = verbose
    )

    test_predictions <- predict_with_adaptive_regions(
      model       = model,
      dataset     = test_ds,
      scale_model = calib_output$scale_model,
      lambda_hat  = calib_output$lambda_hat,
      scaler      = scaler,
      region_type = region_type,
      Gamma_t     = calib_output$Gamma_t
    )

    eval_results <- evaluate_conformal_regions(test_predictions, alpha = alpha)
  }

  # ----------------------------------------------------------------------------
  # ETAPA 6: RELATÓRIO ESTATÍSTICO CONSOLIDADO
  # ----------------------------------------------------------------------------
  if (verbose) {
    cat("==============================================================================\n")
    cat("                    RELATÓRIO ESTATÍSTICO DE COBERTURA                       \n")
    cat("==============================================================================\n")
    if (conformal_choice == "crc") {
      cat(sprintf(" Cobertura Empírica Global da Trajetória: %.1f%% (Alvo: >= %.1f%%)\n",
                  eval_results$overall_trajectory_coverage * 100, (1 - alpha) * 100))
      cat(sprintf(" Perda Média por Jogada E[L_nova]:        %.3f (Garantia: <= %.2f)\n",
                  eval_results$mean_play_loss, alpha))
      cat("\n Cobertura Pontual por Passo Temporal:\n")
      print(eval_results$pointwise)
    } else if (conformal_choice == "naive") {
      cat(sprintf(" Cobertura Empírica Global:               %.1f%% (Alvo: >= %.1f%%)\n",
                  eval_results$summary$simultaneous_coverage$empirical_coverage * 100, (1 - alpha) * 100))
      cat(sprintf(" Cobertura por Jogada (Sorteios MC):      %.1f%% ± %.2f%%\n",
                  eval_results$by_play$mean_coverage * 100, eval_results$by_play$sd_coverage * 100))
      cat("\n Cobertura Pontual por Passo Temporal:\n")
      print(eval_results$summary$pointwise_coverage)
    } else {
      cat(sprintf(" Cobertura Empírica Global:               %.1f%% (Alvo: >= %.1f%%)\n",
                  eval_results$simultaneous_coverage$empirical_coverage * 100, (1 - alpha) * 100))
      cat("\n Cobertura Pontual por Passo Temporal:\n")
      print(eval_results$pointwise_coverage)
    }
    cat("==============================================================================\n\n")
  }

  list(
    model            = model,
    fit_result       = fit_result,
    test_metrics     = test_metrics,
    calib_output     = calib_output,
    test_predictions = test_predictions,
    eval_results     = eval_results,
    scaler           = scaler,
    config           = list(
      arch_choice      = arch_choice,
      conformal_choice = conformal_choice,
      region_type      = region_type,
      alpha            = alpha,
      seed             = seed
    )
  )
}

# ==============================================================================
# 3. EXECUÇÃO DIRETA (QUANDO EXECUTADO COMO SCRIPT PRINCIPAL)
# ==============================================================================
if (sys.nframe() == 0L) {
  result <- run_pipeline()
}
