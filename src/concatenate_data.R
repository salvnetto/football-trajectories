library(tidyverse)
library(fs)
library(vroom)
library(xml2)

# Source the modeling script to get transform_tracking_to_long
source("src/modeling.R")

# 1. Define Paths
interim_dir <- path("data", "interim")
processed_dir <- path("data", "processed")
raw_dir <- path("data", "raw")

dir_create(processed_dir)

# -------------------------------------------------------------------------
# 1.5 Create Player Database from XMLs
# -------------------------------------------------------------------------
cat("Extracting Player Database from XMLs...\n")
xml_files <- dir_ls(raw_dir, regexp = "matchinformation.*\\.xml$")

parse_xml_to_players <- function(xml_path) {
  doc <- read_xml(xml_path)
  
  # Find all teams
  teams <- xml_find_all(doc, ".//Team")
  
  players_list <- map(teams, function(team) {
    team_id <- xml_attr(team, "TeamId")
    team_name <- xml_attr(team, "TeamName")
    
    players <- xml_find_all(team, ".//Player")
    
    if(length(players) > 0) {
      tibble(
        player_id = xml_attr(players, "PersonId"),
        real_name = xml_attr(players, "Shortname"),
        shirt_number = as.numeric(xml_attr(players, "ShirtNumber")),
        original_position = xml_attr(players, "PlayingPosition"),
        team_id = team_id,
        team_name = team_name
      )
    } else {
      NULL
    }
  })
  
  bind_rows(players_list)
}

player_db <- xml_files |> 
  map(parse_xml_to_players) |> 
  list_rbind() |> 
  distinct(player_id, .keep_all = TRUE) |> 
  mutate(is_goalkeeper = ifelse(original_position == "TW", 1, 0))

write_csv(player_db, path(processed_dir, "players_database.csv"))
cat("Player database saved with", nrow(player_db), "unique players.\n")

# -------------------------------------------------------------------------
# 2. Get file lists
# -------------------------------------------------------------------------
events_files <- dir_ls(interim_dir, regexp = "_events\\.csv$")
tracking_files <- dir_ls(interim_dir, regexp = "_tracking\\.csv$")

# 3. Define processing functions
process_events_data <- function(df, file_path) {
  match_id <- path_file(file_path) |> str_remove("_events\\.csv$")
  # Add the match_id as a column to identify the source of each row
  df <- df |> mutate(match_id = match_id, .before = 1)
  
  
  return(df)
}

process_tracking_data <- function(df, file_path) {
  match_id <- path_file(file_path) |> str_remove("_tracking\\.csv$")
  
  # Utiliza a função transform_tracking_to_long criada no modeling.R
  # para converter de Wide para Long e escalar as coordenadas para o tamanho do campo.
  df <- transform_tracking_to_long(df, match_id)
  
  # Join with player database to get is_goalkeeper and team_id
  df <- df |> 
    left_join(player_db |> select(player_id, team_id, is_goalkeeper_xml = is_goalkeeper), by = "player_id") |>
    mutate(
      is_goalkeeper_xml = replace_na(is_goalkeeper_xml, 0),
      is_goalkeeper = as.integer(is_goalkeeper_xml)
    ) |>
    select(-is_goalkeeper_xml)

  # Downsample from 25fps to 1fps
  df <- df |>
    mutate(time_sec_orig_round = floor(time_sec_orig)) |>
    group_by(match_id, period_id, time_sec_orig_round, player_id) |>
    slice_head(n = 1) |>
    ungroup() |>
    select(-time_sec_orig_round)
  
  return(df)
}

# 4. Read, process, and concatenate
cat("Processing Events Data...\n")
events_combined <- events_files |>
  map(\(path) {
    df <- read_csv(path, show_col_types = FALSE, col_types = cols(card_type = col_character()))
    process_events_data(df, path)
  }) |>
  list_rbind()

cat("Processing Tracking Data...\n")
tracking_combined <- tracking_files |>
  map(\(path) {
    df <- vroom(path, show_col_types = FALSE, progress = FALSE)
    process_tracking_data(df, path)
  }) |>
  list_rbind()

# 5. Output / Save
write_csv(events_combined, path(processed_dir, "events.csv"))
write_csv(tracking_combined, path(processed_dir, "tracking.csv"))

cat("Concatenation complete!\n")
