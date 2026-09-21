# src/modeling.R

library(tidyverse)
library(sf)

prepare_data <- function(events, tracking, event_type_filter = "SHOT", 
                         event_subtype = NA, start_time = 29, 
                         end_time = 0, pred_time_event = 0,
                         keep_all_teams = TRUE,
                         attacking_half_only = TRUE) {
  
  # Standard pitch dimensions
  pitch_length <- 105
  pitch_width <- 68
  
  period_col_track <- if ("period" %in% names(tracking)) "period" else "period_id"
  period_col_events <- if ("period" %in% names(events)) "period" else "period_id"
  
  # Deduce team attacking direction using goalkeepers
  # A goalkeeper's mean X position will be near their own goal (defending goal)
  gk_means <- tracking %>%
    filter(is_goalkeeper == 1, !is.na(team_id)) %>%
    group_by(match_id, !!sym(period_col_track), team_id) %>%
    summarize(gk_mean_x = mean(x_pitch, na.rm = TRUE), .groups = "drop")
  
  # If the goalkeeper is on the right side (mean X > 52.5), the team is defending the right goal
  # and attacking the left goal. Thus, their coordinates need to be flipped so they attack right.
  team_dirs <- gk_means %>%
    mutate(flip_required = gk_mean_x > (pitch_length / 2)) %>%
    select(match_id, !!sym(period_col_track), team_id, flip_required)
  
  # Filter out goalkeepers from the tracking data for the analysis
  tracking <- tracking %>% filter(is.na(is_goalkeeper) | is_goalkeeper == 0)
  
  # Filter the events that act as the 'anchor' (e.g., Shots)
  # Filter the events that act as the 'anchor' (e.g., Shots)
  target_events <- events %>%
    filter(event_type == event_type_filter)
  
  if ("set_piece_type" %in% names(target_events)) {
    if (is.na(event_subtype)) {
      # Default: Open Play shots only (exclude Free Kicks and Penalties)
      target_events <- target_events %>% filter(is.na(set_piece_type))
    } else {
      target_events <- target_events %>% filter(set_piece_type == !!event_subtype)
    }
  } else if ("event_subtype" %in% names(target_events)) {
    if (!is.na(event_subtype)) {
      target_events <- target_events %>% filter(event_subtype == !!event_subtype)
    }
  }
  
  # Join direction to events to compute aligned shot coordinates
  join_by_list <- setNames(c("match_id", period_col_track, "team_id"), 
                           c("match_id", period_col_events, "team_with_poss"))
  
  target_events <- target_events %>%
    left_join(team_dirs, by = join_by_list) %>%
    mutate(
      flip_required = replace_na(flip_required, FALSE),
      shot_x = if_else(flip_required, pitch_length - (coordinates_x * pitch_length), coordinates_x * pitch_length),
      shot_y = if_else(flip_required, pitch_width - (coordinates_y * pitch_width), coordinates_y * pitch_width)
    )
  
  # All shots must be in the attacking half (right side after flip)
  target_events <- target_events %>% filter(shot_x >= (pitch_length / 2))
  
  dataset_list <- list()
  
  # For each target event, extract the temporal window from the tracking data
  for (i in seq_len(nrow(target_events))) {
    evt <- target_events[i, ]
    m_id <- evt$match_id
    
    # Identify the time of the event
    # Assumes a time_seconds column exists in events
    evt_time <- evt$time_seconds 
    
    t_start <- evt_time - start_time
    t_end <- evt_time - end_time
    
    # Extract the tracking window for this specific event
    track_window <- tracking %>%
      filter(match_id == m_id, 
             !!sym(period_col_track) == evt[[period_col_events]],
             time_sec_orig >= t_start, 
             time_sec_orig <= t_end)
    
    # Flip the entire tracking window if the attacking team was attacking left
    if (isTRUE(evt$flip_required)) {
      track_window <- track_window %>%
        mutate(
          x_pitch = pitch_length - x_pitch,
          y_pitch = pitch_width - y_pitch
        )
    }
    
    # Create the relative time_sec index (e.g., 1 to 30)
    # Ensure it doesn't exceed the intended window size due to floating point rounding
    max_time_sec <- start_time - end_time + 1
    track_window <- track_window %>%
      mutate(
        time_sec = pmin(floor(time_sec_orig - t_start) + 1, max_time_sec)
      )
    
    # Label attacking vs defending team
    if ("team_with_poss" %in% names(evt) && "team_id" %in% names(track_window)) {
      track_window <- track_window %>%
        mutate(team_code = if_else(team_id == evt$team_with_poss, "Attack", "Defense"))
      
      if (!keep_all_teams) {
        track_window <- track_window %>% filter(team_code == "Attack")
      }
    }
    
    # Ensure necessary coordinates exist and format output
    # (Assuming long format tracking has x_pitch and y_pitch)
    if (nrow(track_window) > 0) {
      track_window <- track_window %>%
        select(x = x_pitch, y = y_pitch, time_sec, match_id, player_id, any_of(c("team_code", "team_id"))) %>%
        mutate(anchor_event_id = evt$event_id)
      
      dataset_list[[i]] <- track_window
    }
  }
  
  # Combine all windows into a single point pattern dataset
  dataset <- bind_rows(dataset_list) %>% 
    drop_na(x, y, time_sec) %>%
    mutate(
      x = pmin(pmax(x, 0), pitch_length),
      y = pmin(pmax(y, 0), pitch_width)
    )
  
  return(dataset)
}

# -------------------------------------------------------------------------
# 4. Spatial Normalization & Helpers
# -------------------------------------------------------------------------
#' Standardize attacking direction
#' For spatial point processes, we typically want all attacking events
#' (like shots) to be directed towards the same side of the pitch (e.g., Right goal).
#' This function flips the coordinates (x, y) if a team is attacking left to right.
own_goal_direction <- function(tracking_data, pitch_length = 105, pitch_width = 68) {
  # If the data does not have 'is_goalkeeper', add a dummy to avoid breaking older scripts
  if (!"is_goalkeeper" %in% names(tracking_data)) {
    warning("Column 'is_goalkeeper' not found. Adding dummy column is_goalkeeper = 0 to prevent errors.")
    tracking_data <- tracking_data %>% mutate(is_goalkeeper = 0)
  }
  
  # Robust determination of attacking direction using goalkeepers:
  # In each match and period, the team with the higher mean X position for their goalkeeper
  # is defending the Right goal (x = 105) and attacking the Left goal (x = 0).
  # We flip that team's coordinates so all teams attack towards x = 105 (Right goal).
  period_col <- if ("period" %in% names(tracking_data)) "period" else "period_id"
  
  if ("team_id" %in% names(tracking_data) && period_col %in% names(tracking_data)) {
    team_dirs <- tracking_data %>%
      filter(is_goalkeeper == 1, !is.na(team_id)) %>%
      group_by(match_id, !!sym(period_col), team_id) %>%
      summarize(gk_mean_x = mean(x_pitch, na.rm = TRUE), .groups = "drop") %>%
      mutate(flip_required = gk_mean_x > (pitch_length / 2)) %>%
      select(match_id, !!sym(period_col), team_id, flip_required)
    
    tracking_data <- tracking_data %>%
      left_join(team_dirs, by = c("match_id", period_col, "team_id")) %>%
      mutate(
        flip_required = replace_na(flip_required, FALSE),
        x_pitch = if_else(flip_required, pitch_length - x_pitch, x_pitch),
        y_pitch = if_else(flip_required, pitch_width - y_pitch, y_pitch)
      ) %>%
      select(-flip_required)
  } else {
    warning("Could not find team_id or period to determine attacking direction.")
  }
  
  return(tracking_data)
}

# Helper function to convert raw DFL wide tracking data into the required long format
transform_tracking_to_long <- function(tracking_wide, match_id_val) {
  
  # Pitch dimensions to un-normalize coordinates
  pitch_length <- 105
  pitch_width <- 68
  
  # Use pivot_longer to convert columns like DFL-OBJ-XXXX_x, DFL-OBJ-XXXX_y into long format
  tracking_long <- tracking_wide %>%
    pivot_longer(
      cols = matches("DFL-OBJ-.*_[xyds]$"),
      names_to = c("player_id", ".value"),
      names_pattern = "(DFL-OBJ-[A-Z0-9]+)_(.)"
    ) %>%
    # Remove rows where player wasn't on pitch (missing coordinates)
    filter(!is.na(x), !is.na(y)) %>%
    # Un-normalize coordinates
    mutate(
      match_id = match_id_val,
      time_sec_orig = time_seconds,
      x_pitch = x * pitch_length,
      y_pitch = y * pitch_width
    )
  
  return(tracking_long)
}
