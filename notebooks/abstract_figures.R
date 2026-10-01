# ==========================================================================
# SSAC 2027 ABSTRACT FIGURES
# ==========================================================================
# Figure 1: best-case single attacking player with 90% conformal regions.
# Figure 2: three attacking players of one play with 90% conformal regions.
#
# Usage:
#   source(here::here("notebooks", "architecture.R"))
#   fig1 <- plot_best_player(test_predictions, events_prepared,
#                            team_filter = "Attack", exclude_events = 3716)
#   fig2 <- plot_multiplayer_event(test_predictions, events_prepared,
#                                  8753, c("DFL-OBJ-J0130T", "DFL-OBJ-002GMO",
#                                          "DFL-OBJ-J01KJ5"))
#   ggsave(here::here("output", "img", "best_player.pdf"), fig1$plot, ...)

# Shared white pitch base for all abstract figures.
base_pitch <- function() {
  ggplot() +
    ggsoccer::annotate_pitch(
      dimensions = ggsoccer::pitch_international,
      fill = NA,
      colour = "grey30",
      limits = FALSE
    ) +
    ggsoccer::theme_pitch() +
    theme(
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA),
      legend.background = element_rect(fill = "white", colour = NA),
      legend.key = element_rect(fill = "white", colour = NA),
      legend.text = element_text(color = "black", size = 9),
      legend.title = element_text(color = "black", size = 10, face = "bold"),
      plot.margin = margin(6, 6, 6, 6)
    )
}

# Distinct hues that do not collide with the teal (predicted) and
# amber (actual) trajectories.
horizon_scale <- function(horizons) {
  scale_color_manual(
    values = c("1" = "#d7191c", "3" = "#2c7fb8", "5" = "#6a51a3"),
    name = "Horizon (s)",
    breaks = as.character(horizons)
  )
}

# Selects the (play, player) pair with the lowest ADE among players who
# actually moved in the horizon, and plots its 5 s forecast.
plot_best_player <- function(test_predictions, events_prepared, alpha = 0.10,
                             min_path_length = 10, horizons = c(1, 3, 5),
                             team_filter = NULL, exclude_events = NULL) {
  best <- test_predictions |>
    group_by(event_id, player_id) |>
    arrange(time_step, .by_group = TRUE) |>
    summarize(
      team_code = first(team_code),
      team_name = first(team_name),
      ADE = mean(distance, na.rm = TRUE),
      FDE = distance[which.max(time_step)],
      path_length = sum(sqrt(diff(true_x)^2 + diff(true_y)^2), na.rm = TRUE),
      .groups = "drop"
    ) |>
    filter(path_length >= min_path_length) |>
    (\(x) if (!is.null(team_filter)) filter(x, team_code == team_filter) else x)() |>
    (\(x) if (!is.null(exclude_events)) filter(x, !event_id %in% exclude_events) else x)() |>
    arrange(ADE, FDE) |>
    slice_head(n = 1)

  target_event  <- best$event_id
  target_player <- best$player_id

  pred_data <- test_predictions |>
    filter(event_id == target_event, player_id == target_player) |>
    arrange(time_step)

  hist_data <- events_prepared |>
    filter(event_id == target_event, time_sec <= 25, player_id == target_player) |>
    arrange(time_sec)

  p <- base_pitch() +
    # 90% conformal regions as outlined discs, shown only at t = 1, 3, 5 s.
    # Sorted largest-first so earlier horizons are drawn on top (1 over 3 over 5).
    ggforce::geom_circle(
      data = pred_data |>
        filter(time_step %in% horizons) |>
        mutate(time_step = factor(time_step, levels = rev(horizons))) |>
        arrange(time_step),
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, colour = time_step),
      fill = NA,
      linewidth = 0.8
    ) +
    horizon_scale(horizons) +
    # Observed history: last 5 frames before the horizon
    geom_path(
      data = hist_data |> slice_tail(n = 5),
      aes(x = x, y = y),
      color = "grey35", alpha = 0.6, linewidth = 0.5, linetype = "dashed"
    ) +
    geom_point(
      data = hist_data |> slice_tail(n = 1),
      aes(x = x, y = y),
      color = "black", size = 2.2, shape = 21, fill = "white"
    ) +
    # Predicted trajectory (5 s)
    geom_path(
      data = pred_data,
      aes(x = pred_x, y = pred_y),
      color = "#009688", linewidth = 0.9
    ) +
    geom_point(
      data = pred_data,
      aes(x = pred_x, y = pred_y),
      color = "#009688", size = 1.8
    ) +
    # Ground-truth trajectory
    geom_path(
      data = pred_data,
      aes(x = true_x, y = true_y),
      color = "#e69f00", linewidth = 0.9
    ) +
    geom_point(
      data = pred_data,
      aes(x = true_x, y = true_y),
      color = "#e69f00", size = 1.8
    ) +
    coord_fixed(
      xlim = c(0, 105),
      ylim = c(0, 68),
      expand = FALSE
    )

  list(plot = p, best = best)
}

# Plots the 5 s forecasts of several players of one play, with 90%
# conformal regions shown only at t = 1, 3, 5 s.
plot_multiplayer_event <- function(test_predictions, events_prepared,
                                   target_event_id, target_player_ids,
                                   horizons = c(1, 3, 5)) {
  pred_data <- test_predictions |>
    filter(event_id == target_event_id, player_id %in% target_player_ids) |>
    arrange(player_id, time_step)

  hist_data <- events_prepared |>
    filter(event_id == target_event_id, time_sec <= 25,
           player_id %in% target_player_ids) |>
    arrange(player_id, time_sec)

  base_pitch() +
    # 90% conformal regions as outlined discs, shown only at t = 1, 3, 5 s.
    # Sorted largest-first so earlier horizons are drawn on top (1 over 3 over 5).
    ggforce::geom_circle(
      data = pred_data |>
        filter(time_step %in% horizons) |>
        mutate(time_step = factor(time_step, levels = rev(horizons))) |>
        arrange(time_step),
      aes(x0 = pred_x, y0 = pred_y, r = conf_radius, colour = time_step),
      fill = NA,
      linewidth = 0.8
    ) +
    horizon_scale(horizons) +
    # Observed history: last 5 frames before the horizon
    geom_path(
      data = hist_data |>
        group_by(player_id) |> slice_tail(n = 5) |> ungroup(),
      aes(x = x, y = y, group = player_id),
      color = "grey35", alpha = 0.6, linewidth = 0.5, linetype = "dashed"
    ) +
    geom_point(
      data = hist_data |>
        group_by(player_id) |> slice_tail(n = 1) |> ungroup(),
      aes(x = x, y = y),
      color = "black", size = 2.2, shape = 21, fill = "white"
    ) +
    # Predicted trajectories (5 s)
    geom_path(
      data = pred_data,
      aes(x = pred_x, y = pred_y, group = player_id),
      color = "#009688", linewidth = 0.9
    ) +
    geom_point(
      data = pred_data,
      aes(x = pred_x, y = pred_y),
      color = "#009688", size = 1.8
    ) +
    # Ground-truth trajectories
    geom_path(
      data = pred_data,
      aes(x = true_x, y = true_y, group = player_id),
      color = "#e69f00", linewidth = 0.9
    ) +
    geom_point(
      data = pred_data,
      aes(x = true_x, y = true_y),
      color = "#e69f00", size = 1.8
    ) +
    coord_fixed(
      xlim = c(0, 105),
      ylim = c(0, 68),
      expand = FALSE
    )
}
