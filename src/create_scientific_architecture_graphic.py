#!/usr/bin/env python3
"""
Scientific Diagram: Social-LSTM Architecture + Data Split + Conformal Region
Matches the style of Alahi et al. (2016), Olah's LSTM diagram, and modern clean neural network schematics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch
import numpy as np
import os

# Matplotlib configuration for publication quality
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure(figsize=(24, 13.5), facecolor='#ffffff')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 24)
ax.set_ylim(0, 13.5)
ax.axis('off')

# ==============================================================================
# HEADER BANNER (Clean, Modern, Pastel aesthetic like Image 3)
# ==============================================================================
ax.text(1.0, 12.8, "SPATIOTEMPORAL TRAJECTORY PREDICTION & MULTI-AGENT CONFORMAL REGIONS",
        fontsize=17, fontweight='bold', color='#0f172a', va='center')
ax.text(1.0, 12.4, "Social-LSTM Multi-Agent Backbone (Alahi et al., CVPR 2016)  ✦  Play-Stratified Data Split  ✦  Adaptive Conformal Risk Control (Angelopoulos et al. 2024)",
        fontsize=10.5, color='#475569', va='center')

# Author Tag / Dissertation pill badge
badge_diss = FancyBboxPatch((18.6, 12.35), 4.4, 0.75, boxstyle="round,pad=0.04,rounding_size=0.15",
                            facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.2)
ax.add_patch(badge_diss)
ax.text(20.8, 12.78, "Dissertação de Mestrado — Estatística", fontsize=9.5, fontweight='bold', color='#1e40af', ha='center', va='center')
ax.text(20.8, 12.52, "Rafael Izbicki (2026) · Sugestões 4 a 6", fontsize=8.2, color='#3b82f6', ha='center', va='center')


# ==============================================================================
# 3 MAIN PANEL CONTAINERS (Light, clean cards with soft grey borders)
# ==============================================================================
card_y = 0.6
card_h = 11.4

# Card 1: Social-LSTM (Width: 7.7)
card1 = FancyBboxPatch((0.8, card_y), 7.4, card_h, boxstyle="round,pad=0.06,rounding_size=0.18",
                       facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
ax.add_patch(card1)

# Card 2: Data Split (Width: 6.8)
card2 = FancyBboxPatch((8.6, card_y), 6.9, card_h, boxstyle="round,pad=0.06,rounding_size=0.18",
                       facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
ax.add_patch(card2)

# Card 3: Conformal Regions (Width: 7.3)
card3 = FancyBboxPatch((15.9, card_y), 7.3, card_h, boxstyle="round,pad=0.06,rounding_size=0.18",
                       facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
ax.add_patch(card3)

# Function to draw clean panel headers (Pill badge style like Image 3)
def draw_panel_header(x, y, pill_text, pill_color, title_text, subtitle_text):
    # Pill badge
    pill = FancyBboxPatch((x, y), 2.2, 0.45, boxstyle="round,pad=0.03,rounding_size=0.1",
                          facecolor=pill_color[0], edgecolor=pill_color[1], linewidth=1.2)
    ax.add_patch(pill)
    ax.text(x + 1.1, y + 0.22, pill_text, fontsize=8.5, fontweight='bold', color=pill_color[2], ha='center', va='center')
    
    # Title & subtitle
    ax.text(x + 2.4, y + 0.32, title_text, fontsize=11.5, fontweight='bold', color='#0f172a', va='center')
    ax.text(x, y - 0.28, subtitle_text, fontsize=8.5, color='#64748b', va='center')

draw_panel_header(1.1, 11.3, "ARCHITECTURE", ("#e0f2fe", "#7dd3fc", "#0369a1"),
                  "Social-LSTM Multi-Agent", "Spatial Grid Pooling & Shared LSTM Cells (Alahi et al. 2016)")

draw_panel_header(8.9, 11.3, "DATA SPLIT", ("#ede9fe", "#c4b5fd", "#5b21b6"),
                  "Exchangeable Partition", "Play-Stratified Sampling & Out-of-Fold Hierarchy")

draw_panel_header(16.2, 11.3, "CONFORMAL", ("#dcfce7", "#86efac", "#166534"),
                  "Adaptive Conformal Region", "Contextual Scale Model ŝ_t(x_i) & Conformal Risk Control")


# ==============================================================================
# PANEL 1: SOCIAL-LSTM ARCHITECTURE (Faithful to Image 1 & Image 2)
# ==============================================================================
# A. Perspective Pitch Slice (Time t and Time t+1)
pitch_y = 7.35

# Perspective pitch 1 at Time t=25
p1_coords = np.array([
    [1.1, pitch_y],
    [3.7, pitch_y],
    [4.3, pitch_y + 1.25],
    [1.7, pitch_y + 1.25]
])
poly1 = Polygon(p1_coords, closed=True, facecolor='#f0fdf4', edgecolor='#86efac', linewidth=1.2)
ax.add_patch(poly1)

# Pitch markings on perspective plane 1
ax.plot([1.4, 4.0], [pitch_y + 0.62, pitch_y + 0.62], color='#bbf7d0', lw=0.8, ls='--')
ax.plot([2.7, 2.7], [pitch_y, pitch_y + 1.25], color='#bbf7d0', lw=0.8)

ax.text(1.2, pitch_y + 0.15, "Pitch Plane (t = 25s)", fontsize=7.5, fontweight='bold', color='#166534')

# Players on Plane 1
# Player 1 (Red / Attacker)
ax.add_patch(Circle((2.2, pitch_y + 0.75), 0.08, facecolor='#ef4444', edgecolor='#991b1b', lw=0.8))
# Player 2 (Blue / Midfielder)
ax.add_patch(Circle((2.8, pitch_y + 0.85), 0.08, facecolor='#3b82f6', edgecolor='#1d4ed8', lw=0.8))
# Player 3 (Black / Reference Player i)
ax.add_patch(Circle((2.5, pitch_y + 0.45), 0.10, facecolor='#0f172a', edgecolor='#ffffff', lw=1.2))
# Player 4 (Orange / Defender)
ax.add_patch(Circle((3.4, pitch_y + 0.40), 0.08, facecolor='#f97316', edgecolor='#c2410c', lw=0.8))

# Trajectory curves on pitch 1
ax.plot([1.8, 2.0, 2.2], [pitch_y + 0.68, pitch_y + 0.72, pitch_y + 0.75], color='#ef4444', lw=1.2)
ax.plot([2.3, 2.5, 2.8], [pitch_y + 0.80, pitch_y + 0.82, pitch_y + 0.85], color='#3b82f6', lw=1.2)
ax.plot([2.0, 2.2, 2.5], [pitch_y + 0.38, pitch_y + 0.40, pitch_y + 0.45], color='#0f172a', lw=1.4)
ax.plot([3.0, 3.2, 3.4], [pitch_y + 0.45, pitch_y + 0.42, pitch_y + 0.40], color='#f97316', lw=1.2)

# Vertical projection lines from agents on pitch down to S-Pooling (Image 1 style)
ax.plot([2.5, 2.5, 1.95], [pitch_y + 0.35, 7.15, 6.6], color='#0f172a', ls=':', lw=1.2, alpha=0.6)
ax.plot([2.2, 2.2, 1.65], [pitch_y + 0.65, 7.15, 6.6], color='#ef4444', ls=':', lw=1.0, alpha=0.5)
ax.plot([2.8, 2.8, 2.25], [pitch_y + 0.75, 7.15, 6.6], color='#3b82f6', ls=':', lw=1.0, alpha=0.5)
ax.plot([3.4, 3.4, 2.35], [pitch_y + 0.30, 7.15, 6.6], color='#f97316', ls=':', lw=1.0, alpha=0.5)

# Perspective pitch 2 at Time t+1 (Future forecast t=26)
p2_coords = np.array([
    [4.8, pitch_y],
    [7.4, pitch_y],
    [8.0, pitch_y + 1.25],
    [5.4, pitch_y + 1.25]
])
poly2 = Polygon(p2_coords, closed=True, facecolor='#f0fdf4', edgecolor='#86efac', linewidth=1.2)
ax.add_patch(poly2)
ax.text(4.9, pitch_y + 0.15, "Forecast (t = 26s)", fontsize=7.5, fontweight='bold', color='#166534')

# Players forecast positions
ax.add_patch(Circle((5.9, pitch_y + 0.82), 0.08, facecolor='#ef4444', edgecolor='#991b1b', lw=0.8))
ax.add_patch(Circle((6.6, pitch_y + 0.90), 0.08, facecolor='#3b82f6', edgecolor='#1d4ed8', lw=0.8))
ax.add_patch(Circle((6.3, pitch_y + 0.52), 0.10, facecolor='#0f172a', edgecolor='#ffffff', lw=1.2))
ax.add_patch(Circle((7.1, pitch_y + 0.38), 0.08, facecolor='#f97316', edgecolor='#c2410c', lw=0.8))

# Time evolution arrow on top
time_arr = FancyArrowPatch((2.5, pitch_y + 1.45), (6.5, pitch_y + 1.45), arrowstyle='->', mutation_scale=14, color='#0f172a', linewidth=1.5)
ax.add_patch(time_arr)
ax.text(4.5, pitch_y + 1.58, "Time Horizon Evolution: t  →  t + 1", fontsize=8.5, fontweight='bold', color='#0f172a', ha='center')

# B. Social Pooling Mechanism Detail Box (Exact adaptation of Image 1 bottom detail)
spool_box = FancyBboxPatch((1.1, 5.0), 6.8, 2.1, boxstyle="round,pad=0.03,rounding_size=0.1",
                           facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.0)
ax.add_patch(spool_box)
ax.text(1.25, 6.85, "Spatial Grid Pooling Mechanism (for Agent i in black):", fontsize=8.5, fontweight='bold', color='#0f172a')

# Step 1: Neighborhood window
ax.add_patch(patches.Rectangle((1.3, 5.25), 1.3, 1.3, facecolor='#ffffff', edgecolor='#cbd5e1', lw=0.8))
# Sub-grid lines
for gi in range(1, 4):
    ax.plot([1.3 + gi*1.3/4, 1.3 + gi*1.3/4], [5.25, 6.55], color='#f1f5f9', lw=0.6)
    ax.plot([1.3, 2.6], [5.25 + gi*1.3/4, 5.25 + gi*1.3/4], color='#f1f5f9', lw=0.6)
# Center player i
ax.add_patch(Circle((1.3 + 0.65, 5.25 + 0.65), 0.07, facecolor='#0f172a', edgecolor='white', lw=0.8))
ax.add_patch(Circle((1.3 + 0.35, 5.25 + 0.95), 0.06, facecolor='#ef4444', edgecolor='none'))
ax.add_patch(Circle((1.3 + 0.95, 5.25 + 1.05), 0.06, facecolor='#3b82f6', edgecolor='none'))
ax.add_patch(Circle((1.3 + 1.05, 5.25 + 0.35), 0.06, facecolor='#f97316', edgecolor='none'))
ax.text(1.95, 5.12, "10x10m Window (32x32)", fontsize=7, color='#64748b', ha='center')

# Arrow
arr_s1 = FancyArrowPatch((2.7, 5.9), (3.0, 5.9), arrowstyle='->', mutation_scale=10, color='#64748b', linewidth=1.2)
ax.add_patch(arr_s1)

# Step 2: Sum pooling tensor
ax.add_patch(patches.Rectangle((3.1, 5.35), 1.4, 1.15, facecolor='#ffffff', edgecolor='#0284c7', lw=0.9))
# 3D representation
p3d_top = Polygon([[3.1, 6.5], [3.3, 6.7], [4.7, 6.7], [4.5, 6.5]], closed=True, facecolor='#e0f2fe', edgecolor='#0284c7', lw=0.8)
p3d_side = Polygon([[4.5, 5.35], [4.7, 5.55], [4.7, 6.7], [4.5, 6.5]], closed=True, facecolor='#bae6fd', edgecolor='#0284c7', lw=0.8)
ax.add_patch(p3d_top)
ax.add_patch(p3d_side)
# Hidden states inside tensor
ax.add_patch(patches.Rectangle((3.3, 5.85), 0.95, 0.45, facecolor='#3b82f6', alpha=0.8, edgecolor='none'))
ax.text(3.78, 6.07, r"$h_2^{t-1}$", fontsize=7.5, color='white', ha='center', va='center', fontweight='bold')
ax.add_patch(patches.Rectangle((3.45, 5.45), 0.85, 0.35, facecolor='#f97316', alpha=0.8, edgecolor='none'))
ax.text(3.88, 5.62, r"$h_4^{t-1}$", fontsize=7.5, color='white', ha='center', va='center', fontweight='bold')
ax.text(3.8, 5.12, r"Social Tensor $H_i^t$", fontsize=7.5, fontweight='bold', color='#0284c7', ha='center')

# Arrow
arr_s2 = FancyArrowPatch((4.8, 5.9), (5.1, 5.9), arrowstyle='->', mutation_scale=10, color='#64748b', linewidth=1.2)
ax.add_patch(arr_s2)

# Step 3: Social Embedding
sub_semb = FancyBboxPatch((5.2, 5.35), 2.5, 1.25, boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor="#f0f9ff", edgecolor="#38bdf8", linewidth=1.0)
ax.add_patch(sub_semb)
ax.text(6.45, 6.35, r"Embedding $a_i^t \in \mathbb{R}^{64}$", fontsize=8.2, fontweight='bold', color='#0369a1', ha='center')
ax.text(6.45, 6.05, r"$a_i^t = \text{ReLU}(W_a H_i^t)$", fontsize=8, color='#0f172a', ha='center')
ax.text(6.45, 5.75, r"Sum-pool $8 \times 8 \to 4 \times 4 \times 128$", fontsize=7.2, color='#475569', ha='center')
ax.text(6.45, 5.48, r"Achatamento $\to$ Densa (64-d)", fontsize=7.2, color='#64748b', ha='center')

# C. Detailed LSTM Cell (Adapted from Olah's standard diagram in Image 2)
lstm_box = FancyBboxPatch((1.1, 1.95), 6.8, 2.75, boxstyle="round,pad=0.04,rounding_size=0.12",
                          facecolor="#f1f5f9", edgecolor="#64748b", linewidth=1.2)
ax.add_patch(lstm_box)
ax.text(1.25, 4.45, "Shared LSTM Cell Internals (D = 128, Parameter Sharing):", fontsize=8.5, fontweight='bold', color='#1e293b')

# Cell State Horizontal Flow (Orange bar across top, exactly like Image 2)
ax.plot([1.4, 7.6], [4.15, 4.15], color='#f97316', lw=2.0)
# Badges for Previous / Next Cell State
ax.add_patch(Circle((1.4, 4.15), 0.16, facecolor='#ea580c', edgecolor='none'))
ax.text(1.4, 4.15, r"$C_{t-1}$", fontsize=6.8, color='white', ha='center', va='center', fontweight='bold')
ax.add_patch(Circle((7.6, 4.15), 0.16, facecolor='#ea580c', edgecolor='none'))
ax.text(7.6, 4.15, r"$C_t$", fontsize=6.8, color='white', ha='center', va='center', fontweight='bold')

# Hidden State Horizontal Flow (Blue bar across bottom)
ax.plot([2.5, 7.6], [2.5, 2.5], color='#2563eb', lw=2.0)
# Badge for Previous / Next Hidden State
ax.add_patch(Circle((1.4, 3.2), 0.16, facecolor='#1d4ed8', edgecolor='none'))
ax.text(1.4, 3.2, r"$h_{t-1}$", fontsize=6.8, color='white', ha='center', va='center', fontweight='bold')
ax.add_patch(Circle((7.6, 2.5), 0.16, facecolor='#1d4ed8', edgecolor='none'))
ax.text(7.6, 2.5, r"$h_t$", fontsize=6.8, color='white', ha='center', va='center', fontweight='bold')

# Input data badge [e_i^t || a_i^t]
ax.add_patch(Circle((1.4, 2.3), 0.18, facecolor='#0284c7', edgecolor='none'))
ax.text(1.4, 2.3, r"$e_i^t, a_i^t$", fontsize=6.5, color='white', ha='center', va='center', fontweight='bold')

# Combine input data and h_{t-1} line
ax.plot([1.6, 2.4, 2.4], [3.2, 2.5, 2.5], color='#2563eb', lw=1.6)
ax.plot([1.6, 2.4], [2.3, 2.5], color='#0284c7', lw=1.6)

# Gate 1: Forget Gate (Yellow sigma node)
ax.plot([2.8, 2.8], [2.5, 4.15], color='#64748b', lw=1.4)
ax.add_patch(Circle((2.8, 3.2), 0.14, facecolor='#facc15', edgecolor='#ca8a04', lw=0.8))
ax.text(2.8, 3.2, r"$\sigma$", fontsize=8, color='#713f12', ha='center', va='center', fontweight='bold')
# Multiplication on cell line
ax.add_patch(Circle((2.8, 4.15), 0.12, facecolor='#fef08a', edgecolor='#ca8a04', lw=0.8))
ax.text(2.8, 4.15, r"$\times$", fontsize=8, color='#854d0e', ha='center', va='center', fontweight='bold')

# Gate 2: Input Gate (sigma & tanh nodes)
ax.plot([3.9, 3.9], [2.5, 3.6], color='#64748b', lw=1.4)
ax.plot([4.5, 4.5], [2.5, 3.6], color='#64748b', lw=1.4)
ax.add_patch(Circle((3.9, 3.0), 0.14, facecolor='#86efac', edgecolor='#16a34a', lw=0.8))
ax.text(3.9, 3.0, "tanh", fontsize=6.2, color='#14532d', ha='center', va='center', fontweight='bold')
ax.add_patch(Circle((4.5, 3.0), 0.14, facecolor='#facc15', edgecolor='#ca8a04', lw=0.8))
ax.text(4.5, 3.0, r"$\sigma$", fontsize=8, color='#713f12', ha='center', va='center', fontweight='bold')

# Multiply input gate
ax.plot([3.9, 4.2], [3.15, 3.6], color='#64748b', lw=1.2)
ax.plot([4.5, 4.2], [3.15, 3.6], color='#64748b', lw=1.2)
ax.add_patch(Circle((4.2, 3.6), 0.12, facecolor='#fef08a', edgecolor='#ca8a04', lw=0.8))
ax.text(4.2, 3.6, r"$\times$", fontsize=8, color='#854d0e', ha='center', va='center', fontweight='bold')

# Addition on cell state line
ax.plot([4.2, 4.2], [3.72, 4.15], color='#64748b', lw=1.2)
ax.add_patch(Circle((4.2, 4.15), 0.13, facecolor='#fed7aa', edgecolor='#ea580c', lw=0.8))
ax.text(4.2, 4.15, r"$+$", fontsize=8.5, color='#9a3412', ha='center', va='center', fontweight='bold')

# Gate 3: Output Gate
ax.plot([5.6, 5.6], [2.5, 2.5], color='#64748b', lw=1.4)
ax.add_patch(Circle((5.6, 2.5), 0.14, facecolor='#facc15', edgecolor='#ca8a04', lw=0.8))
ax.text(5.6, 2.5, r"$\sigma$", fontsize=8, color='#713f12', ha='center', va='center', fontweight='bold')

# Tanh of new cell state
ax.plot([6.4, 6.4], [4.15, 2.5], color='#64748b', lw=1.2)
ax.add_patch(Circle((6.4, 3.3), 0.14, facecolor='#86efac', edgecolor='#16a34a', lw=0.8))
ax.text(6.4, 3.3, "tanh", fontsize=6.2, color='#14532d', ha='center', va='center', fontweight='bold')

# Output multiplication
ax.add_patch(Circle((6.4, 2.5), 0.12, facecolor='#fef08a', edgecolor='#ca8a04', lw=0.8))
ax.text(6.4, 2.5, r"$\times$", fontsize=8, color='#854d0e', ha='center', va='center', fontweight='bold')

# D. Prediction Head (Output Layer)
sub_out = FancyBboxPatch((1.1, 0.75), 6.8, 1.05, boxstyle="round,pad=0.03,rounding_size=0.08",
                         facecolor="#fffbeb", edgecolor="#f59e0b", linewidth=1.2)
ax.add_patch(sub_out)
ax.text(1.25, 1.55, "Bivariate Gaussian Prediction Head (Linear 128 → 5):", fontsize=8.2, fontweight='bold', color='#92400e')
ax.text(1.25, 1.25, r"Output parameters: $[\mu_x, \mu_y, \log \sigma_x, \log \sigma_y, \text{atanh} \rho]_i^{t+1} = W_p h_i^t$", fontsize=8.5, color='#0f172a')
ax.text(1.25, 0.95, r"Loss: Negative Log-Likelihood (NLL) with Teacher Forcing (Eqs. 3–4)", fontsize=7.8, color='#b45309')


# ==============================================================================
# PANEL 2: PLAY-STRATIFIED DATA SPLIT (Modern Flowchart Style like Image 3)
# ==============================================================================
# A. Permutability & Exchangeable Unit Banner
exch_box = FancyBboxPatch((8.85, 9.6), 6.4, 1.35, boxstyle="round,pad=0.03,rounding_size=0.1",
                          facecolor="#f5f3ff", edgecolor="#8b5cf6", linewidth=1.3)
ax.add_patch(exch_box)
ax.text(9.05, 10.65, "Statistical Permutability Invariant:", fontsize=9.2, fontweight='bold', color='#6d28d9')
ax.text(9.05, 10.35, "Contiguous frames within a sequence share tactical context & inertia.", fontsize=7.8, color='#475569')
ax.text(9.05, 10.08, r"Exchangeable Unit: Entire Offensive Sequence / Shot Event $k \in \{1, \dots, M\}$", fontsize=8.2, fontweight='bold', color='#0f172a')
ax.text(9.05, 9.80, r"$\rightarrow$ Zero Intra-Play Leakage: all 20 players stay in the same split!", fontsize=8.0, color='#dc2626', fontweight='bold')

# Arrow down
arr_d1 = FancyArrowPatch((12.05, 9.6), (12.05, 9.25), arrowstyle='->', mutation_scale=12, color='#8b5cf6', linewidth=1.4)
ax.add_patch(arr_d1)

# B. Four Split Cards with Clear Roles & Colors (Image 3 Style)
def draw_clean_split_card(y, title, pct, n_plays, border_c, fill_c, text_c, role_desc, target_metric):
    card = FancyBboxPatch((8.85, y), 6.4, 1.15, boxstyle="round,pad=0.03,rounding_size=0.08",
                          facecolor=fill_c, edgecolor=border_c, linewidth=1.3)
    ax.add_patch(card)
    # Badge
    badge = FancyBboxPatch((9.0, y + 0.65), 2.2, 0.38, boxstyle="round,pad=0.02,rounding_size=0.08",
                           facecolor=border_c, edgecolor='none')
    ax.add_patch(badge)
    ax.text(10.1, y + 0.84, f"{title} ({pct}%)", fontsize=8.0, fontweight='bold', color='white', ha='center', va='center')
    ax.text(11.35, y + 0.84, f"{n_plays} ({n_plays*20} trajectories)", fontsize=7.8, color='#334155', va='center', fontweight='bold')
    
    ax.text(9.05, y + 0.38, role_desc, fontsize=7.8, color='#0f172a')
    ax.text(9.05, y + 0.14, target_metric, fontsize=7.4, color=text_c, fontweight='bold')

draw_clean_split_card(8.05, "1. TRAINING SET", 60, 98, "#0284c7", "#f0f9ff", "#0369a1",
                      "Optimizes Social-LSTM weights θ* via RMSprop (lr = 0.003)",
                      "Data Augmentation: lateral pitch reflection y' = 68 - y")

draw_clean_split_card(6.75, "2. VALIDATION (OOF)", 15, 24, "#f59e0b", "#fffbeb", "#b45309",
                      "Unseen plays for early stopping & out-of-fold residual collection",
                      "Ajuste do Modelo de Escala Adaptativa ŝ_t(x_i) = exp(g(x_i, t)) (Sug. 4)")

draw_clean_split_card(5.45, "3. CALIBRATION SET", 15, 24, "#10b981", "#f0fdf4", "#15803d",
                      "24 independent plays (475 trajectories) for Conformal Risk Control",
                      "Calibração de λ̂ resolvendo cota exata [n/(n+1)]L̄(λ) + 1/(n+1) ≤ α")

draw_clean_split_card(4.15, "4. HELD-OUT TEST", 10, 18, "#ec4899", "#fdf2f8", "#be185d",
                      "18 unseen evaluation plays (359 trajectories, 1,795 player-steps)",
                      "Aferição empírica: Cobertura Simultânea = 90.8% (Alvo ≥ 90%)")

# C. Cross-Conformal Pipeline Box (Sugestão 5)
cc_box = FancyBboxPatch((8.85, 0.75), 6.4, 3.15, boxstyle="round,pad=0.03,rounding_size=0.1",
                        facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=1.3)
ax.add_patch(cc_box)
ax.text(9.05, 3.65, "Cross-Conformal K-Fold Generalization (Sugestão 5):", fontsize=8.8, fontweight='bold', color='#1d4ed8')
ax.text(9.05, 3.35, "Vovk (2015); Barber et al. (2021); Izbicki (2026, Seção 5):", fontsize=7.5, color='#475569')

# Draw K-fold diagram
kf_y = 2.45
folds_cols = ["#38bdf8", "#818cf8", "#f472b6", "#fbbf24", "#34d399"]
for k in range(5):
    f_box = FancyBboxPatch((9.05 + k*1.2, kf_y), 1.1, 0.65, boxstyle="round,pad=0.02,rounding_size=0.06",
                           facecolor="#ffffff", edgecolor=folds_cols[k], linewidth=1.2)
    ax.add_patch(f_box)
    ax.text(9.6 + k*1.2, kf_y + 0.35, f"Fold {k+1}", fontsize=7.5, fontweight='bold', color='#0f172a', ha='center', va='center')
    ax.text(9.6 + k*1.2, kf_y + 0.16, "~33 plays", fontsize=6.8, color='#64748b', ha='center', va='center')

ax.text(9.05, 2.15, "1. Partition plays into K = 5 folds strictly by event_id", fontsize=7.5, color='#1e293b')
ax.text(9.05, 1.85, "2. For fold k: train LSTM on K-1 folds, evaluate fold k out-of-fold", fontsize=7.5, color='#1e293b')
ax.text(9.05, 1.55, "3. Pool all n out-of-fold risk scores across all 5 folds", fontsize=7.5, color='#1e293b')
ax.text(9.05, 1.25, "4. Calibrate λ̂ on 100% of available data (ideal for small samples)", fontsize=7.5, fontweight='bold', color='#2563eb')
ax.text(9.05, 0.95, "5. Inference: ensemble of K models or model trained on all data", fontsize=7.3, color='#475569')


# ==============================================================================
# PANEL 3: ADAPTIVE CONFORMAL PREDICTION (Sugestões 4 & 6 + Realistic Pitch)
# ==============================================================================
# A. Adaptive Scale Model Box (Sugestão 4)
s4_box = FancyBboxPatch((16.15, 9.6), 6.8, 1.35, boxstyle="round,pad=0.03,rounding_size=0.1",
                        facecolor="#f0fdf4", edgecolor="#22c55e", linewidth=1.3)
ax.add_patch(s4_box)
ax.text(16.35, 10.68, "Sugestão 4: Escala Adaptativa Condicional ŝ_t(x_i)", fontsize=9.2, fontweight='bold', color='#15803d')
ax.text(16.35, 10.38, r"Substitui a mediana estática por regressão log-linear em resíduos out-of-fold:", fontsize=7.8, color='#475569')
ax.text(16.35, 10.08, r"$\log(e_{i,t} + \varepsilon) = g(x_i, t) + \text{erro} \rightarrow \hat{s}_t(x_i) = \exp\{\hat{g}(x_i, t)\}$", fontsize=8.8, color='#0f172a', fontweight='bold')
ax.text(16.35, 9.78, r"Covariáveis $x_i$ ($t = 25$): velocidade $\|v_i\|$, dist. bola $d_i^{\text{ball}}$, papel tático, $t, t^2$", fontsize=7.6, color='#166534')

# B. Conformal Risk Control Box (Sugestão 6)
s6_box = FancyBboxPatch((16.15, 7.85), 6.8, 1.6, boxstyle="round,pad=0.03,rounding_size=0.1",
                        facecolor="#fefce8", edgecolor="#eab308", linewidth=1.3)
ax.add_patch(s6_box)
ax.text(16.35, 9.22, "Sugestão 6: Conformal Risk Control (Angelopoulos et al. 2024)", fontsize=9.2, fontweight='bold', color='#a16207')
ax.text(16.35, 8.92, r"Perda por jogada: $L_k(\lambda) = \frac{1}{N_k} \sum_{i=1}^{N_k} \mathbf{1}\left( \max_{t} \frac{e_{k,i,t}}{\hat{s}_t(x_{k,i})} > \lambda \right)$ (fração que escapa)", fontsize=8.2, color='#0f172a')
ax.text(16.35, 8.58, r"Critério exato de calibração em amostra finita com $n = 24$ jogadas:", fontsize=7.8, color='#713f12')
ax.text(16.35, 8.28, r"$\hat{\lambda} = \inf \left\{ \lambda \geq 0 : \frac{n}{n+1} \bar{L}(\lambda) + \frac{1}{n+1} \leq \alpha \right\} \rightarrow \hat{\lambda} = 2.597$", fontsize=8.6, color='#b45309', fontweight='bold')
ax.text(16.35, 7.98, r"Garantia teórica formal: $\mathbb{E}[L_{\text{nova}}(\hat{\lambda})] \leq \alpha$ (válida sem descartar trajetórias)", fontsize=7.5, color='#475569')

# C. Pitch Canvas with Adaptive Tubes (105 x 68 m)
pitch_card = FancyBboxPatch((16.15, 0.75), 6.8, 6.95, boxstyle="round,pad=0.03,rounding_size=0.12",
                            facecolor="#ffffff", edgecolor="#10b981", linewidth=1.4)
ax.add_patch(pitch_card)

ax.text(16.35, 7.42, "Output Conformal no Campo Oficial (105 x 68 m):", fontsize=9.2, fontweight='bold', color='#0f172a')
ax.text(16.35, 7.15, r"Raio individual: $r_{i,t} = \hat{\lambda} \cdot \hat{s}_t(x_i) \rightarrow \mathcal{C}_{i,t} = \{p : \|p - \hat{p}_{i,t}\|_2 \leq r_{i,t}\}$", fontsize=8.0, color='#15803d', fontweight='bold')

# Football pitch visualization box
fpx0, fpy0 = 16.35, 1.3
fpw, fph = 6.4, 5.6

# Realistic Green Turf
turf = FancyBboxPatch((fpx0, fpy0), fpw, fph, boxstyle="round,pad=0.01,rounding_size=0.06",
                      facecolor="#047857", edgecolor="#065f46", linewidth=1.2)
ax.add_patch(turf)

# Official Pitch Lines (White crisp markings)
ax.plot([fpx0, fpx0 + fpw, fpx0 + fpw, fpx0, fpx0], [fpy0, fpy0, fpy0 + fph, fpy0 + fph, fpy0], color='#ffffff', lw=1.0, alpha=0.9)
# Halfway line & center circle
ax.plot([fpx0 + fpw/2, fpx0 + fpw/2], [fpy0, fpy0 + fph], color='#ffffff', lw=0.9, alpha=0.9)
ax.add_patch(Circle((fpx0 + fpw/2, fpy0 + fph/2), fph*0.22, fill=False, edgecolor='#ffffff', lw=0.9, alpha=0.9))
# Penalty boxes
ax.plot([fpx0, fpx0 + fpw*0.18, fpx0 + fpw*0.18, fpx0],
        [fpy0 + fph*0.2, fpy0 + fph*0.2, fpy0 + fph*0.8, fpy0 + fph*0.8], color='#ffffff', lw=0.8, alpha=0.8)
ax.plot([fpx0 + fpw, fpx0 + fpw*0.82, fpx0 + fpw*0.82, fpx0 + fpw],
        [fpy0 + fph*0.2, fpy0 + fph*0.2, fpy0 + fph*0.8, fpy0 + fph*0.8], color='#ffffff', lw=0.8, alpha=0.8)

import matplotlib.patheffects as pe

# PLAYER 1: SPRINTING ATTACKER (v = 7.82 m/s) -> Large Expanding Adaptive Conformal Tube
p1_ox = [fpx0 + fpw*0.35, fpx0 + fpw*0.44, fpx0 + fpw*0.52]
p1_oy = [fpy0 + fph*0.65, fpy0 + fph*0.67, fpy0 + fph*0.69]
ax.plot(p1_ox, p1_oy, color='#ffffff', lw=1.3, ls='--', alpha=0.9)
ax.scatter([p1_ox[-1]], [p1_oy[-1]], s=36, color='#ffffff', edgecolor='#0f172a', zorder=5)

p1_px = np.linspace(p1_ox[-1], fpx0 + fpw*0.88, 5)
p1_py = np.linspace(p1_oy[-1], fpy0 + fph*0.78, 5)
p1_rads = [0.22, 0.38, 0.55, 0.72, 0.90]  # Adaptive: large expanding radii
ax.plot(p1_px, p1_py, color='#00f5d4', lw=2.2, zorder=6)

# Ground truth path for player 1 (Orange, inside the tube)
p1_tx = p1_px + np.array([0, 0.05, 0.11, 0.17, 0.24])
p1_ty = p1_py + np.array([0, -0.06, -0.14, -0.21, -0.26])
ax.plot(p1_tx, p1_ty, color='#ff7a00', lw=2.0, ls='-', zorder=6)
ax.scatter(p1_tx, p1_ty, s=22, color='#ff7a00', zorder=7)

for pt, pr in zip(range(5), p1_rads):
    circ = Circle((p1_px[pt], p1_py[pt]), pr, facecolor='#10b981', edgecolor='#34d399', alpha=0.20, lw=1.2, zorder=4)
    ax.add_patch(circ)

txt1 = ax.text(p1_px[2], p1_py[2] + 0.75, "Atacante em Sprint ($v = 7.8$ m/s)\nRegião Adaptativa Ampla ($r_5 = 38.8$ m) $\\rightarrow$ Resgata GT",
        fontsize=7.8, color='#fef08a', fontweight='bold', ha='center', zorder=10)
txt1.set_path_effects([pe.withStroke(linewidth=2.5, foreground='#064e3b')])

# PLAYER 2: POSITIONAL DEFENDER (v = 0.04 m/s) -> Compact Efficient Conformal Tube
p2_ox = [fpx0 + fpw*0.28, fpx0 + fpw*0.29, fpx0 + fpw*0.30]
p2_oy = [fpy0 + fph*0.30, fpy0 + fph*0.29, fpy0 + fph*0.28]
ax.plot(p2_ox, p2_oy, color='#ffffff', lw=1.3, ls='--', alpha=0.9)
ax.scatter([p2_ox[-1]], [p2_oy[-1]], s=36, color='#ffffff', edgecolor='#0f172a', zorder=5)

p2_px = np.linspace(p2_ox[-1], fpx0 + fpw*0.42, 5)
p2_py = np.linspace(p2_oy[-1], fpy0 + fph*0.23, 5)
p2_rads = [0.09, 0.14, 0.19, 0.24, 0.30]  # Adaptive: compact tight radii
ax.plot(p2_px, p2_py, color='#00f5d4', lw=2.2, zorder=6)

p2_tx = p2_px + np.array([0, 0.02, 0.04, 0.06, 0.08])
p2_ty = p2_py + np.array([0, -0.01, -0.03, -0.04, -0.05])
ax.plot(p2_tx, p2_ty, color='#ff7a00', lw=2.0, ls='-', zorder=6)
ax.scatter(p2_tx, p2_ty, s=22, color='#ff7a00', zorder=7)

for pt, pr in zip(range(5), p2_rads):
    circ = Circle((p2_px[pt], p2_py[pt]), pr, facecolor='#10b981', edgecolor='#34d399', alpha=0.28, lw=1.2, zorder=4)
    ax.add_patch(circ)

txt2 = ax.text(p2_px[2], p2_py[2] - 0.45, "Defensor Posicional ($v = 0.04$ m/s)\nRegião Compacta ($r_5 = 12.9$ m) $\\rightarrow$ 55% Área Economizada",
        fontsize=7.8, color='#a7f3d0', fontweight='bold', ha='center', zorder=10)
txt2.set_path_effects([pe.withStroke(linewidth=2.5, foreground='#064e3b')])

# Legend below pitch
lx = fpx0 + 0.3
ly = fpy0 + 0.25
ax.plot([lx, lx + 0.4], [ly, ly], color='#00f5d4', lw=2.2)
ax.text(lx + 0.45, ly, "Previsto (Social-LSTM)", fontsize=7.2, color='#ffffff', va='center')

ax.plot([lx + 2.1, lx + 2.5], [ly, ly], color='#ff7a00', lw=2.2)
ax.text(lx + 2.55, ly, "Real (Ground Truth)", fontsize=7.2, color='#ffffff', va='center')

ax.add_patch(Circle((lx + 4.2, ly), 0.12, facecolor='#10b981', edgecolor='#34d399', alpha=0.4))
ax.text(lx + 4.4, ly, "Tubo Conforme Adaptativo", fontsize=7.2, color='#ffffff', va='center')


# ==============================================================================
# PIPELINE CONNECTING ARROWS BETWEEN PANELS
# ==============================================================================
# Arrow Panel 1 -> Panel 2
arr_p1_p2 = FancyArrowPatch((8.25, 6.3), (8.55, 6.3), arrowstyle='simple,tail_width=3.5,head_width=8.5,head_length=7.5',
                            facecolor='#0284c7', edgecolor='#0369a1', lw=0.6)
ax.add_patch(arr_p1_p2)
ax.text(8.4, 6.65, "Weights\n$\\theta^*$", fontsize=7.5, fontweight='bold', color='#0284c7', ha='center')

# Arrow Panel 2 -> Panel 3
arr_p2_p3 = FancyArrowPatch((15.55, 6.3), (15.85, 6.3), arrowstyle='simple,tail_width=3.5,head_width=8.5,head_length=7.5',
                            facecolor='#10b981', edgecolor='#047857', lw=0.6)
ax.add_patch(arr_p2_p3)
ax.text(15.7, 6.65, "Residuals\n& $\\hat{\\lambda}$", fontsize=7.5, fontweight='bold', color='#10b981', ha='center')


# ==============================================================================
# FOOTER / CITATION BAR
# ==============================================================================
footer = FancyBboxPatch((0.8, 0.12), 22.4, 0.38, boxstyle="round,pad=0.02,rounding_size=0.06",
                        facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=1.0)
ax.add_patch(footer)
ax.text(1.1, 0.31, "Pipeline Estatístico Completo: Social-LSTM (Alahi et al. 2016)  ✦  Permutabilidade de Jogadas (Vovk 2015)  ✦  Escala Adaptativa & Conformal Risk Control (Angelopoulos et al. 2024; Izbicki 2026)",
        fontsize=8.0, color='#475569', va='center')
ax.text(23.0, 0.31, "Dissertação de Mestrado — Modelagem Espaçotemporal de Futebol", fontsize=8.0, color='#0284c7', ha='right', va='center', fontweight='bold')

# Save high-resolution PNG
os.makedirs("output", exist_ok=True)
os.makedirs("docs", exist_ok=True)

out_file = "output/scientific_pipeline_architecture_clean.png"
docs_file = "docs/scientific_pipeline_architecture_clean.png"

plt.savefig(out_file, dpi=300, facecolor='#ffffff', edgecolor='none', bbox_inches='tight')
plt.savefig(docs_file, dpi=300, facecolor='#ffffff', edgecolor='none', bbox_inches='tight')
print(f"Diagram created successfully:")
print(f"  -> {out_file}")
print(f"  -> {docs_file}")
