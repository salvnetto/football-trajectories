#!/usr/bin/env python3
"""
Pipeline Graphic: Social-LSTM Architecture + Spatiotemporal Data Split + Adaptive Conformal Region
Publication-grade visual overview for football trajectory forecasting dissertation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, PathPatch
import matplotlib.patheffects as pe
from matplotlib.path import Path
import numpy as np
import os

# Set style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# Canvas size: 24 x 13.5 inches at 300 dpi
fig = plt.figure(figsize=(24, 13.5), facecolor='#0b0f19')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 24)
ax.set_ylim(0, 13.5)
ax.axis('off')

# ==============================================================================
# HEADER & TITLE BANNER
# ==============================================================================
# Top Title Card
title_box = FancyBboxPatch((0.6, 12.0), 22.8, 1.15,
                           boxstyle="round,pad=0.08,rounding_size=0.18",
                           facecolor="#131d31", edgecolor="#2a3f68", linewidth=1.5)
ax.add_patch(title_box)

ax.text(0.9, 12.75, "SPATIOTEMPORAL TRAJECTORY FORECASTING & MULTI-AGENT CONFORMAL PREDICTION",
        fontsize=16, fontweight='bold', color='#38bdf8', va='center')
ax.text(0.9, 12.32, "End-to-End Pipeline: Social-LSTM Backbone (Alahi et al. 2016)  ✦  Exchangeable Event-Stratified Split  ✦  Adaptive Risk-Controlled Regions (Angelopoulos et al. 2024)",
        fontsize=10.5, color='#94a3b8', va='center')

# Author / Method Tag
tag_box = FancyBboxPatch((19.4, 12.2), 3.8, 0.75,
                         boxstyle="round,pad=0.04,rounding_size=0.12",
                         facecolor="#1e293b", edgecolor="#38bdf8", linewidth=1.2)
ax.add_patch(tag_box)
ax.text(21.3, 12.57, "Rafael Izbicki (2026) Notes", fontsize=9, fontweight='bold', color='#f8fafc', ha='center', va='center')
ax.text(21.3, 12.35, "Suggestions 4–6 Implementation", fontsize=8, color='#38bdf8', ha='center', va='center')


# ==============================================================================
# 3 MAIN STAGE CONTAINERS
# ==============================================================================
card_y = 0.5
card_h = 11.2

# Stage 1 Container: Social-LSTM
card1 = FancyBboxPatch((0.6, card_y), 7.4, card_h,
                       boxstyle="round,pad=0.06,rounding_size=0.2",
                       facecolor="#111827", edgecolor="#1e293b", linewidth=1.8)
ax.add_patch(card1)

# Stage 2 Container: Data Split
card2 = FancyBboxPatch((8.3, card_y), 7.4, card_h,
                       boxstyle="round,pad=0.06,rounding_size=0.2",
                       facecolor="#111827", edgecolor="#1e293b", linewidth=1.8)
ax.add_patch(card2)

# Stage 3 Container: Conformal Prediction
card3 = FancyBboxPatch((16.0, card_y), 7.4, card_h,
                       boxstyle="round,pad=0.06,rounding_size=0.2",
                       facecolor="#111827", edgecolor="#1e293b", linewidth=1.8)
ax.add_patch(card3)

# Headers for each card
def draw_card_header(x, y, w, number, title, subtitle, color):
    hdr = FancyBboxPatch((x, y - 0.95), w, 0.95,
                         boxstyle="round,pad=0.02,rounding_size=0.16",
                         facecolor="#1a2333", edgecolor=color, linewidth=1.4)
    ax.add_patch(hdr)
    # Circle badge for number
    badge = Circle((x + 0.45, y - 0.47), 0.32, facecolor=color, edgecolor='none')
    ax.add_patch(badge)
    ax.text(x + 0.45, y - 0.47, str(number), fontsize=12, fontweight='bold', color='#ffffff', ha='center', va='center')
    
    ax.text(x + 0.95, y - 0.33, title, fontsize=11.5, fontweight='bold', color='#f8fafc', va='center')
    ax.text(x + 0.95, y - 0.65, subtitle, fontsize=8.5, color='#94a3b8', va='center')

draw_card_header(0.7, 11.6, 7.2, "1", "SOCIAL-LSTM ARCHITECTURE", "Multi-Agent Spatial Grid Pooling (Alahi et al. 2016)", "#0284c7")
draw_card_header(8.4, 11.6, 7.2, "2", "EXCHANGEABLE DATA SPLIT", "Play-Stratified Hierarchy & Out-of-Fold Estimation", "#6366f1")
draw_card_header(16.1, 11.6, 7.2, "3", "ADAPTIVE CONFORMAL REGION", "Conformal Risk Control & Finite-Sample Coverage", "#10b981")


# ==============================================================================
# SECTION 1: SOCIAL-LSTM ARCHITECTURE DETAILS
# ==============================================================================
# A. Input Representation
sub1 = FancyBboxPatch((0.85, 9.4), 6.9, 1.15, boxstyle="round,pad=0.03,rounding_size=0.1",
                      facecolor="#1e293b", edgecolor="#334155", linewidth=1)
ax.add_patch(sub1)
ax.text(1.05, 10.25, "Input Trajectories (25 Observed Frames at 1 Hz)", fontsize=9.5, fontweight='bold', color='#38bdf8')
ax.text(1.05, 9.9, r"Observed: $p_i^t = (x_i^t, y_i^t) \in \mathbb{R}^2, \quad t \in \{1, \dots, 25\}, \quad i \in \{1, \dots, N\}$", fontsize=8.5, color='#e2e8f0')
ax.text(1.05, 9.6, r"Features: $\Delta x_i, \Delta y_i$, ball distance $d_i^{\text{ball}}$, ball speed, role index (Attack / Defense)", fontsize=8, color='#94a3b8')

# Connector arrow
arr1 = FancyArrowPatch((4.3, 9.4), (4.3, 9.0), arrowstyle='->', mutation_scale=14, color='#38bdf8', linewidth=1.6)
ax.add_patch(arr1)

# B. Dual-Stream Feature Extraction: Coordinate Embedding + Social Grid Pooling
# Coordinate Stream Box (Left)
sub_emb = FancyBboxPatch((0.85, 7.3), 3.25, 1.7, boxstyle="round,pad=0.03,rounding_size=0.1",
                         facecolor="#1e293b", edgecolor="#0284c7", linewidth=1.2)
ax.add_patch(sub_emb)
ax.text(1.0, 8.7, "1. Coordinate Embedding", fontsize=8.5, fontweight='bold', color='#38bdf8')
ax.text(1.0, 8.25, r"$e_i^t = \phi(x_i^t, y_i^t; W_e)$", fontsize=9, color='#ffffff')
ax.text(1.0, 7.8, r"Dense Linear (2 $\to$ 64) + ReLU", fontsize=7.8, color='#94a3b8')
ax.text(1.0, 7.5, r"Dropout ($p = 0.5$) for small $N$", fontsize=7.8, color='#64748b')

# Social Pooling Stream Box (Right)
sub_soc = FancyBboxPatch((4.45, 6.7), 3.3, 2.3, boxstyle="round,pad=0.03,rounding_size=0.1",
                         facecolor="#1e293b", edgecolor="#0ea5e9", linewidth=1.2)
ax.add_patch(sub_soc)
ax.text(4.6, 8.7, "2. Social Grid Pooling", fontsize=8.5, fontweight='bold', color='#38bdf8')
ax.text(4.6, 8.35, r"Window: $10 \times 10$ m, $32 \times 32$ grid", fontsize=7.8, color='#e2e8f0')
ax.text(4.6, 7.95, r"$H_i^t(m, n) = \sum_{j \neq i} \mathbf{1}_{mn} h_j^{t-1}$", fontsize=8.5, color='#ffffff')
ax.text(4.6, 7.55, r"Sum-pool $8 \times 8 \to 4 \times 4 \times 128$", fontsize=7.8, color='#cbd5e1')
ax.text(4.6, 7.15, r"$a_i^t = \text{MLP}(H_i^t; W_a) \in \mathbb{R}^{64}$", fontsize=8.5, color='#38bdf8')
ax.text(4.6, 6.85, "Captures neighbor interaction", fontsize=7.5, color='#64748b')

# Draw mini grid illustration inside social box
grid_x0, grid_y0 = 6.85, 7.85
grid_sz = 0.75
ax.add_patch(patches.Rectangle((grid_x0, grid_y0), grid_sz, grid_sz, facecolor='#0f172a', edgecolor='#0284c7', linewidth=0.8))
for gi in range(1, 4):
    ax.plot([grid_x0 + gi*grid_sz/4, grid_x0 + gi*grid_sz/4], [grid_y0, grid_y0 + grid_sz], color='#1e3a5f', lw=0.6)
    ax.plot([grid_x0, grid_x0 + grid_sz], [grid_y0 + gi*grid_sz/4, grid_y0 + gi*grid_sz/4], color='#1e3a5f', lw=0.6)
# Center player i
ax.add_patch(Circle((grid_x0 + grid_sz/2, grid_y0 + grid_sz/2), 0.05, facecolor='#38bdf8', edgecolor='white', lw=0.5))
# Neighbors j
ax.add_patch(Circle((grid_x0 + grid_sz*0.25, grid_y0 + grid_sz*0.75), 0.04, facecolor='#f59e0b', edgecolor='none'))
ax.add_patch(Circle((grid_x0 + grid_sz*0.75, grid_y0*1.02), 0.04, facecolor='#ef4444', edgecolor='none'))

# Arrows converging to LSTM
arr_emb = FancyArrowPatch((2.47, 7.3), (3.6, 6.2), arrowstyle='->', mutation_scale=12, color='#38bdf8', linewidth=1.4)
arr_soc = FancyArrowPatch((6.1, 6.7), (5.0, 6.2), arrowstyle='->', mutation_scale=12, color='#38bdf8', linewidth=1.4)
ax.add_patch(arr_emb)
ax.add_patch(arr_soc)

# C. Shared LSTM Cell
sub_lstm = FancyBboxPatch((1.6, 4.6), 5.4, 1.55, boxstyle="round,pad=0.03,rounding_size=0.12",
                          facecolor="#1e1b4b", edgecolor="#6366f1", linewidth=1.5)
ax.add_patch(sub_lstm)
ax.text(1.8, 5.85, "Shared LSTM Trajectory Cell (D = 128)", fontsize=9.5, fontweight='bold', color='#a5b4fc')
ax.text(1.8, 5.45, r"Input: $[e_i^t \,\|\, a_i^t] \in \mathbb{R}^{128}$ (Positional + Social Context)", fontsize=8.5, color='#ffffff')
ax.text(1.8, 5.08, r"Hidden State: $h_i^t = \text{LSTM}(h_i^{t-1}, [e_i^t, a_i^t]; W_l)$", fontsize=8.8, color='#e0e7ff')
ax.text(1.8, 4.78, "Shared weights across all $N$ players (Parameter parsimony)", fontsize=8, color='#c7d2fe')

# Arrow down to output head
arr_lstm = FancyArrowPatch((4.3, 4.6), (4.3, 4.0), arrowstyle='->', mutation_scale=14, color='#6366f1', linewidth=1.6)
ax.add_patch(arr_lstm)

# D. Bivariate Gaussian Prediction Head
sub_head = FancyBboxPatch((0.85, 2.2), 6.9, 1.8, boxstyle="round,pad=0.03,rounding_size=0.1",
                          facecolor="#1e293b", edgecolor="#f59e0b", linewidth=1.3)
ax.add_patch(sub_head)
ax.text(1.05, 3.65, "Bivariate Gaussian Output Head (Eqs. 3–4)", fontsize=9.5, fontweight='bold', color='#fbbf24')
ax.text(1.05, 3.25, r"Dense Linear: $[\mu_x, \mu_y, \log \sigma_x, \log \sigma_y, \text{atanh} \rho]_i^{t+1} = W_p h_i^t$", fontsize=8.8, color='#ffffff')
ax.text(1.05, 2.85, r"Next displacement: $\Delta p_i^{t+1} \sim \mathcal{N}_2(\boldsymbol{\mu}_i^{t+1}, \boldsymbol{\Sigma}_i^{t+1})$", fontsize=8.8, color='#fef3c7')
ax.text(1.05, 2.45, r"Optimization: NLL loss with Teacher Forcing ($t \in 26..30$)", fontsize=8, color='#94a3b8')

# E. Latent State Extraction Bridge
sub_bridge = FancyBboxPatch((0.85, 0.75), 6.9, 1.15, boxstyle="round,pad=0.03,rounding_size=0.08",
                            facecolor="#0f172a", edgecolor="#38bdf8", linewidth=1.1, linestyle='--')
ax.add_patch(sub_bridge)
ax.text(1.05, 1.55, "Feature Bridge to Conformal Scale Model (Sug. 4)", fontsize=8.5, fontweight='bold', color='#38bdf8')
ax.text(1.05, 1.22, r"Freezes network $\theta^* \rightarrow$ extracts encoder state $h_i^{(25)}$ & covariates $x_i$", fontsize=8, color='#e2e8f0')
ax.text(1.05, 0.92, r"Feeds out-of-fold residuals to calibrate $\hat{s}_t(x_i)$ without in-sample overfitting", fontsize=7.8, color='#94a3b8')


# ==============================================================================
# SECTION 2: SPATIOTEMPORAL DATA SPLIT DETAILS
# ==============================================================================
# A. Exchangeability Principle Box
sub_exch = FancyBboxPatch((8.55, 8.85), 6.9, 1.55, boxstyle="round,pad=0.03,rounding_size=0.1",
                          facecolor="#1e1b4b", edgecolor="#818cf8", linewidth=1.3)
ax.add_patch(sub_exch)
ax.text(8.75, 10.15, "Statistical Invariant: Play-Level Permutability", fontsize=9.2, fontweight='bold', color='#a5b4fc')
ax.text(8.75, 9.80, "Frames and intra-play trajectories are strongly correlated.", fontsize=8.0, color='#cbd5e1')
ax.text(8.75, 9.50, r"Exchangeable Unit: Entire Offensive Sequence / Shot Event $k \in \{1, \dots, M\}$", fontsize=8.0, fontweight='bold', color='#ffffff')
ax.text(8.75, 9.22, r"$\rightarrow$ Frames and players from same play NEVER leak across splits", fontsize=7.8, color='#fca5a5')
ax.text(8.75, 8.98, r"Split-conformal guarantee: $P(Y_{n+1} \in \hat{C}(X_{n+1})) \geq 1 - \alpha$ requires exchangeable $k$", fontsize=7.5, color='#94a3b8')

# Connector arrow
arr_split = FancyArrowPatch((12.0, 8.85), (12.0, 8.5), arrowstyle='->', mutation_scale=12, color='#818cf8', linewidth=1.4)
ax.add_patch(arr_split)

# B. Partition Breakdown
def draw_split_bar(y, title, pct, events_str, color, desc, formula_str):
    bar = FancyBboxPatch((8.55, y), 6.9, 1.2, boxstyle="round,pad=0.03,rounding_size=0.09",
                         facecolor="#1e293b", edgecolor=color, linewidth=1.2)
    ax.add_patch(bar)
    # Badge
    bg = FancyBboxPatch((8.7, y + 0.72), 2.2, 0.38, boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor=color, edgecolor='none')
    ax.add_patch(bg)
    ax.text(9.8, y + 0.91, f"{title} ({pct}%)", fontsize=8, fontweight='bold', color='#ffffff', ha='center', va='center')
    ax.text(11.1, y + 0.91, events_str, fontsize=8, color='#cbd5e1', va='center')
    ax.text(8.75, y + 0.44, desc, fontsize=7.8, color='#f8fafc')
    ax.text(8.75, y + 0.16, formula_str, fontsize=7.4, color='#94a3b8')

draw_split_bar(7.15, "1. TRAINING", 60, "98 plays (1,960 trajectories)", "#0284c7",
               r"Optimizes Social-LSTM weights $\theta^*$ with data augmentation",
               r"Lateral pitch mirroring ($y \rightarrow 68 - y$) doubles effective sample size")

draw_split_bar(5.75, "2. VALIDATION (OOF)", 15, "24 plays (475 trajectories)", "#f59e0b",
               r"Generates strictly out-of-fold residuals $e_{i,t} = \|p_{i,t} - \hat{p}_{i,t}\|_2$",
               r"Fits Adaptive Scale Model $\log(e_{i,t} + \varepsilon) = g(x_i, t)$ (Sug. 4)")

draw_split_bar(4.35, "3. CALIBRATION", 15, "24 plays (475 trajectories)", "#10b981",
               r"Evaluates play-level loss profiles $L_k(\lambda)$ (Sug. 6 / CRC)",
               r"Solves exact finite-sample infimum $\hat{\lambda}$ for nominal $\alpha = 0.10$")

draw_split_bar(2.95, "4. HELD-OUT TEST", 10, "18 plays (359 trajectories)", "#ec4899",
               "Independent benchmark of empirical simultaneous trajectory coverage",
               "Evaluates trajectory coverage (90.8%), pointwise coverage & region areas")

# C. Cross-Conformal Extension (Sug. 5) Box
sub_cross = FancyBboxPatch((8.55, 0.75), 6.9, 1.95, boxstyle="round,pad=0.03,rounding_size=0.1",
                           facecolor="#172554", edgecolor="#3b82f6", linewidth=1.2)
ax.add_patch(sub_cross)
ax.text(8.75, 2.4, "Cross-Conformal K-Fold Pipeline (Sugestão 5)", fontsize=9.2, fontweight='bold', color='#60a5fa')
ax.text(8.75, 2.05, r"1. Partition $M$ plays into $K=5$ folds strictly by offensive sequence", fontsize=8, color='#e2e8f0')
ax.text(8.75, 1.72, r"2. For fold $k$: train LSTM on $K-1$ folds, fit $\hat{s}_t(x_i)$ on internal split", fontsize=8, color='#cbd5e1')
ax.text(8.75, 1.39, r"3. Evaluate fold $k$ out-of-fold $\rightarrow$ pool all $n$ plays for calibration", fontsize=8, color='#cbd5e1')
ax.text(8.75, 1.05, r"Benefit: 100% of plays contribute to calibration in small-sample regime", fontsize=7.8, color='#93c5fd')


# ==============================================================================
# SECTION 3: ADAPTIVE CONFORMAL REGION & TACTICAL PREDICTION
# ==============================================================================
# A. Adaptive Scale Model (Sug. 4)
sub_s4 = FancyBboxPatch((16.25, 8.75), 6.9, 1.65, boxstyle="round,pad=0.03,rounding_size=0.1",
                        facecolor="#1e293b", edgecolor="#10b981", linewidth=1.3)
ax.add_patch(sub_s4)
ax.text(16.45, 10.10, "Sugestão 4: Escala Adaptativa Condicional", fontsize=9.2, fontweight='bold', color='#34d399')
ax.text(16.45, 9.75, r"Substitui a forma estática $\hat{s}_t = \text{mediana}(e_t)$ por função condicional:", fontsize=8.0, color='#e2e8f0')
ax.text(16.45, 9.42, r"$\log(e_{i,t} + \varepsilon) = g(x_i, t) + \text{erro} \rightarrow \hat{s}_t(x_i) = \exp\{\hat{g}(x_i, t)\}$", fontsize=8.5, color='#ffffff')
ax.text(16.45, 9.12, r"Covariáveis $x_i$ ($t = 25$): velocidade $\|v_i\|$, dist. bola $d_i^{\text{ball}}$, papel tático, $t, t^2$", fontsize=7.7, color='#a7f3d0')
ax.text(16.45, 8.87, r"Escore supremo normalizado: $M_{k,i} = \max_{t \in \{1..5\}} [e_{k,i,t} \,/\, \hat{s}_t(x_{k,i})]$", fontsize=7.9, color='#fef08a')

# B. Conformal Risk Control (Sug. 6)
sub_s6 = FancyBboxPatch((16.25, 6.75), 6.9, 1.85, boxstyle="round,pad=0.03,rounding_size=0.1",
                        facecolor="#1e293b", edgecolor="#f59e0b", linewidth=1.3)
ax.add_patch(sub_s6)
ax.text(16.45, 8.32, "Sugestão 6: Conformal Risk Control (CRC)", fontsize=9.2, fontweight='bold', color='#fbbf24')
ax.text(16.45, 7.98, "Angelopoulos et al. (ICLR 2024); Izbicki (2026, Seção 6):", fontsize=7.9, color='#cbd5e1')
ax.text(16.45, 7.64, r"Perda da jogada: $L_k(\lambda) = \frac{1}{N_k} \sum_{i=1}^{N_k} \mathbf{1}\left( M_{k,i} > \lambda \right)$ (fração que escapa)", fontsize=8.2, color='#ffffff')
ax.text(16.45, 7.32, r"Critério de calibração exato em amostra finita ($n$ jogadas):", fontsize=7.7, color='#cbd5e1')
ax.text(16.45, 7.00, r"$\hat{\lambda} = \inf \left\{ \lambda \geq 0 : \frac{n}{n+1} \bar{L}(\lambda) + \frac{1}{n+1} \leq \alpha \right\} \rightarrow \hat{\lambda} = 2.597$", fontsize=8.3, color='#fef08a')

# C. Mini Football Pitch Graphic with Adaptive Regions
pitch_x0, pitch_y0 = 16.35, 1.45
pitch_w, pitch_h = 6.7, 5.45

# Tactical pitch card
sub_pitch = FancyBboxPatch((pitch_x0, pitch_y0), pitch_w, pitch_h, boxstyle="round,pad=0.03,rounding_size=0.12",
                           facecolor="#0f172a", edgecolor="#22c55e", linewidth=1.4)
ax.add_patch(sub_pitch)

ax.text(pitch_x0 + 0.2, pitch_y0 + pitch_h - 0.35, "Adaptive Conformal Prediction Tube on Pitch (105 x 68 m)",
        fontsize=9, fontweight='bold', color='#4ade80')
ax.text(pitch_x0 + 0.2, pitch_y0 + pitch_h - 0.65, r"Individual radius: $r_{i,t} = \hat{\lambda} \cdot \hat{s}_t(x_i) \Rightarrow \mathbb{E}[L_{\text{new}}(\hat{\lambda})] \leq \alpha$ (Coverage: 90.8%)",
        fontsize=8, color='#cbd5e1')

# Draw miniature tactical pitch
px_min, px_max = pitch_x0 + 0.35, pitch_x0 + pitch_w - 0.35
py_min, py_max = pitch_y0 + 0.35, pitch_y0 + pitch_h - 0.95
pw = px_max - px_min
ph = py_max - py_min

# Green turf background
turf = FancyBboxPatch((px_min, py_min), pw, ph, boxstyle="round,pad=0.01,rounding_size=0.06",
                      facecolor="#064e3b", edgecolor="#10b981", linewidth=1.2)
ax.add_patch(turf)

# Pitch markings
ax.plot([px_min, px_max, px_max, px_min, px_min], [py_min, py_min, py_max, py_max, py_min], color='#a7f3d0', lw=0.9, alpha=0.8)
# Halfway line & center circle
ax.plot([(px_min+px_max)/2, (px_min+px_max)/2], [py_min, py_max], color='#a7f3d0', lw=0.8, alpha=0.7)
ax.add_patch(Circle(((px_min+px_max)/2, (py_min+py_max)/2), ph*0.22, fill=False, edgecolor='#a7f3d0', lw=0.8, alpha=0.7))
# Penalty box left
ax.plot([px_min, px_min + pw*0.16, px_min + pw*0.16, px_min],
        [(py_min+py_max)/2 - ph*0.35, (py_min+py_max)/2 - ph*0.35, (py_min+py_max)/2 + ph*0.35, (py_min+py_max)/2 + ph*0.35],
        color='#a7f3d0', lw=0.8, alpha=0.7)
# Penalty box right
ax.plot([px_max, px_max - pw*0.16, px_max - pw*0.16, px_max],
        [(py_min+py_max)/2 - ph*0.35, (py_min+py_max)/2 - ph*0.35, (py_min+py_max)/2 + ph*0.35, (py_min+py_max)/2 + ph*0.35],
        color='#a7f3d0', lw=0.8, alpha=0.7)

# Player 1: Fast Attacker (High velocity sprint) -> Large adaptive expanding circles
p1_obs_x = [px_min + pw*0.40, px_min + pw*0.48, px_min + pw*0.56]
p1_obs_y = [py_min + ph*0.62, py_min + ph*0.64, py_min + ph*0.66]
ax.plot(p1_obs_x, p1_obs_y, color='#ffffff', lw=1.2, ls='--', alpha=0.9)
ax.scatter([p1_obs_x[-1]], [p1_obs_y[-1]], s=30, color='#ffffff', edgecolor='#0f172a', zorder=5)

p1_pred_x = np.linspace(p1_obs_x[-1], px_min + pw*0.88, 5)
p1_pred_y = np.linspace(p1_obs_y[-1], py_min + ph*0.75, 5)
p1_radii = [0.18, 0.32, 0.48, 0.65, 0.82]  # Adaptive: large expanding radii
ax.plot(p1_pred_x, p1_pred_y, color='#00f5d4', lw=2.0, zorder=6)

# Ground truth path for player 1
p1_gt_x = p1_pred_x + np.array([0, 0.04, 0.09, 0.14, 0.20])
p1_gt_y = p1_pred_y + np.array([0, -0.05, -0.12, -0.18, -0.22])
ax.plot(p1_gt_x, p1_gt_y, color='#f97316', lw=1.8, ls='-', zorder=6)
ax.scatter(p1_gt_x, p1_gt_y, s=18, color='#f97316', zorder=7)

for pt, pr in zip(range(5), p1_radii):
    circ = Circle((p1_pred_x[pt], p1_pred_y[pt]), pr, facecolor='#22c55e', edgecolor='#4ade80', alpha=0.15, lw=1.0, zorder=4)
    ax.add_patch(circ)

ax.text(p1_pred_x[2], p1_pred_y[2] + 0.65, "Attacker Sprint (v = 7.8 m/s)\nAdaptive: $r_5 = 38.8$ m (Rescues GT)",
        fontsize=7.2, color='#fef08a', fontweight='bold', ha='center')

# Player 2: Stationary / Positional Defender (Low velocity) -> Tight compact circles
p2_obs_x = [px_min + pw*0.30, px_min + pw*0.31, px_min + pw*0.32]
p2_obs_y = [py_min + ph*0.28, py_min + ph*0.27, py_min + ph*0.26]
ax.plot(p2_obs_x, p2_obs_y, color='#ffffff', lw=1.2, ls='--', alpha=0.9)
ax.scatter([p2_obs_x[-1]], [p2_obs_y[-1]], s=30, color='#ffffff', edgecolor='#0f172a', zorder=5)

p2_pred_x = np.linspace(p2_obs_x[-1], px_min + pw*0.44, 5)
p2_pred_y = np.linspace(p2_obs_y[-1], py_min + ph*0.22, 5)
p2_radii = [0.08, 0.12, 0.17, 0.22, 0.28]  # Adaptive: small compact radii
ax.plot(p2_pred_x, p2_pred_y, color='#00f5d4', lw=2.0, zorder=6)

p2_gt_x = p2_pred_x + np.array([0, 0.02, 0.04, 0.05, 0.07])
p2_gt_y = p2_pred_y + np.array([0, -0.01, -0.03, -0.04, -0.05])
ax.plot(p2_gt_x, p2_gt_y, color='#f97316', lw=1.8, ls='-', zorder=6)
ax.scatter(p2_gt_x, p2_gt_y, s=18, color='#f97316', zorder=7)

for pt, pr in zip(range(5), p2_radii):
    circ = Circle((p2_pred_x[pt], p2_pred_y[pt]), pr, facecolor='#22c55e', edgecolor='#4ade80', alpha=0.22, lw=1.0, zorder=4)
    ax.add_patch(circ)

ax.text(p2_pred_x[2], p2_pred_y[2] - 0.45, "Positional Defender (v = 0.04 m/s)\nAdaptive: $r_5 = 12.9$ m (55% Area Saved)",
        fontsize=7.2, color='#67e8f9', fontweight='bold', ha='center')

# Legend for pitch
leg_y = pitch_y0 + 0.45
ax.plot([pitch_x0 + 0.4, pitch_x0 + 0.8], [leg_y, leg_y], color='#00f5d4', lw=2.0)
ax.text(pitch_x0 + 0.85, leg_y, "Predicted Path", fontsize=7.2, color='#cbd5e1', va='center')

ax.plot([pitch_x0 + 2.3, pitch_x0 + 2.7], [leg_y, leg_y], color='#f97316', lw=2.0)
ax.text(pitch_x0 + 2.75, leg_y, "Ground Truth", fontsize=7.2, color='#cbd5e1', va='center')

ax.add_patch(Circle((pitch_x0 + 4.4, leg_y), 0.12, facecolor='#22c55e', edgecolor='#4ade80', alpha=0.4))
ax.text(pitch_x0 + 4.65, leg_y, "Adaptive Conformal Region", fontsize=7.2, color='#cbd5e1', va='center')


# ==============================================================================
# PIPELINE CONNECTING ARROWS BETWEEN STAGES
# ==============================================================================
# Arrow Stage 1 -> Stage 2
arr_1_to_2 = FancyArrowPatch((8.0, 6.2), (8.3, 6.2), arrowstyle='simple,tail_width=3.5,head_width=9,head_length=8',
                             facecolor='#0284c7', edgecolor='#38bdf8', lw=0.8)
ax.add_patch(arr_1_to_2)
ax.text(8.15, 6.6, "Weights\n$\\theta^*$", fontsize=7.5, fontweight='bold', color='#38bdf8', ha='center')

# Arrow Stage 2 -> Stage 3
arr_2_to_3 = FancyArrowPatch((15.7, 6.2), (16.0, 6.2), arrowstyle='simple,tail_width=3.5,head_width=9,head_length=8',
                             facecolor='#6366f1', edgecolor='#818cf8', lw=0.8)
ax.add_patch(arr_2_to_3)
ax.text(15.85, 6.6, "OOF Resids\n& $\\hat{\\lambda}$", fontsize=7.5, fontweight='bold', color='#a5b4fc', ha='center')


# ==============================================================================
# FOOTER BANNER
# ==============================================================================
footer = FancyBboxPatch((0.6, 0.08), 22.8, 0.35,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        facecolor="#0f172a", edgecolor="#1e293b", linewidth=1.0)
ax.add_patch(footer)
ax.text(1.0, 0.25, "Key Guarantees: Finite-sample validity $P(\\text{Trajectory miscoverage}) \\leq \\alpha$ via play-level exchangeability  ✦  Heteroscedastic contextual adaptation: $r_{i,t} = \\hat{\\lambda} \\cdot \\hat{s}_t(x_i)$",
        fontsize=8.2, color='#94a3b8', va='center')
ax.text(23.0, 0.25, "Dissertação de Mestrado — Estatística Espaçotemporal", fontsize=8.2, color='#38bdf8', ha='right', va='center')

# Save outputs
os.makedirs("output", exist_ok=True)
os.makedirs("docs", exist_ok=True)

out_png = "output/social_lstm_conformal_pipeline_architecture.png"
docs_png = "docs/social_lstm_conformal_pipeline_architecture.png"

plt.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.savefig(docs_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
print(f"Successfully generated pipeline architecture graphic:")
print(f"  -> {out_png}")
print(f"  -> {docs_png}")
