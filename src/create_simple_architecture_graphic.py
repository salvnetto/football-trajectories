#!/usr/bin/env python3
"""
Simple & Clear Pipeline Diagram: Social-LSTM Architecture + Data Split + Conformal Region
Clean, minimal, high-contrast, pedagogical style (matching Image 2 & Image 3).
Explicitly shows where covariates enter the architecture and the conformal model.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
import os

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# Canvas: 24 x 13.5 inches at 300 DPI, clean white background
fig = plt.figure(figsize=(24, 13.5), facecolor='#ffffff')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 24)
ax.set_ylim(0, 13.5)
ax.axis('off')

# ==============================================================================
# MAIN TITLE (Simple, No Citations or Clutter)
# ==============================================================================
ax.text(1.2, 12.7, "Pipeline de Modelagem: Arquitetura, Divisão dos Dados e Região Conforme",
        fontsize=18, fontweight='bold', color='#0f172a', va='center')
ax.text(1.2, 12.3, "Fluxo completo de previsão de trajetórias no futebol com garantias de cobertura estatística",
        fontsize=11, color='#64748b', va='center')


# ==============================================================================
# 3 MAIN STAGE COLUMNS
# ==============================================================================
col_y = 0.8
col_h = 11.1

# Container 1: Arquitetura
c1 = FancyBboxPatch((0.8, col_y), 7.6, col_h, boxstyle="round,pad=0.05,rounding_size=0.15",
                    facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
ax.add_patch(c1)

# Container 2: Divisão dos Dados
c2 = FancyBboxPatch((8.7, col_y), 6.8, col_h, boxstyle="round,pad=0.05,rounding_size=0.15",
                    facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
ax.add_patch(c2)

# Container 3: Região Conforme
c3 = FancyBboxPatch((15.8, col_y), 7.4, col_h, boxstyle="round,pad=0.05,rounding_size=0.15",
                    facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
ax.add_patch(c3)

# Column Header Helper (Pill badge style like Image 3)
def draw_col_header(x, y, badge_text, badge_color, title_text):
    badge = FancyBboxPatch((x, y), 2.2, 0.45, boxstyle="round,pad=0.03,rounding_size=0.1",
                           facecolor=badge_color[0], edgecolor=badge_color[1], linewidth=1.2)
    ax.add_patch(badge)
    ax.text(x + 1.1, y + 0.22, badge_text, fontsize=9, fontweight='bold', color=badge_color[2], ha='center', va='center')
    ax.text(x + 2.4, y + 0.22, title_text, fontsize=13, fontweight='bold', color='#0f172a', va='center')

draw_col_header(1.1, 11.2, "PARTE 1", ("#e0f2fe", "#7dd3fc", "#0369a1"), "Arquitetura da Rede")
draw_col_header(9.0, 11.2, "PARTE 2", ("#ede9fe", "#c4b5fd", "#5b21b6"), "Divisão dos Dados")
draw_col_header(16.1, 11.2, "PARTE 3", ("#dcfce7", "#86efac", "#166534"), "Região Conforme")


# ==============================================================================
# PARTE 1: ARQUITETURA DA REDE (CLARA, PEDAGÓGICA, COVARIÁVEIS EXPLÍCITAS)
# ==============================================================================
# Explicando o fluxo de forma simples:
# 1. ENTRADA: Covariáveis do jogador e da bola
# 2. EMBEDDING: Projeção densa das características
# 3. SOCIAL POOLING: Influência dos outros jogadores
# 4. LSTM: Memória temporal
# 5. SAÍDA: Próxima posição (x, y) e incerteza

# A. BLOCO 1: ENTRADA / COVARIÁVEIS
sub_in = FancyBboxPatch((1.1, 8.4), 7.0, 2.5, boxstyle="round,pad=0.04,rounding_size=0.1",
                        facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.2)
ax.add_patch(sub_in)
ax.text(1.3, 10.6, "1. Variáveis de Entrada (por Jogador i no instante t)", fontsize=10, fontweight='bold', color='#0f172a')

# 4 caixas claras para cada grupo de covariáveis
def draw_covar_pill(x, y, w, h, title, details, color):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor=color[0], edgecolor=color[1], linewidth=1.1)
    ax.add_patch(p)
    ax.text(x + 0.15, y + h - 0.22, title, fontsize=8.5, fontweight='bold', color=color[2])
    ax.text(x + 0.15, y + 0.18, details, fontsize=7.8, color='#334155')

draw_covar_pill(1.3, 9.4, 3.2, 0.95, "Coordenadas Espaciais", "Posição do jogador: (x, y) em metros", ("#e0f2fe", "#38bdf8", "#0369a1"))
draw_covar_pill(4.7, 9.4, 3.2, 0.95, "Cinemática do Jogador", "Velocidade: (vel_x, vel_y) e aceleração", ("#fef3c7", "#f59e0b", "#b45309"))
draw_covar_pill(1.3, 8.55, 3.2, 0.75, "Contexto da Bola", "Distância à bola + vel. da bola", ("#ffedd5", "#fb923c", "#c2410c"))
draw_covar_pill(4.7, 8.55, 3.2, 0.75, "Papel Tático (role)", "Ataque (posse) ou Defesa", ("#f3e8ff", "#c084fc", "#7e22ce"))

# Seta para baixo
arr_in = FancyArrowPatch((4.6, 8.4), (4.6, 7.8), arrowstyle='->', mutation_scale=14, color='#0284c7', linewidth=2.0)
ax.add_patch(arr_in)
ax.text(4.75, 8.1, "Vetor de características x_i", fontsize=8.2, color='#0284c7', fontweight='bold')

# B. BLOCO 2: COMBINAÇÃO DAS COVARIÁVEIS + INFLUÊNCIA SOCIAL (LADO A LADO)
# Caixa da Esquerda: Embedding das Covariáveis
box_emb = FancyBboxPatch((1.1, 5.7), 3.2, 1.9, boxstyle="round,pad=0.03,rounding_size=0.1",
                         facecolor="#f0f9ff", edgecolor="#0284c7", linewidth=1.3)
ax.add_patch(box_emb)
ax.text(1.25, 7.35, "2A. Embedding do Jogador", fontsize=9.2, fontweight='bold', color='#0369a1')
ax.text(1.25, 6.95, "Camada Densa (MLP + ReLU)", fontsize=8.2, color='#0f172a')
ax.text(1.25, 6.6, "Funde coordenadas +", fontsize=8, color='#475569')
ax.text(1.25, 6.35, "velocidade + bola + papel", fontsize=8, color='#475569')
ax.text(1.25, 5.95, "→ Vetor e_i (64 dimensões)", fontsize=8.5, fontweight='bold', color='#0284c7')

# Caixa da Direita: Social Pooling (Outros Jogadores)
box_soc = FancyBboxPatch((4.7, 5.7), 3.4, 1.9, boxstyle="round,pad=0.03,rounding_size=0.1",
                         facecolor="#f0fdf4", edgecolor="#16a34a", linewidth=1.3)
ax.add_patch(box_soc)
ax.text(4.85, 7.35, "2B. Influência dos Vizinhos", fontsize=9.2, fontweight='bold', color='#15803d')
ax.text(4.85, 6.95, "Social Pooling (S-Pooling)", fontsize=8.2, color='#0f172a')
ax.text(4.85, 6.6, "Coleta o que os outros 19", fontsize=8, color='#475569')
ax.text(4.85, 6.35, "jogadores estão fazendo ao redor", fontsize=8, color='#475569')
ax.text(4.85, 5.95, "→ Vetor social a_i (64 dim.)", fontsize=8.5, fontweight='bold', color='#16a34a')

# Setas convergindo para a Célula LSTM
arr_e = FancyArrowPatch((2.7, 5.7), (3.8, 5.0), arrowstyle='->', mutation_scale=12, color='#0284c7', linewidth=1.8)
arr_a = FancyArrowPatch((6.4, 5.7), (5.4, 5.0), arrowstyle='->', mutation_scale=12, color='#16a34a', linewidth=1.8)
ax.add_patch(arr_e)
ax.add_patch(arr_a)

# C. BLOCO 3: CÉLULA LSTM COMPARTILHADA
box_lstm = FancyBboxPatch((1.1, 3.1), 7.0, 1.8, boxstyle="round,pad=0.04,rounding_size=0.12",
                          facecolor="#fdf4ff", edgecolor="#c084fc", linewidth=1.4)
ax.add_patch(box_lstm)
ax.text(1.3, 4.6, "3. Célula Recorrente (LSTM) — Memória Temporal", fontsize=10, fontweight='bold', color='#7e22ce')
ax.text(1.3, 4.25, "Recebe a junção: [ Vetor de Covariáveis e_i  +  Contexto Social a_i ]  (128 dimensões)", fontsize=8.5, color='#0f172a')
ax.text(1.3, 3.85, "• Atualiza a memória de movimento do jogador ao longo do tempo (t = 1 até 25s)", fontsize=8.2, color='#475569')
ax.text(1.3, 3.5, "• Pesos compartilhados: a mesma rede prevê atacantes e defensores (eficiência de parâmetros)", fontsize=8.2, color='#475569')
ax.text(1.3, 3.2, "→ Produz o Estado Oculto final h_i (representação rica de toda a jogada)", fontsize=8.5, fontweight='bold', color='#9333ea')

# Seta para baixo
arr_lstm = FancyArrowPatch((4.6, 3.1), (4.6, 2.5), arrowstyle='->', mutation_scale=14, color='#9333ea', linewidth=2.0)
ax.add_patch(arr_lstm)

# D. BLOCO 4: CABEÇA DE SAÍDA E PREVISÃO
box_out = FancyBboxPatch((1.1, 1.1), 7.0, 1.3, boxstyle="round,pad=0.04,rounding_size=0.1",
                         facecolor="#fffbeb", edgecolor="#f59e0b", linewidth=1.3)
ax.add_patch(box_out)
ax.text(1.3, 2.15, "4. Previsão dos Próximos 5 Segundos (t = 26 a 30s)", fontsize=9.8, fontweight='bold', color='#b45309')
ax.text(1.3, 1.75, "Camada Linear 128 → 5 parâmetros da distribuição Gaussiana Bivariada:", fontsize=8.5, color='#0f172a')
ax.text(1.3, 1.35, "• Posição Média Prevista: (μ_x, μ_y)   • Incerteza do Modelo: (σ_x, σ_y, correlação ρ)", fontsize=8.5, color='#92400e', fontweight='bold')


# ==============================================================================
# PARTE 2: DIVISÃO DOS DADOS (ESTILO IMAGEM 3, SIMPLES E DIRETO)
# ==============================================================================
# Banner de Regra de Ouro
rule_box = FancyBboxPatch((9.0, 9.8), 6.2, 1.1, boxstyle="round,pad=0.03,rounding_size=0.09",
                          facecolor="#fef2f2", edgecolor="#ef4444", linewidth=1.3)
ax.add_patch(rule_box)
ax.text(9.2, 10.6, "Regra Estrita de Amostragem:", fontsize=9.2, fontweight='bold', color='#dc2626')
ax.text(9.2, 10.3, "• O dado é dividido por JOGADA COMPLETA (não por frame)", fontsize=8.2, color='#0f172a')
ax.text(9.2, 10.0, "• Todos os 20 jogadores do mesmo lance ficam no mesmo grupo", fontsize=8.0, color='#475569')

# Total Database Card
tot_box = FancyBboxPatch((9.0, 8.8), 6.2, 0.75, boxstyle="round,pad=0.03,rounding_size=0.08",
                         facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.1)
ax.add_patch(tot_box)
ax.text(9.2, 9.25, "Banco Total: 164 Jogadas de Finalização (Shot Events)", fontsize=8.8, fontweight='bold', color='#0f172a')
ax.text(9.2, 8.95, "Total de 3.280 trajetórias completas rastreadas a 1 Hz", fontsize=7.8, color='#64748b')

# 4 Caixas limpas e coloridas para os 4 conjuntos
def draw_clean_split_card(y, title, pct, n_plays, border_c, bg_c, title_c, desc_line1, desc_line2):
    c = FancyBboxPatch((9.0, y), 6.2, 1.25, boxstyle="round,pad=0.03,rounding_size=0.09",
                       facecolor=bg_c, edgecolor=border_c, linewidth=1.3)
    ax.add_patch(c)
    # Header badge
    ax.text(9.2, y + 0.95, f"{title} ({pct}%)", fontsize=9.2, fontweight='bold', color=title_c)
    ax.text(12.3, y + 0.95, f"{n_plays} jogadas", fontsize=8.2, color='#475569', ha='right')
    ax.text(9.2, y + 0.58, desc_line1, fontsize=8.0, color='#0f172a')
    ax.text(9.2, y + 0.28, desc_line2, fontsize=7.8, color=title_c, fontweight='bold')

draw_clean_split_card(7.3, "1. Conjunto de Treino", 60, "98", "#0284c7", "#f0f9ff", "#0369a1",
                      "Ajusta os pesos da rede Social-LSTM (RMSprop)",
                      "Aumento de dados: espelhamento do campo (y → 68 - y)")

draw_clean_split_card(5.85, "2. Conjunto de Validação", 15, "24", "#f59e0b", "#fffbeb", "#b45309",
                      "Dados não vistos para early stopping do treino",
                      "Gera resíduos out-of-fold para ajustar a Escala ŝ_t(x_i)")

draw_clean_split_card(4.4, "3. Conjunto de Calibração", 15, "24", "#10b981", "#f0fdf4", "#15803d",
                      "Usado para calibrar a garantia conforme (CRC)",
                      "Calcula o multiplicador global λ̂ sem tocar no teste")

draw_clean_split_card(2.95, "4. Conjunto de Teste", 10, "18", "#ec4899", "#fdf2f8", "#be185d",
                      "18 jogadas inéditas (359 trajetórias) para avaliação",
                      "Comprova a cobertura simultânea da trajetória: 90.8%")

# Resumo Cross-Conformal no final da coluna
cross_box = FancyBboxPatch((9.0, 1.1), 6.2, 1.6, boxstyle="round,pad=0.03,rounding_size=0.08",
                           facecolor="#f5f3ff", edgecolor="#8b5cf6", linewidth=1.2)
ax.add_patch(cross_box)
ax.text(9.2, 2.45, "Alternativa: Cross-Conformal em 5 Dobras", fontsize=8.8, fontweight='bold', color='#6d28d9')
ax.text(9.2, 2.15, "• Divide as jogadas em 5 dobras por lance", fontsize=7.8, color='#334155')
ax.text(9.2, 1.85, "• Treina em 4 dobras e avalia na dobra restante", fontsize=7.8, color='#334155')
ax.text(9.2, 1.55, "• Junta todas as dobras para calibrar λ̂ com 100% dos dados", fontsize=7.8, color='#6d28d9', fontweight='bold')
ax.text(9.2, 1.25, "• Ideal para regimes de dados limitados (amostras pequenas)", fontsize=7.5, color='#64748b')


# ==============================================================================
# PARTE 3: REGIÃO CONFORME ADAPTATIVA (ONDE AS COVARIÁVEIS ENTRAM!)
# ==============================================================================
# A. Explicação Clara: Como as Covariáveis entram na Região Conforme
scale_box = FancyBboxPatch((16.1, 8.85), 7.0, 2.05, boxstyle="round,pad=0.04,rounding_size=0.1",
                           facecolor="#f0fdf4", edgecolor="#16a34a", linewidth=1.3)
ax.add_patch(scale_box)
ax.text(16.3, 10.6, "1. Onde as Covariáveis entram na Região Conforme?", fontsize=9.8, fontweight='bold', color='#15803d')
ax.text(16.3, 10.25, "O raio da região NÃO é fixo. Ele depende das covariáveis x_i do jogador:", fontsize=8.2, color='#0f172a')

# Fórmula limpa e explicada
f_box = FancyBboxPatch((16.3, 9.45), 6.6, 0.65, boxstyle="round,pad=0.02,rounding_size=0.06",
                       facecolor="#ffffff", edgecolor="#86efac", linewidth=1.1)
ax.add_patch(f_box)
ax.text(19.6, 9.77, r"Raio Adaptativo:   $r_{i,t} = \hat{\lambda} \cdot \hat{s}_t(x_i)$",
        fontsize=10.5, fontweight='bold', color='#15803d', ha='center', va='center')

ax.text(16.3, 9.15, "• λ̂ = 2.597: Multiplicador global fixado na calibração (CRC)", fontsize=8.0, color='#334155')
ax.text(16.3, 8.92, "• ŝ_t(x_i) = exp( modelo com velocidade, distância da bola e papel )", fontsize=8.0, color='#15803d', fontweight='bold')

# B. Lógica de Adaptação (Contraste Velocidade)
logic_box = FancyBboxPatch((16.1, 6.95), 7.0, 1.7, boxstyle="round,pad=0.03,rounding_size=0.09",
                           facecolor="#fffbeb", edgecolor="#f59e0b", linewidth=1.2)
ax.add_patch(logic_box)
ax.text(16.3, 8.35, "2. Como as Covariáveis mudam o tamanho da Região:", fontsize=9.2, fontweight='bold', color='#b45309')

# Caso Rápido
ax.text(16.3, 7.95, "▲ Jogador Rápido (Sprint, v > 7 m/s):", fontsize=8.5, fontweight='bold', color='#b45309')
ax.text(16.5, 7.68, "Incerteza física alta → ŝ_t cresce → Raio maior (até 38 m)", fontsize=8.0, color='#0f172a')

# Caso Lento
ax.text(16.3, 7.32, "▼ Jogador Lento / Posicional (v < 0.2 m/s):", fontsize=8.5, fontweight='bold', color='#0369a1')
ax.text(16.5, 7.05, "Movimento previsível → ŝ_t diminui → Raio compacto (12 m, economiza 55% de área)", fontsize=8.0, color='#0f172a')

# C. Mini Campo com o Desenho dos Círculos Adaptativos
pitch_card = FancyBboxPatch((16.1, 1.1), 7.0, 5.65, boxstyle="round,pad=0.03,rounding_size=0.1",
                            facecolor="#ffffff", edgecolor="#10b981", linewidth=1.4)
ax.add_patch(pitch_card)

ax.text(16.3, 6.45, "3. Visualização Prática no Campo (105 x 68 m):", fontsize=9.2, fontweight='bold', color='#0f172a')

# Desenho do gramado
gx, gy = 16.35, 1.7
gw, gh = 6.5, 4.5

turf = FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.01,rounding_size=0.06",
                      facecolor="#059669", edgecolor="#047857", linewidth=1.2)
ax.add_patch(turf)

# Linhas do campo
ax.plot([gx, gx + gw, gx + gw, gx, gx], [gy, gy, gy + gh, gy + gh, gy], color='#ffffff', lw=1.0)
ax.plot([gx + gw/2, gx + gw/2], [gy, gy + gh], color='#ffffff', lw=0.9)
ax.add_patch(Circle((gx + gw/2, gy + gh/2), gh*0.22, fill=False, edgecolor='#ffffff', lw=0.9))
# Áreas
ax.plot([gx, gx + gw*0.18, gx + gw*0.18, gx], [gy + gh*0.2, gy + gh*0.2, gy + gh*0.8, gy + gh*0.8], color='#ffffff', lw=0.8)
ax.plot([gx + gw, gx + gw*0.82, gx + gw*0.82, gx + gw], [gy + gh*0.2, gy + gh*0.2, gy + gh*0.8, gy + gh*0.8], color='#ffffff', lw=0.8)

# Atacante Rápido (Sprint)
p1_x = np.linspace(gx + gw*0.48, gx + gw*0.88, 5)
p1_y = np.linspace(gy + gh*0.68, gy + gh*0.78, 5)
p1_r = [0.22, 0.38, 0.55, 0.72, 0.90]

# Trajetória prevista (azul claro) e real (laranja)
ax.plot(p1_x, p1_y, color='#00f5d4', lw=2.2, zorder=6)
p1_gt_x = p1_x + np.array([0, 0.04, 0.09, 0.15, 0.22])
p1_gt_y = p1_y + np.array([0, -0.05, -0.12, -0.18, -0.24])
ax.plot(p1_gt_x, p1_gt_y, color='#ff7a00', lw=2.0, zorder=6)
ax.scatter(p1_gt_x, p1_gt_y, s=20, color='#ff7a00', zorder=7)

for pt in range(5):
    c = Circle((p1_x[pt], p1_y[pt]), p1_r[pt], facecolor='#10b981', edgecolor='#6ee7b7', alpha=0.20, lw=1.2, zorder=4)
    ax.add_patch(c)

# Badge texto Atacante
b1 = FancyBboxPatch((gx + gw*0.46, gy + gh*0.85), 2.8, 0.45, boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor="#064e3b", edgecolor="#34d399", lw=0.8, zorder=8)
ax.add_patch(b1)
ax.text(gx + gw*0.60, gy + gh*0.85 + 0.22, "Atacante em Sprint: Círculos Maiores", fontsize=7.2, color='#ffffff', fontweight='bold', ha='center', va='center', zorder=9)

# Defensor Lento (Posicional)
p2_x = np.linspace(gx + gw*0.30, gx + gw*0.44, 5)
p2_y = np.linspace(gy + gh*0.32, gy + gh*0.25, 5)
p2_r = [0.09, 0.14, 0.19, 0.24, 0.30]

ax.plot(p2_x, p2_y, color='#00f5d4', lw=2.2, zorder=6)
p2_gt_x = p2_x + np.array([0, 0.02, 0.04, 0.06, 0.08])
p2_gt_y = p2_y + np.array([0, -0.01, -0.03, -0.04, -0.05])
ax.plot(p2_gt_x, p2_gt_y, color='#ff7a00', lw=2.0, zorder=6)
ax.scatter(p2_gt_x, p2_gt_y, s=20, color='#ff7a00', zorder=7)

for pt in range(5):
    c = Circle((p2_x[pt], p2_y[pt]), p2_r[pt], facecolor='#10b981', edgecolor='#6ee7b7', alpha=0.30, lw=1.2, zorder=4)
    ax.add_patch(c)

# Badge texto Defensor
b2 = FancyBboxPatch((gx + gw*0.18, gy + gh*0.12), 3.0, 0.45, boxstyle="round,pad=0.02,rounding_size=0.06",
                    facecolor="#064e3b", edgecolor="#34d399", lw=0.8, zorder=8)
ax.add_patch(b2)
ax.text(gx + gw*0.33, gy + gh*0.12 + 0.22, "Defensor Lento: Círculos Compactos", fontsize=7.2, color='#ffffff', fontweight='bold', ha='center', va='center', zorder=9)

# Legenda simples abaixo do campo
ax.plot([gx + 0.4, gx + 0.8], [gy - 0.35, gy - 0.35], color='#00f5d4', lw=2.5)
ax.text(gx + 0.9, gy - 0.35, "Trajetória Prevista", fontsize=7.8, color='#0f172a', va='center')

ax.plot([gx + 2.5, gx + 2.9], [gy - 0.35, gy - 0.35], color='#ff7a00', lw=2.5)
ax.text(gx + 3.0, gy - 0.35, "Trajetória Real", fontsize=7.8, color='#0f172a', va='center')

ax.add_patch(Circle((gx + 4.6, gy - 0.35), 0.12, facecolor='#10b981', edgecolor='#059669', alpha=0.4))
ax.text(gx + 4.85, gy - 0.35, "Região Conforme (r)", fontsize=7.8, color='#0f172a', va='center')


# ==============================================================================
# SETAS CONECTANDO AS 3 PARTES
# ==============================================================================
# Seta Parte 1 -> Parte 2
arr_1_2 = FancyArrowPatch((8.4, 6.3), (8.7, 6.3), arrowstyle='simple,tail_width=3,head_width=8,head_length=7',
                          facecolor='#0284c7', edgecolor='none')
ax.add_patch(arr_1_2)

# Seta Parte 2 -> Parte 3
arr_2_3 = FancyArrowPatch((15.5, 6.3), (15.8, 6.3), arrowstyle='simple,tail_width=3,head_width=8,head_length=7',
                          facecolor='#10b981', edgecolor='none')
ax.add_patch(arr_2_3)

# Salvar arquivo
os.makedirs("output", exist_ok=True)
os.makedirs("docs", exist_ok=True)

out_png = "output/pipeline_arquitetura_simples.png"
docs_png = "docs/pipeline_arquitetura_simples.png"

plt.savefig(out_png, dpi=300, facecolor='#ffffff', edgecolor='none', bbox_inches='tight')
plt.savefig(docs_png, dpi=300, facecolor='#ffffff', edgecolor='none', bbox_inches='tight')
print(f"Gráfico simplificado gerado com sucesso:")
print(f"  -> {out_png}")
print(f"  -> {docs_png}")
