# Football Trajectory Prediction with Conformal Regions

## Introduction

Predicting where every player will move next is central to football analytics. We build upon pedestrian trajectory forecasting architectures to propose an alternative for football. Furthermore, trajectory forecasting is rarely quantified with uncertainty. We incorporate uncertainty quantification using conformal prediction. Specifically, we train a dynamic social-decoder network on roughly one hundred plays to forecast all outfield players five seconds ahead, and wrap it in conformal prediction to obtain finite-sample regions for entire trajectories. Broadcast graphics, tactical dashboards, and scouting tools need both accurate forecasts and coverage guarantees.

## Methods

**Data.** Open Bundesliga tracking data: seven matches, ten teams, 164 open-play shot plays, 3,253 player trajectories. Each play provides 25 observed seconds plus a 5-second horizon, orientation-normalized; plays split 60/15/15/10 into train/validation/calibration/test prevents frame leakage across sets.

**Architecture.** The model adapts pedestrian-trajectory predictors (Social-BiGAT/STGAT, Kosaraju et al., 2019; Huang et al., 2019) to deterministic forecasting: a spatiotemporal encoder (per-frame multi-head attention, then LSTM) and an autoregressive decoder recomputing graph attention over all players' states at each future step, predicting residual displacement against a constant-velocity prior; lateral mirroring doubles training data.

**Conformal regions.** Around the point predictor, for play $k$, player $i$, horizon $t$, define the error $e_{i,t}=\|p_i(t)-\hat p_i(t)\|_2$ and trajectory-level nonconformity score $R_{k,i}=\max_t e_{i,t}/\hat s_t(x_i)$. The band shape $\hat s_t$ is estimated from out-of-fold training residuals; the adaptive scale $\hat s_t(x_i)=\exp\{\hat g(x_i,t)\}$ uses ridge regression on the frozen latent state $h_i(T_{\mathrm{obs}})$ plus horizon $t$. Calibration yields the order statistic $\hat q=R_{(\lceil(n+1)(1-\alpha)\rceil)}$, $\alpha=0.10$, giving regions $C_t=\{p:\|p-\hat p(t)\|_2\le\hat q\,\hat s_t(x_i)\}$. Trajectories within a play being dependent, the calibration unit is the (play, player) pair; exact finite-sample control follows from conformal risk control (Angelopoulos et al. 2024) and cross-conformal (Vovk 2015; Barber et al. 2021) with five play folds. Region efficiency is measured by mean area $\pi r^2$.

## Results

On validation, the final model attains average displacement error (ADE) 4.06 m and final displacement error (FDE) 6.55 m; on the held-out test set, ADE 3.86 m and FDE 6.28 m, improving both attacking (3.78/6.25 m) and defending (3.93/6.31 m) players. At $\alpha=0.10$, the split-conformal pipeline reaches 88.9% empirical simultaneous coverage (target 90%) with $\hat q=3.42$; with only 24 calibration plays, however, $\hat q$ is a noisy statistic, motivating the refinements below. The four conformal variants were compared head-to-head: [FALTA PREENCHER].

## Conclusion

We adapt multi-agent trajectory prediction to football under limited data and equip forecasts with distribution-free, play-level uncertainty. The dynamic social decoder delivers accurate five-second forecasts, while conformal refinements turn them into principled regions. For practitioners, calibrated confidence bands around player trajectories support broadcast, scouting, and decision tools, offering a template for small-sample, interdependent-trajectory uncertainty quantification.
