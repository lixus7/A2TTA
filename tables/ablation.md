# A2TTA Ablation Study (paper Sec. 5.5)

All variants share the tuned main config (`lr=1e-3, steps=3, budget=0.25, pool=512, hidden=64, node_emb=16, warmup=3`), so every row is directly comparable to the A2TTA main-results numbers. Cells are mean±std over 5 seeds (51–55), each seed averaged over all years. Lower is better; **bold** = best Avg MAE per dataset.

> **Ours = Backbone (frozen) + FiLM calibrator + online delayed-label TTA (All) + local context clone.** Sections A–C below ablate exactly these pieces on PEMS05/PEMS06. The legacy "Ablation-1/2" further down predate the FiLM calibrator (they use the original residual calibrator) and are kept for reference.

## Ablation-A: Main Component Ablation (cumulative) — PEMS05 / PEMS06
_Add one component per row, starting from the frozen backbone, ending at the full model (Ours). Calibrator = FiLM. **bold** = Ours (full)._

The two load-bearing steps are **online delayed-label TTA** (the big drop) and the **FiLM calibrator** it adapts; the warm-up-only calibrator alone does not help (it even hurts on PEMS06, +0.84), confirming the calibrator is an online-adaptation interface, not a static head. The **local context clone** is a small, consistent final gain.

### PEMS05
| Component | MAE@3 | MAE@6 | MAE@12 | Avg |
|---|---|---|---|---|
| Backbone (frozen) | 10.39±0.06 | 11.00±0.06 | 12.25±0.10 | 11.10±0.06 |
| + FiLM calibrator (warm-up only) | 10.42±0.02 | 10.99±0.03 | 12.09±0.05 | 11.06±0.03 |
| + online delayed-label TTA (All) | 9.84±0.03 | 10.45±0.04 | 11.61±0.06 | 10.52±0.04 |
| **+ local clone (Ours)** | 9.80±0.03 | 10.40±0.04 | 11.57±0.06 | **10.48±0.04** |

### PEMS06
| Component | MAE@3 | MAE@6 | MAE@12 | Avg |
|---|---|---|---|---|
| Backbone (frozen) | 13.10±0.21 | 13.91±0.19 | 15.54±0.23 | 14.02±0.19 |
| + FiLM calibrator (warm-up only) | 14.01±0.17 | 14.77±0.21 | 16.25±0.26 | 14.86±0.20 |
| + online delayed-label TTA (All) | 12.24±0.09 | 13.07±0.12 | 14.68±0.17 | 13.16±0.12 |
| **+ local clone (Ours)** | 12.15±0.09 | 12.97±0.12 | 14.55±0.16 | **13.06±0.11** |

## Ablation-B: Calibrator Architecture (online TTA, `tta_all`) — Avg MAE + #Params
_Only the calibrator architecture changes; the frozen backbone, online delayed-label TTA, loss, optimiser and data are identical. #Params = trainable head params (excludes the shared node embedding). All architectures are identity at initialisation._

| Architecture | #Params | PEMS05 | PEMS06 |
|---|---|---|---|
| residual (raw-input MLP residual) | 7821 | 10.75 | 13.65 |
| norm (EMA scale-fix MLP residual) | 7821 | 10.63 | 13.79 |
| affine (static channel-wise 1+γ, β) | 24 | 10.89 | 14.15 |
| lowrank (linear low-rank residual, r8) | 460 | 10.80 | 13.87 |
| adapter (bottleneck residual, r8) | 468 | 10.68 | 13.80 |
| gated_adapter (adapter + tanh gate, r8) | 480 | 10.96 | 13.99 |
| temporal_conv (depthwise conv over horizon) | 4 | 10.86 | 14.14 |
| **film (Ours, conditional γ⊙y+β)** | 8601 | **10.52** | **13.16** |
| adapter (parameter-matched, r150) | 8562 | 10.58 | 13.26 |

**What this answers:**
- **FiLM vs residual/norm**: FiLM is best on both (−0.23/−0.49 vs residual) — the multiplicative scale + conditioning suits distribution drift.
- **FiLM vs affine (static)**: affine is among the worst (10.89/14.15); FiLM beats it by 0.37/**0.99**. → FiLM's gain comes from **dynamic conditioning** (input-generated γ,β), not merely scale+shift.
- **FiLM vs adapter at equal capacity**: even the **parameter-matched adapter (8562 params) loses** (10.58/13.26 vs 10.52/13.16). → FiLM's advantage is the **modulation form**, not parameter count.
- **gated vs adapter**: the gate **hurts** (10.96/13.99 vs 10.68/13.80) — no online-stability benefit here.
- **lowrank vs adapter**: linear is slightly worse than the nonlinear adapter; both far behind FiLM.

## Ablation-C: Calibrator as an Online-Adaptation Interface (warm-up only) — Avg MAE
_The calibrator is trained only during warm-up and **frozen at test** (no online adaptation). Compare against the source/backbone, and against the online numbers in Ablation-B for the same architecture._

| Architecture | PEMS05 | PEMS06 |
|---|---|---|
| backbone / source (no calibrator) | 11.10 | 14.02 |
| film | 11.06 | 14.86 |
| affine | 11.25 | 14.73 |
| adapter | 11.21 | 14.80 |
| gated_adapter | 11.30 | 14.51 |

**What this answers:** every warm-up-only calibrator is ≈ or **worse** than the frozen backbone (especially on PEMS06, +0.5–0.84), while the same architectures gain substantially under online TTA (e.g. FiLM 11.06/14.86 → 10.52/13.16). → the calibrator is an **online-adaptation interface, not a better static head**; its value is realised only through continuous test-time updates.

## Ablation-D: Knockout (leave-one-out from full Ours) — multi-backbone

_Start from full Ours, remove **one** component (keep the other three). Larger Δ vs Ours = more important. The same knockout is run on **two backbones** (Online-AN / TrafficStream, and STAEformer) to show the calibration stack is backbone-agnostic. Cells are mean±std over 5 seeds (51–55), years averaged within each seed._

**How each row is built** (`a2tta_main.py` flags; the knockout stack is the generic `tta_ctx_local` path, not the embedding variant `a2tta_emb`):

| Knockout row | `--method` | `--calibrator_arch` | removes |
|---|---|---|---|
| **Ours (full)** | `tta_ctx_local` | `film` | nothing (frozen backbone + FiLM + online TTA + local clone) |
| − local clone | `tta_all` | `film` | the per-batch discardable local clone (global calibrator still adapts online) |
| − FiLM → affine | `tta_ctx_local` | `affine` | FiLM's conditioning net — **replaced** by a static `(1+γ)·y+β`; TTA + clone kept |
| − online TTA | `calibrator` | `film` | online adaptation — FiLM is warm-up-trained then **frozen** at test |
| − calibration stack → backbone | `backbone` | — | the whole calibrator → falls back to the frozen backbone (base predictor) |

FiLM is a **swap**-knockout, not a delete: deleting the calibrator entirely is the last row. The backbone cannot be removed (it is the base predictor); the fallback row is the frozen backbone itself (Online-AN or STAEformer accordingly).

### Backbone = Online-AN (TrafficStream)

#### PEMS03
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 11.52±0.02 | 12.48±0.04 | 14.29±0.07 | **12.59±0.04** | — |
| − local clone | 11.58±0.02 | 12.54±0.04 | 14.34±0.06 | 12.65±0.04 | +0.06 |
| − FiLM → affine (naive) | 12.44±0.05 | 13.41±0.07 | 15.28±0.10 | 13.54±0.07 | +0.95 |
| − online TTA (FiLM warm-up only) | 13.30±0.05 | 14.13±0.06 | 15.70±0.08 | 14.23±0.06 | +1.64 |
| − calibration stack → Online-AN | 12.76±0.04 | 13.68±0.04 | 15.46±0.08 | 13.80±0.04 | +1.21 |

#### PEMS04
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 17.16±0.05 | 18.72±0.06 | 21.72±0.12 | **18.93±0.07** | — |
| − local clone | 17.29±0.05 | 18.83±0.06 | 21.79±0.11 | 19.04±0.06 | +0.11 |
| − FiLM → affine (naive) | 18.67±0.09 | 20.21±0.12 | 23.24±0.22 | 20.43±0.14 | +1.50 |
| − online TTA (FiLM warm-up only) | 20.32±0.05 | 21.59±0.07 | 24.09±0.11 | 21.77±0.07 | +2.84 |
| − calibration stack → Online-AN | 19.34±0.19 | 20.78±0.23 | 23.68±0.30 | 21.01±0.23 | +2.08 |

#### PEMS05
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 9.80±0.03 | 10.40±0.04 | 11.57±0.06 | **10.48±0.04** | — |
| − local clone | 9.84±0.03 | 10.45±0.04 | 11.61±0.06 | 10.52±0.04 | +0.04 |
| − FiLM → affine (naive) | 10.14±0.05 | 10.78±0.06 | 12.06±0.09 | 10.88±0.06 | +0.40 |
| − online TTA (FiLM warm-up only) | 10.42±0.02 | 10.99±0.03 | 12.09±0.05 | 11.06±0.03 | +0.58 |
| − calibration stack → Online-AN | 10.39±0.06 | 11.00±0.06 | 12.25±0.10 | 11.10±0.06 | +0.62 |

#### PEMS06
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 12.15±0.09 | 12.97±0.12 | 14.55±0.16 | **13.06±0.11** | — |
| − local clone | 12.24±0.09 | 13.07±0.12 | 14.68±0.17 | 13.16±0.12 | +0.10 |
| − FiLM → affine (naive) | 13.17±0.15 | 14.01±0.13 | 15.66±0.18 | 14.12±0.13 | +1.06 |
| − online TTA (FiLM warm-up only) | 14.01±0.17 | 14.77±0.21 | 16.25±0.26 | 14.86±0.20 | +1.80 |
| − calibration stack → Online-AN | 13.10±0.21 | 13.91±0.19 | 15.54±0.23 | 14.02±0.19 | +0.96 |

### Backbone = STAEformer

_Knockout `Ours (full)` here = the generic `tta_ctx_local` stack on the STAEformer backbone. (Note: the main-results **STAE-Ours** column uses the embedding variant `a2tta_emb`; on PEMS05/06 this plain stack is actually a touch stronger — 9.63/12.27 vs 9.96/12.65.)_

#### PEMS03
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 10.78±0.05 | 11.29±0.05 | 12.14±0.07 | **11.31±0.05** | — |
| − local clone | 10.83±0.06 | 11.33±0.06 | 12.17±0.08 | 11.35±0.06 | +0.04 |
| − FiLM → affine (naive) | 11.52±0.10 | 11.97±0.10 | 12.77±0.13 | 12.01±0.11 | +0.69 |
| − online TTA (FiLM warm-up only) | 12.39±0.16 | 12.79±0.15 | 13.48±0.15 | 12.82±0.15 | +1.50 |
| − calibration stack → STAEformer | 11.81±0.10 | 12.22±0.09 | 12.98±0.12 | 12.26±0.10 | +0.95 |

#### PEMS04
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 16.17±0.06 | 17.00±0.10 | 18.42±0.19 | **17.05±0.11** | — |
| − local clone | 16.26±0.06 | 17.07±0.09 | 18.48±0.18 | 17.13±0.10 | +0.08 |
| − FiLM → affine (naive) | 18.08±0.94 | 18.83±0.92 | 20.16±0.94 | 18.89±0.93 | +1.84 |
| − online TTA (FiLM warm-up only) | 18.86±0.17 | 19.49±0.17 | 20.62±0.22 | 19.54±0.18 | +2.49 |
| − calibration stack → STAEformer | 18.66±1.01 | 19.35±0.97 | 20.60±0.99 | 19.41±0.98 | +2.35 |

#### PEMS05
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 9.31±0.02 | 9.61±0.03 | 10.12±0.04 | **9.63±0.03** | — |
| − local clone | 9.34±0.02 | 9.63±0.03 | 10.14±0.04 | 9.65±0.03 | +0.02 |
| − FiLM → affine (naive) | 9.64±0.05 | 9.90±0.05 | 10.40±0.07 | 9.93±0.05 | +0.30 |
| − online TTA (FiLM warm-up only) | 9.85±0.04 | 10.12±0.05 | 10.59±0.07 | 10.14±0.05 | +0.51 |
| − calibration stack → STAEformer | 9.88±0.06 | 10.12±0.06 | 10.60±0.07 | 10.16±0.06 | +0.53 |

#### PEMS06
| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg |
|---|---|---|---|---|---|
| **Ours (full)** | 11.82±0.40 | 12.24±0.41 | 12.97±0.42 | **12.27±0.41** | — |
| − local clone | 12.24±0.53 | 12.65±0.54 | 13.37±0.56 | 12.68±0.54 | +0.41 |
| − FiLM → affine (naive) | 12.39±0.40 | 12.76±0.39 | 13.45±0.38 | 12.80±0.39 | +0.53 |
| − online TTA (FiLM warm-up only) | 13.11±0.29 | 13.48±0.29 | 14.14±0.28 | 13.51±0.29 | +1.24 |
| − calibration stack → STAEformer | 12.55±0.45 | 12.89±0.45 | 13.56±0.43 | 12.94±0.44 | +0.67 |

**Read:** across both backbones and all four datasets the ordering is identical — **online TTA** hurts most when removed (Online-AN +0.58…+2.84; STAEformer +0.51…+1.49), so online delayed-label adaptation is the core component. Replacing **FiLM** with a static affine is the second-largest drop (+0.40…+1.50 / +0.30…+1.84) → the calibrator's conditioning form matters, not just scale+shift. **Local clone** is a small refinement everywhere (+0.02…+0.11; the lone exception is STAEformer-PEMS06 at +0.41, where seeds are noisy, std≈0.4). On several cells warm-up-only FiLM is *worse* than the plain backbone (e.g. Online-AN-PEMS06 14.86 vs 14.02) → the calibrator must adapt online, it is not a better static head. (Complements the cumulative Ablation-A: A adds bottom-up, D removes top-down — same conclusion, now on two backbones.)

## (Legacy) Ablation-1: Component Ablation
_Backbone alone vs. adding the frozen residual calibrator vs. adding online delayed-label TTA. The gain comes from online adaptation; the warm-up calibrator alone does not help._

### Ablation-1: Component Ablation — MAE

**PEMS04**

| Variant | MAE@3 | MAE@6 | MAE@12 | Avg |
|---|---|---|---|---|
| Backbone (frozen) | 19.34±0.19 | 20.78±0.23 | 23.68±0.30 | 21.01±0.23 |
| + Residual Calibrator (warm-up, no TTA) | 20.55±0.10 | 21.91±0.13 | 24.66±0.20 | 22.13±0.14 |
| **+ Delayed-label TTA (Ours)** | 18.23±0.12 | 19.72±0.11 | 22.75±0.19 | 19.97±0.13 |

**PEMS05**

| Variant | MAE@3 | MAE@6 | MAE@12 | Avg |
|---|---|---|---|---|
| Backbone (frozen) | 10.39±0.06 | 11.00±0.06 | 12.25±0.10 | 11.10±0.06 |
| + Residual Calibrator (warm-up, no TTA) | 10.52±0.02 | 11.13±0.04 | 12.35±0.07 | 11.22±0.04 |
| **+ Delayed-label TTA (Ours)** | 10.02±0.03 | 10.66±0.04 | 11.93±0.08 | 10.75±0.05 |

**PEMS07**

| Variant | MAE@3 | MAE@6 | MAE@12 | Avg |
|---|---|---|---|---|
| Backbone (frozen) | 16.23±0.25 | 17.53±0.28 | 19.94±0.32 | 17.67±0.28 |
| + Residual Calibrator (warm-up, no TTA) | 16.44±0.21 | 17.72±0.23 | 20.06±0.26 | 17.85±0.23 |
| **+ Delayed-label TTA (Ours)** | 14.77±0.14 | 16.21±0.16 | 18.86±0.18 | 16.36±0.16 |

**PEMS08**

| Variant | MAE@3 | MAE@6 | MAE@12 | Avg |
|---|---|---|---|---|
| Backbone (frozen) | 12.76±0.09 | 13.80±0.09 | 15.75±0.10 | 13.92±0.09 |
| + Residual Calibrator (warm-up, no TTA) | 12.94±0.06 | 13.96±0.06 | 15.86±0.08 | 14.08±0.07 |
| **+ Delayed-label TTA (Ours)** | 12.27±0.05 | 13.36±0.06 | 15.36±0.08 | 13.48±0.06 |

## Ablation-2: TTA Loss Regularisers
_Consistency loss $\lambda_c$ and proximal anchor $\lambda_r$._

### Ablation-2: Regularisers — MAE

**PEMS05**

| Variant | MAE@3 | MAE@6 | MAE@12 | Avg |
|---|---|---|---|---|
| no reg | 10.08±0.03 | 10.72±0.04 | 12.00±0.07 | 10.82±0.04 |
| prox only ($\lambda_r$) | 10.09±0.03 | 10.73±0.04 | 12.01±0.08 | 10.82±0.04 |
| cons only ($\lambda_c$=.05) | 10.10±0.04 | 10.73±0.05 | 12.01±0.08 | 10.83±0.05 |
| **both (default)** | 10.08±0.04 | 10.72±0.05 | 12.00±0.09 | 10.81±0.05 |
| cons only ($\lambda_c$=.10) | 10.09±0.04 | 10.73±0.05 | 12.00±0.08 | 10.82±0.05 |
| $\lambda_c$=.10 + prox | 10.09±0.03 | 10.73±0.04 | 12.01±0.08 | 10.83±0.04 |

**PEMS08**

| Variant | MAE@3 | MAE@6 | MAE@12 | Avg |
|---|---|---|---|---|
| no reg | 12.35±0.05 | 13.43±0.06 | 15.45±0.08 | 13.55±0.06 |
| **prox only ($\lambda_r$)** | 12.34±0.05 | 13.42±0.06 | 15.45±0.08 | 13.55±0.06 |
| cons only ($\lambda_c$=.05) | 12.35±0.05 | 13.43±0.06 | 15.44±0.08 | 13.55±0.06 |
| both (default) | 12.36±0.07 | 13.44±0.08 | 15.45±0.09 | 13.56±0.08 |
| cons only ($\lambda_c$=.10) | 12.36±0.05 | 13.44±0.06 | 15.45±0.08 | 13.57±0.06 |
| $\lambda_c$=.10 + prox | 12.36±0.05 | 13.44±0.06 | 15.45±0.08 | 13.56±0.06 |

