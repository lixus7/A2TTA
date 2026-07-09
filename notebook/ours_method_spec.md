# A2TTA (Ours) — methodology spec for a figure

> Source of truth: `eac/src/model/a2tta.py` (FiLM calibrator + selector),
> `eac/src/trainer/a2tta_trainer.py` (warm-up + online loop),
> `eac/src/trainer/ctx_local.py` (local-clone context weighting),
> `eac/a2tta_main.py` (per-year orchestration). Everything below is faithful to
> that code; the "Ours" configuration is `method = tta_ctx_local` with a **FiLM**
> calibrator. **The method is backbone-agnostic** — validated on the Online-AN
> (TrafficStream) and STAEformer backbones with an identical mechanism.

---

## 0. One-liner

Freeze **any** pretrained traffic-forecasting backbone; wrap its raw prediction
with a tiny per-node **FiLM calibrator**. Evaluate as a **causal stream** in
which each window's ground truth only arrives `H` steps later (a *delayed
label*). Continuously adapt the calibrator on a **delayed-label pool**; and at
every prediction step additionally spin up a **discardable local clone** of the
calibrator, specialise it for a few gradient steps on **context-weighted** pool
samples, predict with it, then throw it away — so the persistent calibrator is
never biased by any single context.

---

## 1. Notation

| symbol | meaning | shape |
|---|---|---|
| `x` | input window (z-score normalised) | `[B, N, T_in]`, `T_in=12` |
| `y` | ground truth horizon | `[B, N, H]`, `H=12` |
| `f_θ` | **frozen** backbone (any model) | — |
| `y_base = f_θ(x)` | backbone prediction, **raw scale** | `[B·N, H]` |
| `g_φ` | **trainable** FiLM calibrator (Ours) | params ≪ backbone |
| `e_n` | per-node learnable embedding | `[N, d]`, `d=16` |
| `N` | # sensors (grows across years) | — |
| `P` | delayed-label pool (deque) | `maxlen = 512` |
| `ŷ` | calibrated prediction | `[B·N, H]` |

Backbone seed convention: OL-AN backbone_seed = tta_seed; STAEformer
backbone_seed = tta_seed − 9. Only the calibrator is ever trained by us.

---

## 2. Components

### 2.1 Frozen backbone `f_θ` (backbone-agnostic interface)
- Loaded from a per-year checkpoint; **all parameters frozen** (`requires_grad=False`, `eval()`).
- Emits `y_base` = an `H`-step forecast in **raw units**.
- The calibrator consumes only `y_base`, `x`, and node ids — **nothing
  backbone-specific** — so any backbone (Online-AN/TrafficStream, STAEformer,
  GWN, …) drops in unchanged.

### 2.2 FiLM calibrator `g_φ` (the ONLY module we train)
Per node-sample inputs, concatenated:
- `y_base` (`H`) — EMA-standardised to ~unit scale (running mean/std buffer);
- `x_in` (`T_in`) — the z-scored input window;
- temporal stats `[last, mean, std, slope]` (4 scalars) of `x_in`;
- node embedding `e_n` (`d=16`), looked up by global node id.

Body: 2-layer MLP (`Linear→GELU→Dropout` ×2, hidden = 64) → head emits per-sample
`(γ_raw, β_norm)` over the horizon. FiLM affine output:

```
γ = 1 + 0.5·tanh(γ_raw)          # init 1  (head is zero-init)
β = std(y_base)·β_norm           # init 0
ŷ = γ ⊙ y_base + β               # per-horizon affine calibration
```

Because the output head is **zero-initialised**, `g_φ` is the **identity** at
init (`ŷ = y_base`): before any adaptation the wrapped model equals the frozen
backbone exactly. Node-embedding table **grows** with the graph across years
(`expand_nodes`).

### 2.3 Delayed-label candidate pool `P`
A FIFO `deque(maxlen=512)`. A window predicted at chronological index `i` is
**released** into `P` only once `next_idx ≥ i + H` — i.e. after its true horizon
has fully elapsed. **Strictly causal, no label leakage.** Each pool item caches
`(x, y_base, y_true, y_pred, node_idx, idx)`.

---

## 3. Two-phase training/inference

### Phase A — offline warm-up (per year, `warmup_epochs = 3`)
Backbone frozen. Train `g_φ` on the year's **train** split with AdamW + L1 loss
(`ŷ` vs `y`), early-stopped on val MAE. Gives the online phase a **calibrated**
(not identity) starting `φ`.

### Phase B — causal online delayed-label TTA (the test stream)
Iterate the test set **in chronological order**. Per batch:

1. **Release** matured windows from `pending` → pool `P` (once `next_idx ≥ i+H`).
2. **Global update** — if `|P| ≥ max(8, 0.1·512)` and `batch % adapt_every == 0`
   (`adapt_every=1`): take `adapt_steps = 3` AdamW steps updating `φ` on the
   **whole pool** (Ours uses selector mode `all`). Loss:
   `L1(g_φ(pool), y_true) + λ_cons·consistency + λ_reg·‖φ − φ_init‖²`
   (proximal anchor to the warm-up init; `λ` default 0). `φ` **persists across
   batches and across years.**
3. **Predict** the current batch:
   - **Ours (`tta_ctx_local`)**: **deep-copy** `g_φ → g_φ'` (local clone). Take
     `local_steps = 3` AdamW steps on the pool, each row weighted by **relevance
     to the current window** (§4). Predict this batch with `g_φ'`, then
     **discard `g_φ'`**. The persistent `g_φ` is never touched by this step.
   - (fallback if pool not ready: predict with the global `g_φ`.)
4. **Enqueue** the current windows into `pending` with `release_idx = i + H`, for
   future delayed adaptation.
5. Metrics via `cal_metric` (per-horizon MAE/RMSE/MAPE) — identical to the
   offline test path, so numbers are directly comparable.

---

## 4. Local-clone context weighting (`context_row_weights`, mode `hybrid`)
Each delayed sample in the pool gets a scalar weight for the clone's specialise
steps, from three relevance signals to the current target window:
- **Phase match**: closeness in time-of-day (Gaussian on circular ToD distance)
  and day-of-week match;
- **Pattern similarity**: cosine similarity between the pooled `[x̄ ‖ ȳ_base]`
  feature of the sample and that of the current window;
- **Recency**: normalised chronological index.

Combined multiplicatively, `softmax`-normalised (temperature `τ=1`), clipped to
`[1/5, 5]`, renormalised to mean 1, and **ESS-guarded**: if effective sample
size collapses (< 20 %), fall back to **uniform** (`None`). This is the *only*
selection/weighting that consistently beat "adapt on all labels" — all other
selection ideas are archived (`a2tta_back.py`); see the appendix.

---

## 5. What is frozen vs trained
- **Frozen (always):** the entire backbone `f_θ`.
- **Trained online:** the calibrator `φ` (global, persistent) + a **per-batch
  local clone `φ'` that is immediately discarded**.
- Param budget: calibrator ≪ backbone (a 2-layer MLP + node table).
- *(Optional variant `a2tta_emb`, NOT the Ours main path: also nudges a
  STAEformer backbone's adaptive-embedding rows for actively-selected nodes —
  backbone-specific, so excluded from the backbone-agnostic story.)*

---

## 6. Backbone-agnostic evidence
Same mechanism, two backbones:
- **OL-AN** (TrafficStream online-AN) → main-table column *A2TTA*.
- **STAEformer** → main-table column *STAE-Ours* / new-sensor *a2tta-staef*.
HP-sensitivity curves are near-identical across the two backbones, and the
default config transfers without re-tuning → Ours is a **drop-in wrapper** for
any pretrained backbone.

---

## 7. Default hyper-parameters (from the sensitivity study)
`adapt_lr = 1e-3` · `adapt_steps = 3` · `candidate_pool_size = 512` ·
`budget_frac = 0.25` (inert under mode `all`) · `warmup_epochs = 3` ·
`local_steps = 3` · calibrator `hidden = 64`, `node_emb = 16`, `arch = film`.
Only `adapt_lr` is materially sensitive (flat optimum `1e-3–3e-3`).

---

## 8. Suggested figure layout (hand this to GPT)

Two panels side by side.

**(a) Architecture / data flow** (left):
```
 x  ──►  [ Frozen Backbone f_θ ]  ──►  y_base ──►┐
 (input window)      ❄ frozen                    │
                                                  ▼
 node emb e_n ─┐                       ┌─────────────────────────┐
 stats(x)     ─┼──────────────────────►│  FiLM calibrator g_φ    │  🔥 trained
 x_in         ─┘                       │  MLP → (γ, β)           │
                                       │  ŷ = γ·y_base + β       │
                                       └───────────┬─────────────┘
                                                   ▼
                                                   ŷ  (calibrated prediction)
```
Annotate: backbone = snowflake/"frozen, any model"; calibrator = flame/"trained,
≪ backbone"; FiLM box shows `γ⊙y_base+β`, identity at init.

**(b) Causal online delayed-label TTA loop** (right): a horizontal time axis of
streaming windows. For a window predicted at `t`, draw its **label arriving at
`t+H`** (dashed, "delayed label") dropping into a **Pool P (size 512)** buffer.
From the pool draw two arrows:
1. solid → **global calibrator update** (`adapt_steps` on all pool labels;
   "persistent φ");
2. dashed → **local clone φ′** box ("context-weighted, `local_steps`,
   **discarded**") that produces the prediction for the **current** window; a
   small tag "context weights: ToD/DoW · cosine · recency" feeds it.
Show the loop: predict → enqueue → (H later) release → adapt. Mark the whole
strip "causal — no future leakage".

Palette to match the paper figures: OL-AN `#0072B2` (blue), STAEformer `#D55E00`
(vermillion), frozen boxes cool-grey, trained boxes warm accent.

---

## 9. 中文速记(便于你和 GPT 沟通)
- **冻结任意 backbone**,在其**原始预测 y_base** 上套一个**逐节点 FiLM 校准器**
  `ŷ = γ·y_base + β`(零初始化→初始即恒等,等于原 backbone)。校准器是**唯一**被训练的模块,参数远小于 backbone。
- **因果流式评测 + 延迟标签**:某窗口的真值要 `H` 步后才到,到齐才进**延迟标签池 P(512)**,严格无泄露。
- **全局在线更新**:每 batch 用**整池**标签对校准器走 `adapt_steps=3` 步(选择器=all;论文结论:没有哪种持久样本选择能胜过用全部标签)。全局 `φ` 跨 batch、跨年份持续。
- **局部克隆(Ours 关键)**:每次预测前**深拷贝**校准器→克隆体,在池上按**与当前窗口的相关性加权**(时段/星期相位 + 输入&base 预测余弦 + 近因,softmax+ESS 兜底)走 `local_steps=3` 步,用克隆体出预测后**立即丢弃**;全局校准器不被任何单一上下文带偏。
- **两 backbone(OL-AN / STAEformer)机制完全相同、敏感性几乎一致** → 即插即用的通用外壳。
- 默认:`lr1e-3 / steps3 / pool512 / budget0.25 / warmup3 / local3`,`hidden64 / node_emb16 / film`。
