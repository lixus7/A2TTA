"""
Version-robust wrapper around Google's TimesFM time-series foundation model.

This is a *zero-shot* forecaster used as a baseline for the xxltraffic / PEMS
continual-forecasting benchmark. It is intentionally self-contained and does NOT
touch any of the existing model code (src/model/model.py etc.).

Why a wrapper?
--------------
The public `timesfm` package has shipped two materially different Python APIs:

  * "classic"  (timesfm <= 1.2.x, checkpoints google/timesfm-1.0-200m-pytorch and
                google/timesfm-2.0-500m-pytorch):
        tfm = timesfm.TimesFm(hparams=TimesFmHparams(...),
                              checkpoint=TimesFmCheckpoint(huggingface_repo_id=...))
        point, quantile = tfm.forecast(list_of_1d_arrays, freq=[0,...])

  * "2.5"     (timesfm >= 2.5, checkpoint google/timesfm-2.5-200m-pytorch):
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo)
        model.compile(timesfm.ForecastConfig(max_context=..., max_horizon=...))
        point, quantile = model.forecast(horizon=H, inputs=list_of_1d_arrays)

We auto-detect which one is installed and adapt. Pin a known-good version in
scripts/setup_timesfm_env.sh; the wrapper just makes the code resilient to the
exact one that ends up installed.

The forecast contract exposed to the rest of the pipeline is a single method:

    forecaster.forecast(contexts, horizon) -> np.ndarray of shape (B, horizon)

where `contexts` is an array/list of B univariate context windows (each length
`context_len`, raw scale). TimesFM performs its own per-series normalization, so
callers MUST pass raw-scale context (not z-scored) and will receive raw-scale
predictions.
"""

import numpy as np


def _infer_arch_from_repo(repo_id: str):
    """Best-effort architecture hyper-params for the classic TimesFm() API."""
    r = (repo_id or "").lower()
    if "1.0-200m" in r or "1.0_200m" in r:
        # TimesFM-1.0-200m
        return dict(num_layers=20, model_dims=1280, use_positional_embedding=True)
    # default to the 2.0-500m architecture
    return dict(num_layers=50, model_dims=1280, use_positional_embedding=False)


class TimesFMForecaster:
    def __init__(self,
                 repo_id="google/timesfm-2.0-500m-pytorch",
                 context_len=512,
                 horizon_len=128,
                 per_core_batch_size=32,
                 backend="gpu",
                 logger=None):
        """
        repo_id            : HuggingFace checkpoint id.
        context_len        : max context the model is configured for. Inputs
                             shorter than this are padded internally by TimesFM.
                             Must be a multiple of 32 for the classic API.
        horizon_len        : max horizon the model is compiled for (>= the
                             horizon you actually request at forecast time).
        per_core_batch_size: TimesFM internal batch size.
        backend            : "gpu" | "cpu" | "torch" (classic API only).
        """
        self.repo_id = repo_id
        self.context_len = int(context_len)
        self.horizon_len = int(horizon_len)
        self.per_core_batch_size = int(per_core_batch_size)
        self.backend = backend
        self.log = (logger.info if logger is not None else print)
        self.api = None        # "2p5" | "classic"
        self._model = None
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self):
        import timesfm  # imported lazily so the rest of the pipeline never needs it

        # --- Try the 2.5 API first ---------------------------------------
        cls = None
        for name in ("TimesFM_2p5_200M_torch", "TimesFM_2p5_200M",
                     "TimesFm_2p5_200M_torch"):
            cls = getattr(timesfm, name, None)
            if cls is not None:
                break

        if cls is not None and hasattr(timesfm, "ForecastConfig"):
            self.log(f"[TimesFM] using 2.5 API ({cls.__name__}) repo={self.repo_id}")
            model = None
            # from_pretrained(repo) is the documented path; fall back to a
            # bare constructor + load_checkpoint() for older 2.5 dev builds.
            if hasattr(cls, "from_pretrained"):
                try:
                    model = cls.from_pretrained(self.repo_id)
                except Exception as e:  # noqa: BLE001
                    self.log(f"[TimesFM] from_pretrained failed ({e}); trying load_checkpoint")
            if model is None:
                model = cls()
                if hasattr(model, "load_checkpoint"):
                    try:
                        model.load_checkpoint(self.repo_id)
                    except TypeError:
                        model.load_checkpoint()

            cfg_kwargs = dict(max_context=self.context_len,
                              max_horizon=self.horizon_len)
            # These flags exist in current 2.5; guard each so we stay
            # compatible if any are renamed/removed.
            #  - per_core_batch_size: 2.5 defaults this to 1 (one series at a
            #    time!). The real batch is per_core_batch_size * device_count,
            #    so this is the throughput knob — must be set high.
            #  - normalize_inputs=True: TimesFM normalizes each series internally
            #    (per-series mean/std). We rely on this so raw-scale context maps
            #    to raw-scale output.
            #  - infer_is_positive=True: traffic flow is non-negative.
            for k, v in dict(per_core_batch_size=self.per_core_batch_size,
                             normalize_inputs=True,
                             use_continuous_quantile_head=True,
                             force_flip_invariance=True,
                             infer_is_positive=True,
                             fix_quantile_crossing=True).items():
                cfg_kwargs[k] = v
            cfg = self._make_forecast_config(timesfm, cfg_kwargs)
            model.compile(cfg)
            self._model = model
            self.api = "2p5"
            return

        # --- Fall back to the classic TimesFm() API ----------------------
        if hasattr(timesfm, "TimesFm"):
            arch = _infer_arch_from_repo(self.repo_id)
            self.log(f"[TimesFM] using classic API repo={self.repo_id} arch={arch} backend={self.backend}")
            hp_kwargs = dict(backend=self.backend,
                             per_core_batch_size=self.per_core_batch_size,
                             horizon_len=self.horizon_len,
                             context_len=self.context_len,
                             num_layers=arch["num_layers"],
                             use_positional_embedding=arch["use_positional_embedding"])
            # model_dims is accepted by most builds; drop it if rejected.
            try:
                hparams = timesfm.TimesFmHparams(model_dims=arch["model_dims"], **hp_kwargs)
            except TypeError:
                hparams = timesfm.TimesFmHparams(**hp_kwargs)
            ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.repo_id)
            self._model = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)
            self.api = "classic"
            return

        raise RuntimeError(
            "Unrecognized `timesfm` API. Expected either `TimesFM_2p5_*` + "
            "`ForecastConfig`, or `TimesFm`/`TimesFmHparams`. Got attrs: "
            + ", ".join(a for a in dir(timesfm) if not a.startswith("_"))
        )

    @staticmethod
    def _make_forecast_config(timesfm, cfg_kwargs):
        """Build ForecastConfig, dropping kwargs the installed version rejects."""
        cfg_kwargs = dict(cfg_kwargs)
        while True:
            try:
                return timesfm.ForecastConfig(**cfg_kwargs)
            except TypeError as e:
                # strip the offending kwarg named in the error message, retry
                msg = str(e)
                dropped = None
                for k in list(cfg_kwargs.keys()):
                    if k in msg and k not in ("max_context", "max_horizon"):
                        cfg_kwargs.pop(k)
                        dropped = k
                        break
                if dropped is None:
                    raise

    # --------------------------------------------------------------- forecast
    def forecast(self, contexts, horizon, chunk_size=4096):
        """
        contexts : np.ndarray (B, L) or list of B 1-D arrays (raw scale).
        horizon  : number of future steps to predict (<= horizon_len).
        returns  : np.ndarray (B, horizon), raw scale.
        """
        if horizon > self.horizon_len:
            raise ValueError(f"requested horizon {horizon} > compiled horizon_len {self.horizon_len}")

        is_arr = isinstance(contexts, np.ndarray)
        n = contexts.shape[0] if is_arr else len(contexts)

        out = []
        # Build the small per-chunk list only as needed (avoids materializing a
        # multi-million element Python list for the big late-year datasets).
        for s in range(0, n, chunk_size):
            if is_arr:
                sub = contexts[s:s + chunk_size]
                batch = [np.ascontiguousarray(sub[i], dtype=np.float32) for i in range(sub.shape[0])]
            else:
                batch = [np.asarray(x, dtype=np.float32) for x in contexts[s:s + chunk_size]]
            point = self._forecast_batch(batch, horizon)
            point = np.asarray(point, dtype=np.float32)
            if point.ndim == 1:
                point = point[None, :]
            out.append(point[:, :horizon])
        return np.concatenate(out, axis=0)

    def _forecast_batch(self, batch, horizon):
        if self.api == "2p5":
            res = self._model.forecast(horizon=horizon, inputs=batch)
        else:  # classic
            freq = [0] * len(batch)  # 0 = high frequency (sub-hourly), correct for 5-min traffic
            res = self._model.forecast(batch, freq=freq)
        # both APIs return (point_forecast, quantile_forecast); some return just point
        point = res[0] if isinstance(res, (tuple, list)) else res
        return point
