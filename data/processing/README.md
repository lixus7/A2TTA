# Data processing pipeline

These notebooks turn raw Caltrans PEMS dumps into the per-year
`RawData/<year>.npz`, `graph/<year>_adj.npz` (and optionally `FastData/`) tensors
consumed by the models. They are provided for **full reproducibility** — you do not
need to run them if you download the released processed data.

Run order, per dataset `pemsNN`:

| Step | Notebook / script | Produces |
|------|-------------------|----------|
| 1 | `pemsNN_yearly_nodes.ipynb` | the set of active sensors for each year (the expanding-node bookkeeping) |
| 2 | `pemsNN_build_eac_data.ipynb` | `RawData/<year>.npz` flow tensors + `graph/<year>_adj.npz` adjacency |
| — | `pems03_build_adj.ipynb` | reference notebook for adjacency construction (distance/connectivity → weighted graph) |
| — | `pems03_data_quality_check.py` | audits raw years for NaN / missing detectors |
| — | `pems_all_growth_viz.ipynb` | visualizes sensor-count growth across all 9 datasets |

> Paths inside the notebooks point at the original working tree; adjust the input/output
> roots to your local copy before running. The end result must match the layout in
> [`../README.md`](../README.md).
