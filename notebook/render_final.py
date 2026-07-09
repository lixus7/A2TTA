"""
Final synthesized paper figure: PEMS sensor network growth, 3x3 district grid.
Combines the best traits found in the design exploration:
  - uniform SQUARE panels computed in Web-Mercator metres (no latitude-rectangle bug, no lon squish)
  - viridis cohort colormap (perceptually uniform, colorblind-safe, prints well)  [beats turbo]
  - earliest cohorts drawn ON TOP so the original "seed" backbone stays visible in dense districts
  - light geographic basemap for context: mode='carto' (CartoDB Positron tiles) or 'vector' (Natural Earth)
  - wide horizontal colorbar with 4-year ticks, consistent sans fonts, thin panel borders
  - 400-dpi PNG + vector PDF

Run:  python notebook/render_final.py
"""
import os, sys, math, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles
from _cohort_lib import all_cohorts, year_range, DISTRICTS

warnings.filterwarnings("ignore")
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ---- clean consistent fonts (English only) ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,   # embed TrueType so text stays selectable/crisp in the PDF
})

CMAP = "viridis"
MERC = ccrs.Mercator.GOOGLE          # Web Mercator (metres)
PC = ccrs.PlateCarree()
WORLD_M = 2 * math.pi * 6378137.0    # mercator world width in metres


class CartoLight(GoogleTiles):
    """CartoDB Positron light basemap (no API key)."""
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"


def square_extent_merc(lon, lat, margin=0.10):
    """Return a square [x0,x1,y0,y1] extent in Web-Mercator metres covering the points."""
    xy = MERC.transform_points(PC, np.asarray(lon), np.asarray(lat))
    x, y = xy[:, 0], xy[:, 1]
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) / 2 * (1 + margin)
    half = max(half, 3000.0)  # floor so a near-degenerate district still gets a frame
    return [cx - half, cx + half, cy - half, cy + half]


def zoom_for_span(span_m, target_tiles=4.0, lo=6, hi=12):
    z = int(round(math.log2(WORLD_M / span_m * target_tiles)))
    return max(lo, min(hi, z))


def render(mode):
    """mode: 'carto' (light tiles, geographic+road context) | 'vector' (tile-free Natural Earth)."""
    cohorts, snap_years = all_cohorts()
    ymin, ymax = year_range()
    norm = Normalize(vmin=ymin, vmax=ymax)

    fig = plt.figure(figsize=(15, 15.6))
    gs = fig.add_gridspec(3, 3, left=0.035, right=0.965, top=0.93, bottom=0.13,
                          hspace=0.14, wspace=0.10)
    tiler = CartoLight() if mode == "carto" else None

    for i, d in enumerate(DISTRICTS):
        ax = fig.add_subplot(gs[i // 3, i % 3], projection=MERC)
        df = cohorts[d]
        ext = square_extent_merc(df["Longitude"], df["Latitude"])
        ax.set_extent(ext, crs=MERC)
        ax.set_box_aspect(1)  # guarantee identical on-page square panels

        if mode == "carto":
            try:
                ax.add_image(tiler, zoom_for_span(ext[1] - ext[0]),
                             interpolation="spline36", alpha=0.85, zorder=0)
            except Exception as e:
                print(f"  [d{d}] carto tiles failed ({type(e).__name__}); vector fallback")
                mode_local = "vector"
            else:
                mode_local = "carto"
        else:
            mode_local = "vector"
        if mode_local == "vector":
            ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eaf1f5", zorder=0)
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f7f7f5", zorder=0)
            ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#c2c2c2",
                           linewidth=0.4, zorder=1)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#9aa0a6",
                           linewidth=0.4, zorder=1)

        # earliest cohorts ON TOP: draw newest first so the rare old "backbone" survives in dense areas
        order = np.argsort(-df["cohort"].values)
        ax.scatter(df["Longitude"].values[order], df["Latitude"].values[order],
                   c=df["cohort"].values[order], cmap=CMAP, norm=norm,
                   s=3.2, alpha=0.75, linewidths=0, transform=PC,
                   rasterized=True, zorder=5)

        ys = snap_years[d]
        ax.set_title(f"District {d}   ({ys[0]}–{ys[-1]},  N={len(df)})",
                     fontsize=11.5, fontweight="semibold", pad=4)
        for s in ax.spines.values():
            s.set_edgecolor("#9aa0a6"); s.set_linewidth(0.6)

    # ---- shared horizontal colorbar with 4-year ticks ----
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cax = fig.add_axes([0.32, 0.075, 0.36, 0.015])
    ticks = list(range(((ymin + 3) // 4) * 4, ymax + 1, 4))
    if ticks and ticks[0] > ymin:
        ticks = [ymin] + ticks
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.xaxis.set_label_position("top")          # label above the bar -> no clash with caption
    cb.set_label("Sensor first-seen year (deployment cohort)", fontsize=11.5, labelpad=6)
    cb.ax.tick_params(labelsize=10)

    ctx = "CartoDB Positron basemap" if mode == "carto" else "Natural Earth vector basemap"
    fig.suptitle("Geographic growth of the PeMS sensor network, by Caltrans district",
                 fontsize=17, fontweight="bold", y=0.965)
    fig.text(0.5, 0.025, "Each dot = one sensor, colored by the year it first appears.  "
             "All panels are equal-size, equal-scale squares sharing one color scale.  "
             f"{ctx}.", ha="center", fontsize=9, color="#444444")

    png = os.path.join(OUTDIR, f"pems_grid_final_{mode}.png")
    pdf = os.path.join(OUTDIR, f"pems_grid_final_{mode}.pdf")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")  # vector text/axes; scatter+basemap rasterized inside
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(png).size
    print(f"[{mode}] saved {png} ({w}x{h}px, {os.path.getsize(png)//1024} KB) + {os.path.basename(pdf)} "
          f"({os.path.getsize(pdf)//1024} KB)")


if __name__ == "__main__":
    for m in ["carto", "vector"]:
        render(m)
    print("done.")
