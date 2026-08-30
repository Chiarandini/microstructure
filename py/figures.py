"""Render the study's figures, in light and dark.

Four charts, each carrying one result that the README currently states as a
table of numbers:

1. the out-of-sample decay curve, and how it depends on the event clock;
2. predicted move against the spread it would have to cross;
3. the fitted queue intensities, which is the phase-5 finding;
4. the tick-size stratification, which is the universe design paying off.

Colour follows the validated categorical palette: slots are assigned in fixed
order and never cycled, marks carry the series colour while all text stays in
ink tokens, and every chart is capped at three series so it clears the
all-pairs colour-vision gates. Light and dark are separately specified rather
than one being an inversion of the other.

    python3 py/figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_reactive as QR

FIGURES = Path("figures")
BUCKETS = Path("data/buckets")
QUEUE = Path("data/queue")

N_TRAIN_SESSIONS = 4

# The reference palette. Dark is a separate selection stepped for the dark
# surface, not an automatic flip of light.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "critical": "#d03b3b",
    },
}

SYMBOLS = ["INTC", "CSCO", "MSFT", "AAPL", "SPY", "QQQ", "AMZN", "GOOGL"]


def style(theme):
    t = THEMES[theme]
    plt.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "text.color": t["primary"],
            "axes.labelcolor": t["secondary"],
            "axes.edgecolor": t["axis"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 200,
        }
    )
    return t


def save(fig, name, theme):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}-{theme}.{ext}", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def place_labels(fig, ax, points, color, avoid=()):
    """Greedy non-overlapping direct labels.

    Tries a ring of offsets per point and keeps the first that clears every
    label already placed. Stacking colliding labels would detach them from
    their marks, which reads as noise; moving them keeps the association.
    """
    placed = list(avoid)
    offsets = [(0, 15), (15, 0), (0, -19), (-15, 0), (14, 13), (-14, 13), (14, -15), (-14, -15)]
    for x, y, text in points:
        chosen = None
        for dx, dy in offsets:
            ann = ax.annotate(
                text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                va="center", color=color, fontsize=8.5,
            )
            fig.canvas.draw()
            box = ann.get_window_extent().expanded(1.05, 1.15)
            if not any(box.overlaps(b) for b in placed):
                chosen = (ann, box)
                break
            ann.remove()
        if chosen is None:
            ann = ax.annotate(
                text, xy=(x, y), xytext=(13, 0), textcoords="offset points",
                ha="left", va="center", color=color, fontsize=8.5,
            )
            fig.canvas.draw()
            chosen = (ann, ann.get_window_extent())
        placed.append(chosen[1])


def end_dot(ax, x, y, color, surface, size=7):
    """A marker with a 2px surface ring, so it stays legible over a line."""
    ax.plot(
        x, y, "o", color=color, markersize=size,
        markeredgecolor=surface, markeredgewidth=2, zorder=5, clip_on=False,
    )


# --- 1. decay curve -------------------------------------------------------


def fig_decay(theme):
    t = style(theme)
    ks = [50, 200, 1000]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for i, k in enumerate(ks):
        path = Path(f"py/results_ofi_k{k}.csv")
        if not path.exists():
            continue
        r = pd.read_csv(path)
        g = r.groupby("h").r2
        med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        c = t["series"][i]

        # The band is the spread across symbols, not a confidence interval:
        # it shows how much the result depends on which symbol you look at.
        ax.fill_between(med.index, lo, hi, color=c, alpha=0.10, linewidth=0)
        ax.plot(med.index, med.values, color=c, linewidth=2,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)
        # No end labels here: the three lines converge at the right edge, so
        # direct labels would collide. The legend carries identity instead.
        end_dot(ax, med.index[-1], med.values[-1], c, t["surface"])

    ax.axhline(0, color=t["axis"], linewidth=1, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("forward horizon (buckets)")
    ax.set_ylabel("median out-of-sample $R^2$")
    ax.set_title("Order flow imbalance predicts less the further ahead you look",
                 loc="left", color=t["primary"], pad=14)
    ax.text(
        0, 1.02,
        "median across 8 symbols, band shows interquartile range; "
        "trained on 4 sessions, tested on 3",
        transform=ax.transAxes, color=t["muted"], fontsize=8.5, va="bottom",
    )
    ax.margins(x=0.10)

    handles = [Line2D([], [], color=t["series"][i], linewidth=2, label=f"{k} events/bucket")
               for i, k in enumerate(ks)]
    ax.legend(handles=handles, loc="upper right", labelcolor=t["secondary"], fontsize=8.5)
    save(fig, "ofi-decay", theme)


# --- 2. predicted move vs cost -------------------------------------------


def rounded_barh(ax, y, width, height, color, rounding=0.16):
    """Horizontal bar with a rounded data-end, square at the baseline."""
    ax.add_patch(
        FancyBboxPatch(
            (0, y - height / 2), max(width, 1e-9), height,
            boxstyle=f"round,pad=0,rounding_size={rounding * height}",
            facecolor=color, edgecolor="none", mutation_aspect=0.02, zorder=3,
        )
    )


def fig_cost(theme):
    t = style(theme)
    r = pd.read_csv("py/results_ofi_k200.csv")
    r = r[r.h == 1].copy()
    r["ratio"] = r.pred_move / r.half_spread
    r = r.set_index("symbol").reindex(SYMBOLS).dropna().sort_values("ratio")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ys = np.arange(len(r))
    for y, (sym, row) in zip(ys, r.iterrows()):
        ax.barh(y, row.ratio, height=0.5, color=t["series"][0], zorder=3)
        ax.annotate(
            f"{row.ratio:.2f}x", xy=(row.ratio, y), xytext=(6, 0),
            textcoords="offset points", va="center", ha="left",
            color=t["secondary"], fontsize=8.5,
        )

    ax.axvline(1.0, color=t["critical"], linewidth=2, zorder=4)
    ax.annotate(
        "cost of crossing the spread",
        xy=(1.0, (len(r) - 1) / 2), xytext=(-10, 0), textcoords="offset points",
        va="center", ha="right", rotation=90,
        color=t["critical"], fontsize=8.5, fontweight="bold",
    )

    ax.set_yticks(ys)
    ax.set_yticklabels(r.index, color=t["secondary"])
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.75, len(r) - 0.25)
    ax.set_xlabel("predicted move ÷ half the quoted spread")
    ax.set_title("The signal is real and about ten times too small to trade",
                 loc="left", color=t["primary"], pad=14)
    ax.text(
        0, 1.02,
        "predicted move at the 90th percentile of |OFI|, one bucket ahead, "
        "200 events per bucket",
        transform=ax.transAxes, color=t["muted"], fontsize=8.5, va="bottom",
    )
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "ofi-vs-cost", theme)


# --- 3. fitted queue intensities -----------------------------------------


def fig_intensities(theme):
    t = style(theme)
    stats = pd.read_csv(QUEUE / "stats.csv.gz", dtype={"date": "string"})
    train = sorted(stats.date.unique())[:N_TRAIN_SESSIONS]

    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6), sharex=False)
    channels = [("L", "limit orders arriving"), ("C", "cancellations"), ("M", "market orders")]

    for ax, sym in zip(axes.ravel(), SYMBOLS):
        sub = stats[(stats.symbol == sym) & (stats.side == "B") & stats.date.isin(train)]
        pooled = sub.groupby("q", as_index=False).sum(numeric_only=True)
        model = QR.fit(pooled, n_bins=20)
        edges, time_s = model["edges"], model["time_s"]
        centres = np.sqrt(edges[:-1] * edges[1:])
        # Restricted to the occupancy core: the middle 95% of time spent.
        # The tails are real states but the book is almost never in them, and
        # including them stretches the axis across four orders of magnitude
        # to show a regime that is not what the model is about.
        share = np.cumsum(time_s) / time_s.sum()
        core = (share > 0.025) & (share < 0.975) & (time_s > 30.0)

        for i, (ch, _) in enumerate(channels):
            ax.plot(
                centres[core], model["rates"][ch][core], color=t["series"][i],
                linewidth=2, solid_capstyle="round", solid_joinstyle="round",
            )

        # One number per panel: the elasticity of cancellation intensity to
        # queue size. A slope of 1 would mean a constant per-share hazard.
        lx = np.log(centres[core])
        slope = np.polyfit(lx, np.log(model["rates"]["C"][core]), 1)[0]
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Log axes over a narrow range label every minor tick by default,
        # which collides into an unreadable smear.
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
        ax.set_title(sym, loc="left", color=t["primary"])
        # The slope rides the title rather than the plot area: every corner of
        # these panels has a line running through it in at least one symbol.
        ax.set_title(
            f"cancel {slope:+.2f}", loc="right",
            color=t["secondary"], fontsize=8, fontweight="normal",
        )
        ax.tick_params(labelsize=7.5)

    for ax in axes[1]:
        ax.set_xlabel("queue size (shares)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("intensity (events/s)", fontsize=8.5)

    # Lay the panels out first, then hang the titles above them: setting the
    # titles first lets tight_layout move the axes underneath the text.
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.text(
        0.005, 0.985,
        "Deeper queues pull in cancellations; limit orders keep arriving at their own pace",
        color=t["primary"], fontsize=12, fontweight="bold", ha="left", va="top",
    )
    fig.text(
        0.005, 0.938,
        "fitted departure intensities at the best bid, training sessions, log-log, "
        "middle 95% of occupancy; cancellation elasticity is positive for all eight "
        "symbols (median +0.79), limit arrival near flat (median +0.11)",
        color=t["muted"], fontsize=8.5, ha="left", va="top",
    )
    handles = [Line2D([], [], color=t["series"][i], linewidth=2, label=label)
               for i, (_, label) in enumerate(channels)]
    fig.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.0),
        ncol=3, labelcolor=t["secondary"], fontsize=9,
    )
    save(fig, "queue-intensities", theme)


# --- 4. tick-size stratification -----------------------------------------


def fig_stratification(theme):
    t = style(theme)
    panel = pd.read_csv(BUCKETS / "k200.csv.gz", dtype={"date": "string", "symbol": "string"})

    rows = []
    for sym, g in panel.groupby("symbol"):
        ticks = g.spread_mean.median() * 10_000 / 100  # dollars -> cents -> ticks
        acs = []
        for _, gg in g.groupby("date"):
            a = gg.ofi.to_numpy()
            if len(a) > 10:
                acs.append(np.corrcoef(a[:-1], a[1:])[0, 1])
        rows.append({"symbol": sym, "ticks": ticks, "ac1": np.mean(acs)})
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    c = t["series"][0]
    ax.plot(
        d.ticks, d.ac1, "o", color=c, markersize=9,
        markeredgecolor=t["surface"], markeredgewidth=2, linestyle="none", zorder=3,
    )
    ax.margins(0.16)
    fig.canvas.draw()
    marks = [
        ax.transData.transform((r.ticks, r.ac1)) for _, r in d.iterrows()
    ]
    from matplotlib.transforms import Bbox
    avoid = [Bbox.from_bounds(mx - 10, my - 10, 20, 20) for mx, my in marks]
    place_labels(
        fig, ax,
        [(r.ticks, r.ac1, r.symbol) for _, r in d.iterrows()],
        t["secondary"], avoid=avoid,
    )

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("median quoted spread (ticks)")
    ax.set_ylabel("order flow autocorrelation, lag 1")
    ax.set_title("Order flow is far more persistent where the spread is wide",
                 loc="left", color=t["primary"], pad=14)
    ax.text(
        0, 1.02,
        "one point per symbol; the universe was chosen to span this axis "
        "before any result was computed",
        transform=ax.transAxes, color=t["muted"], fontsize=8.5, va="bottom",
    )
    save(fig, "tick-regimes", theme)


def main():
    for theme in ("light", "dark"):
        fig_decay(theme)
        fig_cost(theme)
        fig_intensities(theme)
        fig_stratification(theme)
        print(f"  rendered {theme}")
    print(f"\nwrote {FIGURES}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
