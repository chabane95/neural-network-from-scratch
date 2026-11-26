"""Draw the C program's classification and training results as journal figures."""

import argparse
import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FixedLocator, LogLocator, MaxNLocator, ScalarFormatter
from matplotlib.tri import Triangulation
import numpy as np
from PIL import Image


CLASS_COLORS = ("#1b1b1b", "#155aa2", "#ad322f")
MARKS = ("o", "s", "^")
BAND = "#dedede"
THRESHOLD = 0.5

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "mathtext.fontset": "stix",
    "axes.linewidth": 0.75, "axes.labelsize": 10, "axes.unicode_minus": True,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.size": 3.4, "ytick.major.size": 3.4, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.width": 0.65, "ytick.major.width": 0.65, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "legend.fontsize": 8, "legend.frameon": False, "legend.handlelength": 1.65,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "svg.hashsalt": "spiralnet", "svg.fonttype": "path",
})


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@dataclass
class Surface:
    step: int
    mesh: Triangulation
    p: np.ndarray


def read_surfaces(path):
    with path.open(encoding="utf-8") as stream:
        if stream.readline().strip() != "frame,epoch,x1,x2,p0,p1,p2":
            raise ValueError(f"Unerwarteter Tabellenkopf in {path}")
        data = np.loadtxt(stream, delimiter=",", ndmin=2)
    surfaces = []
    for block in np.split(data, np.flatnonzero(np.diff(data[:, 0]))+1):
        n = math.isqrt(len(block))
        if n < 2 or n*n != len(block):
            raise ValueError(f"Unvollständiges Gitter in {path}")
        grid = block.reshape(n, n, 7)
        xs, ys = grid[0, :, 2], grid[:, 0, 3]
        if (not np.all(grid[:, :, 2] == xs) or not np.all(grid[:, :, 3] == ys[:, None])
                or np.any(np.diff(xs) <= 0) or np.any(np.diff(ys) <= 0)):
            raise ValueError(f"Ungültige Gitterkoordinaten in {path}")
        # Identical diagonal in every cell makes the interpolation reproducible.
        a = np.arange(n*n).reshape(n, n)[:-1, :-1].ravel()
        triangles = np.vstack((np.column_stack((a, a+1, a+n+1)),
                               np.column_stack((a, a+n+1, a+n))))
        mesh = Triangulation(block[:, 2], block[:, 3], triangles)
        surfaces.append(Surface(int(block[0, 1]), mesh, block[:, 4:7]))
    return surfaces


def boundaries(surface):
    """Find p_i=p_j >= p_k exactly for piecewise affine probabilities."""
    tri = surface.mesh.triangles
    xy = np.column_stack((surface.mesh.x, surface.mesh.y))
    segments = []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        delta = surface.p[:, i] - surface.p[:, j]
        winners = (delta[tri] >= 0).sum(axis=1)
        cut = tri[(winners > 0) & (winners < 3)]
        if not len(cut):
            continue
        start = cut
        end = np.roll(cut, -1, axis=1)
        edge = (delta[start] >= 0) != (delta[end] >= 0)
        a, b = start[edge], end[edge]
        t = delta[a] / (delta[a] - delta[b])
        crossings = (xy[a] + t[:, None]*(xy[b]-xy[a])).reshape(-1, 2, 2)
        third = surface.p[:, i] - surface.p[:, k]
        gaps = (third[a] + t*(third[b]-third[a])).reshape(-1, 2)
        # Only draw the part of this pairwise tie on which both scores lead.
        for pt, gap in zip(crossings, gaps):
            if max(gap) < 0:
                continue
            if min(gap) < 0:
                mix = gap[0] / (gap[0]-gap[1])
                pt[np.argmin(gap)] = pt[0] + mix*(pt[1]-pt[0])
            segments.append(pt)
    return segments


def records(records_):
    return np.array([(float(r["x1"]), float(r["x2"]), int(r["label"]), int(r.get("pred", -1)))
                     for r in records_])


def label_panel(ax, index, name, right=""):
    ax.set_title(f"({chr(97+index)})  {name}", fontsize=10, loc="left", pad=5)
    if right:
        ax.set_title(right, fontsize=8, loc="right", pad=5)


def entropy_grid(surface, subdivision=5):
    """Interpolate probabilities on the same triangles before computing H(p)."""
    n = math.isqrt(len(surface.p))
    p = surface.p.reshape(n, n, 3)
    t = np.linspace(0, n-1, subdivision*(n-1)+1)
    index = np.minimum(t.astype(int), n-2)
    x, y = index[None, :], index[:, None]
    dx, dy = (t-index)[None, :, None], (t-index)[:, None, None]
    fine = ((1-np.maximum(dx, dy))*p[y, x] + np.minimum(dx, dy)*p[y+1, x+1]
            + np.maximum(dx-dy, 0)*p[y, x+1] + np.maximum(dy-dx, 0)*p[y+1, x])
    entropy = -np.sum(fine*np.log(np.maximum(fine, 1e-300)), axis=2) / math.log(3)
    xs = np.interp(t, np.arange(n), surface.mesh.x.reshape(n, n)[0])
    ys = np.interp(t, np.arange(n), surface.mesh.y.reshape(n, n)[:, 0])
    return xs, ys, entropy


def map_panel(ax, surface, data):
    mesh = surface.mesh
    xs, ys, entropy = entropy_grid(surface)
    if np.min(entropy) >= THRESHOLD:
        ax.set_facecolor(BAND)
    elif np.max(entropy) > THRESHOLD:
        ax.contourf(xs, ys, entropy, levels=(THRESHOLD, 1.01), colors=[BAND], antialiased=False)
    ax.add_collection(LineCollection(boundaries(surface), colors="#373737", linewidths=0.75))
    for k, (marker, color) in enumerate(zip(MARKS, CLASS_COLORS)):
        sample = data[data[:, 2] == k]
        ax.scatter(sample[:, 0], sample[:, 1], s=9, marker=marker, facecolors="none",
                   edgecolors=color, linewidths=0.61)
    wrong = data[(data[:, 3] >= 0) & (data[:, 3] != data[:, 2])]
    ax.scatter(wrong[:, 0], wrong[:, 1], marker="x", color="black", s=24, linewidths=0.8)
    ax.set(xlim=(mesh.x.min(), mesh.x.max()), ylim=(mesh.y.min(), mesh.y.max()),
           xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")
    ax.xaxis.set_major_locator(FixedLocator([-1, -0.5, 0, 0.5, 1]))
    ax.yaxis.set_major_locator(FixedLocator([-1, -0.5, 0, 0.5, 1]))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))


def panels(count, height=3.45):
    fig, axes = plt.subplots(1, count, figsize=(3.7*count, height), squeeze=False)
    fig.subplots_adjust(left=0.115 if count == 1 else 0.085, right=0.972,
                        bottom=0.18, top=0.915, wspace=0.29)
    return fig, axes[0]


def map_figure(models, displayed, points, status, seed, include_errors):
    fig, axes = panels(len(models), height=3.95 if len(models) == 2 else 4.25)
    fig.subplots_adjust(bottom=0.275, top=0.90)
    for i, (model, surface, ax, pts) in enumerate(zip(models, displayed, axes, points)):
        map_panel(ax, surface, pts)
        label_panel(ax, i, model["name"], f"{model['unit']} {surface.step}")
    handles = [Line2D([], [], ls="none", marker=m, mfc="none", mec=c, ms=4,
                      label=f"Klasse {i}") for i, (m, c) in enumerate(zip(MARKS, CLASS_COLORS))]
    handles.append(Line2D([], [], color="0.2", lw=0.8, label="Klassengrenze"))
    handles.append(Patch(facecolor=BAND, edgecolor="none", label=r"$H/\ln 3 \geq 0{,}5$"))
    if include_errors:
        handles.append(Line2D([], [], ls="none", marker="x", color="black", ms=4, label="Fehler"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.055),
               ncol=6 if len(models) == 2 else 3, columnspacing=1.05, handletextpad=0.35)
    fig.text(0.5, 0.025, f"{status}    ·    Seed {seed}", ha="center", va="center", fontsize=8)
    return fig


def save_figure(fig, out, name):
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    with (out / f"{name}.svg").open("w", encoding="utf-8", newline="\n") as dest:
        dest.writelines(line.rstrip() + "\n" for line in svg.getvalue().splitlines())
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def training(models, out):
    fig, axes = panels(len(models), height=3.35)
    fig.subplots_adjust(bottom=0.18, top=0.90, wspace=0.35)
    vals = [float(r[k]) for model in models for r in model["history"] for k in ("train_loss", "val_loss")]
    if min(vals) <= 0:
        raise ValueError("Für eine logarithmische Verlustachse muss die Kreuzentropie positiv sein.")
    low, high = min(vals)*0.8, max(vals)*1.2
    for i, (ax, model) in enumerate(zip(axes, models)):
        history, best = model["history"], model["best"]
        step = [int(r[model["key"]]) for r in history]
        y = [float(r["val_loss"]) for r in history]
        ax.semilogy(step, [float(r["train_loss"]) for r in history], color="black",
                    lw=1.05, label="Training")
        ax.semilogy(step, y, "--", color=CLASS_COLORS[2], lw=1.12, label="Validierung")
        ax.plot(best, y[step.index(best)], ls="none", marker="o", ms=5,
                mfc="white", mec=CLASS_COLORS[2], label="Gewählter Stand")
        ax.set(xlim=(0, max(step)), ylim=(low, high), xlabel=model["unit"], ylabel="Kreuzentropie")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))
        ax.legend(loc="upper right", handletextpad=0.45)
        label_panel(ax, i, model["name"])
    save_figure(fig, out, "training")


def comparison(data, out):
    fig, axes = panels(2, height=3.35)
    fig.subplots_adjust(bottom=0.18, top=0.90, wspace=0.36)
    seeds = np.array([int(r["seed"]) for r in data])
    for i, (ax, key, name, factor) in enumerate(zip(axes, ("test_loss", "test_accuracy"),
               ("Test-Kreuzentropie", "Trefferquote (%)"), (1, 100))):
        mlp = [factor*float(r[key]) for r in data]
        radial = [factor*float(r["radial_"+key]) for r in data]
        for x, a, b in zip(seeds, mlp, radial):
            ax.plot([x, x], [a, b], color="0.63", lw=0.8, zorder=1)
        ax.plot(seeds, mlp, ls="none", marker="s", ms=5.5, mfc="white", mec=CLASS_COLORS[1], label="MLP")
        ax.plot(seeds, radial, ls="none", marker="o", ms=5.5, mfc="white", mec=CLASS_COLORS[2], label="Radialnetz")
        ax.set(xticks=seeds, xlim=(seeds.min()-0.4, seeds.max()+0.4), xlabel="Seed", ylabel=name)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        if i == 0:
            ax.set_ylim(0, max(max(mlp), max(radial))*1.24)
        else:
            ax.set_ylim(math.floor(min(min(mlp), min(radial)))-0.3, 100.45)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        label_panel(ax, i, "Kreuzentropie" if i == 0 else "Trefferquote")
        if i == 0:
            ax.legend(loc="upper left", handletextpad=0.3)
    save_figure(fig, out, "vergleich")


def render(run, out, study):
    summary = rows(run / "summary.csv")[0]
    models = []
    configs = [("MLP", "", "epoch", "Epoche", "best_epoch")]
    if "radial_step" in summary:
        configs.append(("Radialnetz", "radial_", "step", "Newton-Schritt", "radial_step"))
    for name, prefix, key, unit, chosen in configs:
        model = {"name": name, "unit": unit, "key": key, "best": int(summary[chosen]),
                 "surfaces": read_surfaces(run / (prefix+"surface.csv")),
                 "test": records(rows(run / (prefix+"predictions.csv"))),
                 "history": rows(run / (prefix+"history.csv"))}
        if model["surfaces"][-1].step != model["best"]:
            raise ValueError("Letzter Modellstand stimmt nicht mit dem gewählten Checkpoint überein.")
        models.append(model)
    out.mkdir(parents=True, exist_ok=True)
    tests = [m["test"] for m in models]
    best = [m["surfaces"][-1] for m in models]
    status = f"Ausgewählter Stand · {summary['test_n']} Testpunkte"
    save_figure(map_figure(models, best, tests, status, summary["seed"], True), out, "entscheidungsgrenzen")
    training(models, out)
    study_path = study or run / "vergleich.csv"
    if study or study_path.exists():
        data = rows(study_path)
        if len(data) != 5:
            raise ValueError("Der Vergleich erwartet fünf Läufe.")
        comparison(data, out)
    train = records(r for r in rows(run / "points.csv") if r["split"] == "train")
    count = max(len(m["surfaces"])-1 for m in models)
    frames = []
    for step in range(count+1):
        final = step == count
        if final:
            surfaces = best
        else:
            surfaces = [m["surfaces"][round(step*(len(m["surfaces"])-2)/max(count-1, 1))] for m in models]
        fig = map_figure(models, surfaces, tests if final else [train]*len(models),
                         status if final else f"Training · {len(train)} Punkte", summary["seed"], final)
        fig.set_dpi(150)
        fig.canvas.draw()
        im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE)
        frames.append(im)
        plt.close(fig)
    durations = [430]*len(frames)
    durations[0], durations[-1] = 1200, 2400
    frames[0].save(out / "lernprozess.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description="Publikationsgrafiken aus den C-Ergebnissen zeichnen.")
    parser.add_argument("--run", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument("--study", type=Path, help="CSV aus spiralnet study")
    args = parser.parse_args()
    try:
        render(args.run, args.out, args.study)
    except (OSError, KeyError, IndexError, ValueError) as exc:
        parser.exit(1, f"Visualisierung fehlgeschlagen: {exc}\n")
    print(f"Grafiken gespeichert in {args.out.resolve()}")


if __name__ == "__main__":
    main()
