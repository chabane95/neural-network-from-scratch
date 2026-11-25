"""Render the surfaces, metrics and held-out predictions exported by the C program."""

import argparse
import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont


PAPER = "#FAF9F5"
INK = "#182E30"
MUTED = "#455F62"
RULE = "#BCCBCD"
GRID = "#DFE5E3"
CLASSES = ("#146E7A", "#BB5939", "#674C91")
FIELDS = ("#9AD2D1", "#F2C5AB", "#CCBEE3")
TRAIN, VAL = "#1C7181", "#A64C2D"
# Perceptually ordered dark-to-light ramp for normalized entropy.
ENTROPY = ("#18364C", "#286479", "#4A918D", "#97B790", "#E7D78E", "#FFF2BB")


@lru_cache(maxsize=None)
def font(size, role="body"):
    faces = {
        "body": ("segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
        "bold": ("segoeuib.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"),
        "title": ("georgia.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf"),
    }
    for face in faces[role]:
        try:
            return ImageFont.truetype(face, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def label(draw, xy, value, size=20, color=INK, role="body", anchor="lt"):
    draw.text(xy, str(value), font=font(size, role), fill=color, anchor=anchor)


def decimal(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def percent(value):
    return decimal(100 * value) + " %"


def rgb(colors):
    return np.array([ImageColor.getrgb(c) for c in colors])


def ramp(values, colors):
    stops = rgb(colors)
    scaled = np.clip(values, 0, 1) * (len(stops) - 1)
    lo = np.minimum(scaled.astype(int), len(stops) - 2)
    diff = (scaled - lo)[..., None]
    return np.uint8(np.rint(stops[lo] * (1 - diff) + stops[lo + 1] * diff))


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@dataclass
class Surface:
    epoch: int
    xs: np.ndarray
    ys: np.ndarray
    probabilities: np.ndarray

    def upsample(self, size):
        """Bilinear interpolation of exported probabilities in data coordinates."""
        x = np.interp(np.linspace(self.xs[0], self.xs[-1], size), self.xs, np.arange(len(self.xs)))
        y = np.interp(np.linspace(self.ys[-1], self.ys[0], size), self.ys, np.arange(len(self.ys)))
        left = np.minimum(x.astype(int), len(self.xs) - 2)
        bottom = np.minimum(y.astype(int), len(self.ys) - 2)
        dx, dy = (x - left)[None, :, None], (y - bottom)[:, None, None]
        horizontal = self.probabilities[:, left] * (1 - dx) + self.probabilities[:, left + 1] * dx
        return horizontal[bottom] * (1 - dy) + horizontal[bottom + 1] * dy


def read_surfaces(path):
    with path.open(encoding="utf-8") as stream:
        if stream.readline().strip() != "frame,epoch,x1,x2,p0,p1,p2":
            raise ValueError(f"Unerwarteter Tabellenkopf in {path}")
        data = np.loadtxt(stream, delimiter=",", ndmin=2)
    groups = np.split(data, np.flatnonzero(np.diff(data[:, 0])) + 1)
    surfaces = []
    for block in groups:
        n = math.isqrt(len(block))
        if n < 2 or n * n != len(block):
            raise ValueError(f"Unvollständiges Gitter bei Frame {block[0, 0]:g}")
        grid = block.reshape(n, n, 7)
        xs, ys = grid[0, :, 2], grid[:, 0, 3]
        if (not np.all(grid[:, :, 2] == xs) or not np.all(grid[:, :, 3] == ys[:, None])
                or np.any(np.diff(xs) <= 0) or np.any(np.diff(ys) <= 0)):
            raise ValueError(f"Ungültige Gitterkoordinaten bei Frame {block[0, 0]:g}")
        surfaces.append(Surface(int(block[0, 1]), xs, ys, grid[:, :, 4:]))
    return surfaces


def header(draw, title, subtitle, width):
    label(draw, (62, 38), title, 55, role="title")
    label(draw, (65, 111), subtitle, 22, MUTED)
    draw.line((64, 159, width - 64, 159), fill=RULE, width=2)


def symbol(draw, xy, group, radius):
    x, y = xy
    r = radius
    if group == 0:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=CLASSES[0], outline=PAPER, width=1)
    elif group == 1:
        draw.polygon([(x, y - r - 1), (x - r - 1, y + r), (x + r + 1, y + r)],
                     fill=CLASSES[1], outline=PAPER, width=1)
    else:
        draw.rectangle((x - r, y - r, x + r, y + r), fill=CLASSES[2], outline=PAPER, width=1)


def coordinate_ticks(low, high):
    step = nice_step((high - low) / 4.8)
    return np.arange(math.ceil(low / step - 1e-9), math.floor(high / step + 1e-9) + 1) * step


def nice_step(rough):
    magnitude = 10 ** math.floor(math.log10(rough))
    for multiple in (1, 2, 2.5, 5, 10):
        if multiple * magnitude >= rough - 1e-12:
            return multiple * magnitude


def compact_number(value):
    return f"{value:.4g}".replace(".", ",").replace("-", "−") if abs(value) > 1e-12 else "0"


def map_panel(image, surface, probabilities, box, points=(), entropy=False, small=False):
    left, top, size = box
    draw = ImageDraw.Draw(image)
    # Colors are assigned after interpolation; no artificial model evaluations.
    if entropy:
        safe = np.maximum(probabilities, 1e-300)
        values = -np.sum(probabilities * np.log(safe), axis=2) / math.log(3)
        pixels = ramp(values, ENTROPY)
    else:
        pixels = np.uint8(rgb(FIELDS)[np.argmax(probabilities, axis=2)])
    tile = Image.fromarray(pixels)
    marks = ImageDraw.Draw(tile)
    x0, x1 = surface.xs[[0, -1]]
    y0, y1 = surface.ys[[0, -1]]
    sx = lambda x: round((x - x0) * (size - 1) / (x1 - x0))
    sy = lambda y: round((y1 - y) * (size - 1) / (y1 - y0))
    for x, y, actual in points:
        px, py = sx(x), sy(y)
        if entropy:
            for color, w in ((PAPER, 5), (INK, 2)):
                marks.line((px - 5, py - 5, px + 5, py + 5), fill=color, width=w)
                marks.line((px + 5, py - 5, px - 5, py + 5), fill=color, width=w)
        else:
            symbol(marks, (px, py), actual, 3 if small else 5)
    image.paste(tile, (left, top))
    right, last = left + size - 1, top + size - 1
    draw.rectangle((left, top, right, last), outline="#788D91")
    fs = 15 if small else 18
    for tick in coordinate_ticks(x0, x1):
        xx = left + sx(tick)
        draw.line((xx, last, xx, last + 6), fill=MUTED)
        label(draw, (xx, last + 11), compact_number(tick), fs, MUTED, anchor="mt")
    for tick in coordinate_ticks(y0, y1):
        yy = top + sy(tick)
        draw.line((left - 6, yy, left, yy), fill=MUTED)
        label(draw, (left - 11, yy), compact_number(tick), fs, MUTED, anchor="rm")
    label(draw, (right + 9, last + 4), "x₁", fs + 2, role="bold")
    label(draw, (left - 11, top - 10), "x₂", fs + 2, role="bold", anchor="rb")


def legend(draw, xy, spacing=130, size=19):
    x, y = xy
    for k in range(3):
        symbol(draw, (x + k * spacing + 7, y), k, 7)
        label(draw, (x + k * spacing + 22, y), f"Klasse {k}", size, anchor="lm")


def coordinates(rows):
    return [(float(r["x1"]), float(r["x2"]), int(r["label"])) for r in rows]


def decision_figure(path, surface, predictions, summary):
    image = Image.new("RGB", (1480, 1010), PAPER)
    d = ImageDraw.Draw(image)
    header(d, "Wie das Netz die Spiralen trennt",
           f"Drei-Klassen-Klassifikation   ·   Ausgewähltes Modell aus Epoche {surface.epoch}   ·   "
           f"{summary['test_n']} unabhängige Testpunkte", image.width)
    label(d, (110, 194), "Entscheidungsgebiete", 31, role="bold")
    label(d, (803, 194), "Unsicherheit des Modells", 31, role="bold")
    label(d, (110, 244), "Flächen zeigen die jeweils wahrscheinlichste Klasse.", 20, MUTED)
    label(d, (803, 244), "Normierte Entropie der drei Klassenwahrscheinlichkeiten.", 20, MUTED)
    size = 556
    p = surface.upsample(size)
    errors = [r for r in predictions if r["pred"] != r["label"]]
    map_panel(image, surface, p, (112, 301, size), coordinates(predictions))
    map_panel(image, surface, p, (805, 301, size), coordinates(errors), entropy=True)
    legend(d, (112, 918), spacing=164)
    label(d, (112, 952), "Symbolform und -farbe geben die wahre Klasse an.", 19, MUTED)
    grad = np.broadcast_to(np.linspace(0, 1, size), (13, size))
    image.paste(Image.fromarray(ramp(grad, ENTROPY)), (805, 905))
    label(d, (805, 925), "0  ·  eindeutig", 17, MUTED)
    label(d, (1361, 925), "1  ·  gleichverteilt", 17, MUTED, anchor="rt")
    label(d, (805, 957), f"×  Fehlklassifizierte Testpunkte ({len(errors)})", 19)
    image.save(path)


def series_key(draw, x, y, size=18):
    for off, color, name in ((0, TRAIN, "Training"), (138, VAL, "Validierung")):
        draw.line((x + off, y, x + off + 28, y), fill=color, width=4)
        label(draw, (x + off + 35, y), name, size, anchor="lm")


def plot_metrics(draw, box, epochs, train, val, best, accuracy=False, small=False, until=None):
    left, top, width, height = box
    right, bottom = left + width, top + height
    xmax = max(1, int(epochs[-1]))
    step = 25 if accuracy else nice_step(max(max(train), max(val), 1e-6) / (2.8 if small else 3.5))
    ymax = 100 if accuracy else step * math.ceil(max(max(train), max(val), 1e-6) / step)
    fs = 13 if small else 17
    xx = lambda e: left + width * e / xmax
    yy = lambda v: bottom - height * v / ymax
    for tick in np.arange(0, ymax + step / 2, step):
        draw.line((left, yy(tick), right, yy(tick)), fill=GRID)
        label(draw, (left - 9, yy(tick)), compact_number(tick), fs, MUTED, anchor="rm")
    xstep = max(1, math.ceil(nice_step(xmax / (3.6 if small else 6))))
    ticks = list(np.arange(0, xmax + 0.001, xstep))
    if xmax - ticks[-1] > 0.56 * xstep:
        ticks.append(xmax)
    for tick in ticks:
        draw.line((xx(tick), bottom, xx(tick), bottom + 4), fill=MUTED)
        label(draw, (xx(tick), bottom + 8), str(int(tick)), fs, MUTED, anchor="mt")
    if best is not None:
        for y in range(top, bottom, 10):
            draw.line((xx(best), y, xx(best), min(y + 5, bottom)), fill=MUTED, width=2)
    for series, color in ((train, TRAIN), (val, VAL)):
        factor = 100 if accuracy else 1
        pts = [(xx(e), yy(v * factor)) for e, v in zip(epochs, series) if until is None or e <= until]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2 if small else 3, joint="curve")
        if pts:
            x, y = pts[-1]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def training_figure(path, history, predictions, summary):
    image = Image.new("RGB", (1480, 950), PAPER)
    d = ImageDraw.Draw(image)
    best = int(summary["best_epoch"])
    header(d, "Lernkurven und unabhängiger Test",
           f"{summary['train_n']} Trainingspunkte   ·   {summary['val_n']} Validierungspunkte   ·   "
           f"{summary['test_n']} Testpunkte   ·   Seed {summary['seed']}", image.width)
    label(d, (65, 192), "Verlauf des Trainings", 31, role="bold")
    label(d, (850, 192), "Konfusionsmatrix", 31, role="bold")
    series_key(d, 68, 265)
    d.line((360, 265, 371, 265), fill=MUTED, width=2)
    d.line((376, 265, 387, 265), fill=MUTED, width=2)
    label(d, (397, 265), f"Auswahl · Epoche {best}", 18, anchor="lm")
    e = history["epoch"]
    label(d, (68, 315), "Kreuzentropie", 21, role="bold")
    plot_metrics(d, (123, 372, 617, 152), e, history["train_loss"], history["val_loss"], best)
    label(d, (68, 595), "Trefferquote (%)", 21, role="bold")
    plot_metrics(d, (123, 649, 617, 149), e, history["train_accuracy"], history["val_accuracy"], best, True)
    label(d, (432, 842), "Epoche", 19, MUTED, anchor="mt")

    matrix = np.zeros((3, 3), dtype=int)
    for row in predictions:
        matrix[int(row["label"]), int(row["pred"])] += 1
    label(d, (850, 252), "Anzahl der Testpunkte je Klassenpaar", 20, MUTED)
    left, top, cell = 982, 381, 100
    label(d, (left + 1.5 * cell, 306), "Vorhergesagte Klasse", 20, role="bold", anchor="mt")
    label(d, (850, 358), "Wahre Klasse", 18, MUTED)
    for i in range(3):
        label(d, (left + (i + 0.5) * cell, top - 13), str(i), 20, MUTED, anchor="mb")
        label(d, (left - 18, top + (i + 0.5) * cell), str(i), 20, MUTED, anchor="rm")
        for j in range(3):
            fraction = matrix[i, j] / max(matrix.max(), 1)
            color = tuple(ramp(np.array(fraction), ("#E9EEEC", "#146474")).tolist())
            x, y = left + j * cell, top + i * cell
            d.rectangle((x, y, x + cell - 3, y + cell - 3), fill=color)
            label(d, (x + cell / 2, y + cell / 2), str(matrix[i, j]), 32,
                  PAPER if fraction > 0.58 else INK, "bold", "mm")
    d.line((850, 724, 1410, 724), fill=RULE, width=2)
    label(d, (850, 760), percent(float(summary["test_accuracy"])), 45, TRAIN, "title")
    label(d, (1135, 760), decimal(float(summary["test_loss"]), 3), 45, TRAIN, "title")
    label(d, (850, 823), "Test-Trefferquote", 19, MUTED)
    label(d, (1135, 823), "Test-Kreuzentropie", 19, MUTED)
    label(d, (65, 902), "Modellwahl nach kleinster Validierungs-Kreuzentropie.", 19, MUTED)
    if "linear_test_accuracy" in summary:
        baseline = percent(float(summary["linear_test_accuracy"]))
        label(d, (850, 902), f"Lineare Softmax-Referenz im Test · {baseline}", 19, MUTED)
    image.save(path)


def animate(path, surfaces, history, train_points, predictions, summary):
    frames = []
    best = int(summary["best_epoch"])
    epochs = history["epoch"]
    for i, surface in enumerate(surfaces):
        selected = i == len(surfaces) - 1
        e = surface.epoch
        row = np.flatnonzero(epochs == e)
        if not len(row):
            raise ValueError(f"Trainingsmetriken für Epoche {e} fehlen.")
        index = row[-1]
        image = Image.new("RGB", (960, 626), PAPER)
        d = ImageDraw.Draw(image)
        label(d, (34, 20), "Wie eine Entscheidungsgrenze entsteht", 40, role="title")
        d.line((36, 85, 924, 85), fill=RULE, width=2)
        size = 403
        map_panel(image, surface, surface.upsample(size), (86, 125, size),
                  predictions if selected else train_points, small=True)
        legend(d, (83, 589), spacing=125, size=16)
        x = 561
        label(d, (x, 123), "Ausgewähltes Modell" if selected else "Trainingsverlauf", 22, role="bold")
        label(d, (x, 170), f"Epoche {e}", 39, TRAIN, "title")
        phase = f"{len(predictions)} Testpunkte" if selected else f"{len(train_points)} Trainingspunkte"
        label(d, (x, 229), f"Gezeigte Symbole · {phase}", 17, MUTED)
        label(d, (x, 283), "Kreuzentropie", 18, role="bold")
        plot_metrics(d, (601, 329, 295, 94), epochs, history["train_loss"], history["val_loss"],
                     best if selected else None, small=True, until=e)
        label(d, (890, 451), "Epoche", 14, MUTED, anchor="rt")
        series_key(d, x, 483, size=15)
        d.line((x, 516, 918, 516), fill=RULE)
        value = float(summary["test_accuracy"]) if selected else history["val_accuracy"][index]
        label(d, (x, 535), percent(value), 36, role="title")
        label(d, (x, 585), "Test-Trefferquote" if selected else "Validierungs-Trefferquote", 17, MUTED)
        frames.append(image.quantize(colors=128, dither=Image.Dither.NONE))
    durations = [400] * len(frames)
    durations[0], durations[-1] = 900, 2200
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description="Visualisierungen aus den CSV-Ausgaben des C-Trainings erzeugen.")
    parser.add_argument("--run", type=Path, default=Path("results"), help="Verzeichnis des C-Laufs (results)")
    parser.add_argument("--out", type=Path, default=Path("assets"), help="Ziel für PNG und GIF (assets)")
    args = parser.parse_args()
    try:
        rows = read_rows(args.run / "history.csv")
        history = {k: np.array([float(row[k]) for row in rows]) for k in rows[0]}
        summary = read_rows(args.run / "summary.csv")[0]
        points = read_rows(args.run / "points.csv")
        predictions = read_rows(args.run / "predictions.csv")
        surfaces = read_surfaces(args.run / "surface.csv")
        if surfaces[-1].epoch != int(summary["best_epoch"]):
            raise ValueError("Die letzte Fläche gehört nicht zur ausgewählten Epoche.")
        args.out.mkdir(parents=True, exist_ok=True)
        decision_figure(args.out / "entscheidungsgrenzen.png", surfaces[-1], predictions, summary)
        training_figure(args.out / "training.png", history, predictions, summary)
        animate(args.out / "lernprozess.gif", surfaces, history, coordinates(r for r in points if r["split"] == "train"),
                coordinates(predictions), summary)
    except (OSError, ValueError, IndexError, KeyError) as error:
        parser.exit(1, f"Visualisierung fehlgeschlagen: {error}\n")
    print(f"Drei Grafiken gespeichert in {args.out.resolve()}")


if __name__ == "__main__":
    main()
