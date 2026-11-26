"""Figures from the C classifiers' grid predictions and experimental records."""

import argparse
import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont


NIGHT, LIGHT, PALE, HAIR = "#101923", "#F0F4F6", "#B9CCD3", "#364756"
PAPER, TEXT, SOFT, RULE = "#F5F6F1", "#172833", "#465A66", "#CAD3CF"
COLORS = ("#56C5F1", "#FFA15E", "#CEA1F7")
MLP, RADIAL = "#AD4E2F", "#007270"
TRAIN, VALID = "#496879", "#8A3D85"


@lru_cache(maxsize=None)
def face(size, style):
    options = {
        "text": ("segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
        "heavy": ("segoeuib.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"),
        "mono": ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"),
    }
    for name in options[style]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def text(draw, x, y, value, size, color, style="text", anchor="lt"):
    draw.text((x, y), str(value), font=face(size, style), fill=color, anchor=anchor)


def number(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def fraction(value):
    return number(100 * float(value), 1) + " %"


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



def page(title, subtitle, size, dark=False):
    paper, ink, soft, rule = (NIGHT, LIGHT, PALE, HAIR) if dark else (PAPER, TEXT, SOFT, RULE)
    im = Image.new("RGB", size, paper)
    d = ImageDraw.Draw(im)
    text(d, 64, 40, title, 52, ink, "heavy")
    text(d, 67, 118, subtitle, 22, soft)
    d.line((66, 168, size[0] - 66, 168), fill=rule, width=2)
    return im


def mark(draw, x, y, k, r, fill, outline):
    if k == 0:
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline=outline)
    elif k == 1:
        draw.polygon(((x, y-r-1), (x-r-1, y+r), (x+r+1, y+r)), fill=fill, outline=outline)
    else:
        draw.rectangle((x-r, y-r, x+r, y+r), fill=fill, outline=outline)


def map_tile(im, xy, s, surf, points, small=False):
    """The brightest area in a class has low entropy. Tied logits stay neutral."""
    p = surf.upsample(s)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=-1) / math.log(3)
    certainty = np.clip(1.0 - entropy, 0, 1)[..., None]
    palette = np.array([ImageColor.getrgb(c) for c in COLORS], dtype=float)
    base = np.array([25, 38, 51], dtype=float)
    # A probability-weighted hue avoids a false class at a three-way tie.
    hue = p @ palette
    pixels = base + 0.49 * certainty ** 0.7 * (hue - base)
    tile = Image.fromarray(pixels.astype("uint8"))
    d = ImageDraw.Draw(tile, "RGBA")
    x0, x1 = surf.xs[[0, -1]]
    y0, y1 = surf.ys[[0, -1]]
    tx = lambda x: round((x - x0) / (x1 - x0) * (s - 1))
    ty = lambda y: round((y1 - y) / (y1 - y0) * (s - 1))
    ticks = [t for t in (-1, -0.5, 0, 0.5, 1) if x0 <= t <= x1 and y0 <= t <= y1]
    for t in ticks:
        alpha = 76 if t == 0 else 29
        d.line((tx(t), 0, tx(t), s-1), fill=(217, 229, 232, alpha))
        d.line((0, ty(t), s-1, ty(t)), fill=(217, 229, 232, alpha))
    # Draw error halos last, so nearby data cannot cover them.
    wrong = []
    for px, py, actual, pred in points:
        x, y = tx(px), ty(py)
        mark(d, x, y, actual, 3 if small else 4, COLORS[actual], NIGHT)
        if pred is not None and pred != actual:
            wrong.append((x, y))
    for x, y in wrong:
        r = 6 if small else 8
        d.ellipse((x-r, y-r, x+r, y+r), outline=LIGHT, width=1 if small else 2)
    im.paste(tile, xy)
    d = ImageDraw.Draw(im)
    left, top = xy
    d.rectangle((left, top, left+s-1, top+s-1), outline="#71818E")
    fs = 15 if small else 18
    for tick in ticks:
        tickname = f"{tick:g}".replace(".", ",").replace("-", "−")
        text(d, left+tx(tick), top+s+7, tickname, fs, PALE, "mono", "mt")
        text(d, left-8, top+ty(tick), tickname, fs, PALE, "mono", "rm")
    text(d, left+s+8, top+s-6, "x₁", fs+1, PALE, anchor="lm")
    text(d, left-9, top-8, "x₂", fs+1, PALE, anchor="rb")


def legend(d, x, y, compact=False):
    spacing = 100 if compact else 142
    for k in range(3):
        mark(d, x+k*spacing, y, k, 6, COLORS[k], NIGHT)
        text(d, x+13+k*spacing, y, f"Klasse {k}", 16 if compact else 20, LIGHT, anchor="lm")


def samples(records):
    return [(float(r["x1"]), float(r["x2"]), int(r["label"]), int(r["pred"]) if "pred" in r else None)
            for r in records]


def decisions(path, nets, summary):
    paired = len(nets) == 2
    im = page("Was die Drehsymmetrie verändert" if paired else "Die Entscheidungsfläche des MLP",
              f"Entscheidungsflächen auf denselben {summary['test_n']} Testpunkten  /  Seed {summary['seed']}",
              (1500 if paired else 820, 990), dark=True)
    d = ImageDraw.Draw(im)
    for i, (name, surfaces, _, predictions) in enumerate(nets):
        left = (115, 838)[i]
        acc = fraction(summary[("test_accuracy", "radial_test_accuracy")[i]])
        loss = number(float(summary[("test_loss", "radial_test_loss")[i]]), 4)
        text(d, left, 203, name, 33, LIGHT, "heavy")
        text(d, left, 255, f"{acc} richtig    /    Kreuzentropie {loss}", 20, PALE, "mono")
        map_tile(im, (left, 317), 548, surfaces[-1], predictions)
    legend(d, 115, 940)
    d.ellipse((581, 932, 597, 948), outline=LIGHT, width=2)
    text(d, 608, 940, "Testfehler", 20, LIGHT, anchor="lm")
    if paired:
        text(d, 838, 940, "Dunkle Übergänge zeigen hohe Vorhersageentropie", 19, PALE, anchor="lm")
    im.save(path)


def curve(d, xs, ys, color, dashed=False, width=4):
    remaining, painting = 10.0, True
    for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]):
        if not dashed:
            d.line((x0, y0, x1, y1), fill=color, width=width)
            continue
        dx, dy = x1-x0, y1-y0
        length = math.hypot(dx, dy)
        pos = 0.0
        while pos < length:
            step = min(length-pos, remaining)
            if painting:
                a, b = pos/length, (pos+step)/length
                d.line((x0+a*dx, y0+a*dy, x0+b*dx, y0+b*dy), fill=color, width=width)
            pos += step
            remaining -= step
            if remaining < 1e-8:
                painting = not painting
                remaining = 10.0 if painting else 7.0


def loss_plot(d, left, records, best, xlabel, low, high):
    key = "step" if "step" in records[0] else "epoch"
    steps = np.array([int(r[key]) for r in records])
    tr = np.array([float(r["train_loss"]) for r in records])
    va = np.array([float(r["val_loss"]) for r in records])
    x0, x1, y0, y1 = left+30, left+541, 343, 635
    xmax = max(steps[-1], 1)
    sx = lambda x: x0 + (x1-x0) * x / xmax
    sy = lambda y: y1 - (y1-y0) * (np.log10(y)-math.log10(low)) / (math.log10(high)-math.log10(low))
    for exp in range(math.floor(math.log10(low)), math.ceil(math.log10(high))+1):
        for mantissa in (1, 2, 5):
            tick = mantissa * 10.0 ** exp
            if low <= tick <= high:
                yy = sy(tick)
                d.line((x0, yy, x1, yy), fill=RULE)
                text(d, x0-9, yy, f"{tick:g}".replace(".", ","), 19, SOFT, "mono", "rm")
    for tick in sorted({round(v) for v in np.linspace(0, xmax, min(5, xmax+1))}):
        text(d, sx(tick), y1+13, tick, 18, SOFT, "mono", "mt")
    curve(d, sx(steps), sy(tr), TRAIN, width=3)
    curve(d, sx(steps), sy(va), VALID, dashed=True, width=4)
    xx, yy = sx(best), sy(va[steps == best][0])
    d.ellipse((xx-6, yy-6, xx+6, yy+6), fill=PAPER, outline=VALID, width=3)
    text(d, (x0+x1)/2, y1+57, xlabel, 20, TEXT, anchor="mt")


def training(path, nets, summary):
    paired = len(nets) == 2
    im = page("Wann sich weiteres Training lohnt" if paired else "Der Trainingsverlauf des MLP",
              "Kreuzentropie auf Training und Validierung  /  logarithmische Skala",
              (1500 if paired else 840, 856))
    d = ImageDraw.Draw(im)
    lo = min(float(row[key]) for _, _, h, _ in nets for row in h for key in ("train_loss", "val_loss"))
    hi = max(float(row[key]) for _, _, h, _ in nets for row in h for key in ("train_loss", "val_loss"))
    low = min(0.02, lo * 0.85)
    high = max(1.5, hi * 1.1)
    for i, (name, _, h, _) in enumerate(nets):
        x = (110, 838)[i]
        step = int(summary[("best_epoch", "radial_step")[i]])
        text(d, x, 208, name, 32, TEXT, "heavy")
        text(d, x, 261, f"Ausgewählt in {('Epoche', 'Newton-Schritt')[i]} {step}", 21, SOFT)
        loss_plot(d, x, h, step, ("Epoche", "Newton-Schritt")[i], low, high)
    d.line((65, 746, im.width-65, 746), fill=RULE, width=2)
    d.line((100, 796, 145, 796), fill=TRAIN, width=3)
    text(d, 157, 796, f"Training · {summary['train_n']} Punkte", 21, TEXT, anchor="lm")
    curve(d, [518, 572], [796, 796], VALID, dashed=True)
    text(d, 583, 796, f"Validierung · {summary['val_n']} Punkte", 21, TEXT, anchor="lm")
    if paired:
        d.ellipse((1027, 787, 1045, 805), fill=PAPER, outline=VALID, width=3)
        text(d, 1060, 796, "Geringster Validierungsverlust", 21, TEXT, anchor="lm")
    im.save(path)


def study_chart(path, rows):
    im = page("Der Vergleich über fünf Stichproben",
              "Jede Zeile zeigt beide Modelle auf identischen Testpunkten. Die horizontale Lage gibt den Messwert an.",
              (1500, 805))
    d = ImageDraw.Draw(im)
    metrics = [("Test-Kreuzentropie", "test_loss", 1.0), ("Test-Trefferquote (%)", "test_accuracy", 100.0)]
    text(d, 66, 284, "Seed", 20, SOFT, "mono")
    for col, (name, key, scale) in enumerate(metrics):
        left, right = [(199, 707), (857, 1364)][col]
        text(d, left, 210, name, 29, TEXT, "heavy")
        values = [[scale*float(r[k]) for k in (key, "radial_"+key)] for r in rows]
        if col == 0:
            largest = max(v for pair in values for v in pair)
            magnitude = 10 ** math.floor(math.log10(max(largest / 5, 1e-6)))
            unit = next(s * magnitude for s in (1, 2, 5, 10) if s*magnitude >= largest/5)
            low, high = 0, max(unit, math.ceil(largest / unit) * unit)
            ticks = np.arange(0, high + unit/2, unit)
        else:
            low, high = math.floor(min(v for pair in values for v in pair))-1, 100
            ticks = np.linspace(low, high, 5)
        sx = lambda value: left+(right-left)*(value-low)/(high-low)
        for tick in ticks:
            xx = sx(tick)
            d.line((xx, 316, xx, 650), fill=RULE)
            text(d, xx, 273, number(tick, 2 if col == 0 else 1), 18, SOFT, "mono", "mt")
        for i, (row, (baseline, radial)) in enumerate(zip(rows, values)):
            y = 351+i*65
            if col == 0:
                text(d, 72, y, row["seed"], 25, TEXT, "mono", "lm")
            xx, rx = sx(baseline), sx(radial)
            d.line((xx,y,rx,y), fill="#81919A", width=3)
            d.rectangle((xx-6,y-6,xx+6,y+6), fill=MLP)
            d.ellipse((rx-7,y-7,rx+7,y+7), fill=RADIAL, outline=PAPER)
    d.line((65, 682, 1435, 682), fill=RULE, width=2)
    d.rectangle((83,735,99,751), fill=MLP)
    text(d, 112, 743, "MLP", 23, TEXT, anchor="lm")
    d.ellipse((262,735,278,751), fill=RADIAL)
    text(d, 291, 743, "Radialnetz", 23, TEXT, anchor="lm")
    text(d, 1431, 743, f"{rows[0]['test_n']} Testpunkte je Stichprobe", 20, SOFT, anchor="rm")
    im.save(path)


def animate(path, nets, train, summary):
    frames = []
    count = max(len(surfaces)-1 for _, surfaces, _, _ in nets)
    paired = len(nets) == 2
    for fi in range(count+1):
        final = fi == count
        im = Image.new("RGB", (1100 if paired else 620, 717), NIGHT)
        d = ImageDraw.Draw(im)
        text(d, 37, 27, "Zwei Wege zur Spiralgrenze" if paired else "Wie das MLP lernt", 42, LIGHT, "heavy")
        msg = (f"Auswahl nach Validierungsverlust  /  {summary['test_n']} Testpunkte" if final
               else f"Trainingsverlauf  /  {summary['train_n']} Trainingspunkte")
        if paired and not final:
            msg += "  /  getrennte Trainingsskalen"
        text(d, 40, 90, msg, 20, PALE)
        d.line((39, 132, im.width-40, 132), fill=HAIR, width=2)
        for i, (name, surfaces, _, pred) in enumerate(nets):
            idx = -1 if final else round(fi*(len(surfaces)-2)/max(count-1, 1))
            surf = surfaces[idx]
            left = (83, 634)[i]
            text(d, left, 162, name, 27, LIGHT, "heavy")
            text(d, left, 208, f"{('Epoche', 'Newton-Schritt')[i]} {surf.epoch}", 19, PALE, "mono")
            map_tile(im, (left, 259), 384, surf, pred if final else train, small=True)
        legend(d, 83, 693, compact=True)
        if paired:
            text(d, 634, 693, "Weiße Ringe markieren Testfehler." if final else "Farbe und Symbolform zeigen die Klasse.",
                 17, PALE, anchor="lm")
        frames.append(im.quantize(colors=160, dither=Image.Dither.NONE))
    durations = [380] * len(frames)
    durations[0], durations[-1] = 1100, 2600
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser(description="Die C-Vorhersagen und Messreihen als Grafiken rendern.")
    ap.add_argument("--run", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("assets"))
    ap.add_argument("--study", type=Path, help="CSV aus spiralnet study (Standard: vergleich.csv im Laufordner)")
    args = ap.parse_args()
    try:
        summary = read_rows(args.run / "summary.csv")[0]
        nets = []
        models = [("MLP", "")]
        if "radial_step" in summary:
            models.append(("Radialnetz", "radial_"))
        for name, prefix in models:
            surfaces = read_surfaces(args.run / (prefix+"surface.csv"))
            history = read_rows(args.run / (prefix+"history.csv"))
            pred = samples(read_rows(args.run / (prefix+"predictions.csv")))
            best = int(summary["best_epoch" if not prefix else "radial_step"])
            if surfaces[-1].epoch != best:
                raise ValueError(f"Ausgewählter Modellstand für {name} stimmt nicht mit summary.csv überein.")
            nets.append((name, surfaces, history, pred))
        train = samples(r for r in read_rows(args.run / "points.csv") if r["split"] == "train")
        args.out.mkdir(parents=True, exist_ok=True)
        decisions(args.out / "entscheidungsgrenzen.png", nets, summary)
        training(args.out / "training.png", nets, summary)
        animate(args.out / "lernprozess.gif", nets, train, summary)
        study = args.study or args.run / "vergleich.csv"
        if args.study or study.exists():
            rows = read_rows(study)
            if len(rows) != 5:
                raise ValueError("Der Stichprobenvergleich erwartet fünf Läufe.")
            study_chart(args.out / "vergleich.png", rows)
    except (OSError, KeyError, IndexError, ValueError) as exc:
        ap.exit(1, f"Visualisierung fehlgeschlagen: {exc}\n")
    print(f"Grafiken gespeichert in {args.out.resolve()}")


if __name__ == "__main__":
    main()
