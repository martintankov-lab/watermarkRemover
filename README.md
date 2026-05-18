# Watermark Remover

Batch tool that removes a watermark from every image in a folder. The watermark location is **hardcoded** in the script (rectangle measured against a 2800×2100 reference; auto-scaled to each image's real size), so you can run it on any folder without setup. Uses **LaMa** AI inpainting (a deep-learning model designed for filling masked regions) with a plain OpenCV fallback if LaMa is unavailable.

> Current hardcoded watermark: `x=2606, y=1905, w=125, h=126` against a `2800×2100` reference (bottom-right corner). To change it permanently, edit `HARDCODED_COORDS` near the top of `scripts/remove_watermark.py`. To override per-run, use `--x / --y / --w / --h / --ref-w / --ref-h`, or re-pick interactively with `--reselect`.

Repo layout:

```
watermarkRemover/
├── README.md
├── requirements.txt
├── WatermarkRemover.app/       (macOS double-click launcher)
└── scripts/
    ├── remove_watermark.py     (the actual tool)
    └── watermark_coords.json   (written by --reselect)
```

---

## 1. Requirements

- macOS or Linux
- Python 3.9+ (tested on 3.10 / 3.11)
- ~2 GB free disk (the LaMa model weights are ~200 MB and torch pulls in more)

## 2. Install

```bash
# Clone
git clone https://github.com/martintankov-lab/watermarkRemover.git
cd watermarkRemover

# (Recommended) create a virtualenv so deps don't pollute the system Python
python3 -m venv .venv
source .venv/bin/activate

# Install the libraries
pip install --upgrade pip
pip install -r requirements.txt
```

What each package is for:

| Package | Why |
| --- | --- |
| `opencv-python` | Image I/O, the region-selection window, seamless cloning, OpenCV fallback inpainting |
| `numpy` | Array math for the mask |
| `Pillow` | Image format that LaMa expects on input/output |
| `torch` | LaMa runs on PyTorch (CPU is fine; GPU is faster but not required) |
| `simple-lama-inpainting` | Thin wrapper around the LaMa model — downloads the weights automatically on first use |

> **First run will download the LaMa weights** (~200 MB) into your torch cache. This only happens once.

If `simple-lama-inpainting` fails to install or load, the script keeps working — it just falls back to OpenCV's classic `cv2.inpaint` (lower quality on textured backgrounds).

## 3. Usage

### Option A — double-click the macOS app

Double-click `WatermarkRemover.app`. It opens Terminal and asks you to paste the path to your images folder.

### Option B — run from the terminal

```bash
# Interactive: it will prompt for the input folder
python3 scripts/remove_watermark.py

# Or pass the folder directly — uses the hardcoded watermark region
python3 scripts/remove_watermark.py --input /path/to/images

# Custom output folder (default is "<input>/removed watermark")
python3 scripts/remove_watermark.py --input /path/to/images --output /path/to/out

# Override the hardcoded rectangle for a one-off run
python3 scripts/remove_watermark.py --input /path/to/images \
    --x 2606 --y 1905 --w 125 --h 126 --ref-w 2800 --ref-h 2100

# Open the picker window to draw a new rectangle (also saves it to scripts/watermark_coords.json)
python3 scripts/remove_watermark.py --input /path/to/images --reselect

# Add more padding around the mask (default 4 px)
python3 scripts/remove_watermark.py --input /path/to/images --padding 8
```

### Flow

1. The script scans the input folder for images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`).
2. It uses the **hardcoded watermark rectangle** (or your `--x/--y/--w/--h` overrides) and scales it proportionally to each image's actual dimensions.
3. Cleaned images are written to `<input>/removed watermark/` (or your `--output` folder) with the original filenames.

### Changing the hardcoded rectangle

Two ways:

- **Permanent:** edit `HARDCODED_COORDS` near the top of `scripts/remove_watermark.py`.
- **Interactive pick:** run with `--reselect`. A window opens on the first image — drag a rectangle and press:
  - `ENTER` — confirm
  - `R` — reset the rectangle
  - `Q` / `ESC` — quit

  The selection is written to `scripts/watermark_coords.json` (useful as a one-off override; to make it permanent, copy those values into `HARDCODED_COORDS`).

## 4. Troubleshooting

- **`Cannot open image: …`** — the file is corrupt or an unsupported format. Check the extension is in the list above.
- **LaMa fails to load on startup** — the script prints nothing and silently switches to OpenCV inpainting. You'll see `[warn] LaMa not available, falling back to OpenCV inpainting` per image. Reinstall with `pip install --force-reinstall simple-lama-inpainting torch`.
- **Mac asks "cannot be opened because the developer cannot be verified"** when launching `WatermarkRemover.app` — right-click the app → *Open* → confirm. Only needed once.
- **Watermark position is wrong on some images** — the hardcoded rectangle was measured on a different layout. Run with `--reselect` on a representative image, or pass `--x/--y/--w/--h` directly. To make the new position permanent, update `HARDCODED_COORDS` in `scripts/remove_watermark.py`.
- **GPU users:** `torch` will use CUDA / MPS automatically if available; nothing to configure.
