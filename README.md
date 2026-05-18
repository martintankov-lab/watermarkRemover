# Watermark Remover

Batch tool that removes a watermark from every image in a folder. You pick the watermark rectangle **once** (the first time you run it) and it's remembered for every future run — the rectangle is anchored to its nearest corner and scaled uniformly per image, so it works across **any image size and any aspect ratio**. Uses **LaMa** AI inpainting (a deep-learning model designed for filling masked regions), with a plain OpenCV fallback if LaMa is unavailable.

> Picked coordinates are persisted to `~/Library/Application Support/WatermarkRemover/watermark_coords.json` (macOS) or `~/.config/watermarkremover/watermark_coords.json` (Linux). To re-pick, run with `--reselect` or just delete that file.

Repo layout:

```
watermarkRemover/
├── README.md
├── requirements.txt
├── WatermarkRemover.app/                       (macOS double-click launcher; self-contained)
│   └── Contents/
│       ├── MacOS/WatermarkRemover              (shell launcher that calls the bundled script)
│       └── Resources/remove_watermark.py       (bundled copy — the .app is portable to any folder)
└── scripts/
    └── remove_watermark.py                     (canonical source of the tool)
```

The `.app` bundle is fully self-contained — you can drag it to `/Applications`, your Desktop, or anywhere else and double-clicking will still work, as long as the Python dependencies are installed.

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

Double-click `WatermarkRemover.app`. It opens Terminal and asks you to paste the path to your images folder. On the **very first run** a picker window opens on the first image — drag a rectangle around the watermark and press `ENTER`. From then on, every future run just processes silently using that saved rectangle.

### Option B — run from the terminal

```bash
# Interactive: it prompts for the input folder; auto-opens the picker on first run
python3 scripts/remove_watermark.py

# Pass the folder directly
python3 scripts/remove_watermark.py --input /path/to/images

# Custom output folder (default is "<input>/removed watermark")
python3 scripts/remove_watermark.py --input /path/to/images --output /path/to/out

# Force a re-pick (replaces the saved rectangle)
python3 scripts/remove_watermark.py --input /path/to/images --reselect

# Override the rectangle for a one-off run without saving
python3 scripts/remove_watermark.py --input /path/to/images \
    --x 2606 --y 1905 --w 125 --h 126 --ref-w 2800 --ref-h 2100

# Add more padding around the mask (default 4 px)
python3 scripts/remove_watermark.py --input /path/to/images --padding 8
```

### Flow

1. Scans the input folder for images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`).
2. Resolves the watermark rectangle in this order:
   - explicit `--x/--y/--w/--h` overrides, **or**
   - `--reselect` (opens the picker, saves the result), **or**
   - the persisted saved rectangle from `~/Library/Application Support/WatermarkRemover/watermark_coords.json`, **or**
   - the built-in `HARDCODED_COORDS` fallback. If none of the above and no saved rectangle exists yet, the picker opens automatically on first run.
3. For each image, the rectangle is **anchored to the same image corner** it was picked closest to (top-left / top-right / bottom-left / bottom-right) and scaled by `min(image_w / ref_w, image_h / ref_h)`. Shape preserved, position consistent across any aspect ratio.
4. Cleaned images are written to `<input>/removed watermark/` (or `--output`).

### Changing the saved rectangle

- **Re-pick interactively:** run with `--reselect`. A window opens on the first image — drag, press:
  - `ENTER` — confirm and save (overwrites the persisted file)
  - `R` — reset the rectangle
  - `Q` / `ESC` — abort
- **Reset to first-run state:** delete `~/Library/Application Support/WatermarkRemover/watermark_coords.json`. Next run will auto-open the picker.
- **Change the built-in fallback:** edit `HARDCODED_COORDS` near the top of `scripts/remove_watermark.py`. (Only used if there is no saved file and no CLI overrides.)

## 4. Troubleshooting

- **`Cannot open image: …`** — the file is corrupt or an unsupported format. Check the extension is in the list above.
- **LaMa fails to load on startup** — the script silently switches to OpenCV inpainting. You'll see `[warn] LaMa not available, falling back to OpenCV inpainting` per image. Reinstall with `pip install --force-reinstall simple-lama-inpainting torch`.
- **Mac asks "cannot be opened because the developer cannot be verified"** when launching `WatermarkRemover.app` — right-click the app → *Open* → confirm. Only needed once.
- **Watermark position is wrong on a new batch** — the saved rectangle was picked on an image whose layout doesn't match. Run with `--reselect` on a representative image to overwrite it.
- **GPU users:** `torch` uses CUDA / MPS automatically if available; nothing to configure.
