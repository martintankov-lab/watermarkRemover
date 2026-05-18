"""
Watermark remover — batch version.

First run: opens a window so you can drag a rectangle over the watermark.
Subsequent runs: reuses the saved coordinates automatically.

Usage:
    python scripts/remove_watermark.py --input <folder> --output <folder>
    python scripts/remove_watermark.py --input <folder> --output <folder> --reselect
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    _orig_jit_load = torch.jit.load
    def _cpu_jit_load(path, *args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return _orig_jit_load(path, *args, **kwargs)
    torch.jit.load = _cpu_jit_load

    from simple_lama_inpainting import SimpleLama
    _lama = SimpleLama()
    USE_LAMA = True
    print("LaMa AI inpainting ready.")
except Exception as e:
    USE_LAMA = False

def _coords_file_path() -> Path:
    if sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / "WatermarkRemover"
    else:
        d = Path.home() / ".config" / "watermarkremover"
    d.mkdir(parents=True, exist_ok=True)
    return d / "watermark_coords.json"


COORDS_FILE = _coords_file_path()


def _load_saved_coords():
    if not COORDS_FILE.exists():
        return None
    try:
        with open(COORDS_FILE) as f:
            d = json.load(f)
        return {k: int(d[k]) for k in ("x", "y", "w", "h", "ref_w", "ref_h")}
    except Exception:
        return None


def _save_coords(coords: dict):
    with open(COORDS_FILE, "w") as f:
        json.dump({k: int(coords[k]) for k in ("x", "y", "w", "h", "ref_w", "ref_h")}, f)


def _detect_anchor(coords: dict):
    """Return (vert, horiz) of the image corner closest to the rectangle's centre."""
    cx = coords["x"] + coords["w"] / 2
    cy = coords["y"] + coords["h"] / 2
    horiz = "right"  if cx > coords["ref_w"] / 2 else "left"
    vert  = "bottom" if cy > coords["ref_h"] / 2 else "top"
    return vert, horiz


def _rect_for_image(coords: dict, orig_w: int, orig_h: int):
    """Compute the (x, y, w, h) rectangle on a target image of any aspect ratio,
    by anchoring to the corner the watermark sits closest to in the reference image
    and scaling uniformly so the rectangle's shape is preserved."""
    ref_w, ref_h = coords["ref_w"], coords["ref_h"]
    w, h = coords["w"], coords["h"]
    vert, horiz = _detect_anchor(coords)

    off_x = ref_w - (coords["x"] + w) if horiz == "right"  else coords["x"]
    off_y = ref_h - (coords["y"] + h) if vert  == "bottom" else coords["y"]

    s = min(orig_w / ref_w, orig_h / ref_h)
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    new_off_x = int(round(off_x * s))
    new_off_y = int(round(off_y * s))

    new_x = orig_w - new_off_x - new_w if horiz == "right"  else new_off_x
    new_y = orig_h - new_off_y - new_h if vert  == "bottom" else new_off_y
    return new_x, new_y, new_w, new_h


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Hardcoded watermark region (measured against a 2800x2100 reference image).
# The script scales these proportionally to each input image's actual size.
# Override at runtime with --reselect (or --x/--y/--w/--h/--ref-w/--ref-h).
HARDCODED_COORDS = {
    "x": 2606,
    "y": 1905,
    "w": 125,
    "h": 126,
    "ref_w": 2800,
    "ref_h": 2100,
}

# --- region selector ---------------------------------------------------------

_drag_state = {"drawing": False, "start": None, "rect": None, "clone": None, "display": None}


def _mouse_cb(event, x, y, flags, param):
    s = _drag_state
    if event == cv2.EVENT_LBUTTONDOWN:
        s["drawing"] = True
        s["start"] = (x, y)
        s["rect"] = None
    elif event == cv2.EVENT_MOUSEMOVE and s["drawing"]:
        s["display"] = s["clone"].copy()
        cv2.rectangle(s["display"], s["start"], (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        s["drawing"] = False
        x0, y0 = s["start"]
        s["rect"] = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        s["display"] = s["clone"].copy()
        cv2.rectangle(s["display"], s["start"], (x, y), (0, 255, 0), 2)


def select_region(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot open image: {image_path}")

    _drag_state["clone"] = img.copy()
    _drag_state["display"] = img.copy()
    _drag_state["rect"] = None

    win = "Draw rectangle around watermark — ENTER to confirm, R to reset, Q to quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _mouse_cb)
    print("Draw a rectangle around the watermark, then press ENTER.")

    while True:
        cv2.imshow(win, _drag_state["display"])
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and _drag_state["rect"]:   # ENTER
            break
        if key == ord("r"):
            _drag_state["rect"] = None
            _drag_state["display"] = _drag_state["clone"].copy()
        if key in (ord("q"), 27):               # Q / ESC
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return _drag_state["rect"]


# --- inpainting --------------------------------------------------------------

CONTEXT = 60

def remove_watermark(src: str, dst: str, coords: dict, padding: int = 10) -> bool:
    img_bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"  SKIP (unreadable): {src}")
        return False

    orig_h, orig_w = img_bgr.shape[:2]

    # Anchor to nearest corner of the reference image and scale uniformly,
    # so the rectangle stays in place across any aspect ratio.
    x, y, w, h = _rect_for_image(coords, orig_w, orig_h)

    px1 = max(0, x - padding - CONTEXT)
    py1 = max(0, y - padding - CONTEXT)
    px2 = min(orig_w, x + w + padding + CONTEXT)
    py2 = min(orig_h, y + h + padding + CONTEXT)

    patch = img_bgr[py1:py2, px1:px2]
    if patch.size == 0:
        print(f"  SKIP (watermark outside bounds for {orig_w}x{orig_h} image): {src}")
        return False
    ph, pw = patch.shape[:2]

    mx1 = max(0, (x - padding) - px1)
    my1 = max(0, (y - padding) - py1)
    mx2 = min(pw, mx1 + w + padding * 2)
    my2 = min(ph, my1 + h + padding * 2)
    mask_patch = np.zeros((ph, pw), dtype=np.uint8)
    mask_patch[my1:my2, mx1:mx2] = 255

    if USE_LAMA:
        pil_img  = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask_patch)
        inpainted_patch = cv2.cvtColor(np.array(_lama(pil_img, pil_mask)), cv2.COLOR_RGB2BGR)
        inpainted_patch = cv2.resize(inpainted_patch, (pw, ph), interpolation=cv2.INTER_LANCZOS4)
    else:
        print("  [warn] LaMa not available, falling back to OpenCV inpainting")
        inpainted_patch = cv2.inpaint(patch, mask_patch, inpaintRadius=12, flags=cv2.INPAINT_TELEA)

    center = ((px1 + px2) // 2, (py1 + py2) // 2)
    result = cv2.seamlessClone(inpainted_patch, img_bgr, mask_patch, center, cv2.NORMAL_CLONE)

    cv2.imwrite(dst, result)
    return True


# --- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch watermark remover")
    parser.add_argument("--input",    help="Folder with source images (omit to be prompted)")
    parser.add_argument("--output",   help="Output folder (default: '<input>/removed watermark')")
    parser.add_argument("--reselect", action="store_true", help="Force re-selection of watermark region (ignores hardcoded coords)")
    parser.add_argument("--padding",  type=int, default=4, help="Extra pixels around mask (default 4)")
    parser.add_argument("--x",        type=int, help="Override hardcoded watermark x")
    parser.add_argument("--y",        type=int, help="Override hardcoded watermark y")
    parser.add_argument("--w",        type=int, help="Override hardcoded watermark width")
    parser.add_argument("--h",        type=int, help="Override hardcoded watermark height")
    parser.add_argument("--ref-w",    type=int, dest="ref_w", help="Override hardcoded reference image width")
    parser.add_argument("--ref-h",    type=int, dest="ref_h", help="Override hardcoded reference image height")
    args = parser.parse_args()

    if args.input:
        input_dir = Path(args.input)
    else:
        print("=" * 50)
        print("  Watermark Remover")
        print("=" * 50)
        raw = input("\nPaste the path to your images folder and press ENTER:\n> ").strip().strip("'\"")
        input_dir = Path(raw)

    if not input_dir.is_dir():
        sys.exit(f"Folder not found: {input_dir}")

    output_dir = Path(args.output) if args.output else input_dir / "removed watermark"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No images found in {input_dir}")

    cli_override = any(v is not None for v in (args.x, args.y, args.w, args.h, args.ref_w, args.ref_h))
    saved = _load_saved_coords()

    if not args.reselect and not cli_override and saved is None:
        print("First run — let's pick the watermark region once. It'll be remembered for every future run.\n")
        args.reselect = True

    if args.reselect:
        print(f"Opening {images[0].name} for region selection...\n")
        rect = select_region(str(images[0]))
        if rect is None or rect[2] == 0 or rect[3] == 0:
            sys.exit("No region selected. Exiting.")
        x, y, w, h = rect
        ref_img = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        ref_h, ref_w = ref_img.shape[:2]
        coords = {"x": x, "y": y, "w": w, "h": h, "ref_w": ref_w, "ref_h": ref_h}
        _save_coords(coords)
        print(f"Region saved to {COORDS_FILE} — future runs will reuse it automatically.\n")
    elif cli_override:
        base = saved if saved else HARDCODED_COORDS
        coords = {k: (getattr(args, k) if getattr(args, k) is not None else base[k])
                  for k in ("x", "y", "w", "h", "ref_w", "ref_h")}
        print(f"Using CLI-overridden watermark region ({coords['w']}x{coords['h']} at "
              f"{coords['x']},{coords['y']}, ref {coords['ref_w']}x{coords['ref_h']}). Processing...\n")
    elif saved:
        coords = saved
        print(f"Using saved watermark region ({coords['w']}x{coords['h']} at "
              f"{coords['x']},{coords['y']}, ref {coords['ref_w']}x{coords['ref_h']}) from {COORDS_FILE}. Processing...\n")
    else:
        coords = dict(HARDCODED_COORDS)
        print(f"Using built-in watermark region ({coords['w']}x{coords['h']} at "
              f"{coords['x']},{coords['y']}, ref {coords['ref_w']}x{coords['ref_h']}). Processing...\n")

    v, hz = _detect_anchor(coords)
    print(f"  (anchored to {v}-{hz} corner; uniform scale across all aspect ratios)\n")

    ok = fail = 0
    for i, img_path in enumerate(images, 1):
        out_path = output_dir / img_path.name
        if remove_watermark(str(img_path), str(out_path), coords, args.padding):
            ok += 1
            print(f"  [{i:>2}/{len(images)}] {img_path.name}")
        else:
            fail += 1

    print(f"\nDone: {ok} saved to '{output_dir}'" + (f", {fail} failed" if fail else ""))


if __name__ == "__main__":
    main()
