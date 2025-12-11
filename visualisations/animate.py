"""
Generate Matplotlib animations from the snapshot VTK files in Results_fixed/.
Supports the 2D modes (diffusion_2d, advection_2d, forcing_2d) and the 3D mode
(full_climate_3d, displayed as a mid-plane slice).

Usage examples (from repo root):
  python visualisations/animate.py --mode advection_2d --threads 2 --out advection.mp4
  python visualisations/animate.py --mode forcing_2d --threads 2 --out forcing.gif
  python visualisations/animate.py --mode full_climate_3d --threads 2 --out full3d.mp4
"""
import argparse
import glob
import os
import re
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


VTK_FLOAT_REGEX = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
STEP_REGEX = re.compile(r"step(\d+)")


def read_structured_points(path: str) -> Tuple[Tuple[int, int, int], Tuple[float, float, float], np.ndarray]:
    """
    Minimal ASCII VTK STRUCTURED_POINTS reader for our outputs.
    Returns (dims, spacing, data) where data is shaped (nz, ny, nx).
    """
    with open(path, "r") as f:
        lines = f.readlines()

    dims = None
    spacing = (1.0, 1.0, 1.0)
    data_start = None
    for idx, line in enumerate(lines):
        if line.startswith("DIMENSIONS"):
            parts = line.split()
            dims = (int(parts[1]), int(parts[2]), int(parts[3]))
        elif line.startswith("SPACING"):
            parts = line.split()
            spacing = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif line.strip() == "LOOKUP_TABLE default":
            data_start = idx + 1
            break

    if dims is None or data_start is None:
        raise ValueError(f"Could not parse DIMENSIONS/LOOKUP_TABLE in {path}")

    # Read scalar values
    raw_vals: List[float] = []
    for line in lines[data_start:]:
        raw_vals.extend(float(v) for v in VTK_FLOAT_REGEX.findall(line))
    expected = dims[0] * dims[1] * dims[2]
    if len(raw_vals) < expected:
        raise ValueError(f"Expected {expected} values, got {len(raw_vals)} in {path}")

    arr = np.array(raw_vals[:expected], dtype=float)
    # VTK structured points stores x fastest, then y, then z.
    arr = arr.reshape((dims[2], dims[1], dims[0]))
    return dims, spacing, arr


def load_series(folder: str, threads: int) -> List[str]:
    """Return sorted list of VTK files for given mode/thread."""
    pattern = os.path.join(folder, f"t{threads}_step*.vtk")
    files = glob.glob(pattern)
    files.sort(key=lambda p: int(STEP_REGEX.search(p).group(1)) if STEP_REGEX.search(p) else 0)
    if not files:
        raise FileNotFoundError(f"No VTK files matching {pattern}")
    return files


def animate_2d(mode: str, threads: int, out_path: str, fps: int = 10):
    files = load_series(os.path.join("Results_fixed", mode), threads)
    dims, spacing, arr0 = read_structured_points(files[0])
    nz, ny, nx = arr0.shape
    if nz != 1:
        raise ValueError(f"Expected 2D data for {mode}, got nz={nz}")

    frames = []
    for f in files:
        _, _, arr = read_structured_points(f)
        frames.append(arr[0, :, :])
    frames = np.stack(frames, axis=0)

    fig, ax = plt.subplots()
    extent = [0, nx * spacing[0], 0, ny * spacing[1]]
    im = ax.imshow(frames[0], origin="lower", extent=extent, cmap="coolwarm", animated=True)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("temperature")
    ax.set_title(f"{mode} (threads={threads})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i):
        im.set_data(frames[i])
        return (im,)

    anim = FuncAnimation(fig, update, frames=frames.shape[0], interval=1000 / fps, blit=True)
    save_animation(anim, out_path, fps)


def animate_3d(mode: str, threads: int, out_path: str, fps: int = 10):
    files = load_series(os.path.join("Results_fixed", mode), threads)
    dims, spacing, arr0 = read_structured_points(files[0])
    nz, ny, nx = arr0.shape
    kmid = nz // 2  # middle z-slice

    frames = []
    for f in files:
        _, _, arr = read_structured_points(f)
        frames.append(arr[kmid, :, :])
    frames = np.stack(frames, axis=0)

    fig, ax = plt.subplots()
    extent = [0, nx * spacing[0], 0, ny * spacing[1]]
    im = ax.imshow(frames[0], origin="lower", extent=extent, cmap="coolwarm", animated=True)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("temperature (z mid-plane)")
    ax.set_title(f"{mode} z-slice (threads={threads})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i):
        im.set_data(frames[i])
        return (im,)

    anim = FuncAnimation(fig, update, frames=frames.shape[0], interval=1000 / fps, blit=True)
    save_animation(anim, out_path, fps)


def save_animation(anim: FuncAnimation, out_path: str, fps: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".mp4", ".m4v"):
        try:
            anim.save(out_path, writer="ffmpeg", fps=fps, dpi=150)
            return
        except Exception:
            # ffmpeg missing—fall back to GIF next
            gif_path = os.path.splitext(out_path)[0] + ".gif"
            anim.save(gif_path, writer="pillow", fps=fps)
            print(f"ffmpeg unavailable; saved GIF instead at {gif_path}")
            return
    elif ext in (".gif",):
        anim.save(out_path, writer="pillow", fps=fps)
    else:
        # Unsupported extension, default to GIF
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        anim.save(gif_path, writer="pillow", fps=fps)
        print(f"Unknown extension {ext}; saved GIF instead at {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="Animate climate model snapshots from Results_fixed.")
    parser.add_argument("--mode", required=True,
                        choices=["diffusion_2d", "advection_2d", "forcing_2d", "full_climate_3d"],
                        help="Which simulation to animate.")
    parser.add_argument("--threads", type=int, default=2, help="Thread count in filename prefix (t<threads>_stepN.vtk).")
    parser.add_argument("--out", default=None, help="Output file (mp4/gif). Default is visualisations/<mode>.mp4")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the animation.")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join("visualisations", f"{args.mode}.mp4")

    if args.mode == "full_climate_3d":
        animate_3d(args.mode, args.threads, args.out, args.fps)
    else:
        animate_2d(args.mode, args.threads, args.out, args.fps)

    print(f"Saved animation to {args.out}")


if __name__ == "__main__":
    main()
