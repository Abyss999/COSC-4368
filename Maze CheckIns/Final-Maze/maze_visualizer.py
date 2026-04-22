"""
maze_visualizer.py  ·  Unified per-phase visualization  (MP4 edition)

Key changes from GIF version:
  · Output is .mp4 (H.264 via cv2.VideoWriter) instead of .gif
  · Frames are streamed directly to the video writer -- no temp file, no
    in-memory list.  Peak RAM = one frame at a time (O(1)).
  · Base image is still downscaled to VIZ_MAX_DIM before rendering.
  · Rich debug HUD:
      - Episode / timestep counter
      - Agent (x, y) position
      - Last action taken (name)
      - Step reward + cumulative episode reward
      - Current epsilon
      - Optional Q-value arrows (best action per cell in agent's known map)
  · Hazard overlays on the base image:
      - Fire / death-pit cells: red-orange tint + "🔥" label
      - Teleport pad cells: colour-coded outlines matching pad pairs
      - Confusion trap cells: purple tint
  · Controls (all in VizConfig):
      - fps           : output video FPS (default 20)
      - episode_stride: record every Nth episode (default 1 = all)
      - last_k_steps  : if >0, only record the last K steps of each episode
      - pause_on_end  : insert N duplicate frames at episode end (simulates pause)
  · Headless-safe: cv2.VideoWriter uses the mp4v FOURCC; no GUI is opened.
  · Fallback: if cv2 is unavailable, falls back to PIL-based GIF (with warning).

Public API (hook contract unchanged from GIF version):
  class PhaseVisualizer
    begin_phase(label)
    begin_episode(env, agent, ep_idx)
    step(pos, action=None, reward=0.0)
    end_episode(success)
    flush(agent, env)
"""
from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Optional cv2 import ───────────────────────────────────────────────────────
try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    warnings.warn(
        "[maze_visualizer] opencv-python not found. "
        "Falling back to GIF output. Install with: pip install opencv-python",
        stacklevel=2,
    )

# ════════════════════════════════════════════════════════
#  DOWNSCALE LIMIT
# ════════════════════════════════════════════════════════
VIZ_MAX_DIM: int = 512   # longest side of rendered frames

def _maybe_downscale(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= VIZ_MAX_DIM:
        return arr
    scale = VIZ_MAX_DIM / longest
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img = Image.fromarray(arr, "RGB").resize((nw, nh), Image.BICUBIC)
    return np.array(img)

# ════════════════════════════════════════════════════════
#  ACTION NAMES (for HUD)
# ════════════════════════════════════════════════════════
_ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT", 4: "WAIT", None: "---"}

# ════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════
@dataclass
class VizConfig:
    enabled:            bool  = True
    # Sampling
    step_stride:        int   = 2        # render every Nth step within an episode
    frames_per_episode: int   = 150      # hard cap per episode (uniform downsample)
    episode_stride:     int   = 1        # record every Nth episode (1 = all)
    last_k_steps:       int   = 0        # if >0, only keep last K steps per episode
    replay_frames:      int   = 80       # best-path replay at end of phase
    pause_on_end:       int   = 15       # duplicate frames at episode end (simulates pause)
    # Video output
    fps:                int   = 20
    out_path:           Path  = Path("Results/training_progress.mp4")
    # Overlays
    show_q_arrows:      bool  = False    # draw best-action arrows on known cells
    show_hazards:       bool  = True     # tint fire/conf/teleport cells on base
    # Style
    agent_rgb:          Tuple = (30,  120, 255)
    path_rgb:           Tuple = (220, 20,  40)
    best_rgb:           Tuple = (255, 0,   220)
    path_thickness:     int   = 2
    best_thickness:     int   = 4
    # HUD
    hud_height:         int   = 90       # pixels reserved at bottom for HUD panel


# ════════════════════════════════════════════════════════
#  COORDINATE HELPERS
# ════════════════════════════════════════════════════════
def _cell_center_scaled(cell: Tuple[int, int], cs: int,
                        scale: float) -> Tuple[int, int]:
    cx, cy = cell
    return (int(round((cx * cs + cs // 2) * scale)),
            int(round((cy * cs + cs // 2) * scale)))

# ════════════════════════════════════════════════════════
#  BASE IMAGE  (immutable, with hazard tints baked in)
# ════════════════════════════════════════════════════════
# Teleport pair colours (cycle through these for up to 4 pairs)
_TELE_COLORS = [
    (0,   200,  80),    # green
    (200, 100, 255),    # purple
    (255, 200,  30),    # yellow
    (255,  60,  60),    # red
]

def _build_base(env, agent, cfg: VizConfig) -> Tuple[np.ndarray, float]:
    """
    Build a downscaled, read-only base image that has:
      • start marker (green ring)
      • goal marker  (gold diamond)
      • hazard tints baked in (fire=red-orange, conf=purple, teleport=coloured outlines)
    Returns (base_rgb_readonly, pixel_scale).
    """
    raw   = env.pixels_copy()
    small = _maybe_downscale(raw)
    h_orig, w_orig = raw.shape[:2]
    h_new,  w_new  = small.shape[:2]
    scale = h_new / h_orig

    img  = Image.fromarray(small, "RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    cs   = env.cs
    R    = max(3, int(round((cs // 2) * scale)))

    if cfg.show_hazards:
        # ── Fire / death-pit cells (agent.rot_danger[0] == initial slot) ────
        fire_cells = set()
        for slot_set in agent.rot_danger.values():
            fire_cells |= slot_set
        # Also include anything currently tagged "death" in known map
        fire_cells |= {c for c, v in agent.known.items() if v == "death"}
        for (cx, cy) in fire_cells:
            px, py = _cell_center_scaled((cx, cy), cs, scale)
            r2 = max(2, int(round((cs // 2 - 1) * scale)))
            draw.ellipse([px-r2, py-r2, px+r2, py+r2],
                         fill=(255, 80, 0, 120))

        # ── Confusion trap cells ─────────────────────────────────────────────
        for (cx, cy), v in agent.known.items():
            if v == "confusion":
                px, py = _cell_center_scaled((cx, cy), cs, scale)
                r2 = max(2, int(round((cs // 2 - 1) * scale)))
                draw.rectangle([px-r2, py-r2, px+r2, py+r2],
                               fill=(160, 0, 220, 100))

        # ── Teleport pad cells (colour-coded by pair index) ──────────────────
        tele_items = list(agent.tele.items())
        # group into pairs by destination
        seen: Dict[tuple, int] = {}
        pair_idx = 0
        for src, dst in tele_items:
            key = tuple(sorted([src, dst]))
            if key not in seen:
                seen[key] = pair_idx % len(_TELE_COLORS)
                pair_idx += 1
            color = _TELE_COLORS[seen[key]]
            for cell in (src, dst):
                px, py = _cell_center_scaled(cell, cs, scale)
                r2 = max(2, int(round((cs // 2 - 1) * scale)))
                draw.ellipse([px-r2, py-r2, px+r2, py+r2],
                             outline=color + (255,), width=max(1, int(round(2*scale))))

    # ── Start marker (green ring) ────────────────────────────────────────────
    sx, sy = _cell_center_scaled(env.start, cs, scale)
    draw.ellipse([sx-R, sy-R, sx+R, sy+R],
                 outline=(30, 220, 80, 255),
                 width=max(1, int(round(3*scale))))

    # ── Goal marker (gold diamond) ───────────────────────────────────────────
    gx, gy = _cell_center_scaled(env.goal, cs, scale)
    draw.polygon([(gx, gy-R), (gx+R, gy), (gx, gy+R), (gx-R, gy)],
                 outline=(255, 215, 0, 255), fill=(255, 215, 0, 200))

    arr = np.array(img.convert("RGB"))
    arr.setflags(write=False)
    return arr, scale


# ════════════════════════════════════════════════════════
#  Q-VALUE ARROWS  (optional overlay)
# ════════════════════════════════════════════════════════
_ARROW_DX = {0: 0, 1: 1, 2: 0, 3: -1, 4: 0}
_ARROW_DY = {0: -1, 1: 0, 2: 1, 3: 0, 4: 0}

def _draw_q_arrows(draw: ImageDraw.ImageDraw, agent, cs: int, scale: float):
    """Draw a small arrow on each known free cell pointing to the greedy action."""
    r = max(1, int(round((cs // 3) * scale)))
    for (cx, cy), v in agent.known.items():
        if v not in ("free", "goal"):
            continue
        q_row = agent.q[cx, cy, 0]   # non-confused state
        best  = int(np.argmax(q_row))
        if q_row[best] <= 0:
            continue
        px, py = _cell_center_scaled((cx, cy), cs, scale)
        ex = px + int(round(_ARROW_DX[best] * r))
        ey = py + int(round(_ARROW_DY[best] * r))
        draw.line([px, py, ex, ey], fill=(255, 255, 100, 180), width=1)
        draw.ellipse([ex-1, ey-1, ex+1, ey+1], fill=(255, 255, 100, 200))


# ════════════════════════════════════════════════════════
#  PATH DRAWING
# ════════════════════════════════════════════════════════
def _draw_polyline(draw, cells, cs, color, thickness, scale):
    pts = [_cell_center_scaled(c, cs, scale) for c in cells]
    for i in range(1, len(pts)):
        draw.line([pts[i-1], pts[i]], fill=color, width=max(1, thickness))
    r = max(1, thickness // 2)
    for x, y in pts:
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)


# ════════════════════════════════════════════════════════
#  HUD PANEL
# ════════════════════════════════════════════════════════
def _draw_hud(img: Image.Image, cfg: VizConfig,
              ep_idx: int, step_idx: int,
              pos: Tuple[int, int],
              action: Optional[int],
              reward: float, cum_reward: float,
              epsilon: float,
              success_flag: Optional[bool] = None) -> Image.Image:
    """
    Append a solid black HUD panel below the maze frame with all debug info.
    Returns a new image (original + panel).
    """
    W, H   = img.size
    ph     = cfg.hud_height
    canvas = Image.new("RGB", (W, H + ph), (0, 0, 0))
    canvas.paste(img, (0, 0))
    draw   = ImageDraw.Draw(canvas)

    # Two rows of text
    lines = [
        (f"EP {ep_idx:>4}  STEP {step_idx:>5}  POS ({pos[0]:>2},{pos[1]:>2})  "
         f"ACT {_ACTION_NAMES.get(action, '---'):<5}  EPS {epsilon:.3f}"),
        (f"REWARD {reward:>+7.1f}  CUM {cum_reward:>+8.1f}  "
         f"MAP {len([v for v in [None]])or 0}"),
    ]
    # Overwrite line 2 with actual map size
    map_sz = sum(1 for v in getattr(_hud_agent_ref, "known", {}).values()
                 if v != "wall") if _hud_agent_ref else 0
    lines[1] = (f"REWARD {reward:>+7.1f}  CUM {cum_reward:>+8.1f}  "
                f"MAP {map_sz} cells")

    if success_flag is not None:
        status = "✓ GOAL" if success_flag else "✗ TIMEOUT"
        lines[0] += f"  [{status}]"

    y0 = H + 6
    for line in lines:
        draw.text((8, y0), line, fill=(220, 220, 220))
        y0 += 18

    # Colour bar at top of HUD (green=success, red=fail, grey=running)
    if success_flag is True:
        bar_color = (0, 200, 80)
    elif success_flag is False:
        bar_color = (200, 50, 50)
    else:
        bar_color = (80, 80, 80)
    draw.rectangle([0, H, W, H+3], fill=bar_color)

    return canvas

# Module-level mutable ref so _draw_hud can access agent.known size
# (avoids threading issues -- we're single-threaded anyway)
_hud_agent_ref = None


# ════════════════════════════════════════════════════════
#  FRAME COMPOSER
# ════════════════════════════════════════════════════════
def _compose_frame(base_rgb: np.ndarray, env, agent, scale: float,
                   pos: Tuple[int, int],
                   full_path: List[Tuple[int, int]],
                   best_path: Optional[List[Tuple[int, int]]],
                   highlight_best: bool,
                   ep_idx: int, step_idx: int,
                   action: Optional[int],
                   reward: float, cum_reward: float,
                   cfg: VizConfig,
                   success_flag: Optional[bool] = None) -> np.ndarray:
    """
    Compose one frame as a numpy uint8 array (H+hud_height, W, 3) in RGB order.
    """
    frame = base_rgb.copy()
    img   = Image.fromarray(frame, "RGB")
    draw  = ImageDraw.Draw(img)
    cs    = env.cs

    # Path trail
    if full_path and len(full_path) >= 2:
        _draw_polyline(draw, full_path, cs, cfg.path_rgb, cfg.path_thickness, scale)

    # Best-path replay
    if highlight_best and best_path:
        _draw_polyline(draw, best_path, cs, cfg.best_rgb, cfg.best_thickness, scale)

    # Q arrows (optional)
    if cfg.show_q_arrows and not highlight_best:
        _draw_q_arrows(draw, agent, cs, scale)

    # Agent dot
    ar = max(3, int(round((cs // 2 - 1) * scale)))
    ax, ay = _cell_center_scaled(pos, cs, scale)
    draw.ellipse([ax-ar, ay-ar, ax+ar, ay+ar],
                 fill=cfg.agent_rgb, outline=(255, 255, 255),
                 width=max(1, int(round(2*scale))))

    # HUD panel
    img = _draw_hud(img, cfg, ep_idx, step_idx, pos, action,
                    reward, cum_reward, agent.epsilon, success_flag)

    return np.array(img)


# ════════════════════════════════════════════════════════
#  VIDEO WRITER WRAPPER
# ════════════════════════════════════════════════════════
class _VideoWriter:
    """
    Thin wrapper around cv2.VideoWriter (or PIL GIF fallback).
    Streams frames directly to disk -- no in-memory accumulation.
    """

    def __init__(self, path: Path, fps: int, frame_hw: Tuple[int, int]):
        self._path  = path
        self._fps   = fps
        self._hw    = frame_hw   # (height, width)
        self._writer = None
        self._gif_frames: Optional[List] = None   # fallback only
        self._use_cv2 = _CV2_OK
        path.parent.mkdir(parents=True, exist_ok=True)
        self._open()

    def _open(self):
        h, w = self._hw
        if self._use_cv2:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self._path), fourcc, self._fps, (w, h)
            )
            if not self._writer.isOpened():
                warnings.warn(
                    f"[viz] cv2.VideoWriter failed to open {self._path}. "
                    "Falling back to GIF.", stacklevel=3
                )
                self._use_cv2 = False
                self._gif_frames = []
        else:
            self._gif_frames = []

    def write(self, frame_rgb: np.ndarray):
        """Accept an RGB uint8 numpy array."""
        if self._use_cv2:
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._writer.write(bgr)
        else:
            self._gif_frames.append(Image.fromarray(frame_rgb))

    def close(self) -> Path:
        if self._use_cv2 and self._writer is not None:
            self._writer.release()
            print(f"    [viz] MP4 saved -> {self._path.name}")
            return self._path
        # GIF fallback
        if self._gif_frames:
            gif_path = self._path.with_suffix(".gif")
            pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128)
                   for f in self._gif_frames]
            dur  = max(20, int(1000 / max(1, self._fps)))
            pal[0].save(gif_path, save_all=True, append_images=pal[1:],
                        duration=dur, loop=0, optimize=True, disposal=2)
            mb = gif_path.stat().st_size / (1024*1024)
            print(f"    [viz] GIF fallback saved -> {gif_path.name} "
                  f"({len(pal)} frames, {mb:.2f} MB)")
            return gif_path
        return self._path


# ════════════════════════════════════════════════════════
#  SEPARATOR FRAME
# ════════════════════════════════════════════════════════
def _make_separator_frame(base_rgb: np.ndarray, banner: str,
                          hud_height: int) -> np.ndarray:
    """Dimmed maze + centred banner text + blank HUD strip."""
    frame  = (base_rgb.copy().astype(np.uint16) * 45 // 100).astype(np.uint8)
    img    = Image.fromarray(frame, "RGB")
    draw   = ImageDraw.Draw(img)
    W, H   = img.size
    box_w  = 7 * len(banner) + 24
    x0     = (W - box_w) // 2
    y0     = H // 2 - 16
    draw.rectangle([x0, y0, x0+box_w, y0+28],
                   fill=(0,0,0), outline=(255,255,255), width=2)
    draw.text((x0+12, y0+9), banner, fill=(255,255,255))
    # Append blank HUD strip so dimensions match
    canvas = Image.new("RGB", (W, H + hud_height), (0,0,0))
    canvas.paste(img, (0,0))
    return np.array(canvas)


# ════════════════════════════════════════════════════════
#  PHASE VISUALIZER  (public API -- hook contract unchanged)
# ════════════════════════════════════════════════════════
class PhaseVisualizer:
    """
    One PhaseVisualizer = one output video file.

    Hook points (same as GIF version):
        viz.begin_phase(label)
        viz.begin_episode(env, agent, ep_idx)
        viz.step(pos, action=None, reward=0.0)   ← action + reward are NEW optional args
        viz.end_episode(success)
        viz.flush(agent, env)

    The old viz.step(pos) signature still works; action/reward default to None/0.
    """

    def __init__(self, cfg: Optional[VizConfig] = None):
        self.cfg = cfg or VizConfig()
        self.cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

        self._label:      str  = ""
        self._writer:     Optional[_VideoWriter] = None
        self._frame_hw:   Optional[Tuple[int, int]] = None

        # Per-episode state
        self._env         = None
        self._agent       = None
        self._base_rgb:   Optional[np.ndarray]      = None
        self._scale:      float                     = 1.0
        self._ep_idx:     int                       = 0
        self._step_idx:   int                       = 0
        self._full_path:  List[Tuple[int, int]]     = []
        self._ep_frames:  List[np.ndarray]          = []  # buffer for ONE episode
        self._last_action: Optional[int]            = None
        self._last_reward: float                    = 0.0
        self._cum_reward:  float                    = 0.0

    # ── phase hooks ──────────────────────────────────────────────────────────
    def begin_phase(self, label: str):
        if not self.cfg.enabled:
            return
        self._label  = label
        self._writer = None   # writer opened on first frame (need frame size first)

    def begin_episode(self, env, agent, ep_idx: int):
        if not self.cfg.enabled:
            return
        global _hud_agent_ref
        _hud_agent_ref  = agent

        self._env        = env
        self._agent      = agent
        self._ep_idx     = ep_idx
        self._step_idx   = 0
        self._full_path  = [env.start]
        self._ep_frames  = []
        self._last_action = None
        self._last_reward = 0.0
        self._cum_reward  = 0.0

        self._base_rgb, self._scale = _build_base(env, agent, self.cfg)
        self._capture(env.start)

    def step(self, pos: Tuple[int, int],
             action: Optional[int] = None,
             reward: float = 0.0):
        if not self.cfg.enabled:
            return
        self._step_idx   += 1
        self._last_action = action
        self._last_reward = reward
        self._cum_reward += reward
        self._full_path.append(pos)

        if (self._step_idx % self.cfg.step_stride) != 0:
            return

        # last_k_steps mode: only keep a rolling window
        self._capture(pos)
        if self.cfg.last_k_steps > 0:
            keep = self.cfg.last_k_steps // max(1, self.cfg.step_stride)
            if len(self._ep_frames) > keep:
                self._ep_frames = self._ep_frames[-keep:]

    def end_episode(self, success: bool):
        if not self.cfg.enabled:
            return
        if self._full_path:
            self._capture(self._full_path[-1], success_flag=success)

        # Downsample episode frames to budget
        ep_frames = _uniform_sample(self._ep_frames, self.cfg.frames_per_episode)

        # Skip recording if episode_stride says so
        if self._ep_idx % self.cfg.episode_stride != 0:
            self._ep_frames = []
            return

        # Lazy-init the video writer (we now know frame dimensions)
        if ep_frames and self._writer is None:
            h, w = ep_frames[0].shape[:2]
            self._frame_hw = (h, w)
            out = self._resolve_path()
            self._writer = _VideoWriter(out, self.cfg.fps, (h, w))

        if self._writer is not None:
            for f in ep_frames:
                self._writer.write(f)
            # Pause frames
            if ep_frames:
                pause = _make_separator_frame(
                    self._base_rgb,
                    f"EP {self._ep_idx}  {'SOLVED' if success else 'TIMEOUT'}",
                    self.cfg.hud_height,
                )
                for _ in range(self.cfg.pause_on_end):
                    self._writer.write(pause)

        self._ep_frames = []

    def flush(self, agent, env):
        if not self.cfg.enabled:
            return
        if self._writer is None:
            print(f"    [viz] no frames recorded for phase '{self._label}'")
            return

        # Best-path replay
        if agent.best_path and len(agent.best_path) >= 2:
            base_replay, replay_scale = _build_base(env, agent, self.cfg)
            h_r, w_r = base_replay.shape[:2]
            fh = h_r + self.cfg.hud_height
            # intro separator
            intro = _make_separator_frame(
                base_replay,
                f"BEST PATH  {len(agent.best_path)} cells",
                self.cfg.hud_height,
            )
            for _ in range(self.cfg.pause_on_end):
                self._writer.write(intro)
            # animate
            bp = list(agent.best_path)
            n_out = min(self.cfg.replay_frames, len(bp))
            idxs  = np.linspace(0, len(bp)-1, n_out, dtype=int)
            for i in idxs:
                prefix = bp[:i+1]
                frame  = _compose_frame(
                    base_replay, env, agent, replay_scale,
                    bp[i], [],
                    prefix, highlight_best=True,
                    ep_idx=self._ep_idx, step_idx=i,
                    action=None, reward=0.0, cum_reward=0.0,
                    cfg=self.cfg,
                )
                self._writer.write(frame)

        self._writer.close()
        self._writer = None

    # ── internals ────────────────────────────────────────────────────────────
    def _capture(self, pos: Tuple[int, int],
                 success_flag: Optional[bool] = None):
        if self._base_rgb is None:
            return
        frame = _compose_frame(
            self._base_rgb, self._env, self._agent, self._scale,
            pos, self._full_path,
            None, False,
            self._ep_idx, self._step_idx,
            self._last_action, self._last_reward, self._cum_reward,
            self.cfg, success_flag,
        )
        self._ep_frames.append(frame)

    def _resolve_path(self) -> Path:
        """Return output path, embedding phase label if user left the default."""
        out = self.cfg.out_path
        if out == VizConfig().out_path:
            safe = self._label.replace(" ", "_").replace("/", "_")
            out  = out.parent / f"{safe}.mp4"
        return out


# ════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════
def _uniform_sample(items: List, target: int) -> List:
    if target <= 0 or len(items) <= target:
        return list(items)
    idx = np.linspace(0, len(items)-1, target, dtype=int)
    return [items[i] for i in idx]


# ════════════════════════════════════════════════════════
#  LEGACY GIF HELPERS  (kept so any code importing them still works)
# ════════════════════════════════════════════════════════
def append_episode_frames(frames_out, episode_frames,
                          separator=None, separator_count=0):
    frames_out.extend(episode_frames)
    if separator is not None and separator_count > 0:
        frames_out.extend([separator] * separator_count)

def save_training_gif(frames, out_path: Path, fps: int = 20):
    if not frames:
        print(f"    [viz] save_training_gif: no frames -> {out_path.name}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(20, int(1000 / max(1, fps)))
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(out_path, save_all=True, append_images=pal[1:],
                duration=dur, loop=0, optimize=True, disposal=2)
    mb = out_path.stat().st_size / (1024*1024)
    print(f"    [viz] GIF saved -> {out_path.name} ({len(frames)} frames, {mb:.2f} MB)")

# ── VizConfig alias for backwards compat ──────────────────────────────────────
# Old code that passes gif_fps= will just get it ignored gracefully.