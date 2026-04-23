"""
maze_visualizer.py  ·  COSC 4368 AI  ·  Group 5  (v2 patch)

Changes from maze_visualizer (FX9 version)
-------------------------------------------
V2-FX4 (visualizer side): The visualizer draws fire using
env.current_fire, which is a frozenset of cell coords already
updated by env.step().  No change needed in the visualizer for
V2-FX4 — the env now calls _rotate_fire(turns%4) instead of
(turns//5)%4, so current_fire is always current.  The visualizer
just reads it, so it stays in sync automatically.

All FX9 changes retained: fire overlay reads env.current_fire
(cell coords) directly without any intermediate pixel conversion.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

UP, RIGHT, DOWN, LEFT, WAIT = 0, 1, 2, 3, 4
_ACTION_NAMES = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN",
                 LEFT: "LEFT", WAIT: "WAIT"}

_C_AGENT    = (  0, 220, 255)
_C_PATH_NEW = (255, 160,  20)
_C_PATH_OLD = ( 60,  25,   0)
_C_FIRE     = (255,  50,   0)
_C_HUD_BG   = ( 18,  18,  18)
_C_HUD_FG   = (210, 210, 210)
_C_SUCCESS  = ( 40, 255, 100)
_C_FAIL     = (255,  80,  60)
_C_NEUTRAL  = (180, 180, 180)


@dataclass
class VizConfig:
    enabled           : bool           = True
    step_stride       : int            = 2
    frames_per_episode: int            = 120
    episode_stride    : int            = 1
    replay_frames     : int            = 80
    pause_on_end      : int            = 15
    fps               : int            = 20
    out_path          : Optional[Path] = None
    show_q_arrows     : bool           = False
    show_hazards      : bool           = True
    hud_height        : int            = 90


class PhaseVisualizer:
    """
    Records one MP4 for a complete training/test phase.

    Fire overlay reads env.current_fire (frozenset of cell coords)
    directly.  Because the env now rotates fire every turn (V2-FX4),
    current_fire is always the correct set for the current step.
    No pixel conversion needed — the visualizer draws disks at
    (cx*cs + cs//2, cy*cs + cs//2) for each (cx,cy) in the set.
    """

    def __init__(self, cfg: VizConfig):
        self._cfg   = cfg
        self._label = ""
        self._frames: List[np.ndarray] = []

        self._ep_idx       : int   = 0
        self._ep_step      : int   = 0
        self._ep_frame_cnt : int   = 0
        self._ep_path: List[Optional[Tuple[int, int]]] = []
        self._ep_reward    : float = 0.0
        self._ep_deaths    : int   = 0
        self._last_action  : int   = WAIT

        self._pos: Tuple[int, int] = (0, 0)

        self._base_img: Optional[np.ndarray] = None
        self._cs       : int = 16
        self._env_ref        = None

        self._active = False

    def begin_phase(self, label: str) -> None:
        self._label  = label
        self._frames = []
        self._active = True

    def begin_episode(self, env, agent, ep_idx: int = 0) -> None:
        self._ep_idx       = ep_idx
        self._ep_step      = 0
        self._ep_frame_cnt = 0
        self._ep_path      = []
        self._ep_reward    = 0.0
        self._ep_deaths    = 0
        self._last_action  = WAIT
        self._env_ref      = env
        self._cs           = env.cs

        self._base_img = env.pixels_copy()

        self._pos = env.start
        self._ep_path.append(env.start)

    def step(self,
             pos    : tuple,
             action : int   = WAIT,
             reward : float = 0.0,
             is_dead: bool  = False) -> None:
        if not self._active:
            return

        self._last_action  = action
        self._ep_reward   += reward
        self._ep_step     += 1

        if is_dead:
            self._ep_path.append(None)
            self._ep_deaths += 1

        self._ep_path.append(pos)
        self._pos = pos

        if self._ep_idx % max(1, self._cfg.episode_stride) != 0:
            return
        if self._ep_frame_cnt >= self._cfg.frames_per_episode:
            return
        if self._ep_step % max(1, self._cfg.step_stride) != 0:
            return

        self._ep_frame_cnt += 1
        self._frames.append(
            self._render_frame(f"Ep {self._ep_idx + 1}  Step {self._ep_step}"))

    def end_episode(self, success: bool) -> None:
        if not self._active:
            return
        if self._ep_idx % max(1, self._cfg.episode_stride) != 0:
            return
        colour = _C_SUCCESS if success else _C_FAIL
        txt    = "SUCCESS!" if success else "FAILED"
        for _ in range(self._cfg.pause_on_end):
            self._frames.append(
                self._render_frame(
                    f"Ep {self._ep_idx + 1}  {txt}",
                    overlay_text=txt, overlay_color=colour))

    def flush(self, agent=None, env=None) -> None:
        if not self._active or not self._frames:
            self._active = False
            return
        out = self._cfg.out_path
        if out is None:
            self._active = False
            return
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        n = len(self._frames)
        print(f"    Writing {n} frames -> {out.name} ...", end=" ", flush=True)
        try:
            self._write_mp4(out, self._frames)
            print("done.")
        except Exception as ex:
            print(f"\n    MP4 failed ({ex}), trying GIF ...")
            gif = out.with_suffix(".gif")
            try:
                self._write_gif(gif, self._frames)
                print(f"    Saved GIF -> {gif.name}")
            except Exception as ex2:
                print(f"    GIF also failed: {ex2}")
        self._frames = []
        self._active = False

    def _render_frame(self,
                      label        : str   = "",
                      overlay_text : str   = "",
                      overlay_color: tuple = _C_NEUTRAL) -> np.ndarray:
        """
        Fire cells come directly from env.current_fire (frozenset of
        cell coordinates, already at the correct rotation for this step).
        Draws a disk at (cx*cs + cs//2, cy*cs + cs//2) for each cell.
        """
        cfg = self._cfg
        if self._base_img is None:
            return np.zeros((64 + cfg.hud_height, 64, 3), dtype=np.uint8)

        base = self._base_img.copy()
        h, w = base.shape[:2]
        cs   = self._cs

        # Fire overlay — cell-space coordinates → pixel center
        if cfg.show_hazards and self._env_ref is not None:
            try:
                fire_cells = self._env_ref.current_fire   # frozenset of (cx, cy)
                r_f = max(1, cs // 2 - 1)
                side = 2 * r_f + 1
                yy, xx = np.mgrid[-r_f:r_f+1, -r_f:r_f+1]
                disk = (xx * xx + yy * yy) <= r_f * r_f
                fire_rgb = np.array(_C_FIRE, dtype=np.int32)
                for (cx, cy) in fire_cells:
                    fpx = cx * cs + cs // 2
                    fpy = cy * cs + cs // 2
                    y0, y1 = fpy - r_f, fpy + r_f + 1
                    x0, x1 = fpx - r_f, fpx + r_f + 1
                    dy0 = max(0, -y0); dy1 = side - max(0, y1 - h)
                    dx0 = max(0, -x0); dx1 = side - max(0, x1 - w)
                    y0 = max(0, y0); y1 = min(h, y1)
                    x0 = max(0, x0); x1 = min(w, x1)
                    if y0 >= y1 or x0 >= x1:
                        continue
                    patch_disk = disk[dy0:dy1, dx0:dx1]
                    region = base[y0:y1, x0:x1]
                    blended = (region.astype(np.int32) + fire_rgb) >> 1
                    region[patch_disk] = np.clip(blended, 0, 255).astype(np.uint8)[patch_disk]
            except Exception:
                pass

        # Path trail
        dot_r = max(1, cs // 8)
        side_d = 2 * dot_r + 1
        yy_d, xx_d = np.mgrid[-dot_r:dot_r+1, -dot_r:dot_r+1]
        disk_d = (xx_d * xx_d + yy_d * yy_d) <= dot_r * dot_r
        valid = [p for p in self._ep_path if p is not None]
        n_pts = len(valid)

        for i, pt in enumerate(valid[:-1]):
            recency = i / max(1, n_pts - 2)
            r_c = int(_C_PATH_OLD[0] + (_C_PATH_NEW[0] - _C_PATH_OLD[0]) * recency)
            g_c = int(_C_PATH_OLD[1] + (_C_PATH_NEW[1] - _C_PATH_OLD[1]) * recency)
            b_c = int(_C_PATH_OLD[2] + (_C_PATH_NEW[2] - _C_PATH_OLD[2]) * recency)
            ppx = pt[0] * cs + cs // 2
            ppy = pt[1] * cs + cs // 2
            y0, y1 = ppy - dot_r, ppy + dot_r + 1
            x0, x1 = ppx - dot_r, ppx + dot_r + 1
            dy0 = max(0, -y0); dy1 = side_d - max(0, y1 - h)
            dx0 = max(0, -x0); dx1 = side_d - max(0, x1 - w)
            y0 = max(0, y0); y1 = min(h, y1)
            x0 = max(0, x0); x1 = min(w, x1)
            if y0 >= y1 or x0 >= x1:
                continue
            mask = disk_d[dy0:dy1, dx0:dx1]
            base[y0:y1, x0:x1][mask] = [r_c, g_c, b_c]

        # Agent dot
        ax, ay = self._pos
        apx = ax * cs + cs // 2
        apy = ay * cs + cs // 2
        ar  = max(2, cs // 3)
        side_a = 2 * ar + 1
        yy_a, xx_a = np.mgrid[-ar:ar+1, -ar:ar+1]
        disk_a = (xx_a * xx_a + yy_a * yy_a) <= ar * ar
        y0, y1 = apy - ar, apy + ar + 1
        x0, x1 = apx - ar, apx + ar + 1
        dy0 = max(0, -y0); dy1 = side_a - max(0, y1 - h)
        dx0 = max(0, -x0); dx1 = side_a - max(0, x1 - w)
        y0 = max(0, y0); y1 = min(h, y1)
        x0 = max(0, x0); x1 = min(w, x1)
        if y0 < y1 and x0 < x1:
            mask_a = disk_a[dy0:dy1, dx0:dx1]
            base[y0:y1, x0:x1][mask_a] = list(_C_AGENT)

        maze_pil = Image.fromarray(base)
        total_h  = h + cfg.hud_height
        full     = Image.new("RGB", (w, total_h), color=_C_HUD_BG)
        full.paste(maze_pil, (0, 0))

        draw = ImageDraw.Draw(full)
        hx, hy = 8, h + 5
        lh = 18
        draw.text((hx, hy),          f"Phase: {self._label}",        fill=_C_HUD_FG)
        draw.text((hx, hy + lh),     label,                           fill=_C_HUD_FG)
        draw.text((hx, hy + lh * 2), f"Pos: {self._pos}  "
                                      f"Act: {_ACTION_NAMES.get(self._last_action,'?')}",
                                      fill=_C_HUD_FG)
        draw.text((hx, hy + lh * 3), f"Reward: {self._ep_reward:+.1f}  "
                                      f"Deaths: {self._ep_deaths}",
                                      fill=_C_HUD_FG)

        if overlay_text:
            cw  = len(overlay_text) * 8
            bx0 = w // 2 - cw // 2 - 6
            bx1 = w // 2 + cw // 2 + 6
            by0 = h // 2 - 14
            by1 = h // 2 + 14
            draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0))
            draw.text((w // 2 - cw // 2, h // 2 - 10),
                      overlay_text, fill=overlay_color)

        return np.array(full, dtype=np.uint8)

    def _write_mp4(self, path: Path, frames: list) -> None:
        if _HAS_IMAGEIO:
            try:
                import imageio
                with imageio.get_writer(
                    str(path),
                    fps=self._cfg.fps,
                    codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                ) as wrt:
                    for f in frames:
                        wrt.append_data(f.astype(np.uint8))
                return
            except Exception:
                pass

        if _HAS_CV2:
            import cv2
            fh, fw = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw     = cv2.VideoWriter(str(path), fourcc, self._cfg.fps, (fw, fh))
            for f in frames:
                vw.write(cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR))
            vw.release()
            return

        raise RuntimeError(
            "No MP4 backend available.  "
            "Install `imageio` + `imageio-ffmpeg`, or `opencv-python`."
        )

    def _write_gif(self, path: Path, frames: list) -> None:
        pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        dur = max(20, 1000 // max(1, self._cfg.fps))
        pil_frames[0].save(
            str(path),
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=dur,
            optimize=False,
        )