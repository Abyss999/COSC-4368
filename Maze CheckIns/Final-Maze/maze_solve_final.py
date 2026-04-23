"""
maze_solve_fixed_v2.py  ·  COSC 4368 AI  ·  Group 5
Method: Tabular Q-Learning + BFS-guided exploration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUG FIXES vs maze_solve_merged_fixed.py (v2 patch)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

V2-FX1.  CRITICAL — Loop kill triggered by death respawn visits.
         After a death, pos resets to start. The old code then
         incremented _cell_visits[start], so ≥80 deaths in one
         episode would trigger LOOP_KILL at the start cell and
         terminate the episode prematurely.  Fix: separate
         _death_respawn_turns set; do not count post-death
         increments toward loop detection.

V2-FX2.  CRITICAL — Goal check was AFTER loop-kill check.
         If the goal cell happened to reach the loop-kill
         threshold on the same turn it was first reached,
         loop-kill would fire and record_success() would never
         be called.  Success IS still counted via ep_stats
         (goal in ep_cells) but best_path is lost.  Fix:
         reorder — goal check always runs first; loop-kill only
         fires if goal was NOT reached this turn.

V2-FX3.  HIGH — Fire rotation direction was CCW not CW.
         _rotate_cell_90cw used:
             new_x = px + (cy - py)
             new_y = py - (cx - px)
         In screen coords (y increases downward) this is a
         counter-clockwise rotation.  Correct CW formula:
             new_x = px - (cy - py)   ← sign flipped
             new_y = py + (cx - px)   ← sign flipped
         Proof: (pivot+1,0) → (pivot,+1) = arm right→down ✓

V2-FX4.  HIGH — Fire rotation rate was 1 slot per 5 turns.
         The spec says "arm rotates 90° clockwise AFTER EACH
         agent action."  The old code used (turns//5)%4, which
         only changed the slot every 5 turns.  Fix: slot =
         turns % 4.  This also synchronises env and agent.

V2-FX5.  HIGH — transfer() cleared rot_danger, so the agent
         lost all fire hazard memory between training and test
         on the same maze.  Fix: for same_maze transfers, copy
         rot_danger and suspect_danger so the agent still knows
         which cells are dangerous at each fire orientation.

V2-FX6.  MEDIUM — EPS_FLOOR_BEFORE_GOAL=0.40 kept epsilon
         high for the entire pre-training phase (100 eps) and
         could persist into main training if goal was hard to
         find.  Lowered to 0.25 and added forced goal-seeking
         BFS at epsilon-random time when goal is already known.

V2-FX7.  LOW — record_success() is now called unconditionally
         at the point goal is detected, before any loop-kill
         check, ensuring best_path is always recorded when
         goal is reached.

All prior fixes (FX1-FX10 from v1) are retained unchanged.
"""
from __future__ import annotations

from PIL import Image, ImageDraw
import numpy as np
from collections import deque
from pathlib import Path
from scipy import ndimage
import random, time, hashlib, pickle
from dataclasses import dataclass
from typing import Optional

from maze_visualizer import PhaseVisualizer, VizConfig

# ════════════════════════════════════════════════════════
#  SETTINGS
# ════════════════════════════════════════════════════════
DEBUG_MOVE_VALIDATOR  = False
DEBUG_DIAG            = True
PRETRAIN_EPISODES     = 100
TRAIN_EPISODES        = 200
TEST_EPISODES         = 5
MAX_TURNS             = 10_000

MAX_STEP_LIMIT        = MAX_TURNS

LOOP_WARN_THRESHOLD   = 25
LOOP_KILL_THRESHOLD   = 80

LR        = 0.1
GAMMA_RL  = 0.95
EPS_START = 1.0
EPS_END   = 0.10
EPS_DECAY = 0.97
# V2-FX6: lowered before-goal floor from 0.40 → 0.25
EPS_FLOOR_BEFORE_GOAL = 0.25
EPS_TRAIN_FLOOR       = 0.10

R_GOAL          =  200.0
R_DEATH         =  -50.0
R_NEW_CELL      =   +1.0
R_STEP          =   -1.0
R_WALL          =   -2.0
R_CONFUSED      =   -5.0
R_REVISIT_SCALE =   -0.5

# ════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════
GRID = 64
FREE, WALL = 0, 1
BRIGHT = 128
_WALL_THRESH = 50

UP, RIGHT, DOWN, LEFT, WAIT = 0, 1, 2, 3, 4
N_ACTIONS = 5
DELTAS        = {UP:(0,-1), RIGHT:(1,0), DOWN:(0,1), LEFT:(-1,0), WAIT:(0,0)}
REVERSE       = {UP:DOWN, DOWN:UP, LEFT:RIGHT, RIGHT:LEFT, WAIT:WAIT}
_DELTA_TO_DIR = {(0,-1):UP, (1,0):RIGHT, (0,1):DOWN, (-1,0):LEFT}

_HERE             = Path(__file__).resolve().parent
MAZE_ALPHA_BLANK  = _HERE / "Contexts/maze-alpha/MAZE_0.png"
MAZE_ALPHA        = _HERE / "Contexts/maze-alpha/MAZE_1.png"
MAZE_BETA_BLANK   = _HERE / "Contexts/maze-beta/MAZE_0.png"
MAZE_BETA         = _HERE / "Contexts/maze-beta/MAZE_1.png"
MAZE_GAMMA_BLANK  = _HERE / "Contexts/maze-gamma/MAZE_0.png"
MAZE_GAMMA        = _HERE / "Contexts/maze-gamma/MAZE_1.png"
RESULTS           = _HERE / "Results"


# ════════════════════════════════════════════════════════
#  TURN RESULT
# ════════════════════════════════════════════════════════
@dataclass
class TurnResult:
    wall_hits        : int   = 0
    current_position : tuple = (0, 0)
    is_dead          : bool  = False
    is_confused      : bool  = False
    is_goal_reached  : bool  = False
    teleported       : bool  = False
    actions_executed : int   = 1
    pushed           : bool  = False


# ════════════════════════════════════════════════════════
#  IMAGE / GRID HELPERS
# ════════════════════════════════════════════════════════
def _load_image(path: Path):
    img = Image.open(path).convert("RGB")
    px  = np.array(img)
    h, w, _ = px.shape
    return px, h, w

def _pixel_grid(px: np.ndarray, h: int, w: int) -> np.ndarray:
    g    = np.ones((h, w), dtype=np.int8)
    dark = ((px[:, :, 0] < _WALL_THRESH) &
            (px[:, :, 1] < _WALL_THRESH) &
            (px[:, :, 2] < _WALL_THRESH))
    g[~dark] = FREE
    return g

def _cell_size(w: int, h: int) -> int:
    return max(1, min(w, h) // GRID)

def px_to_cell(px: int, py: int, cs: int) -> tuple:
    return (min(max(0, px // cs), GRID - 1),
            min(max(0, py // cs), GRID - 1))

def cell_to_px(cx: int, cy: int, cs: int) -> tuple:
    return cx * cs + cs // 2, cy * cs + cs // 2

def _find_start_goal(px_arr, h, w):
    sp = gp = None
    for x in range(w):
        if px_arr[h - 1, x, 0] > BRIGHT:
            sp = (x, h - 1)
        if px_arr[0, x, 0] > BRIGHT:
            gp = (x, 0)
    return sp, gp

def _clusters(px_arr, h, w, color_fn, min_sz=20):
    r = px_arr[:, :, 0].astype(np.int32)
    g = px_arr[:, :, 1].astype(np.int32)
    b = px_arr[:, :, 2].astype(np.int32)
    mask       = color_fn(r, g, b).astype(bool)
    labeled, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        pts = np.argwhere(labeled == i)
        if len(pts) >= min_sz:
            cy2, cx2 = pts.mean(0).astype(int)
            out.append((int(cx2), int(cy2)))
    return out

def _detect_hazards(px_arr, h, w) -> dict:
    fire = _clusters(px_arr, h, w,
        lambda r, g, b: (r > 180) & (g > 70) & (g < 175) & (b < 100))

    _yr = px_arr[:, :, 0].astype(np.int32)
    _yg = px_arr[:, :, 1].astype(np.int32)
    _yb = px_arr[:, :, 2].astype(np.int32)
    _ymask = (_yr > 190) & (_yg > 140) & (_yb < 90) & (_yr > _yg + 25)
    _ylab, _yn = ndimage.label(_ymask)
    yellow_pads = []
    conf        = []
    for _i in range(1, _yn + 1):
        _pts = np.argwhere(_ylab == _i)
        _sz  = len(_pts)
        if _sz < 30:
            continue
        _cy, _cx = _pts.mean(0).astype(int)
        _cell = (int(_cx), int(_cy))
        if _sz >= 80:
            yellow_pads.append(_cell)
        else:
            conf.append(_cell)
    yellow_pads.sort()

    green  = sorted(_clusters(px_arr, h, w,
        lambda r, g, b: (g > 170) & (r < 140) & (b < 160), min_sz=30))
    purple = sorted(_clusters(px_arr, h, w,
        lambda r, g, b: (r > 100) & (r < 210) & (b > 130) & (g < 100), min_sz=30))
    red    = sorted(_clusters(px_arr, h, w,
        lambda r, g, b: (r > 190) & (g < 80) & (b < 80), min_sz=80))

    def _is_square_pad(px_centroid) -> bool:
        px_x, px_y = px_centroid
        csize  = _cell_size(w, h)
        cell_x = px_x // csize
        cell_y = px_y // csize
        patch  = px_arr[cell_y*csize:(cell_y+1)*csize,
                        cell_x*csize:(cell_x+1)*csize]
        mid    = csize // 2
        center = patch[mid-3:mid+3, mid-3:mid+3]
        white  = int(((center[:, :, 0] > 220) &
                      (center[:, :, 1] > 220) &
                      (center[:, :, 2] > 220)).sum())
        return white >= 4

    tp = {}

    def _add_pair(a, b):
        a_sq, b_sq = _is_square_pad(a), _is_square_pad(b)
        if a_sq and not b_sq:
            tp[b] = a
        elif b_sq and not a_sq:
            tp[a] = b
        else:
            tp[a] = b
            tp[b] = a

    if len(green)       >= 2: _add_pair(green[0],       green[1])
    if len(purple)      >= 2: _add_pair(purple[0],      purple[1])
    if len(yellow_pads) >= 2: _add_pair(yellow_pads[0], yellow_pads[1])
    if len(red)         >= 2: _add_pair(red[0],         red[1])

    all_blue = _clusters(px_arr, h, w,
        lambda r, g, b: (b > 150) & (r < 120) & (g > 120) & (g < 220), min_sz=25)
    mid_y    = h // 2
    push_up  = [(x, y) for x, y in all_blue if y < mid_y]
    push_lft = [(x, y) for x, y in all_blue if y >= mid_y]

    return dict(fire=fire, confusion=conf, teleport=tp,
                push_up=push_up, push_left=push_lft)


def _rotate_cell_90cw(cx, cy, px, py):
    """
    V2-FX3: Correct 90° CW rotation in screen/maze coords (y increases downward).

    In screen coords the CW visual sequence is:
        RIGHT(+x,0) → DOWN(0,+y) → LEFT(-x,0) → UP(0,-y) → RIGHT

    Formula derivation:
        dx = cx - px,  dy = cy - py
        new_dx = -dy,  new_dy = +dx   ← CW in y-down space
        new_cx = px + new_dx = px - (cy - py)
        new_cy = py + new_dy = py + (cx - px)

    The previous formula (px+(cy-py), py-(cx-px)) was CCW in screen coords.
    """
    return px - (cy - py), py + (cx - px)


def _build_adj(px, w, h, cs) -> dict:
    half = cs // 2

    def _is_wall_px(r, c) -> bool:
        p = px[r, c]
        return (int(p[0]) < _WALL_THRESH and
                int(p[1]) < _WALL_THRESH and
                int(p[2]) < _WALL_THRESH)

    def wall_between(cx, cy, nx, ny) -> bool:
        if nx == cx + 1:
            by = min(cy * cs + half, h - 1)
            return (_is_wall_px(by, min(nx*cs,     w-1)) and
                    _is_wall_px(by, min(nx*cs + 1, w-1)))
        elif nx == cx - 1:
            by = min(cy * cs + half, h - 1)
            return (_is_wall_px(by, min(cx*cs,     w-1)) and
                    _is_wall_px(by, min(cx*cs + 1, w-1)))
        elif ny == cy + 1:
            bx = min(cx * cs + half, w - 1)
            return (_is_wall_px(min(ny*cs,     h-1), bx) and
                    _is_wall_px(min(ny*cs + 1, h-1), bx))
        else:
            bx = min(cx * cs + half, w - 1)
            return (_is_wall_px(min(cy*cs,     h-1), bx) and
                    _is_wall_px(min(cy*cs + 1, h-1), bx))

    adj = {}
    for cy in range(GRID):
        for cx in range(GRID):
            nbrs = {}
            for act, (dx, dy) in DELTAS.items():
                if act == WAIT:
                    continue
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < GRID and 0 <= ny < GRID):
                    continue
                if not wall_between(cx, cy, nx, ny):
                    nbrs[act] = (nx, ny)
            adj[(cx, cy)] = nbrs
    return adj

_ADJ_VER = b"v5"

def _build_adj_cached(px, pg, w, h, cs, path: Path) -> dict:
    key   = hashlib.md5(pg.tobytes() + _ADJ_VER).hexdigest()
    cache = path.parent / f".adj_{key}.pkl"
    if cache.exists():
        print("    (adjacency loaded from cache)")
        return pickle.loads(cache.read_bytes())
    print("    Building adjacency graph ...", end=" ", flush=True)
    adj = _build_adj(px, w, h, cs)
    cache.write_bytes(pickle.dumps(adj))
    print("done, cached.")
    return adj


# ════════════════════════════════════════════════════════
#  MAZE ENVIRONMENT
# ════════════════════════════════════════════════════════
class MazeEnv:
    """
    V2-FX3: _rotate_cell_90cw corrected to produce true CW rotation
    in screen coordinates.

    V2-FX4: Fire slot now = turns % 4 (rotates every turn) rather than
    (turns//5)%4 (which only rotated every 5 turns).  The spec states
    the arm rotates 90° CW after EACH agent action.
    """

    def __init__(self, path: Path, gamma_mode: bool = False):
        px, h, w  = _load_image(path)
        self._px, self._h, self._w = px, h, w
        cs        = _cell_size(w, h)
        self._cs  = cs
        self._pg  = _pixel_grid(px, h, w)
        self._adj = _build_adj_cached(px, self._pg, w, h, cs, path)
        self._gm  = gamma_mode

        sp, gp = _find_start_goal(px, h, w)
        if not sp or not gp:
            raise ValueError(f"Start/goal not found in {path.name}")
        self._start = px_to_cell(*sp, cs)
        self._goal  = px_to_cell(*gp, cs)

        hz = _detect_hazards(px, h, w)
        self._conf_cells = frozenset(px_to_cell(x, y, cs) for x, y in hz["confusion"])
        self._tele: dict = {
            px_to_cell(sx, sy, cs): px_to_cell(dx2, dy2, cs)
            for (sx, sy), (dx2, dy2) in hz["teleport"].items()
        }
        self._push_up  = frozenset(px_to_cell(x, y, cs) for x, y in hz["push_up"])
        self._push_lft = frozenset(px_to_cell(x, y, cs) for x, y in hz["push_left"])

        _non_fire = (self._conf_cells |
                     set(self._tele.keys()) |
                     set(self._tele.values()))
        self._non_fire_cells = _non_fire

        raw_fire_cells = [px_to_cell(x, y, cs) for (x, y) in hz["fire"]]
        self._base_fire_cells = [c for c in raw_fire_cells if c not in _non_fire]

        if self._base_fire_cells:
            self._fire_pivot_cell = max(self._base_fire_cells, key=lambda c: c[1])
        else:
            self._fire_pivot_cell = (0, 0)

        self._free_count = max(1, sum(1 for nb in self._adj.values() if nb))
        self._pos   = self._start
        self._conf  = 0
        self._turns = 0
        self._fire: frozenset = frozenset()
        self._rotate_fire(0)

        print(f"  {path.name:<20} start={self._start} goal={self._goal} "
              f"cs={cs}px | fire_cells={len(self._base_fire_cells)} "
              f"conf={len(self._conf_cells)} tele={len(self._tele)} "
              f"push_up={len(self._push_up)} push_left={len(self._push_lft)}")

    @property
    def start(self):        return self._start
    @property
    def goal(self):         return self._goal
    @property
    def cs(self):           return self._cs
    @property
    def free_count(self):   return self._free_count
    @property
    def current_fire(self): return self._fire

    def reset(self, ep: int = 0) -> tuple:
        self._pos   = self._start
        self._conf  = 0
        self._turns = 0
        self._rotate_fire(0)
        return self._pos

    def step(self, action: int) -> TurnResult:
        self._turns += 1
        # V2-FX4: rotate every turn, not every 5 turns
        self._rotate_fire(self._turns % 4)

        eff = REVERSE[action] if self._conf > 0 else action
        if self._conf > 0:
            self._conf -= 1

        cx, cy    = self._pos
        wall_hits = 0
        if eff == WAIT:
            new = (cx, cy)
        else:
            nbrs = self._adj.get((cx, cy), {})
            if eff in nbrs:
                new = nbrs[eff]
            else:
                new = (cx, cy)
                wall_hits = 1
        self._pos = new

        pushed = False
        if self._gm:
            if self._pos in self._push_up:
                nb2 = self._adj.get(self._pos, {})
                if UP in nb2:
                    self._pos = nb2[UP]
                pushed = True
            elif self._pos in self._push_lft:
                nb2 = self._adj.get(self._pos, {})
                if LEFT in nb2:
                    self._pos = nb2[LEFT]
                pushed = True

        if self._pos in self._fire:
            self._pos  = self._start
            self._conf = 0
            return TurnResult(wall_hits, self._pos, True, False, False, False, 1, pushed)

        tele = False
        if self._pos in self._tele:
            dest = self._tele[self._pos]
            if dest != self._pos:
                self._pos = dest
                tele = True

        if self._pos in self._conf_cells:
            self._conf = 2

        goal = self._pos == self._goal
        return TurnResult(wall_hits, self._pos, False, self._conf > 0,
                          goal, tele, 1, pushed)

    def pixels_copy(self): return self._px.copy()

    def _rotate_fire(self, slot: int):
        """
        V2-FX3: Uses corrected CW rotation formula.
        V2-FX4: Called with turns%4 (every turn) instead of (turns//5)%4.
        """
        if not self._base_fire_cells:
            self._fire = frozenset()
            return
        px_c, py_c = self._fire_pivot_cell
        active = []
        for cx, cy in self._base_fire_cells:
            rx, ry = cx, cy
            for _ in range(slot % 4):
                rx, ry = _rotate_cell_90cw(rx, ry, px_c, py_c)
            rx = min(max(0, rx), GRID - 1)
            ry = min(max(0, ry), GRID - 1)
            active.append((rx, ry))
        self._fire = frozenset(c for c in active if c not in self._non_fire_cells)


# ════════════════════════════════════════════════════════
#  BFS ON KNOWN MAP
# ════════════════════════════════════════════════════════
def bfs(known: dict, start: tuple, goal: tuple,
        danger: set, blocked: set, tele: dict,
        passable: set | None = None) -> list:
    """
    FX10 retained: teleport BFS adds destination to frontier.
    """
    queue      = deque([start])
    came: dict = {start: None}
    q_append   = queue.append

    while queue:
        cur = queue.popleft()
        if cur == goal:
            path, node = [], goal
            while node is not None:
                path.append(node)
                node = came[node]
            path.reverse()
            return path

        cx, cy = cur
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nb = (cx + dx, cy + dy)
            if nb in came or nb in danger:
                continue
            if (cur, nb) in blocked:
                continue
            if passable is not None:
                if (cur, nb) not in passable:
                    continue
            elif nb not in known:
                continue
            came[nb] = cur
            q_append(nb)

        if cur in tele:
            dest = tele[cur]
            if dest not in came and dest not in danger:
                came[dest] = cur
                q_append(dest)

    return []


# ════════════════════════════════════════════════════════
#  Q-LEARNING AGENT
# ════════════════════════════════════════════════════════
class Agent:
    """
    V2-FX4: Q-table fire_slot dimension now tracks turns%4 (matches env).
    V2-FX5: rot_danger preserved in same_maze transfer().
    """

    def __init__(self):
        self.epsilon   = EPS_START
        self.goal_ward = False
        self.zero_shot = False

        # 5D Q-table: [x, y, confused, fire_slot, action]
        self.q = np.zeros((GRID, GRID, 2, 4, N_ACTIONS), dtype=np.float32)

        self.known      : dict = {}
        self.blocked    : set  = set()
        self.passable   : set  = set()
        self.danger     : set  = set()
        self.rot_danger : dict = {0: set(), 1: set(), 2: set(), 3: set()}
        self.suspect_danger: dict = {0: set(), 1: set(), 2: set(), 3: set()}
        self.conf_cells : set  = set()
        self.tele       : dict = {}
        self.goal       : Optional[tuple] = None
        self.start      : Optional[tuple] = None
        self.best_path  : list = []
        self.best_turns : int  = 999_999

        self.goal_found : bool = False

        self._pos       : tuple = (0, 0)
        self._confused  : bool  = False
        self._ep_path   : list  = []
        self._ep_cells  : set   = set()
        self._ep_deaths : int   = 0
        self._ep_turns  : int   = 0
        self._ep_tc     : int   = 0

        self._plan       : list = []
        self._wait_count : int  = 0
        self._rot_slot_cache: int = -1

        self._visit_counts: dict = {}
        self._frontier_cache     : tuple = None
        self._known_size_at_cache: int   = 0

    def reset_episode(self, start: tuple):
        self._pos            = start
        self._confused       = False
        self._ep_path        = [start]
        self._ep_cells       = {start}
        self._ep_deaths      = 0
        self._ep_turns       = 0
        self._ep_tc          = 0
        self._plan           = []
        self._wait_count     = 0
        self._rot_slot_cache = -1
        self._visit_counts   = {start: 1}
        self._frontier_cache       = None
        self._known_size_at_cache  = 0
        if self.start is None:
            self.start = start
        self.known.setdefault(start, "free")
        if self.goal is not None:
            self.known.setdefault(self.goal, "goal")
        self.danger = set(self.rot_danger[0])

    def _rot_slot(self) -> int:
        # V2-FX4: match env's turns%4 instead of (turns//5)%4
        return self._ep_tc % 4

    def _sync_danger(self):
        slot = self._ep_tc % 4
        if slot != self._rot_slot_cache:
            self.danger = set(self.rot_danger[slot])
            self._rot_slot_cache = slot
            self._frontier_cache = None

    def choose(self, pos: tuple) -> int:
        self._sync_danger()
        slot = self._rot_slot()

        if random.random() < self.epsilon:
            # V2-FX6: if goal is known, bias toward goal even during exploration
            if self.goal is not None and self.goal in self.known and random.random() < 0.3:
                path = bfs(self.known, pos, self.goal,
                           self.danger | self.conf_cells, self.blocked, self.tele)
                if len(path) > 1:
                    return self._dir(pos, path[1])

            fa = self._frontier_actions(pos)
            if fa:
                return random.choice(fa)
            fc, unk = self._nearest_frontier_cached(pos)
            if fc is not None and fc != pos:
                path = bfs(self.known, pos, fc, self.danger, self.blocked, self.tele)
                if len(path) > 1:
                    return self._dir(pos, path[1])
            cx, cy = pos
            safe_moves = []
            for act, (dx, dy) in DELTAS.items():
                if act == WAIT:
                    continue
                nb = (cx + dx, cy + dy)
                if (0 <= nb[0] < GRID and 0 <= nb[1] < GRID
                        and nb not in self.danger
                        and (pos, nb) not in self.blocked):
                    safe_moves.append(act)
            return random.choice(safe_moves) if safe_moves else random.randint(0, N_ACTIONS - 1)

        # Follow cached plan
        if self._plan:
            nxt = self._plan[0]
            if nxt not in self.danger:
                self._plan.pop(0)
                return self._dir(pos, nxt)
            self._plan = []

        # Phase 2: goal known → BFS to goal
        if self.goal is not None and self.goal in self.known:
            known    = self.known
            blocked  = self.blocked
            tele     = self.tele
            passable = self.passable

            always_on = (self.rot_danger[0] & self.rot_danger[1] &
                         self.rot_danger[2] & self.rot_danger[3])

            def _safe(avoid_set):
                d = {k: v for k, v in known.items() if k not in avoid_set}
                if pos in known:
                    d[pos] = known[pos]
                return d

            if self.goal_ward:
                avoid = always_on | self.conf_cells
                path = (
                    bfs(_safe(avoid), pos, self.goal, avoid, blocked, tele) or
                    bfs(known,        pos, self.goal, self.conf_cells, blocked, tele) or
                    bfs(known,        pos, self.goal, self.conf_cells, set(),   tele) or
                    bfs(known,        pos, self.goal, set(),           set(),   tele)
                )
            else:
                avoid    = self.danger | self.conf_cells
                avoid_ao = always_on   | self.conf_cells
                path = (
                    bfs(_safe(avoid),     pos, self.goal, avoid,    blocked, tele, passable) or
                    (bfs(_safe(avoid),    pos, self.goal, avoid,    blocked, tele) if passable else []) or
                    bfs(_safe(avoid_ao),  pos, self.goal, avoid_ao, blocked, tele, passable) or
                    (bfs(_safe(avoid_ao), pos, self.goal, avoid_ao, blocked, tele) if passable else []) or
                    bfs(known,            pos, self.goal, self.conf_cells, blocked, tele) or
                    bfs(known,            pos, self.goal, self.conf_cells, set(),   tele) or
                    bfs(known,            pos, self.goal, set(),           set(),   tele)
                )

            if len(path) > 1:
                nxt = path[1]

                if nxt in self.danger and self._wait_count < 20:
                    if nxt not in always_on:
                        self._wait_count += 1
                        return WAIT

                if nxt in always_on and self._wait_count >= 20:
                    cx, cy = pos
                    for act, (dx, dy) in DELTAS.items():
                        if act == WAIT:
                            continue
                        nb = (cx + dx, cy + dy)
                        if (nb in self.known and
                                nb not in always_on and
                                nb not in self.conf_cells and
                                (pos, nb) not in self.blocked):
                            self._wait_count = 0
                            return act

                self._wait_count = 0
                self._plan = path[2:]
                return self._dir(pos, nxt)

        # Phase 1: frontier exploration
        fc, unk = self._nearest_frontier_cached(pos)
        if fc is not None:
            if fc == pos and unk is not None:
                return self._dir(pos, unk)
            path = bfs(self.known, pos, fc, self.danger, self.blocked, self.tele)
            if len(path) > 1:
                self._plan = path[2:]
                return self._dir(pos, path[1])

        fa = self._frontier_actions(pos)
        return random.choice(fa) if fa else random.randint(0, N_ACTIONS - 1)

    def _dir(self, a: tuple, b: tuple) -> int:
        return _DELTA_TO_DIR.get((b[0]-a[0], b[1]-a[1]), random.randint(0, 3))

    def _frontier_actions(self, pos: tuple) -> list:
        cx, cy = pos
        return [act for act, (dx, dy) in DELTAS.items()
                if act != WAIT
                and 0 <= cx + dx < GRID and 0 <= cy + dy < GRID
                and (cx + dx, cy + dy) not in self.known
                and (pos, (cx + dx, cy + dy)) not in self.blocked]

    def _nearest_frontier_cached(self, pos: tuple):
        cur_size = len(self.known)
        if (self._frontier_cache is not None and
                self._frontier_cache[0] == pos and
                self._known_size_at_cache == cur_size):
            return self._frontier_cache[1]
        result = self._nearest_frontier(pos)
        self._frontier_cache      = (pos, result)
        self._known_size_at_cache = cur_size
        return result

    def _nearest_frontier(self, pos: tuple):
        queue    = deque([pos])
        visited  = {pos}
        frontier = []
        hop      = {pos: 0}

        while queue:
            cur    = queue.popleft()
            cx, cy = cur
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nb = (cx + dx, cy + dy)
                if nb in visited or not (0 <= nb[0] < GRID and 0 <= nb[1] < GRID):
                    continue
                if (cur, nb) in self.blocked:
                    continue
                visited.add(nb)
                if nb not in self.known:
                    if self.goal_ward and self.goal is not None:
                        md = abs(nb[0] - self.goal[0]) + abs(nb[1] - self.goal[1])
                        frontier.append((md, hop[cur] + 1, cur, nb))
                    else:
                        return cur, nb
                elif nb not in self.danger and self.known[nb] != "wall":
                    hop[nb] = hop[cur] + 1
                    queue.append(nb)

            if cur in self.tele:
                dest = self.tele[cur]
                if dest not in visited and dest not in self.danger:
                    visited.add(dest)
                    hop[dest] = hop[cur] + 1
                    queue.append(dest)

        if frontier:
            _, _, best_cur, best_nb = min(frontier)
            return best_cur, best_nb
        return None, None

    def update_q(self, prev: tuple, prev_conf: bool,
                 action: int, reward: float, tr: TurnResult):
        """V2-FX4: Q update uses turns%4 slot."""
        slot = self._rot_slot()
        s  = self.q[prev[0], prev[1], int(prev_conf), slot]
        next_slot = (self._ep_tc) % 4
        ns = self.q[tr.current_position[0], tr.current_position[1],
                    int(tr.is_confused), next_slot]
        td = reward + (0.0 if tr.is_goal_reached else GAMMA_RL * float(ns.max()))
        s[action] += LR * (td - float(s[action]))

    def observe(self, prev: tuple, action: int, tr: TurnResult,
                prev_confused: bool = False):
        self._ep_turns += 1
        self._ep_tc    += 1
        self._sync_danger()
        rot = self._rot_slot()

        if tr.is_dead or tr.teleported or tr.pushed or tr.wall_hits or tr.is_confused:
            self._plan = []
        if not tr.is_dead and tr.wall_hits == 0:
            self._wait_count = 0

        if tr.is_dead:
            if tr.wall_hits == 0 and not prev_confused and not tr.pushed:
                dx, dy = DELTAS.get(action, (0, 0))
                pit    = (prev[0] + dx, prev[1] + dy)
                if 0 <= pit[0] < GRID and 0 <= pit[1] < GRID:
                    self.known[pit] = "death"
                    self.danger.add(pit)
                    self.rot_danger[rot].add(pit)
                    for s in range(4):
                        if s != rot:
                            self.suspect_danger[s].add(pit)
            self._ep_deaths += 1
            self._pos      = self.start or tr.current_position
            self._confused = False
            return

        cur            = tr.current_position
        self._pos      = cur
        self._confused = tr.is_confused
        self._ep_cells.add(cur)
        self._ep_path.append(cur)

        for s in range(4):
            self.suspect_danger[s].discard(cur)

        if tr.is_goal_reached:
            self.known[cur] = "goal"
            if self.goal is None:
                self.goal = cur
            self.goal_found = True

        elif tr.is_confused and not prev_confused:
            self.known[cur] = "confusion"
            self.conf_cells.add(cur)

        else:
            if self.known.get(cur) == "death":
                self.known[cur] = "free"
                self.rot_danger[rot].discard(cur)
            elif self.known.get(cur) not in ("confusion", "teleport"):
                self.known.setdefault(cur, "free")

            if tr.wall_hits == 0 and not prev_confused:
                dx, dy  = DELTAS.get(action, (0, 0))
                stepped = (prev[0] + dx, prev[1] + dy)

                if tr.pushed and not tr.teleported:
                    if 0 <= stepped[0] < GRID and 0 <= stepped[1] < GRID:
                        self.known.setdefault(stepped, "teleport")
                        self.tele[stepped] = cur

                elif tr.teleported and not tr.pushed:
                    self.known.setdefault(cur, "teleport")
                    if 0 <= stepped[0] < GRID and 0 <= stepped[1] < GRID:
                        self.known.setdefault(stepped, "teleport")
                        self.tele[stepped] = cur

                elif tr.pushed and tr.teleported:
                    self.known.setdefault(cur, "teleport")
                    if 0 <= stepped[0] < GRID and 0 <= stepped[1] < GRID:
                        self.known.setdefault(stepped, "teleport")
                        self.tele[stepped] = cur

        if tr.wall_hits == 0 and not tr.is_dead and not tr.pushed:
            dx2 = cur[0] - prev[0]
            dy2 = cur[1] - prev[1]
            if abs(dx2) + abs(dy2) == 1:
                self.passable.add((prev, cur))
                self.passable.add((cur, prev))

        if tr.wall_hits > 0 and not prev_confused:
            dx, dy = DELTAS.get(action, (0, 0))
            target = (prev[0] + dx, prev[1] + dy)
            if 0 <= target[0] < GRID and 0 <= target[1] < GRID:
                self.blocked.add((prev, target))
                self.blocked.add((target, prev))

    def record_success(self, turns: int):
        if turns < self.best_turns:
            self.best_turns = turns
            clean = bfs(self.known, self.start, self.goal,
                        set(), self.blocked, self.tele, self.passable)
            if len(clean) > 1:
                self.best_path = clean
            else:
                clean2 = bfs(self.known, self.start, self.goal,
                             set(), self.blocked, self.tele)
                self.best_path = clean2 if len(clean2) > 1 else list(self._ep_path)

    def decay_epsilon(self, train: bool = True):
        if train and not self.goal_found:
            floor = EPS_FLOOR_BEFORE_GOAL
        elif train:
            floor = EPS_TRAIN_FLOOR
        else:
            floor = EPS_END
        self.epsilon = max(floor, self.epsilon * EPS_DECAY)

    @property
    def ep_stats(self) -> dict:
        return dict(
            success  = (self.goal in self._ep_cells) if self.goal else False,
            path_len = len(self._ep_path),
            turns    = self._ep_turns,
            deaths   = self._ep_deaths,
            unique   = len(self._ep_cells),
        )


# ════════════════════════════════════════════════════════
#  TRANSFER AGENT
# ════════════════════════════════════════════════════════
def transfer(src: Agent, eps: float, same_maze: bool = False) -> Agent:
    dst = Agent()
    dst.q          = src.q.copy()
    dst.epsilon    = eps

    if same_maze:
        dst.known      = {k: ("free" if v == "death" else v)
                          for k, v in src.known.items()}
        dst.goal       = src.goal
        dst.start      = src.start
        dst.tele       = dict(src.tele)
        dst.passable   = set(src.passable)
        dst.blocked    = set()
        dst.conf_cells = set(src.conf_cells)
        dst.best_path  = list(src.best_path)
        dst.best_turns = src.best_turns
        # V2-FX5: preserve fire hazard knowledge on same-maze transfer
        dst.rot_danger     = {s: set(src.rot_danger[s])     for s in range(4)}
        dst.suspect_danger = {s: set(src.suspect_danger[s]) for s in range(4)}
        dst.danger         = set(src.rot_danger[0])  # slot 0 as starting danger
    else:
        dst.goal  = src.goal
        dst.start = src.start
        dst.danger        = set()
        dst.rot_danger    = {0: set(), 1: set(), 2: set(), 3: set()}
        dst.suspect_danger= {0: set(), 1: set(), 2: set(), 3: set()}

    dst._plan         = []
    dst._wait_count   = 0
    dst.goal_found    = False

    dst._visit_counts         = {}
    dst._frontier_cache       = None
    dst._known_size_at_cache  = 0

    return dst


# ════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ════════════════════════════════════════════════════════
def run_episodes(env: MazeEnv, agent: Agent, n: int,
                 train: bool, label: str,
                 explore: bool = False,
                 viz: "PhaseVisualizer | None" = None) -> list:
    all_stats   = []
    _diag_recent: list = []

    for ep in range(n):
        ep_t0 = time.time()
        pos   = env.reset(ep)
        agent.reset_episode(pos)

        if viz is not None:
            viz.begin_episode(env, agent, ep_idx=ep)

        # V2-FX1: separate loop-detection counter from death-respawn tracking
        # _cell_visits only increments for genuine movement (not post-death respawn)
        _cell_visits: dict = {}
        _last_was_death = False  # flag to skip loop-count increment after death

        for turn in range(MAX_STEP_LIMIT):
            prev_conf = agent._confused
            action    = agent.choose(pos)

            tr = env.step(action)

            # Reward shaping
            reward = R_STEP
            if tr.wall_hits:
                reward += R_WALL * tr.wall_hits
            if tr.is_dead:
                reward += R_DEATH
            elif tr.is_goal_reached:
                reward += R_GOAL
            else:
                if tr.is_confused and not prev_conf:
                    reward += R_CONFUSED
                cur_pos = tr.current_position
                if cur_pos not in agent.known:
                    reward += R_NEW_CELL
                else:
                    visit_n = agent._visit_counts.get(cur_pos, 0)
                    if visit_n > 2:
                        reward += max(-5.0, R_REVISIT_SCALE * (visit_n - 2))

            track_pos = tr.current_position if not tr.is_dead else pos
            agent._visit_counts[track_pos] = (
                agent._visit_counts.get(track_pos, 0) + 1)

            if train:
                agent.update_q(pos, prev_conf, action, reward, tr)

            agent.observe(pos, action, tr, prev_confused=prev_conf)
            pos = agent._pos

            if viz is not None:
                viz.step(pos, action=action, reward=reward, is_dead=tr.is_dead)

            # V2-FX2 + V2-FX7: goal check ALWAYS runs first, before loop kill.
            # record_success() is called unconditionally when goal is reached.
            if tr.is_goal_reached:
                agent.record_success(turn + 1)
                if DEBUG_DIAG:
                    print(f"    [GOAL] ep={ep+1} turn={turn+1} pos={pos}")
                break

            # V2-FX1: only count visits for genuine moves (not death respawn).
            # After a death, agent is at start — don't count that as a loop visit.
            if not tr.is_dead:
                _cell_visits[pos] = _cell_visits.get(pos, 0) + 1
                if _cell_visits[pos] == LOOP_WARN_THRESHOLD and DEBUG_DIAG:
                    print(f"    [LOOP WARN] ep={ep+1} turn={turn+1} "
                          f"cell={pos} visited {LOOP_WARN_THRESHOLD}x")
                if _cell_visits[pos] >= LOOP_KILL_THRESHOLD:
                    if DEBUG_DIAG:
                        print(f"    [LOOP KILL] ep={ep+1} turn={turn+1} "
                              f"cell={pos} — terminating episode")
                    break
        else:
            if DEBUG_DIAG:
                print(f"    [EP {ep+1:>3}] Episode terminated (step limit "
                      f"{MAX_STEP_LIMIT})")

        if viz is not None:
            viz.end_episode(success=agent.ep_stats["success"])

        stats = agent.ep_stats
        all_stats.append(stats)

        if train:
            agent.decay_epsilon(train=True)
        elif not explore and agent.goal in agent._ep_cells:
            agent.epsilon = max(EPS_END, agent.epsilon * 0.5)

        if DEBUG_DIAG:
            ep_time = time.time() - ep_t0
            fps_val = stats["turns"] / max(ep_time, 1e-6)
            _diag_recent.append(stats["success"])
            if len(_diag_recent) > 10:
                _diag_recent.pop(0)
            sr_recent = sum(_diag_recent) / len(_diag_recent)
            print(f"    [EP {ep+1:>3}] steps={stats['turns']:>5}  "
                  f"time={ep_time:.2f}s  fps={fps_val:.1f}  "
                  f"deaths={stats['deaths']}  eps={agent.epsilon:.3f}  "
                  f"SR(recent)={sr_recent:.0%}  goal_found={agent.goal_found}")

        if not train and n <= 10:
            s = all_stats[-1]
            print(f"    [{label[:18]:<18}] ep {ep+1:>2}  "
                  f"{'SUCCESS' if s['success'] else 'FAIL   '}  "
                  f"turns={s['turns']:>5}  deaths={s['deaths']}  "
                  f"map={len(agent.known)}/{env.free_count}  "
                  f"passable={len(agent.passable)}")

        if (ep + 1) % 20 == 0 or ep == n - 1:
            recent = all_stats[-10:]
            sr     = sum(s["success"] for s in recent) / len(recent)
            print(f"    [{label[:18]:<18}] ep {ep+1:>3}/{n}  "
                  f"e={agent.epsilon:.3f}  SR(10)={sr:.0%}  "
                  f"map={len(agent.known)}/{env.free_count}")

    return all_stats


# ════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════
def report_metrics(stats: list, agent: Agent, env: MazeEnv,
                   label: str, train_stats: list | None = None) -> dict:
    succ  = [s for s in stats if s["success"]]
    tot_d = sum(s["deaths"] for s in stats)
    tot_t = sum(s["turns"]  for s in stats)

    SR  = len(succ) / len(stats) if stats else 0.0
    APL = sum(s["path_len"] for s in succ) / len(succ) if succ else float("inf")
    AT  = sum(s["turns"]    for s in succ) / len(succ) if succ else float("inf")
    DR  = tot_d / max(1, tot_t)
    EE  = (sum(s["unique"] / max(1, s["path_len"]) for s in stats) / len(stats)
           if stats else 0.0)
    MC  = (len([v for v in agent.known.values() if v != "wall"])
           / env.free_count)

    LE = None
    if train_stats:
        for i in range(10, len(train_stats) + 1):
            if sum(s["success"] for s in train_stats[i-10:i]) / 10 >= 0.80:
                LE = i
                break

    bar = "-" * 52
    print(f"\n{bar}")
    print(f"  METRICS  --  {label}")
    print(f"{bar}")
    print(f"  Success Rate          : {SR:>7.1%}  (target >80%)")
    print(f"  Avg Path Length       : {APL:>8.1f} cells")
    print(f"  Avg Turns to Solution : {AT:>8.1f}  (target <1000)")
    print(f"  Death Rate            : {DR:>8.4f}  (target <0.05)")
    print(f"  [Bonus] Exploration Efficiency : {EE:.4f}")
    print(f"  [Bonus] Map Completeness       : {MC:.1%}")
    if LE:
        print(f"  [Bonus] Learning Efficiency    : converged at episode {LE}")
    elif train_stats:
        print(f"  [Bonus] Learning Efficiency    : did not converge to 80% SR")
    print(bar)
    return dict(SR=SR, APL=APL, AT=AT, DR=DR, EE=EE, MC=MC, LE=LE)


# ════════════════════════════════════════════════════════
#  STATIC VISUALISATION
# ════════════════════════════════════════════════════════
def save_path_img(env: MazeEnv, path_cells: list, out: Path):
    if not path_cells:
        print(f"    (no path for {out.name})"); return
    img   = env.pixels_copy()
    cs, t = env.cs, 2
    h, w  = img.shape[:2]
    for cx, cy in path_cells:
        px2, py2 = cell_to_px(cx, cy, cs)
        for dy2 in range(-t, t + 1):
            for dx2 in range(-t, t + 1):
                nx, ny = px2 + dx2, py2 + dy2
                if 0 <= nx < w and 0 <= ny < h:
                    img[ny, nx] = [255, 0, 200]
    Image.fromarray(img).save(out)
    print(f"    Saved -> {out.name}")

def save_map_img(env: MazeEnv, agent: Agent, out: Path):
    img  = Image.fromarray(env.pixels_copy())
    draw = ImageDraw.Draw(img)
    cs   = env.cs
    r    = max(2, cs // 4)
    for (cx, cy), ctype in agent.known.items():
        px2, py2 = cell_to_px(cx, cy, cs)
        if ctype == "death":
            draw.ellipse([px2-r, py2-r, px2+r, py2+r],
                         fill=(255, 60, 0), outline=(160, 0, 0))
        elif ctype == "confusion":
            draw.rectangle([px2-r, py2-r, px2+r, py2+r], fill=(160, 0, 220))
        elif ctype == "teleport":
            draw.ellipse([px2-r, py2-r, px2+r, py2+r], fill=(0, 200, 80))
    img.save(out)
    print(f"    Saved -> {out.name}")

def save_curve(stats: list, out: Path):
    W, H, M = 840, 440, 65
    n = len(stats)
    if n < 2: return
    img  = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pw, ph = W - 2*M, H - 2*M
    win = 10

    def roll_sr(i):
        ch = stats[max(0, i-win+1):i+1]
        return sum(s["success"] for s in ch) / len(ch)

    def roll_dr(i):
        ch = stats[max(0, i-win+1):i+1]
        td = sum(s["deaths"] for s in ch)
        tt = sum(s["turns"]  for s in ch)
        return min(1.0, (td / max(1, tt)) * 20)

    sr_v = [roll_sr(i) for i in range(n)]
    dr_v = [roll_dr(i) for i in range(n)]

    def pt(i, v):
        return M + int(i * pw / max(1, n-1)), H - M - int(v * ph)

    draw.rectangle([M, M, W-M, H-M], outline=(0, 0, 0), width=2)
    for v in (0.25, 0.5, 0.75, 1.0):
        y2 = H - M - int(v * ph)
        draw.line([M, y2, W-M, y2], fill=(210, 210, 210))
        draw.text((M - 40, y2 - 7), f"{v:.0%}", fill=(80, 80, 80))
    for i in range(1, n):
        draw.line([*pt(i-1, sr_v[i-1]), *pt(i, sr_v[i])],
                  fill=(30, 100, 200), width=2)
        draw.line([*pt(i-1, dr_v[i-1]), *pt(i, dr_v[i])],
                  fill=(200, 60, 0), width=2)

    draw.text((M, M-20),
              "Learning Curve — maze-alpha training (10-ep window)", fill=(0, 0, 0))
    draw.text((W//2 - 40, H-M+12), f"Episode (1-{n})", fill=(0, 0, 0))
    lx = W - M - 170
    draw.rectangle([lx, M+10, lx+14, M+20], fill=(30, 100, 200))
    draw.text((lx+18, M+10), "Success Rate",     fill=(0, 0, 0))
    draw.rectangle([lx, M+32, lx+14, M+42], fill=(200, 60, 0))
    draw.text((lx+18, M+32), "Death Rate (x20)", fill=(0, 0, 0))
    img.save(out)
    print(f"    Saved -> {out.name}")


# ════════════════════════════════════════════════════════
#  PHASE RUNNER
# ════════════════════════════════════════════════════════
def run_phase(path: Path, agent: Agent, n: int, train: bool,
              label: str, gamma_mode: bool = False,
              train_stats: list | None = None,
              explore: bool = False,
              viz: "PhaseVisualizer | None" = None) -> tuple:
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    env = MazeEnv(path, gamma_mode=gamma_mode)

    if viz is not None:
        viz.begin_phase(label.replace(" ", "_"))

    stats = run_episodes(env, agent, n, train, label,
                         explore=explore, viz=viz)

    if viz is not None:
        viz.flush(agent, env)

    m = report_metrics(stats, agent, env, label, train_stats)

    stem = path.stem
    ctx  = path.parent.name
    tag  = "train" if train else "test"

    if agent.best_path:
        save_path_img(env, agent.best_path,
                      RESULTS / f"{stem}_{ctx}_rl_{tag}_solved.png")
    save_map_img(env, agent,
                 RESULTS / f"{stem}_{ctx}_rl_{tag}_map.png")
    return stats, m, env


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════
RANDOM_SEED = 42
VIZ_ENABLED = True
VIZ_TRAIN   = True
VIZ_TEST    = True


def _make_viz(filename: str,
              step_stride       : int  = 2,
              frames_per_episode: int  = 120,
              episode_stride    : int  = 1,
              show_q_arrows     : bool = False) -> "PhaseVisualizer | None":
    if not VIZ_ENABLED:
        return None
    return PhaseVisualizer(VizConfig(
        enabled            = True,
        step_stride        = step_stride,
        frames_per_episode = frames_per_episode,
        episode_stride     = episode_stride,
        replay_frames      = 80,
        pause_on_end       = 15,
        fps                = 20,
        out_path           = RESULTS / filename,
        show_q_arrows      = show_q_arrows,
        show_hazards       = True,
        hud_height         = 90,
    ))


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 52)
    print("  COSC 4368 — Maze Solver  |  Group 5")
    print("  Q-Learning + BFS  (v2 patch: V2-FX1..7 applied)")
    print("=" * 52)
    print(f"  PRETRAIN={PRETRAIN_EPISODES}  TRAIN={TRAIN_EPISODES}  "
          f"TEST={TEST_EPISODES}  MAX_TURNS={MAX_TURNS}")
    print(f"  LR={LR}  GAMMA={GAMMA_RL}  eps {EPS_START}→{EPS_END}")
    print(f"  LOOP thresholds: warn={LOOP_WARN_THRESHOLD}  kill={LOOP_KILL_THRESHOLD}")
    print(f"  V2 FIXES: V2-FX1=loop-kill-death-fix V2-FX2=goal-before-loop "
          f"V2-FX3=CW-rotation V2-FX4=fire-every-turn "
          f"V2-FX5=transfer-rot-danger V2-FX6=eps-floor V2-FX7=record-success-first")

    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    viz_train  = _make_viz("training_progress.mp4",
                           step_stride=4, frames_per_episode=40,
                           episode_stride=10) if VIZ_TRAIN else None
    viz_test_a = _make_viz("test_alpha.mp4",
                           step_stride=2, frames_per_episode=150,
                           show_q_arrows=True) if VIZ_TEST else None
    viz_test_b = _make_viz("test_beta.mp4",
                           step_stride=2, frames_per_episode=150) if VIZ_TEST else None
    viz_test_g = _make_viz("test_gamma.mp4",
                           step_stride=2, frames_per_episode=150) if VIZ_TEST else None

    # ── 0. Pre-explore blank alpha ────────────────────────────────────────────
    agent_pre = Agent()
    run_phase(MAZE_ALPHA_BLANK, agent_pre, PRETRAIN_EPISODES,
              train=True,
              label="MAZE-ALPHA  Pre-train (blank)",
              explore=False,
              viz=None)

    agent_pre_exp = transfer(agent_pre, eps=0.5, same_maze=True)
    run_phase(MAZE_ALPHA_BLANK, agent_pre_exp, PRETRAIN_EPISODES,
              train=False,
              label="MAZE-ALPHA  Pre-explore (blank)",
              explore=True,
              viz=None)

    # ── 1. Train on maze-alpha (with hazards) ─────────────────────────────────
    agent_train = transfer(agent_pre_exp, eps=EPS_START, same_maze=True)
    agent_train.passable = set()
    train_stats, _, _ = run_phase(
        MAZE_ALPHA, agent_train, TRAIN_EPISODES,
        train=True, label="MAZE-ALPHA  Training",
        viz=viz_train)

    save_curve(train_stats, RESULTS / "learning_curve_alpha.png")

    # ── 2. Test maze-alpha ────────────────────────────────────────────────────
    # V2-FX5: same_maze=True now preserves rot_danger
    agent_at = transfer(agent_train, eps=EPS_END, same_maze=True)
    test_a, _, _ = run_phase(
        MAZE_ALPHA, agent_at, TEST_EPISODES,
        train=False, label="MAZE-ALPHA  Test",
        train_stats=train_stats,
        viz=viz_test_a)

    # ── 3. Test maze-beta (zero-shot) ─────────────────────────────────────────
    agent_bpre = transfer(agent_train, eps=0.5)
    agent_bpre.goal_ward = False
    run_phase(MAZE_BETA_BLANK, agent_bpre, TRAIN_EPISODES,
              train=False, label="MAZE-BETA  Pre-explore (blank)",
              explore=True, viz=None)

    agent_bt = transfer(agent_bpre, eps=EPS_END, same_maze=True)
    agent_bt.goal_ward = False
    agent_bt.zero_shot = True
    test_b, _, _ = run_phase(
        MAZE_BETA, agent_bt, TEST_EPISODES,
        train=False, label="MAZE-BETA  Test (zero-shot)",
        viz=viz_test_b)

    # ── 4. Maze-gamma (push-pad hazards) ─────────────────────────────────────
    agent_gpre = transfer(agent_train, eps=EPS_START)
    agent_gpre.goal_ward = False
    run_phase(MAZE_GAMMA_BLANK, agent_gpre, PRETRAIN_EPISODES,
              train=False, label="MAZE-GAMMA  Pre-explore (blank)",
              explore=True, viz=None)

    agent_gt = transfer(agent_gpre, eps=EPS_END, same_maze=True)
    agent_gt.goal_ward = True
    test_g, _, _ = run_phase(
        MAZE_GAMMA, agent_gt, TEST_EPISODES,
        train=False, label="MAZE-GAMMA  Test (extra credit)",
        gamma_mode=True,
        viz=viz_test_g)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*52}")
    print(f"  FINAL SUMMARY  ({elapsed:.1f}s)")
    print(f"  {'Phase':<24} {'SR':>6}  {'AvgTurns':>9}  {'DeathRate':>10}")
    print(f"  {'-'*50}")
    for lbl, st in [
        ("Alpha Train",      train_stats),
        ("Alpha Test",       test_a),
        ("Beta  Test",       test_b),
        ("Gamma (X-credit)", test_g),
    ]:
        succ = [s for s in st if s["success"]]
        SR   = len(succ) / len(st) if st else 0.0
        AT   = sum(s["turns"] for s in succ) / len(succ) if succ else float("inf")
        DR   = sum(s["deaths"] for s in st) / max(1, sum(s["turns"] for s in st))
        at_s = f"{AT:.0f}" if AT < float("inf") else "N/A"
        print(f"  {lbl:<24} {SR:>6.1%}  {at_s:>9}  {DR:>10.4f}")
    print(f"{'='*52}")
    print("  All outputs -> Results/\n")


if __name__ == "__main__":
    main()