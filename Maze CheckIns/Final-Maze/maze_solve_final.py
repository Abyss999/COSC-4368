"""
maze_solve_final.py  ·  COSC 4368 AI  ·  Group 5
Method: Tabular Q-Learning + BFS-guided exploration

Flow:
  1. Train on maze-alpha
  2. Test  on maze-alpha  (no retraining)
  3. Test  on maze-beta   (zero-shot)
  4. Test  on maze-gamma  (extra credit -- push-pad hazards)
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

# ════════════════════════════════════════════════════════
#  SETTINGS  <- change freely
# ════════════════════════════════════════════════════════
DEBUG_MOVE_VALIDATOR = False   # set True to assert every move is legal under env adjacency
PRETRAIN_EPISODES = 40   # blank maze (MAZE_0) warm-up
TRAIN_EPISODES = 200
TEST_EPISODES  = 5
MAX_TURNS      = 4000

LR        = 0.1    # Q-learning rate
GAMMA_RL  = 0.95   # discount factor
EPS_START = 1.0    # initial exploration rate
EPS_END   = 0.10   # minimum exploration rate
EPS_DECAY = 0.985  # epsilon multiplied by this each episode

R_GOAL     =  200.0
R_DEATH    =  -50.0
R_NEW_CELL =   +0.5
R_STEP     =   -1.0
R_WALL     =   -2.0
R_CONFUSED =   -5.0

# ════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════
GRID = 64           # cells per side
FREE, WALL = 0, 1
BRIGHT = 128        # kept for reference; wall detection now uses full RGB
_WALL_THRESH = 50   # pixel is a wall iff r<50 AND g<50 AND b<50

UP, RIGHT, DOWN, LEFT, WAIT = 0, 1, 2, 3, 4
N_ACTIONS = 5
DELTAS  = {UP:(0,-1), RIGHT:(1,0), DOWN:(0,1), LEFT:(-1,0), WAIT:(0,0)}
REVERSE = {UP:DOWN, DOWN:UP, LEFT:RIGHT, RIGHT:LEFT, WAIT:WAIT}
_DELTA_TO_DIR = {(0,-1):UP, (1,0):RIGHT, (0,1):DOWN, (-1,0):LEFT}  # constant lookup

_HERE           = Path(__file__).resolve().parent
MAZE_ALPHA_BLANK = _HERE / "Contexts/maze-alpha/MAZE_0.png"
MAZE_ALPHA      = _HERE / "Contexts/maze-alpha/MAZE_1.png"
MAZE_BETA_BLANK  = _HERE / "Contexts/maze-beta/MAZE_0.png"
MAZE_BETA       = _HERE / "Contexts/maze-beta/MAZE_1.png"
MAZE_GAMMA_BLANK = _HERE / "Contexts/maze-gamma/MAZE_0.png"
MAZE_GAMMA      = _HERE / "Contexts/maze-gamma/MAZE_1.png"
RESULTS         = _HERE / "Results"


# ════════════════════════════════════════════════════════
#  TURN RESULT
# ════════════════════════════════════════════════════════
@dataclass
class TurnResult:
    wall_hits       : int   = 0
    current_position: tuple = (0, 0)
    is_dead         : bool  = False
    is_confused     : bool  = False
    is_goal_reached : bool  = False
    teleported      : bool  = False
    actions_executed: int   = 1
    pushed          : bool  = False   # gamma only


# ════════════════════════════════════════════════════════
#  IMAGE / GRID HELPERS
# ════════════════════════════════════════════════════════
def _load_image(path: Path):
    img = Image.open(path).convert("RGB")
    px  = np.array(img)
    h, w, _ = px.shape
    return px, h, w

def _pixel_grid(px: np.ndarray, h: int, w: int) -> np.ndarray:
    # Wall = dark pixel (r<50 AND g<50 AND b<50); avoids misclassifying colored hazards as free
    g = np.ones((h, w), dtype=np.int8)
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
        if px_arr[0, x, 0] > BRIGHT:
            sp = (x, 0)
        if px_arr[h - 1, x, 0] > BRIGHT:
            gp = (x, h - 1)
    return sp, gp

def _clusters(px_arr, h, w, color_fn, min_sz=20):
    r = px_arr[:, :, 0].astype(np.int32)
    g = px_arr[:, :, 1].astype(np.int32)
    b = px_arr[:, :, 2].astype(np.int32)
    mask = color_fn(r, g, b).astype(bool)
    labeled, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        pts = np.argwhere(labeled == i)
        if len(pts) >= min_sz:
            cy2, cx2 = pts.mean(0).astype(int)
            out.append((int(cx2), int(cy2)))
    return out

def _detect_hazards(px_arr, h, w) -> dict:
    # Fire pits: orange-red
    fire = _clusters(px_arr, h, w,
        lambda r, g, b: (r > 180) & (g > 70) & (g < 175) & (b < 100))

    # Yellow-hue clusters: solid pads (~114px, no embedded face features) are
    # teleport pads; smaller clusters (~48-51px) with embedded dark face features
    # (X-eyes / zigzag mouth) are confusion-trap emojis. Size cleanly separates them.
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

    # Teleport pads -- pair colour families by sorted order.
    green  = sorted(_clusters(px_arr, h, w,
        lambda r, g, b: (g > 170) & (r < 140) & (b < 160), min_sz=30))
    purple = sorted(_clusters(px_arr, h, w,
        lambda r, g, b: (b > 140) & (r > 80) & (r < 230) & (b > r) & (b > g + 40), min_sz=30))
    red    = sorted(_clusters(px_arr, h, w,
        lambda r, g, b: (r > 190) & (g < 80) & (b < 80), min_sz=80))

    def _is_square_pad(px_centroid):
        """Square-star exit pads have a white star icon in the center; spheres do not."""
        px_x, px_y = px_centroid
        csize = _cell_size(w, h)
        cell_x, cell_y = px_x // csize, px_y // csize
        patch = px_arr[cell_y*csize:(cell_y+1)*csize, cell_x*csize:(cell_x+1)*csize]
        # Star icon pixels are bright-white in the center; spheres have no white core.
        # Use a tight center crop to avoid counting corridor/background white at edges.
        mid = csize // 2
        center = patch[mid-3:mid+3, mid-3:mid+3]
        white = int(((center[:,:,0] > 220) & (center[:,:,1] > 220) & (center[:,:,2] > 220)).sum())
        return white >= 4

    def _add_pair(a, b):
        # If one pad is a square-star exit, only map sphere->square (one-way).
        # Square exit pads should not re-teleport the agent back on arrival.
        # If both are spheres, map bidirectionally.
        a_sq, b_sq = _is_square_pad(a), _is_square_pad(b)
        if a_sq and not b_sq:
            tp[b] = a          # sphere b -> square a
        elif b_sq and not a_sq:
            tp[a] = b          # sphere a -> square b
        elif not a_sq and not b_sq:
            tp[a] = b          # both spheres: bidirectional
            tp[b] = a
        # both squares: degenerate -- skip (no valid teleport)

    tp = {}
    if len(green)       >= 2: _add_pair(green[0],       green[1])
    if len(purple)      >= 2: _add_pair(purple[0],      purple[1])
    if len(yellow_pads) >= 2: _add_pair(yellow_pads[0], yellow_pads[1])
    if len(red)         >= 2: _add_pair(red[0],         red[1])

    # Gamma push pads: teal/blue arrows
    all_blue = _clusters(px_arr, h, w,
        lambda r, g, b: (b > 150) & (r < 120) & (g > 120) & (g < 220), min_sz=25)
    mid_y    = h // 2
    push_up  = [(x, y) for x, y in all_blue if y < mid_y]
    push_lft = [(x, y) for x, y in all_blue if y >= mid_y]

    return dict(fire=fire, confusion=conf, teleport=tp,
                push_up=push_up, push_left=push_lft)

def _rotate_90cw(x, y, px, py):
    return px + (y - py), py - (x - px)

def _build_adj(px, w, h, cs) -> dict:
    """
    Tile-based adjacency using wall detection at cell boundaries.
    Each cell is 16px wide; walls are 2px lines at the leading edge of each
    cell block. Two cells are connected iff the first 2 pixels of the neighbor's
    leading edge (sampled at the midpoint row/col) are both bright (not black).
    """
    half = cs // 2

    def _is_wall_px(r, c) -> bool:
        # Dark pixel = wall (all channels below threshold)
        p = px[r, c]
        return int(p[0]) < _WALL_THRESH and int(p[1]) < _WALL_THRESH and int(p[2]) < _WALL_THRESH

    def wall_between(cx, cy, nx, ny) -> bool:
        if nx == cx + 1:   # moving RIGHT: check 2px at left edge of nx
            by = min(cy * cs + half, h - 1)
            return (_is_wall_px(by, min(nx*cs,   w-1)) and
                    _is_wall_px(by, min(nx*cs+1, w-1)))
        elif nx == cx - 1: # moving LEFT: check 2px at left edge of cx
            by = min(cy * cs + half, h - 1)
            return (_is_wall_px(by, min(cx*cs,   w-1)) and
                    _is_wall_px(by, min(cx*cs+1, w-1)))
        elif ny == cy + 1: # moving DOWN: check 2px at top edge of ny
            bx = min(cx * cs + half, w - 1)
            return (_is_wall_px(min(ny*cs,   h-1), bx) and
                    _is_wall_px(min(ny*cs+1, h-1), bx))
        else:              # moving UP: check 2px at top edge of cy
            bx = min(cx * cs + half, w - 1)
            return (_is_wall_px(min(cy*cs,   h-1), bx) and
                    _is_wall_px(min(cy*cs+1, h-1), bx))

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

_ADJ_VER = b"v5"   # bumped: wall detection now uses full RGB instead of red-channel only

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
    Simulates the maze from a PNG.
    The agent ONLY receives TurnResult feedback -- never inspects internals.
    """

    def __init__(self, path: Path, gamma_mode: bool = False, flip_start_goal: bool = False):
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
        s_cell = px_to_cell(*sp, cs)
        g_cell = px_to_cell(*gp, cs)
        if flip_start_goal:
            self._start, self._goal = g_cell, s_cell
        else:
            self._start, self._goal = s_cell, g_cell

        hz = _detect_hazards(px, h, w)
        self._conf_cells  = frozenset(px_to_cell(x, y, cs) for x, y in hz["confusion"])
        self._tele: dict  = {
            px_to_cell(sx, sy, cs): px_to_cell(dx2, dy2, cs)
            for (sx, sy), (dx2, dy2) in hz["teleport"].items()
        }
        self._push_up  = frozenset(px_to_cell(x, y, cs) for x, y in hz["push_up"])
        self._push_lft = frozenset(px_to_cell(x, y, cs) for x, y in hz["push_left"])
        # Strip teleport pads (both entry and exit), and confusion cells from fire.
        # Yellow/orange pad pixels share hue with fire; destinations must also be
        # excluded so an agent arriving via teleport doesn't instantly die.
        _non_fire = self._conf_cells | set(self._tele.keys()) | set(self._tele.values())
        self._base_fire = [
            (x, y) for (x, y) in hz["fire"]
            if px_to_cell(x, y, cs) not in _non_fire
        ]

        self._free_count = max(1, sum(1 for nb in self._adj.values() if nb))

        self._pos    = self._start
        self._conf   = 0
        self._turns  = 0
        self._cur_rotation = 0

        # Pre-compute all 4 rotation sets once at init (avoids recomputing each step)
        _non_fire = self._conf_cells | set(self._tele.keys()) | set(self._tele.values())
        self._rotation_sets: dict = {}
        for slot in range(4):
            self._rotation_sets[slot] = self._compute_rotation_set(slot, _non_fire)
        self._fire: frozenset = self._rotation_sets[0]

        print(f"  {path.name:<20} start={self._start} goal={self._goal} "
              f"cs={cs}px | fire={len(self._base_fire)} "
              f"conf={len(self._conf_cells)} tele={len(self._tele)} "
              f"push_up={len(self._push_up)} push_left={len(self._push_lft)}")

    @property
    def start(self): return self._start
    @property
    def goal(self):  return self._goal
    @property
    def cs(self):    return self._cs
    @property
    def free_count(self): return self._free_count

    def reset(self, ep: int = 0) -> tuple:
        self._pos          = self._start
        self._conf         = 0
        self._turns        = 0
        self._cur_rotation = 0
        self._fire         = self._rotation_sets[0]
        return self._pos

    def _rotate_deathpits(self):
        """Advance fire rotation by one slot; incrementally update only changed cells."""
        next_rot           = (self._cur_rotation + 1) % 4
        self._fire         = self._rotation_sets[next_rot]
        self._cur_rotation = next_rot

    def step(self, action: int) -> TurnResult:
        # Rotate fire every 5 turns (spec requirement)
        self._turns += 1
        if self._turns % 5 == 0:
            self._rotate_deathpits()

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

        # Gamma push pads
        pushed = False
        if self._gm:
            if self._pos in self._push_up:
                nb2 = self._adj.get(self._pos, {})
                if UP in nb2: self._pos = nb2[UP]
                pushed = True
            elif self._pos in self._push_lft:
                nb2 = self._adj.get(self._pos, {})
                if LEFT in nb2: self._pos = nb2[LEFT]
                pushed = True

        # Fire pit -> instant death, respawn at start
        if self._pos in self._fire:
            self._pos  = self._start
            self._conf = 0
            return TurnResult(wall_hits, self._pos, True, False, False, False, 1, pushed)

        # Teleport
        tele = False
        if self._pos in self._tele:
            dest = self._tele[self._pos]
            if dest != self._pos:   # guard against degenerate self-loop
                self._pos = dest
                tele = True

        # Confusion trap
        if self._pos in self._conf_cells:
            self._conf = 2

        goal = self._pos == self._goal
        return TurnResult(wall_hits, self._pos, False, self._conf > 0,
                          goal, tele, 1, pushed)

    def pixels_copy(self): return self._px.copy()

    def _compute_rotation_set(self, slot: int, non_fire: frozenset) -> frozenset:
        """Compute the set of active fire cells for a given rotation slot."""
        if not self._base_fire:
            return frozenset()
        ppx, ppy = max(self._base_fire, key=lambda p: p[1])
        active   = []
        for x, y in self._base_fire:
            rx, ry = float(x), float(y)
            for _ in range(slot % 4):
                rx, ry = _rotate_90cw(rx, ry, ppx, ppy)
            xi = min(max(0, int(round(rx))), self._w - 1)
            yi = min(max(0, int(round(ry))), self._h - 1)
            active.append(px_to_cell(xi, yi, self._cs))
        return frozenset(c for c in active if c not in non_fire)


# ════════════════════════════════════════════════════════
#  BFS ON KNOWN MAP
# ════════════════════════════════════════════════════════
def bfs(known: dict, start: tuple, goal: tuple,
        danger: set, blocked: set, tele: dict,
        passable: set | None = None) -> list:
    """
    BFS strictly through the agent's discovered cells.
    Skips danger cells and blocked directed edges. Follows teleport edges.
    If passable is provided, only traverses confirmed-passable edges.
    Returns path list (start...goal), or [] if unreachable.
    """
    # start not in known means it has never been visited -- no edges to expand from.
    # Allow start even if absent from known (agent is already there; also needed in
    # passable-mode where gap cells aren't in known). Also allow start even if it is
    # in danger (agent is standing on a rotating-fire cell -- plan FROM there, just
    # don't route THROUGH other dangerous cells).
    if start not in known and passable is None:
        return []
    # Local bindings avoid repeated global lookups in the hot loop.
    _known = known; _danger = danger; _blocked = blocked; _passable = passable
    queue = deque([start])
    # Exclude start from danger check: agent is already standing there.
    came: dict = {start: None}
    q_append = queue.append
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
            if nb in came or nb in _danger:
                continue
            if (cur, nb) in _blocked:
                continue
            if _passable is not None:
                # Passable-gated: allow traversal through confirmed edges even if
                # nb is not yet in known (gap cells in the map).
                if (cur, nb) not in _passable:
                    continue
            elif nb not in _known:
                # No passable gate: restrict to known cells only.
                continue
            came[nb] = cur
            q_append(nb)
        if cur in tele:
            dest = tele[cur]
            if dest not in came and dest not in _danger:
                came[dest] = cur
                q_append(dest)
    return []


# ════════════════════════════════════════════════════════
#  Q-LEARNING AGENT
# ════════════════════════════════════════════════════════
class Agent:
    """
    Tabular Q-Learning agent.
    State: (x, y, confused: 0|1) -- 5 actions per state.
    Builds its own internal map purely from TurnResult feedback.
    Never inspects MazeEnv internals.
    """

    def __init__(self):
        self.epsilon    = EPS_START
        self.goal_ward  = False   # set True for zero-shot phases to bias frontier toward goal
        self.zero_shot  = False   # set True for beta/gamma test: enables danger-ignoring BFS fallback

        # Persistent across all episodes
        self.q          = np.zeros((GRID, GRID, 2, N_ACTIONS), dtype=np.float32)
        self.known      : dict  = {}   # (x,y) -> "free"|"death"|"teleport"|"confusion"|"goal"
        self.blocked    : set   = set()   # (from_cell, to_cell) confirmed impassable
        self.passable   : set   = set()   # (from_cell, to_cell) confirmed traversable
        self.danger     : set   = set()   # cells lethal at current rotation slot
        self.rot_danger : dict  = {0:set(), 1:set(), 2:set(), 3:set()}
        self.conf_cells : set   = set()   # confusion trap cells (cached; avoids dict scan each turn)
        self.tele       : dict  = {}   # pad -> dest (inferred)
        self.goal       : Optional[tuple] = None
        self.start      : Optional[tuple] = None
        self.best_path  : list = []
        self.best_turns : int  = 999_999

        # Per-episode (reset each episode)
        self._pos      : tuple = (0, 0)
        self._confused : bool  = False
        self._ep_path  : list  = []
        self._ep_cells : set   = set()
        self._ep_deaths: int   = 0
        self._ep_turns : int   = 0
        self._ep_tc    : int   = 0    # turn count within episode (drives fire rotation)
        self._rot_slot_cache: int = -1  # last computed rotation slot; avoids redundant set copy

        # BFS plan cache -- invalidated on death/teleport/push or when path is consumed
        self._plan      : list  = []   # cached step sequence toward goal/frontier
        self._wait_count: int   = 0    # consecutive WAITs issued; breaks infinite wait loops

    def reset_episode(self, start: tuple):
        self._pos       = start
        self._confused  = False
        self._ep_path   = [start]
        self._ep_cells  = {start}
        self._ep_deaths = 0
        self._ep_turns  = 0
        self._ep_tc     = 0
        self._plan      = []
        self._wait_count = 0
        self._rot_slot_cache = -1
        if self.start is None:
            self.start = start
        self.known.setdefault(start, "free")
        # Seed goal into known so BFS can aim toward it even before it's explored.
        # The goal cell coordinate is consistent across all three mazes.
        if self.goal is not None:
            self.known.setdefault(self.goal, "goal")
        self.danger = set(self.rot_danger[0])

    def _rot_slot(self) -> int:
        return (self._ep_tc // 5) % 4

    def _sync_danger(self):
        slot = (self._ep_tc // 5) % 4
        if slot != self._rot_slot_cache:
            self.danger = set(self.rot_danger[slot])
            self._rot_slot_cache = slot

    # ── Action selection ──────────────────────────────────────────────────────
    def choose(self, pos: tuple) -> int:
        self._sync_danger()

        if random.random() < self.epsilon:
            fa = self._frontier_actions(pos)
            return random.choice(fa) if fa else random.randint(0, N_ACTIONS - 1)

        # Follow cached plan; invalidate if next step is now dangerous.
        if self._plan:
            nxt = self._plan[0]
            if nxt not in self.danger:
                self._plan.pop(0)
                dir_action = self._dir(pos, nxt)
                q_vals = self.q[pos[0], pos[1], int(self._confused)]
                best_q = int(np.argmax(q_vals))
                return dir_action if random.random() < 0.8 else best_q
            self._plan = []   # fire rotated into cached path -- replan

        # Phase 2: goal known -> BFS to goal.
        if self.goal is not None and self.goal in self.known:
            conf_cells = self.conf_cells   # pre-maintained set; no dict scan needed
            known = self.known
            blocked = self.blocked
            tele = self.tele
            passable = self.passable

            # Build safe sub-map that always includes pos so BFS can start even
            # when agent is standing on a rotating-fire cell.
            def _safe(avoid_set):
                d = {k: v for k, v in known.items() if k not in avoid_set}
                if pos in known:
                    d[pos] = known[pos]
                return d

            always_on = (self.rot_danger[0] & self.rot_danger[1] &
                         self.rot_danger[2] & self.rot_danger[3])
            if self.goal_ward:
                avoid        = always_on | conf_cells
                safe_avoid   = _safe(avoid)
                path = (bfs(safe_avoid, pos, self.goal, avoid, blocked, tele) or
                        bfs(known, pos, self.goal, conf_cells, blocked, tele) or
                        bfs(known, pos, self.goal, conf_cells, set(), tele) or
                        bfs(known, pos, self.goal, set(), set(), tele))
            else:
                avoid        = self.danger | conf_cells
                avoid_ao     = always_on  | conf_cells
                safe_avoid   = _safe(avoid)
                safe_avoid_ao = _safe(avoid_ao)
                path = (bfs(safe_avoid, pos, self.goal, avoid, blocked, tele, passable) or
                        (bfs(safe_avoid, pos, self.goal, avoid, blocked, tele) if passable else []) or
                        bfs(safe_avoid_ao, pos, self.goal, avoid_ao, blocked, tele, passable) or
                        (bfs(safe_avoid_ao, pos, self.goal, avoid_ao, blocked, tele) if passable else []) or
                        bfs(known, pos, self.goal, conf_cells, blocked, tele) or
                        bfs(known, pos, self.goal, conf_cells, set(), tele) or
                        bfs(known, pos, self.goal, set(), set(), tele))

            if len(path) > 1:
                nxt = path[1]
                if nxt in self.danger:
                    if nxt in always_on or self._wait_count >= 20:
                        # Permanently blocked or waited too long — replan around it
                        self._plan = []
                        self._wait_count = 0
                    else:
                        # Rotating fire — wait for it to move away
                        self._wait_count += 1
                        return WAIT
                else:
                    self._wait_count = 0
                    self._plan = path[2:]
                    return self._dir(pos, nxt)

        # Phase 1: goal unknown -> BFS to nearest frontier.
        fc, unk = self._nearest_frontier(pos)
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

    def _nearest_frontier(self, pos: tuple):
        # Frontier BFS uses only blocked-gating (not passable).
        # passable is for goal-routing (exploit known safe paths).
        # Exploration must be able to reach new frontiers even through unconfirmed
        # known cells -- otherwise the agent gets trapped in its passable island.
        #
        # Goal-ward bias: collect ALL reachable frontier edges, then pick the one
        # whose unexplored neighbor is closest to the goal (Manhattan distance).
        # This pulls exploration toward the goal during zero-shot transfer phases
        # where the agent must discover a path in very few episodes.
        queue    = deque([pos])
        visited  = {pos}
        frontier = []   # (manhattan_to_goal, hop_count, cur, nb)
        hop      = {pos: 0}
        while queue:
            cur    = queue.popleft()
            cx, cy = cur
            for dx, dy in ((0,-1),(1,0),(0,1),(-1,0)):
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
                        # No goal-ward bias: first frontier found (BFS order = closest)
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

    # ── Q-table update ─────────────────────────────────────────────────────────
    def update_q(self, prev: tuple, prev_conf: bool,
                 action: int, reward: float, tr: TurnResult):
        s  = self.q[prev[0], prev[1], int(prev_conf)]
        ns = self.q[tr.current_position[0], tr.current_position[1], int(tr.is_confused)]
        td = reward + (0.0 if tr.is_goal_reached else GAMMA_RL * float(ns.max()))
        s[action] += LR * (td - float(s[action]))

    # ── Observe TurnResult and update internal map ─────────────────────────────
    def observe(self, prev: tuple, action: int, tr: TurnResult, prev_confused: bool = False):
        """Update internal map from TurnResult only. Never reads MazeEnv directly."""
        self._ep_turns += 1
        self._ep_tc    += 1
        self._sync_danger()
        rot = self._rot_slot()

        # Invalidate plan on any event that can desync expected position or path safety.
        if tr.is_dead or tr.teleported or tr.pushed or tr.wall_hits or tr.is_confused:
            self._plan = []
        # Reset wait counter only when agent actually changed position (not on WAIT action).
        if not tr.is_dead and tr.wall_hits == 0 and tr.current_position != prev:
            self._wait_count = 0

        if tr.is_dead:
            # Fire-pit inference is only reliable when the intended action direction
            # actually equals the attempted move. Confusion reverses the action, and
            # gamma push pads displace the agent after the move -- in either case the
            # raw `action` no longer identifies the fatal cell. Skip inference in
            # those cases rather than polluting the danger map with bad entries.
            if tr.wall_hits == 0 and not prev_confused and not tr.pushed:
                dx, dy = DELTAS.get(action, (0, 0))
                pit    = (prev[0] + dx, prev[1] + dy)
                if 0 <= pit[0] < GRID and 0 <= pit[1] < GRID:
                    self.known[pit] = "death"
                    self.danger.add(pit)
                    self.rot_danger[rot].add(pit)
            self._ep_deaths += 1
            self._pos        = self.start or tr.current_position
            self._confused   = False
            return

        cur            = tr.current_position   # always use TurnResult.current_position
        self._pos      = cur
        self._confused = tr.is_confused
        self._ep_cells.add(cur)
        self._ep_path.append(cur)

        if tr.is_goal_reached:
            self.known[cur] = "goal"
            if self.goal is None:
                self.goal = cur
        elif tr.is_confused and not prev_confused:
            # Only label confusion on actual trap entry (transition not-confused -> confused);
            # merely passing through a cell while under ongoing confusion must not tag it.
            self.known[cur] = "confusion"
            self.conf_cells.add(cur)
        else:
            if self.known.get(cur) == "death":
                self.known[cur] = "free"   # fire rotated away -- cell safe now
            elif self.known.get(cur) not in ("confusion", "teleport"):
                self.known.setdefault(cur, "free")

            if tr.wall_hits == 0 and not prev_confused:
                dx, dy = DELTAS.get(action, (0, 0))
                stepped = (prev[0] + dx, prev[1] + dy)   # cell agent physically entered

                if tr.pushed or tr.teleported:
                    # Unified: whatever combination of push/teleport occurred,
                    # the net effect for BFS is stepped -> cur. Record directly.
                    self.known.setdefault(cur, "teleport")
                    if 0 <= stepped[0] < GRID and 0 <= stepped[1] < GRID:
                        self.known.setdefault(stepped, "teleport")
                        self.tele[stepped] = cur

        # Record confirmed traversable edge. An actually-walked cardinal step proves
        # the edge is wall-free in BOTH directions (walls are symmetric), so record
        # both (prev,cur) and (cur,prev). Previously one-way recording blocked BFS
        # from planning reverse traversal and produced long detours.
        if tr.wall_hits == 0 and not tr.is_dead and not tr.pushed:
            dx2 = cur[0] - prev[0]
            dy2 = cur[1] - prev[1]
            if abs(dx2) + abs(dy2) == 1:
                self.passable.add((prev, cur))
                self.passable.add((cur, prev))

        # Record impassable edge from wall hit (bidirectional -- walls block both ways).
        # Skip when confused: effective direction is reversed, so `action` doesn't
        # identify the blocked edge.
        if tr.wall_hits > 0 and not prev_confused:
            dx, dy = DELTAS.get(action, (0, 0))
            target = (prev[0] + dx, prev[1] + dy)
            if 0 <= target[0] < GRID and 0 <= target[1] < GRID:
                self.blocked.add((prev, target))
                self.blocked.add((target, prev))

    def record_success(self, turns: int):
        if turns < self.best_turns:
            self.best_turns = turns
            # Try passable-gated BFS first (guaranteed wall-safe shortest path).
            # Fall back to blocked-only BFS (uses all known cells, still respects walls).
            # Only use the raw walked path if both BFS variants fail.
            clean = bfs(self.known, self.start, self.goal,
                        set(), self.blocked, self.tele, self.passable)
            if len(clean) > 1:
                self.best_path = clean
            else:
                clean2 = bfs(self.known, self.start, self.goal,
                             set(), self.blocked, self.tele)
                self.best_path = clean2 if len(clean2) > 1 else list(self._ep_path)

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

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
#  TRANSFER AGENT (alpha -> beta/gamma)
# ════════════════════════════════════════════════════════
def transfer(src: Agent, eps: float, same_maze: bool = False) -> Agent:
    """
    Clone trained agent for use in another run.

    same_maze=True  (alpha pre-train -> alpha train, or alpha train -> alpha test):
        Carry over Q-table, known map, passable edges, goal/start.
        The wall layout is identical so previously confirmed edges remain valid.

    same_maze=False (alpha -> beta/gamma):
        Carry over Q-table and goal/start (coordinates are the same across all
        three mazes), but clear known, passable, blocked, tele.
        Beta/gamma have different wall layouts from alpha -- reusing alpha's
        passable edges causes BFS to route through walls that exist in
        beta/gamma but not in alpha.
    """
    dst = Agent()
    dst.q          = src.q.copy()
    dst.epsilon    = eps

    if same_maze:
        # Remap transient cell labels: "death" may have rotated away (safe now),
        # "confusion" is a permanent trap so keep it to continue avoiding it.
        dst.known      = {k: ("free" if v == "death" else v)
                          for k, v in src.known.items()}
        dst.goal       = src.goal
        dst.start      = src.start
        dst.tele       = dict(src.tele)
        dst.passable   = set(src.passable)
        dst.conf_cells = set(src.conf_cells)
        dst.blocked    = set()          # reset wall hits; passable is canonical
        dst.best_path  = list(src.best_path)
        dst.best_turns = src.best_turns
        dst._plan       = []            # never carry a stale plan across episodes
        dst._wait_count = 0
    else:
        # Carry goal/start so BFS knows where to aim even before re-discovering goal.
        # All three mazes share the same start/goal cell coordinates.
        dst.goal  = src.goal
        dst.start = src.start
        # known, passable, blocked, tele, conf_cells stay empty (Agent() defaults)

    dst.danger     = set()
    dst.rot_danger = {0:set(), 1:set(), 2:set(), 3:set()}
    return dst


# ════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ════════════════════════════════════════════════════════
def run_episodes(env: MazeEnv, agent: Agent, n: int,
                 train: bool, label: str, explore: bool = False) -> list:
    all_stats = []
    for ep in range(n):
        pos = env.reset(ep)
        agent.reset_episode(pos)

        for turn in range(MAX_TURNS):
            prev_conf = agent._confused
            action    = agent.choose(pos)

            if DEBUG_MOVE_VALIDATOR and action != WAIT:
                eff = REVERSE[action] if prev_conf else action
                adj_nbrs = env._adj.get(pos, {})
                expected = adj_nbrs.get(eff)
                # Pre-validate: if eff not in adj_nbrs the env will bounce (wall_hits=1)

            tr        = env.step(action)

            if DEBUG_MOVE_VALIDATOR and action != WAIT and not tr.is_dead and not tr.pushed:
                eff = REVERSE[action] if prev_conf else action
                adj_nbrs = env._adj.get(pos, {})
                expected = adj_nbrs.get(eff, pos)
                if not tr.teleported and tr.wall_hits == 0:
                    assert tr.current_position == expected, (
                        f"ILLEGAL MOVE ep={ep} turn={turn}: from={pos} action={action} "
                        f"eff={eff} expected={expected} got={tr.current_position}"
                    )

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
                if tr.current_position not in agent.known:
                    reward += R_NEW_CELL

            if train:
                agent.update_q(pos, prev_conf, action, reward, tr)

            agent.observe(pos, action, tr, prev_confused=prev_conf)
            pos = agent._pos   # use agent's pos (handles death respawn correctly)

            if tr.is_goal_reached:
                agent.record_success(turn + 1)
                break

        all_stats.append(agent.ep_stats)
        if train:
            agent.decay_epsilon()
        elif not explore and agent.goal in agent._ep_cells:
            # Goal was found this episode -- ratchet epsilon down so future
            # episodes exploit the discovered path.  Skipped in explore mode
            # (pre-exploration passes) so epsilon stays high for full coverage.
            agent.epsilon = max(EPS_END, agent.epsilon * 0.5)

        if not train and n <= 10:
            # Per-episode detail for short test runs
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
        print(f"  [Bonus] Learning Efficiency    : did not converge to 80% during training")
    print(bar)
    return dict(SR=SR, APL=APL, AT=AT, DR=DR, EE=EE, MC=MC, LE=LE)


# ════════════════════════════════════════════════════════
#  VISUALISATION
# ════════════════════════════════════════════════════════
def save_path_img(env: MazeEnv, path_cells: list, out: Path):
    if not path_cells:
        print(f"    (no path for {out.name})"); return
    img    = env.pixels_copy()
    cs, t  = env.cs, 2
    h, w   = img.shape[:2]
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
        draw.line([*pt(i-1, sr_v[i-1]), *pt(i, sr_v[i])], fill=(30, 100, 200), width=2)
        draw.line([*pt(i-1, dr_v[i-1]), *pt(i, dr_v[i])], fill=(200, 60, 0),  width=2)

    draw.text((M, M-20),
              "Learning Curve -- maze-alpha training (10-ep window)", fill=(0,0,0))
    draw.text((W//2-40, H-M+12), f"Episode (1-{n})", fill=(0, 0, 0))
    lx = W - M - 170
    draw.rectangle([lx, M+10, lx+14, M+20], fill=(30,100,200))
    draw.text((lx+18, M+10), "Success Rate",     fill=(0,0,0))
    draw.rectangle([lx, M+32, lx+14, M+42], fill=(200,60,0))
    draw.text((lx+18, M+32), "Death Rate (x20)", fill=(0,0,0))
    img.save(out)
    print(f"    Saved -> {out.name}")


# ════════════════════════════════════════════════════════
#  PHASE RUNNER
# ════════════════════════════════════════════════════════
def run_phase(path: Path, agent: Agent, n: int, train: bool,
              label: str, gamma_mode: bool = False,
              train_stats: list | None = None,
              explore: bool = False,
              flip_start_goal: bool = False) -> tuple:
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    env   = MazeEnv(path, gamma_mode=gamma_mode, flip_start_goal=flip_start_goal)
    stats = run_episodes(env, agent, n, train, label, explore=explore)
    m     = report_metrics(stats, agent, env, label, train_stats)

    stem = path.stem
    ctx  = path.parent.name
    tag  = "train" if train else "test"

    if agent.best_path:
        save_path_img(env, agent.best_path,
                      RESULTS / f"{stem}_{ctx}_rl_{tag}_solved.png")
    save_map_img(env, agent, RESULTS / f"{stem}_{ctx}_rl_{tag}_map.png")
    return stats, m, env


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════
RANDOM_SEED = 42

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "="*52)
    print("  COSC 4368 -- Maze Solver  |  Group 5")
    print("  Q-Learning + BFS exploration")
    print("="*52)
    print(f"  PRETRAIN={PRETRAIN_EPISODES}  TRAIN={TRAIN_EPISODES}  TEST={TEST_EPISODES}  MAX_TURNS={MAX_TURNS}")
    print(f"  LR={LR}  GAMMA={GAMMA_RL}  e {EPS_START}->{EPS_END} (x{EPS_DECAY}/ep)")

    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    # 0 -- Pre-train on blank maze-alpha (MAZE_0) to learn layout without hazards
    agent_pre = Agent()
    agent_pre.goal_ward = True   # bias frontier toward goal to find it faster
    _, _, _ = run_phase(
        MAZE_ALPHA_BLANK, agent_pre, PRETRAIN_EPISODES,
        train=True, label="MAZE-ALPHA  Pre-train (blank)", flip_start_goal=True)

    # 1 -- Train on maze-alpha (with hazards), warm-started from pre-training
    # Keep passable edges from pretrain: MAZE_0 and MAZE_1 share the same wall layout,
    # so pretrain passable edges are valid. Clearing them causes the agent to lose all
    # path knowledge and die repeatedly in fire before re-learning routes.
    agent_train = transfer(agent_pre, eps=EPS_START, same_maze=True)
    agent_train.goal_ward = True
    train_stats, _, _ = run_phase(
        MAZE_ALPHA, agent_train, TRAIN_EPISODES,
        train=True, label="MAZE-ALPHA  Training", flip_start_goal=True)

    save_curve(train_stats, RESULTS / "learning_curve_alpha.png")

    # 2 -- Test maze-alpha (Q-table + map transferred, no retraining)
    agent_at = transfer(agent_train, eps=EPS_END, same_maze=True)
    test_a, _, _ = run_phase(
        MAZE_ALPHA, agent_at, TEST_EPISODES,
        train=False, label="MAZE-ALPHA  Test",
        train_stats=train_stats, flip_start_goal=True)

    # 3 -- Test maze-beta (zero-shot)
    # Alpha was flipped (start=bottom, goal=top). Beta is normal (start=top, goal=bottom).
    # Swap goal/start so the agent aims at the correct beta goal (32,63) not alpha's goal (31,0).
    agent_bpre = transfer(agent_train, eps=0.5)
    agent_bpre.goal, agent_bpre.start = agent_bpre.start, agent_bpre.goal  # un-flip for beta
    agent_bpre.goal_ward = False
    run_phase(MAZE_BETA_BLANK, agent_bpre, TRAIN_EPISODES,
              train=False, label="MAZE-BETA  Pre-explore (blank)", explore=True)

    agent_bt = transfer(agent_bpre, eps=EPS_END, same_maze=True)
    agent_bt.goal_ward = False
    agent_bt.zero_shot = True
    test_b, _, _ = run_phase(
        MAZE_BETA, agent_bt, TEST_EPISODES,
        train=False, label="MAZE-BETA  Test (zero-shot)")

    # 4 -- Test maze-gamma (extra credit: push-pad hazards)
    # Gamma is normal orientation (start=top, goal=bottom), same as beta.
    # Transfer from alpha_train and un-flip goal/start to match gamma's orientation.
    agent_gpre = transfer(agent_train, eps=0.5)
    agent_gpre.goal, agent_gpre.start = agent_gpre.start, agent_gpre.goal  # un-flip for gamma
    agent_gpre.goal_ward = False
    run_phase(MAZE_GAMMA_BLANK, agent_gpre, TRAIN_EPISODES,
              train=False, label="MAZE-GAMMA  Pre-explore (blank)",
              gamma_mode=True, explore=True)

    agent_gt = transfer(agent_gpre, eps=EPS_END, same_maze=True)
    agent_gt.goal_ward = False
    agent_gt.zero_shot = True
    test_g, _, _ = run_phase(
        MAZE_GAMMA, agent_gt, TEST_EPISODES,
        train=False, label="MAZE-GAMMA  Test (zero-shot)",
        gamma_mode=True)

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*52}")
    print(f"  FINAL SUMMARY  ({elapsed:.1f}s)")
    print(f"  {'Phase':<24} {'SR':>6}  {'AvgTurns':>9}  {'DeathRate':>10}")
    print(f"  {'-'*50}")
    for lbl, st in [
        ("Alpha Train",      train_stats),
        ("Alpha Test",       test_a),
        ("Beta  Test",       test_b),
        ("Gamma Test",       test_g),
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
