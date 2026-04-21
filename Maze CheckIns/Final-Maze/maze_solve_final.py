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
PRETRAIN_EPISODES = 100   # blank maze (MAZE_0) warm-up
TRAIN_EPISODES = 200
TEST_EPISODES  = 5
MAX_TURNS      = 10_000

LR        = 0.1    # Q-learning rate
GAMMA_RL  = 0.95   # discount factor
EPS_START = 1.0    # initial exploration rate
EPS_END   = 0.05   # minimum exploration rate
EPS_DECAY = 0.95   # epsilon multiplied by this each episode

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

    # Confusion traps: bright yellow emoji
    conf = _clusters(px_arr, h, w,
        lambda r, g, b: (r > 210) & (g > 160) & (b < 70) & (np.abs(r - g) < 70),
        min_sz=50)

    # Teleport pads -- pair colour families by sorted order
    green  = sorted(_clusters(px_arr, h, w,
        lambda r, g, b: (g > 170) & (r < 140) & (b < 160), min_sz=30))
    purple = sorted(_clusters(px_arr, h, w,
        lambda r, g, b: (r > 100) & (r < 210) & (b > 130) & (g < 100), min_sz=30))
    yellow = sorted(_clusters(px_arr, h, w,
        lambda r, g, b: (r > 190) & (g > 140) & (b < 90) & (r > g + 25), min_sz=30))
    red    = sorted(_clusters(px_arr, h, w,
        lambda r, g, b: (r > 190) & (g < 80) & (b < 80), min_sz=80))
    tp = {}
    if len(green)  >= 2: tp[green[0]]  = green[1]
    if len(purple) >= 2: tp[purple[0]] = purple[1]
    if len(yellow) >= 2: tp[yellow[0]] = yellow[1]
    if len(red)    >= 2: tp[red[0]]    = red[1]

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
        self._base_fire   = hz["fire"]
        self._conf_cells  = frozenset(px_to_cell(x, y, cs) for x, y in hz["confusion"])
        self._tele: dict  = {
            px_to_cell(sx, sy, cs): px_to_cell(dx2, dy2, cs)
            for (sx, sy), (dx2, dy2) in hz["teleport"].items()
        }
        self._push_up  = frozenset(px_to_cell(x, y, cs) for x, y in hz["push_up"])
        self._push_lft = frozenset(px_to_cell(x, y, cs) for x, y in hz["push_left"])

        self._free_count = max(1, sum(1 for nb in self._adj.values() if nb))

        self._pos    = self._start
        self._conf   = 0
        self._turns  = 0          # global turn counter (drives fire rotation)
        self._fire: frozenset = frozenset()
        self._rotate_fire(0)

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
        self._pos   = self._start
        self._conf  = 0
        self._turns = 0
        self._rotate_fire(0)
        return self._pos

    def step(self, action: int) -> TurnResult:
        # Advance turn counter and rotate fire every 5 turns (spec requirement)
        self._turns += 1
        self._rotate_fire((self._turns // 5) % 4)

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
            self._pos = self._tele[self._pos]
            tele = True
            if self._pos in self._tele:      # chained teleport
                self._pos = self._tele[self._pos]

        # Confusion trap
        if self._pos in self._conf_cells:
            self._conf = 2

        goal = self._pos == self._goal
        return TurnResult(wall_hits, self._pos, False, self._conf > 0,
                          goal, tele, 1, pushed)

    def pixels_copy(self): return self._px.copy()

    def _rotate_fire(self, slot: int):
        if not self._base_fire:
            self._fire = frozenset()
            return
        ppx, ppy = max(self._base_fire, key=lambda p: p[1])
        active   = []
        for x, y in self._base_fire:
            rx, ry = float(x), float(y)
            for _ in range(slot % 4):
                rx, ry = _rotate_90cw(rx, ry, ppx, ppy)
            xi = min(max(0, int(round(rx))), self._w - 1)
            yi = min(max(0, int(round(ry))), self._h - 1)
            active.append(px_to_cell(xi, yi, self._cs))
        self._fire = frozenset(active)


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
    if start not in known:
        return []
    queue = deque([start])
    came: dict = {start: None}
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
            if nb in came or nb not in known or nb in danger:
                continue
            if (cur, nb) in blocked:
                continue
            if passable is not None and (cur, nb) not in passable:
                continue
            came[nb] = cur
            queue.append(nb)
        if cur in tele:
            dest = tele[cur]
            if dest not in came and dest in known and dest not in danger:
                came[dest] = cur
                queue.append(dest)
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

        # Persistent across all episodes
        self.q          = np.zeros((GRID, GRID, 2, N_ACTIONS), dtype=np.float32)
        self.known      : dict  = {}   # (x,y) -> "free"|"death"|"teleport"|"confusion"|"goal"
        self.blocked    : set   = set()   # (from_cell, to_cell) confirmed impassable
        self.passable   : set   = set()   # (from_cell, to_cell) confirmed traversable
        self.danger     : set   = set()   # cells lethal at current rotation slot
        self.rot_danger : dict  = {0:set(), 1:set(), 2:set(), 3:set()}
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

    def reset_episode(self, start: tuple):
        self._pos       = start
        self._confused  = False
        self._ep_path   = [start]
        self._ep_cells  = {start}
        self._ep_deaths = 0
        self._ep_turns  = 0
        self._ep_tc     = 0
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
        self.danger = set(self.rot_danger[self._rot_slot()])

    # ── Action selection ──────────────────────────────────────────────────────
    def choose(self, pos: tuple) -> int:
        self._sync_danger()

        if random.random() < self.epsilon:
            fa = self._frontier_actions(pos)
            return random.choice(fa) if fa else random.randint(0, N_ACTIONS - 1)

        # Phase 2: goal known -> BFS to goal through safe known cells.
        # Zero-shot mode: use blocked-only BFS (shortest path through full known map).
        #   Passable edges from a first-pass random walk are a long circuitous route;
        #   blocked-only BFS finds a shorter path through already-mapped cells.
        # Trained mode: use passable-gated BFS (confirmed wall-safe edges only) first,
        #   falling back to blocked-only if passable gaps break connectivity.
        if self.goal is not None and self.goal in self.known:
            if self.goal_ward:
                # Zero-shot: avoid only cells dangerous at ALL rotation slots (always-on fire).
                # Single-slot fire can be waited out; always-on fire is truly impassable.
                always_danger = (self.rot_danger[0] & self.rot_danger[1] &
                                 self.rot_danger[2] & self.rot_danger[3])
                safe = {k: v for k, v in self.known.items() if k not in always_danger}
                path = bfs(safe, pos, self.goal, always_danger, self.blocked, self.tele)
                if not path:
                    # No path even through always-danger-free map -- allow current-slot
                    # danger cells too and accept the death/retry cost.
                    path = bfs(self.known, pos, self.goal, set(), self.blocked, self.tele)
            else:
                safe = {k: v for k, v in self.known.items() if k not in self.danger}
                path = bfs(safe, pos, self.goal, self.danger, self.blocked, self.tele, self.passable)
                if not path and self.passable:
                    path = bfs(safe, pos, self.goal, self.danger, self.blocked, self.tele)
            if len(path) > 1:
                nxt = path[1]
                # Fire-dodge: if next step is in current danger, WAIT for rotation.
                # Fire rotates every 5 turns -- waiting costs 5 turns but avoids death.
                if self.goal_ward and nxt in self.danger:
                    return WAIT
                return self._dir(pos, nxt)

        # Phase 1: goal unknown -> BFS to nearest frontier
        # No passable gate: exploration must route through unconfirmed known cells.
        fc, unk = self._nearest_frontier(pos)
        if fc is not None:
            if fc == pos and unk is not None:
                return self._dir(pos, unk)
            path = bfs(self.known, pos, fc, self.danger, self.blocked, self.tele)
            if len(path) > 1:
                return self._dir(pos, path[1])

        fa = self._frontier_actions(pos)
        return random.choice(fa) if fa else random.randint(0, N_ACTIONS - 1)

    def _dir(self, a: tuple, b: tuple) -> int:
        dx, dy = b[0] - a[0], b[1] - a[1]
        return {(0,-1):UP, (1,0):RIGHT, (0,1):DOWN, (-1,0):LEFT}.get(
            (dx, dy), random.randint(0, 3))

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
                    # Goal-ward bias: pull exploration toward goal in zero-shot phases
                    # where the agent has no pre-built passable map. Disabled during
                    # trained phases (alpha) to preserve broad map coverage.
                    if self.goal_ward and self.goal is not None:
                        md = abs(nb[0] - self.goal[0]) + abs(nb[1] - self.goal[1])
                    else:
                        md = 0
                    frontier.append((md, hop[cur] + 1, cur, nb))
                elif nb not in self.danger and self.known[nb] != "wall":
                    hop[nb] = hop[cur] + 1
                    queue.append(nb)
            if cur in self.tele:
                dest = self.tele[cur]
                if dest not in visited and dest in self.known and dest not in self.danger:
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
        elif tr.teleported:
            self.known.setdefault(cur, "teleport")
            # Infer the pad cell stepped onto before being teleported
            if tr.wall_hits == 0 and not prev_confused:
                dx, dy = DELTAS.get(action, (0, 0))
                pad    = (prev[0] + dx, prev[1] + dy)
                if 0 <= pad[0] < GRID and 0 <= pad[1] < GRID:
                    self.known.setdefault(pad, "teleport")
                    self.tele[pad] = cur
        elif tr.is_confused and not prev_confused:
            # Only label confusion on actual trap entry (transition not-confused -> confused);
            # merely passing through a cell while under ongoing confusion must not tag it.
            self.known[cur] = "confusion"
        else:
            if self.known.get(cur) == "death":
                self.known[cur] = "free"   # fire rotated away -- cell safe now
            elif self.known.get(cur) != "confusion":
                self.known.setdefault(cur, "free")

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
        dst.known      = {k: ("free" if v == "death" else v)
                          for k, v in src.known.items()}
        dst.goal       = src.goal
        dst.start      = src.start
        dst.tele       = dict(src.tele)
        dst.passable   = set(src.passable)
        dst.blocked    = set()          # reset wall hits; passable is canonical
        dst.best_path  = list(src.best_path)
        dst.best_turns = src.best_turns
    else:
        # Carry goal/start so BFS knows where to aim even before re-discovering goal.
        # All three mazes share the same start/goal cell coordinates.
        dst.goal  = src.goal
        dst.start = src.start
        # known, passable, blocked, tele stay empty (Agent() defaults)

    dst.danger     = set()
    dst.rot_danger = {0:set(), 1:set(), 2:set(), 3:set()}
    return dst


# ════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ════════════════════════════════════════════════════════
def run_episodes(env: MazeEnv, agent: Agent, n: int,
                 train: bool, label: str) -> list:
    all_stats = []
    for ep in range(n):
        pos = env.reset(ep)
        agent.reset_episode(pos)

        for turn in range(MAX_TURNS):
            prev_conf = agent._confused
            action    = agent.choose(pos)
            tr        = env.step(action)

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
        elif agent.goal in agent._ep_cells:
            # Goal was found this episode -- path is now in passable.
            # Ratchet epsilon down so future episodes exploit the discovered path
            # rather than wandering randomly. Q-table is not updated so this is
            # not policy drift; it only controls BFS-vs-explore balance.
            agent.epsilon = max(EPS_END, agent.epsilon * 0.5)

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
              train_stats: list | None = None) -> tuple:
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    env   = MazeEnv(path, gamma_mode=gamma_mode)
    stats = run_episodes(env, agent, n, train, label)
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
    _, _, _ = run_phase(
        MAZE_ALPHA_BLANK, agent_pre, PRETRAIN_EPISODES,
        train=True, label="MAZE-ALPHA  Pre-train (blank)")

    # 1 -- Train on maze-alpha (with hazards), warm-started from pre-training
    # same_maze=True: MAZE_0 and MAZE_1 share the same wall layout (MAZE_1 adds hazards only)
    agent_train = transfer(agent_pre, eps=agent_pre.epsilon, same_maze=True)
    train_stats, _, _ = run_phase(
        MAZE_ALPHA, agent_train, TRAIN_EPISODES,
        train=True, label="MAZE-ALPHA  Training")

    save_curve(train_stats, RESULTS / "learning_curve_alpha.png")

    # 2 -- Test maze-alpha (Q-table + map transferred, no retraining)
    agent_at = transfer(agent_train, eps=EPS_END, same_maze=True)
    test_a, _, _ = run_phase(
        MAZE_ALPHA, agent_at, TEST_EPISODES,
        train=False, label="MAZE-ALPHA  Test",
        train_stats=train_stats)

    # 3 -- Test maze-beta (zero-shot -- DO NOT train)
    # Use higher epsilon: alpha Q-values don't generalize to a different wall layout,
    # so the agent must explore more to build a valid passable map for beta.
    agent_bt = transfer(agent_train, eps=0.30)
    agent_bt.goal_ward = True   # bias frontier exploration toward goal
    test_b, _, _ = run_phase(
        MAZE_BETA, agent_bt, TEST_EPISODES,
        train=False, label="MAZE-BETA  Test (zero-shot)")

    # 4 -- Extra credit: maze-gamma (push-pad hazards)
    agent_gt = transfer(agent_train, eps=0.30)
    agent_gt.goal_ward = True   # bias frontier exploration toward goal
    test_g, _, _ = run_phase(
        MAZE_GAMMA, agent_gt, TEST_EPISODES,
        train=False, label="MAZE-GAMMA  Test (extra credit)",
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
