"""
maze_solve_final.py  —  COSC 4368 AI  |  Final Check-In
Group 5

Method: Tabular Q-Learning with BFS-guided exploration

Why Q-Learning:
  - State space is tractable: 64×64 cells × 2 (confused flag) = 8 192 states
  - Off-policy — memory and Q-table persist across episodes (spec permits this)
  - Handles non-stationary fire rotations naturally via continued Q-updates
  - No neural network needed; interpretable and easy to demo

Flow:
  1. Train on maze-alpha  (TRAIN_EPISODES)
  2. Test  on maze-alpha  (TEST_EPISODES,  no retraining)
  3. Test  on maze-beta   (TEST_EPISODES,  zero-shot — no training at all)
  4. Test  on maze-gamma  (TEST_EPISODES,  extra credit — push-pad hazards)

Metrics reported
  Primary : success rate, avg path length, avg turns to solution, death rate
  Bonus   : exploration efficiency, map completeness, learning efficiency
"""
from __future__ import annotations

# ── Imports ───────────────────────────────────────────────────────────────────
from PIL import Image, ImageDraw
import numpy as np
from collections import deque, defaultdict, namedtuple
from pathlib import Path
from scipy import ndimage
import hashlib, pickle
import random
import time

# ═════════════════════════════════════════════════════════════════════════════
#  TUNABLE PARAMETERS  ← change these freely
# ═════════════════════════════════════════════════════════════════════════════
TRAIN_EPISODES = 200      # episodes to train on maze-alpha
TEST_EPISODES  = 5        # evaluation episodes per maze
MAX_TURNS      = 10_000   # per-episode turn limit (spec max)

# Q-Learning hyperparameters
ALPHA     = 0.1           # learning rate
GAMMA_RL  = 0.95          # discount factor
EPS_START = 1.0           # initial exploration rate (ε)
EPS_END   = 0.05          # minimum ε
EPS_DECAY = 0.95          # multiply ε by this each episode


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
WALL        = 1
FREE        = 0
NUM_CELLS   = 64          # maze grid size (spec: 64 × 64)
BRIGHTNESS  = 128         # pixel-brightness threshold (FREE if > BRIGHTNESS)

# Actions
UP, RIGHT, DOWN, LEFT, WAIT = 0, 1, 2, 3, 4
NUM_ACTIONS = 5
DELTAS  = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0), WAIT: (0, 0)}
REVERSE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT, WAIT: WAIT}

# Rewards
R_GOAL      =  200
R_DEATH     =  -50
R_NEW_CELL  =  +0.5
R_STEP      =   -1
R_WALL_HIT  =   -2
R_CONFUSION =   -5

# ── Maze file paths (relative to this script) ─────────────────────────────────
_HERE      = Path(__file__).resolve().parent
MAZE_ALPHA = _HERE / "Contexts/maze-alpha/NewMaze_1.png"
MAZE_BETA  = _HERE / "Contexts/maze-beta/NewMaze_1.png"
MAZE_GAMMA = _HERE / "Contexts/maze-gamma/NewMaze_1.png"

# ═════════════════════════════════════════════════════════════════════════════
#  NAMED TUPLES
# ═════════════════════════════════════════════════════════════════════════════
TurnResult = namedtuple("TurnResult", [
    "wall_hits",          # int   – wall collisions this action
    "current_position",   # (cx, cy) – agent cell after action (respawn if dead)
    "is_dead",            # bool  – stepped into fire pit
    "is_confused",        # bool  – confusion flag active now
    "is_goal_reached",    # bool  – reached goal cell
    "teleported",         # bool  – teleport was triggered
    "actions_executed",   # int   – always 1 in this implementation
    "pushed",             # bool  – pushed by directional pad (gamma only)
])

EpisodeStats = namedtuple("EpisodeStats", [
    "success",        # bool
    "path_length",    # total cells traversed (incl. revisits)
    "turns",          # turns used this episode
    "deaths",         # number of deaths
    "unique_cells",   # unique cells visited
    "total_cells",    # total cell visits (same as path_length)
])

# ═════════════════════════════════════════════════════════════════════════════
#  IMAGE / GRID HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _load_image(path: Path):
    img = Image.open(path).convert("RGB")
    px  = np.array(img)
    h, w, _ = px.shape
    return px, h, w


def _pixel_grid(pixels: np.ndarray, h: int, w: int) -> np.ndarray:
    """Binary (WALL=1 / FREE=0) pixel-level grid using red channel brightness."""
    g = np.ones((h, w), dtype=np.int8)
    g[pixels[:, :, 0] > BRIGHTNESS] = FREE
    return g


def _find_start_goal_px(pixels, h, w):
    """Scan top row for start, bottom row for goal; return pixel coords."""
    sp = gp = None
    for x in range(w):
        if pixels[0, x, 0] > BRIGHTNESS:
            sp = (x, 0)
        if pixels[h - 1, x, 0] > BRIGHTNESS:
            gp = (x, h - 1)
    return sp, gp


def _find_hazard_px(pixels, h, w, color_fn, min_sz: int = 20):
    """
    Return list of (cx, cy) pixel-space centres for colour clusters
    that pass color_fn and contain at least min_sz matching pixels.
    color_fn receives numpy arrays (r, g, b) and must return a boolean array
    using numpy operators (&, |, ~) rather than Python `and`/`or`.
    """
    r_ch = pixels[:, :, 0].astype(np.int32)
    g_ch = pixels[:, :, 1].astype(np.int32)
    b_ch = pixels[:, :, 2].astype(np.int32)
    mask = color_fn(r_ch, g_ch, b_ch).astype(bool)
    labeled, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        pts = np.argwhere(labeled == i)
        if len(pts) >= min_sz:
            cy, cx = pts.mean(0).astype(int)
            out.append((int(cx), int(cy)))
    return out


def _detect_hazards(pixels, h, w) -> dict:
    """Detect all hazard types from the maze image; return pixel-space data.
    All color_fn lambdas use numpy bitwise operators (&, |) for vectorised
    evaluation across the full image in a single pass.
    """

    # ── Fire pits 🔥  (orange-red: R high, G mid, B low) ────────────────────
    fire = _find_hazard_px(pixels, h, w,
        lambda r, g, b: (r > 180) & (g > 70) & (g < 175) & (b < 100))

    # ── Confusion traps 😵  (bright yellow face: R≈G both high, B very low) ─
    confusion = _find_hazard_px(pixels, h, w,
        lambda r, g, b: (r > 210) & (g > 160) & (b < 70) & (np.abs(r - g) < 70),
        min_sz=50)

    # ── Teleport pads: detect colour families, pair by x-sorted order ────────
    # Green 🟢  (G dominant)
    green = sorted(_find_hazard_px(pixels, h, w,
        lambda r, g, b: (g > 170) & (r < 140) & (b < 160), min_sz=30))

    # Purple 🟣  (R+B elevated, G low)
    purple = sorted(_find_hazard_px(pixels, h, w,
        lambda r, g, b: (r > 100) & (r < 210) & (b > 130) & (g < 100), min_sz=30))

    # Yellow/gold 🟡  (R+G high, B low, R noticeably > G)
    yellow = sorted(_find_hazard_px(pixels, h, w,
        lambda r, g, b: (r > 190) & (g > 140) & (b < 90) & (r > g + 25), min_sz=30))

    # Red 🔴  (R dominant, G and B both low; larger footprint than fire emojis)
    red = sorted(_find_hazard_px(pixels, h, w,
        lambda r, g, b: (r > 190) & (g < 80) & (b < 80), min_sz=80))

    tp: dict = {}
    if len(green)  >= 2: tp[green[0]]  = green[1]
    if len(purple) >= 2: tp[purple[0]] = purple[1]
    if len(yellow) >= 2: tp[yellow[0]] = yellow[1]
    if len(red)    >= 2: tp[red[0]]    = red[1]

    # ── Gamma push pads ⬆️⬅️  (teal / blue arrows) ──────────────────────────
    all_blue = _find_hazard_px(pixels, h, w,
        lambda r, g, b: (b > 150) & (r < 120) & (g > 120) & (g < 220), min_sz=25)
    mid_y    = h // 2
    push_up  = [(x, y) for x, y in all_blue if y < mid_y]
    push_lft = [(x, y) for x, y in all_blue if y >= mid_y]

    return dict(
        fire       = fire,
        confusion  = confusion,
        teleport   = tp,
        push_up    = push_up,
        push_left  = push_lft,
    )


# ── Cell-space helpers ─────────────────────────────────────────────────────────

def _cell_size(w: int, h: int) -> int:
    return max(1, min(w, h) // NUM_CELLS)


def px_to_cell(px: int, py: int, cs: int) -> tuple:
    return (min(max(0, px // cs), NUM_CELLS - 1),
            min(max(0, py // cs), NUM_CELLS - 1))


def cell_to_px(cx: int, cy: int, cs: int) -> tuple:
    return cx * cs + cs // 2, cy * cs + cs // 2


def _build_adjacency(pg, w, h, cs):
    adj = {}
    free_centres = set()
    for cy in range(NUM_CELLS):
        for cx in range(NUM_CELLS):
            c1x = min(cx * cs + cs // 2, w - 1)
            c1y = min(cy * cs + cs // 2, h - 1)
            if pg[c1y, c1x] == FREE:
                free_centres.add((cx, cy))

    for (cx, cy) in free_centres:
        nbrs = {}
        for act, (dx, dy) in DELTAS.items():
            if act == WAIT:
                continue
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < NUM_CELLS and 0 <= ny < NUM_CELLS):
                continue
            if (nx, ny) not in free_centres:
                continue
            # Boundary slice - precomputed once per direction
            if dx == 1:
                x1 = min(cx * cs + cs // 2, w - 1)
                x2 = min(nx * cs + cs // 2, w - 1)
                has_free = np.any(pg[cy*cs : min((cy+1)*cs, h), x1:x2+1] == FREE)
            elif dx == -1:
                x1 = min(nx * cs + cs // 2, w - 1)
                x2 = min(cx * cs + cs // 2, w - 1)
                has_free = np.any(pg[cy*cs : min((cy+1)*cs, h), x1:x2+1] == FREE)
            elif dy == 1:
                # Scan the full vertical span between the two cell centres
                y1 = min(cy * cs + cs // 2, h - 1)
                y2 = min(ny * cs + cs // 2, h - 1)
                has_free = np.any(pg[y1:y2+1, cx*cs : min((cx+1)*cs, w)] == FREE)
            else:  # dy == -1
                y1 = min(ny * cs + cs // 2, h - 1)
                y2 = min(cy * cs + cs // 2, h - 1)
                has_free = np.any(pg[y1:y2+1, cx*cs : min((cx+1)*cs, w)] == FREE)
            if has_free:
                nbrs[act] = (nx, ny)
        adj[(cx, cy)] = nbrs
    return adj

def _build_adjacency_cached(pg, w, h, cs, path):
    cache_key = hashlib.md5(pg.tobytes()).hexdigest()
    cache_file = path.parent / f".adj_cache_{cache_key}.pkl"
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())
    adj = _build_adjacency(pg, w, h, cs)
    cache_file.write_bytes(pickle.dumps(adj))
    return adj

def _rotate_90cw(x: float, y: float, px: float, py: float):
    """Rotate point (x, y) 90° clockwise around pivot (px, py)."""
    return px + (y - py), py - (x - px)


# ═════════════════════════════════════════════════════════════════════════════
#  MAZE ENVIRONMENT  (server-side — agent cannot inspect its internals)
# ═════════════════════════════════════════════════════════════════════════════

class MazeEnvironment:
    """
    Simulates the maze from a PNG image.
    The agent ONLY receives TurnResult feedback — no internal grid or
    hazard positions are exposed to the agent.
    """

    def __init__(self, path: Path, is_gamma: bool = False):
        px, h, w  = _load_image(path)
        self._px  = px
        self._h   = h
        self._w   = w
        self._cs  = _cell_size(w, h)
        self._pg  = _pixel_grid(px, h, w)
        self._adj = _build_adjacency(self._pg, w, h, self._cs)
        
        self._gamma = is_gamma

        # Start / goal
        sp, gp = _find_start_goal_px(px, h, w)
        if sp is None or gp is None:
            raise ValueError(f"Start or goal not found in {path.name}")
        self._start = px_to_cell(*sp, self._cs)
        self._goal  = px_to_cell(*gp,  self._cs)

        # Hazards (pixel space → cell space)
        hz = _detect_hazards(px, h, w)

        self._base_fire_px = hz["fire"]   # original fire positions before rotation

        self._conf_cells = frozenset(
            px_to_cell(x, y, self._cs) for x, y in hz["confusion"])

        self._tele_cells: dict = {
            px_to_cell(sx, sy, self._cs): px_to_cell(dx, dy, self._cs)
            for (sx, sy), (dx, dy) in hz["teleport"].items()
        }

        # Gamma push pads
        self._push_up_cells   = frozenset(
            px_to_cell(x, y, self._cs) for x, y in hz["push_up"])
        self._push_left_cells = frozenset(
            px_to_cell(x, y, self._cs) for x, y in hz["push_left"])

        # Approximate number of navigable cells (for metric normalisation)
        self._free_count = max(1, sum(
            1 for cell, nbrs in self._adj.items() if nbrs
        ))

        # Episode state
        self._ep   = 0
        self._pos  = self._start
        self._conf = 0        # confusion turns remaining
        self._fire: frozenset = frozenset()

        print(
            f"  Loaded {path.name:<22} | "
            f"start={self._start} goal={self._goal} cs={self._cs}px | "
            f"fire={len(self._base_fire_px)} "
            f"conf={len(self._conf_cells)} "
            f"tele={len(self._tele_cells)} "
            f"push_up={len(self._push_up_cells)} "
            f"push_left={len(self._push_left_cells)}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, episode: int | None = None) -> tuple:
        """Reset to start position, rotate fire pits for new episode."""
        self._ep  = episode if episode is not None else self._ep + 1
        self._pos = self._start
        self._conf = 0
        self._rotate_fire(self._ep)
        return self._pos

    def step(self, action: int) -> TurnResult:
        """Execute one action; return TurnResult feedback."""
        wall_hits  = 0
        teleported = False
        pushed     = False

        # Confusion reversal
        eff_action = REVERSE[action] if self._conf > 0 else action
        if self._conf > 0:
            self._conf -= 1

        # Movement
        cx, cy = self._pos
        if eff_action == WAIT:
            new_pos = (cx, cy)
        else:
            nbrs = self._adj.get((cx, cy), {})
            if eff_action in nbrs:
                new_pos = nbrs[eff_action]
            else:
                new_pos = (cx, cy)
                wall_hits = 1

        self._pos = new_pos

        # ── Gamma push pads ──────────────────────────────────────────────────
        if self._gamma:
            if self._pos in self._push_up_cells:
                nbrs2 = self._adj.get(self._pos, {})
                if UP in nbrs2:
                    self._pos = nbrs2[UP]
                pushed = True
            elif self._pos in self._push_left_cells:
                nbrs2 = self._adj.get(self._pos, {})
                if LEFT in nbrs2:
                    self._pos = nbrs2[LEFT]
                pushed = True

        # ── Fire pit ─────────────────────────────────────────────────────────
        if self._pos in self._fire:
            self._pos  = self._start
            self._conf = 0
            return TurnResult(wall_hits, self._pos, True,
                              False, False, False, 1, pushed)

        # ── Teleport ─────────────────────────────────────────────────────────
        if self._pos in self._tele_cells:
            self._pos  = self._tele_cells[self._pos]
            teleported = True
            # Handle chained teleport
            if self._pos in self._tele_cells:
                self._pos = self._tele_cells[self._pos]

        # ── Confusion trap ───────────────────────────────────────────────────
        if self._pos in self._conf_cells:
            self._conf = 2

        goal = (self._pos == self._goal)
        return TurnResult(wall_hits, self._pos, False,
                          self._conf > 0, goal, teleported, 1, pushed)

    # ── Properties for visualiser (NOT used by the agent) ────────────────────

    @property
    def start_cell(self) -> tuple:
        return self._start

    @property
    def goal_cell(self) -> tuple:
        return self._goal

    @property
    def cell_size(self) -> int:
        return self._cs

    @property
    def pixels_copy(self) -> np.ndarray:
        return self._px.copy()

    @property
    def num_free_cells(self) -> int:
        return self._free_count

    # ── Internal ─────────────────────────────────────────────────────────────

    def _rotate_fire(self, episode: int):
        """
        Rotate the fire-pit cluster 90° CW × episode around the bottommost pit
        (highest y = bottom of the V).  Only positions landing on FREE cells
        remain active.
        """
        if not self._base_fire_px:
            self._fire = frozenset()
            return

        # Pivot = pixel with maximum y (visual bottom of the V arrangement)
        ppx, ppy = max(self._base_fire_px, key=lambda p: p[1])
        rot      = episode % 4
        active   = []
        for (x, y) in self._base_fire_px:
            rx, ry = float(x), float(y)
            for _ in range(rot):
                rx, ry = _rotate_90cw(rx, ry, ppx, ppy)
            xi = int(round(rx))
            yi = int(round(ry))
            xi = min(max(0, xi), self._w - 1)
            yi = min(max(0, yi), self._h - 1)
            active.append(px_to_cell(xi, yi, self._cs))

        self._fire = frozenset(active)


# ═════════════════════════════════════════════════════════════════════════════
#  BFS ON KNOWN MAP  (used internally by the agent)
# ═════════════════════════════════════════════════════════════════════════════

def _bfs_known(known: dict, start: tuple, goal: tuple,
               danger: set, blocked: set | None = None,
               tele_map: dict | None = None) -> list:
    """
    BFS strictly through the agent's DISCOVERED free cells.
    - Only traverses cells present in `known` with value != "wall"/"death".
    - Skips cells in `danger`.
    - Skips directed edges in `blocked` (set of (from, to) tuples).
    - Follows teleport edges in `tele_map` (pad_cell → dest_cell).
    Returns a path list, or [] if goal is unreachable within known map.
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
            if nb in came:
                continue
            if nb not in known:
                continue
            if nb in danger:
                continue
            if blocked and (cur, nb) in blocked:
                continue
            queue.append(nb)
            came[nb] = cur
        # Follow teleport edge from this pad cell (if any)
        if tele_map and cur in tele_map:
            dest = tele_map[cur]
            if dest not in came and dest in known:
                if known[dest] not in ("wall", "death") and dest not in danger:
                    came[dest] = cur
                    queue.append(dest)
    return []


# ═════════════════════════════════════════════════════════════════════════════
#  Q-LEARNING AGENT  (client-side — only sees TurnResult)
# ═════════════════════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Tabular Q-Learning agent.
    State  : (cx, cy, confused: 0|1)
    Actions: UP RIGHT DOWN LEFT WAIT  (5 actions)

    The agent never inspects MazeEnvironment's private attributes.
    It builds its own internal map purely from TurnResult feedback.
    """

    def __init__(self, free_cells_hint: int = 2048):
        self.epsilon = EPS_START
        self._hint   = free_cells_hint   # updated when maze is loaded

        # ── Persistent across all episodes ───────────────────────────────────
        self.q_table = np.zeros((NUM_CELLS, NUM_CELLS, 2, NUM_ACTIONS), dtype=np.float32)
        # known_map: visited free/special cells only (no "wall" entries)
        self.known_map: dict = {}       # (cx,cy) → "free"|"death"|
                                        #           "confusion"|"teleport"|"goal"
        # Blocked directed edges: (from_cell, to_cell) confirmed impassable
        self._blocked: set  = set()
        self._danger: set   = set()     # cells known to be lethal
        self._tele_map: dict = {}        # pad_cell → dest_cell (inferred from TurnResult)
        self._rot_danger: dict = {0: set(), 1: set(), 2: set(), 3: set()}  # ep%4 → known fire pits
        self._ep_count: int = 0          # total episodes seen (drives fire rotation inference)
        self._goal_cell: tuple | None  = None
        self._start_cell: tuple | None = None

        self.best_path: list | None = None
        self.best_turns: int        = 999_999

        # ── Per-episode (reset each call to reset_episode) ───────────────────
        self._frontier_cache     = None
        self._map_size_at_cache  = 0
        self._pos:       tuple = (0, 0)
        self._confused:  bool  = False
        self._ep_path:   list  = []
        self._ep_deaths: int   = 0
        self._ep_turns:  int   = 0
        self._ep_cells:  set   = set()

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def reset_episode(self, start: tuple):
        self._pos       = start
        self._confused  = False
        self._ep_path   = [start]
        self._ep_deaths = 0
        self._ep_turns  = 0
        self._ep_cells  = {start}
        if self._start_cell is None:
            self._start_cell = start
        self.known_map.setdefault(start, "free")
        self._ep_count += 1
        rot = self._ep_count % 4
        self._danger = set(self._rot_danger[rot])  # preload this rotation's known fire pits

    # ── Action selection ──────────────────────────────────────────────────────

    def choose_action(self, pos: tuple, confused: bool) -> int:
        """
        BFS-guided action selection with epsilon as a random override:
          epsilon chance  → random walk (frontier-biased for exploration).
          1-epsilon chance:
            Phase 2 (goal known)   : BFS to goal through known safe cells.
            Phase 1 (goal unknown) : BFS to nearest frontier, step into unknown.
        BFS always runs when epsilon does NOT trigger — this ensures the agent
        actually navigates rather than spinning in place once epsilon drops.
        """
        # ── Small epsilon-random override (prevents loops, handles hazards) ───
        if random.random() < self.epsilon:
            frontier_acts = self._frontier_actions(pos)
            if frontier_acts:
                return random.choice(frontier_acts)
            return random.randint(0, NUM_ACTIONS - 1)

        # ── Deterministic BFS component ───────────────────────────────────────

        # Phase 2: goal found → navigate to it
        if self._goal_cell is not None:
            safe = {k: v for k, v in self.known_map.items()
                    if k not in self._danger}
            path = _bfs_known(safe, pos, self._goal_cell, self._danger,
                              blocked=self._blocked, tele_map=self._tele_map)
            if len(path) > 1:
                return self._delta_to_action(pos, path[1])

        # Phase 1: goal unknown → BFS to nearest frontier, step into unknown
        fc, unk = self._find_nearest_frontier(pos)
        if fc is not None:
            if fc == pos:
                # Already at a frontier cell — step into the unknown cell
                if unk is not None:
                    return self._delta_to_action(pos, unk)
            else:
                path = _bfs_known(self.known_map, pos, fc, self._danger,
                                  blocked=self._blocked, tele_map=self._tele_map)
                if len(path) > 1:
                    return self._delta_to_action(pos, path[1])

        # ── Fallback: random walk ─────────────────────────────────────────────
        frontier_acts = self._frontier_actions(pos)
        if frontier_acts:
            return random.choice(frontier_acts)
        return random.randint(0, NUM_ACTIONS - 1)

    def _delta_to_action(self, pos: tuple, nxt: tuple) -> int:
        dx, dy = nxt[0] - pos[0], nxt[1] - pos[1]
        return {(0,-1): UP, (1,0): RIGHT, (0,1): DOWN, (-1,0): LEFT}.get(
            (dx, dy), random.randint(0, 3))

    def _frontier_actions(self, pos: tuple) -> list:
        """Return actions whose immediate neighbour has not yet been discovered."""
        cx, cy = pos
        out = []
        for act, (dx, dy) in DELTAS.items():
            if act == WAIT:
                continue
            nb = (cx + dx, cy + dy)
            if (0 <= nb[0] < NUM_CELLS and 0 <= nb[1] < NUM_CELLS
                    and nb not in self.known_map):
                out.append(act)
        return out

    def _find_nearest_frontier(self, pos: tuple):
        current_size = len(self.known_map)
        if (self._frontier_cache is not None
                and self._frontier_cache[0] == pos
                and self._map_size_at_cache == current_size):
            return self._frontier_cache[1]
        result = self._find_nearest_frontier_uncached(pos)
        self._frontier_cache = (pos, result)
        self._map_size_at_cache = current_size
        return result

    def _find_nearest_frontier_uncached(self, pos: tuple):
        """
        BFS strictly through known (non-wall, non-danger) cells to find the
        nearest cell that has an undiscovered neighbour reachable via a
        non-blocked edge.
        Returns (frontier_cell, unknown_neighbour) or (None, None).
        """
        # Check current position first
        for act, (dx, dy) in DELTAS.items():
            if act == WAIT:
                continue
            nb = (pos[0]+dx, pos[1]+dy)
            if (0 <= nb[0] < NUM_CELLS and 0 <= nb[1] < NUM_CELLS
                    and nb not in self.known_map
                    and (pos, nb) not in self._blocked):
                return pos, nb

        queue   = deque([pos])
        visited = {pos}
        while queue:
            cur    = queue.popleft()
            cx, cy = cur
            for dx, dy in ((0,-1),(1,0),(0,1),(-1,0)):
                nb = (cx+dx, cy+dy)
                if nb in visited:
                    continue
                if not (0 <= nb[0] < NUM_CELLS and 0 <= nb[1] < NUM_CELLS):
                    continue
                if (cur, nb) in self._blocked:
                    continue
                visited.add(nb)
                nb_type = self.known_map.get(nb)
                if nb_type is None:
                    # nb is unknown and reachable — cur is the frontier cell
                    return cur, nb
                if nb in self._danger:
                    continue
                queue.append(nb)
            # Follow teleport edge from this pad cell (if any)
            if cur in self._tele_map:
                dest = self._tele_map[cur]
                if dest not in visited:
                    visited.add(dest)
                    dest_type = self.known_map.get(dest)
                    if dest_type is None:
                        return cur, dest  # frontier reachable via teleport
                    if dest_type not in ("wall", "death") and dest not in self._danger:
                        queue.append(dest)
        return None, None

    # ── Q-table update ────────────────────────────────────────────────────────

    def update_q(self, prev: tuple, prev_confused: bool,
                 action: int, reward: float, tr: TurnResult):
        s_q  = self.q_table[prev[0], prev[1], int(prev_confused)]
        ns_q = self.q_table[tr.current_position[0], tr.current_position[1], int(tr.is_confused)]
        target = reward if tr.is_goal_reached else reward + GAMMA_RL * float(ns_q.max())
        s_q[action] += ALPHA * (target - float(s_q[action]))

    # ── Observation / internal map update ────────────────────────────────────

    def observe(self, prev: tuple, action: int, tr: TurnResult):
        """Update internal map and per-episode stats from TurnResult."""
        cur = tr.current_position
        self._ep_turns += 1

        if tr.is_dead:
            # The cell we stepped INTO was a fire pit.
            # Infer its location from prev + action direction.
            if tr.wall_hits == 0:
                dx, dy  = DELTAS.get(action, (0, 0))
                pit_loc = (prev[0] + dx, prev[1] + dy)
                if 0 <= pit_loc[0] < NUM_CELLS and 0 <= pit_loc[1] < NUM_CELLS:
                    self.known_map[pit_loc] = "death"
                    self._danger.add(pit_loc)
                    self._rot_danger[self._ep_count % 4].add(pit_loc)
            self._ep_deaths += 1
            self._pos      = self._start_cell if self._start_cell else cur
            self._confused = False
            return

        # Successful move
        self._pos      = cur
        self._confused = tr.is_confused
        self._ep_cells.add(cur)
        self._ep_path.append(cur)

        if tr.is_goal_reached:
            self.known_map[cur] = "goal"
            if self._goal_cell is None:
                self._goal_cell = cur
        elif tr.teleported:
            self.known_map.setdefault(cur, "teleport")
            # Infer and record the teleport pad cell + the pad→dest edge
            if tr.wall_hits == 0:
                dx, dy = DELTAS.get(action, (0, 0))
                pad = (prev[0] + dx, prev[1] + dy)
                if 0 <= pad[0] < NUM_CELLS and 0 <= pad[1] < NUM_CELLS:
                    self.known_map.setdefault(pad, "teleport")
                    self._tele_map[pad] = cur
        elif tr.is_confused:
            self.known_map.setdefault(cur, "confusion")
        else:
            # Overwrite stale "death" from a past fire-pit rotation
            if self.known_map.get(cur) == "death":
                self.known_map[cur] = "free"
            else:
                self.known_map.setdefault(cur, "free")

        # Record a blocked directed edge — the move prev→target was rejected.
        # We do NOT mark target as "wall" because the cell may be reachable
        # from another direction; we only know THIS edge is impassable.
        if tr.wall_hits > 0:
            dx, dy = DELTAS.get(action, (0, 0))
            target = (prev[0] + dx, prev[1] + dy)
            if 0 <= target[0] < NUM_CELLS and 0 <= target[1] < NUM_CELLS:
                self._blocked.add((prev, target))

    def record_success(self, turns: int):
        if turns < self.best_turns:
            self.best_turns = turns
            self.best_path  = list(self._ep_path)

    def get_episode_stats(self) -> EpisodeStats:
        return EpisodeStats(
            success      = self._goal_cell in self._ep_cells,
            path_length  = len(self._ep_path),
            turns        = self._ep_turns,
            deaths       = self._ep_deaths,
            unique_cells = len(self._ep_cells),
            total_cells  = len(self._ep_path),
        )

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


# ═════════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_episodes(env: MazeEnvironment, agent: QLearningAgent,
                 n: int, train: bool, label: str) -> list:
    """
    Run n episodes.
    If train=True  → update Q-table after each step.
    If train=False → map-building + BFS only; Q-table unchanged.
    Epsilon always decays to let the agent exploit its growing map.
    """
    stats = []
    for ep in range(n):
        pos = env.reset(ep)
        agent.reset_episode(pos)
        agent.reset_episode(pos)
                
        agent._hint = env.num_free_cells

        for turn in range(MAX_TURNS):
            prev_confused = agent._confused
            action = agent.choose_action(pos, prev_confused)
            tr     = env.step(action)

            # Reward shaping
            reward = R_STEP
            if tr.wall_hits:
                reward += R_WALL_HIT * tr.wall_hits
            if tr.is_dead:
                reward += R_DEATH
            elif tr.is_goal_reached:
                reward += R_GOAL
            else:
                if tr.is_confused and not prev_confused:
                    reward += R_CONFUSION
                if tr.current_position not in agent.known_map:
                    reward += R_NEW_CELL

            if train:
                agent.update_q(pos, prev_confused, action, reward, tr)

            agent.observe(pos, action, tr)
            pos = agent._pos   # may differ from tr.current_position after death

            if tr.is_goal_reached:
                agent.record_success(turn + 1)
                break

        stats.append(agent.get_episode_stats())
        agent.decay_epsilon()   # always decay — helps test episodes exploit map

        if (ep + 1) % 20 == 0 or ep == n - 1:
            recent = stats[-min(10, len(stats)):]
            sr     = sum(s.success for s in recent) / len(recent)
            print(f"    [{label[:20]:<20}] ep {ep+1:>3}/{n}  "
                  f"ε={agent.epsilon:.3f}  "
                  f"SR(last10)={sr:.0%}  "
                  f"map={len(agent.known_map)}/{env.num_free_cells}")

    return stats


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════════════

def report_metrics(stats: list, agent: QLearningAgent,
                   env: MazeEnvironment, label: str,
                   train_stats: list | None = None) -> dict:
    """Compute and print all required + bonus metrics."""
    successes  = [s for s in stats if s.success]
    total_d    = sum(s.deaths for s in stats)
    total_t    = sum(s.turns  for s in stats)

    SR  = len(successes) / len(stats) if stats else 0.0
    APL = (sum(s.path_length for s in successes) / len(successes)
           if successes else float("inf"))
    AT  = (sum(s.turns for s in successes) / len(successes)
           if successes else float("inf"))
    DR  = total_d / max(1, total_t)
    EE  = (sum(s.unique_cells / max(1, s.total_cells) for s in stats) / len(stats)
           if stats else 0.0)
    MC  = (len([v for v in agent.known_map.values() if v != "wall"])
           / env.num_free_cells)

    # Learning efficiency: first episode where rolling-10 SR ≥ 80 %
    LE_ep = None
    if train_stats:
        window = 10
        for i in range(window, len(train_stats) + 1):
            chunk = train_stats[i - window: i]
            if sum(s.success for s in chunk) / window >= 0.80:
                LE_ep = i
                break

    print(f"\n{'─'*54}")
    print(f"  METRICS  —  {label}")
    print(f"{'─'*54}")
    print(f"  Success Rate            : {SR:>7.1%}   (target >80 %)")
    print(f"  Avg Path Length         : {APL:>8.1f} cells")
    print(f"  Avg Turns to Solution   : {AT:>8.1f}   (target <1 000)")
    print(f"  Death Rate              : {DR:>8.4f}   (target <0.05)")
    print(f"  [Bonus] Exploration Eff.: {EE:.4f}")
    print(f"  [Bonus] Map Completeness: {MC:.1%}")
    if LE_ep is not None:
        print(f"  [Bonus] Learning Eff.   : converged at episode {LE_ep}")
    elif train_stats:
        print(f"  [Bonus] Learning Eff.   : did not reach 80 % SR during training")
    print(f"{'─'*54}")

    return dict(SR=SR, APL=APL, AT=AT, DR=DR, EE=EE, MC=MC, LE=LE_ep)


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def _save_path_image(env: MazeEnvironment, path_cells: list, out: Path):
    """Draw best-found solution path (magenta) on the maze image and save."""
    if not path_cells:
        print(f"    (no path to save for {out.name})")
        return
    img = env.pixels_copy
    cs  = env.cell_size
    t   = 2
    h, w = img.shape[:2]
    for (cx, cy) in path_cells:
        px, py = cell_to_px(cx, cy, cs)
        for dy in range(-t, t + 1):
            for dx in range(-t, t + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h:
                    img[ny, nx] = [255, 0, 200]
    Image.fromarray(img).save(out)
    print(f"    Saved → {out.name}")


def _save_map_overlay(env: MazeEnvironment, agent: QLearningAgent, out: Path):
    """Overlay the agent's discovered map (death/confusion/teleport cells)."""
    img  = Image.fromarray(env.pixels_copy)
    draw = ImageDraw.Draw(img)
    cs   = env.cell_size
    r    = max(2, cs // 4)
    for (cx, cy), ctype in agent.known_map.items():
        px, py = cell_to_px(cx, cy, cs)
        if ctype == "death":
            draw.ellipse([px-r, py-r, px+r, py+r],
                         fill=(255, 60, 0), outline=(160, 0, 0))
        elif ctype == "confusion":
            draw.rectangle([px-r, py-r, px+r, py+r],
                           fill=(160, 0, 220), outline=(80, 0, 140))
        elif ctype in ("teleport", "teleport_dest"):
            draw.ellipse([px-r, py-r, px+r, py+r],
                         fill=(0, 200, 80), outline=(255, 255, 255))
    img.save(out)
    print(f"    Saved → {out.name}")


def _save_learning_curve(all_stats: list, out: Path):
    """Save a learning-curve chart (success rate + death rate) as a PNG."""
    W, H, M = 840, 440, 65
    n       = len(all_stats)
    if n < 2:
        return

    img  = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pw   = W - 2 * M
    ph   = H - 2 * M
    window = 10

    def rolling_sr(i):
        chunk = all_stats[max(0, i - window + 1): i + 1]
        return sum(s.success for s in chunk) / len(chunk)

    def rolling_dr(i):
        chunk  = all_stats[max(0, i - window + 1): i + 1]
        td = sum(s.deaths for s in chunk)
        tt = sum(s.turns  for s in chunk)
        return min(1.0, (td / max(1, tt)) * 20)   # scaled ×20 for visibility

    sr_vals = [rolling_sr(i) for i in range(n)]
    dr_vals = [rolling_dr(i) for i in range(n)]

    def pt(i, v):
        x = M + int(i * pw / max(1, n - 1))
        y = H - M - int(v * ph)
        return x, y

    # Grid lines
    draw.rectangle([M, M, W - M, H - M], outline=(0, 0, 0), width=2)
    for v in (0.25, 0.5, 0.75, 1.0):
        y = H - M - int(v * ph)
        draw.line([M, y, W - M, y], fill=(210, 210, 210), width=1)
        draw.text((M - 40, y - 7), f"{v:.0%}", fill=(80, 80, 80))

    # Series
    for i in range(1, n):
        draw.line([*pt(i-1, sr_vals[i-1]), *pt(i, sr_vals[i])],
                  fill=(30, 100, 200), width=2)
    for i in range(1, n):
        draw.line([*pt(i-1, dr_vals[i-1]), *pt(i, dr_vals[i])],
                  fill=(200, 60, 0), width=2)

    # Labels
    draw.text((M, M - 20),
              "Learning Curve — maze-alpha training  (rolling 10-ep window)",
              fill=(0, 0, 0))
    draw.text((W // 2 - 40, H - M + 12), f"Episode  (1 – {n})", fill=(0, 0, 0))
    lx = W - M - 170
    draw.rectangle([lx, M + 10, lx + 14, M + 20], fill=(30, 100, 200))
    draw.text((lx + 18, M + 10), "Success Rate",      fill=(0, 0, 0))
    draw.rectangle([lx, M + 32, lx + 14, M + 42], fill=(200, 60, 0))
    draw.text((lx + 18, M + 32), "Death Rate (×20)",  fill=(0, 0, 0))

    img.save(out)
    print(f"    Saved → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  PER-MAZE RUNNER  (wraps environment + agent + visualisation)
# ═════════════════════════════════════════════════════════════════════════════

def run_maze_phase(
        maze_path: Path,
        agent: QLearningAgent,
        n_ep: int,
        train: bool,
        label: str,
        is_gamma: bool = False,
        train_stats_for_LE: list | None = None,
) -> tuple:
    """Load maze, run episodes, report metrics, save output images."""
    print(f"\n{'═'*55}")
    print(f"  {label}")
    print(f"{'═'*55}")

    env = MazeEnvironment(maze_path, is_gamma=is_gamma)
    agent._hint = env.num_free_cells

    stats   = run_episodes(env, agent, n_ep, train=train, label=label)
    metrics = report_metrics(stats, agent, env, label,
                             train_stats=train_stats_for_LE)

    stem = maze_path.stem
    tag  = "train" if train else "test"
    if agent.best_path:
        _save_path_image(env, agent.best_path,
                         maze_path.parent / f"{stem}_rl_{tag}_solved.png")
    _save_map_overlay(env, agent,
                      maze_path.parent / f"{stem}_rl_{tag}_map.png")

    return stats, metrics, env


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*55)
    print("  COSC 4368 — Final Maze  |  Q-Learning Agent")
    print("  Method: Tabular Q-Learning + BFS exploration")
    print("═"*55)
    print(f"  TRAIN={TRAIN_EPISODES}ep  TEST={TEST_EPISODES}ep  "
          f"MAX_TURNS={MAX_TURNS}  α={ALPHA}  γ={GAMMA_RL}  "
          f"ε {EPS_START}→{EPS_END} (×{EPS_DECAY}/ep)")

    t0 = time.time()

    # ── 1. TRAIN on maze-alpha ────────────────────────────────────────────────
    agent_train = QLearningAgent()
    train_stats, _, _ = run_maze_phase(
        MAZE_ALPHA, agent_train, TRAIN_EPISODES,
        train=True, label="MAZE-ALPHA  Training")

    _save_learning_curve(train_stats, _HERE / "learning_curve_alpha.png")

    # ── 2. TEST maze-alpha (transfer Q-table + map; no new training) ─────────
    agent_at           = QLearningAgent()
    agent_at.q_table   = agent_train.q_table.copy()
    agent_at.known_map = dict(agent_train.known_map)
    agent_at._danger   = set(agent_train._danger)
    agent_at._goal_cell  = agent_train._goal_cell
    agent_at._start_cell = agent_train._start_cell
    agent_at.best_path   = agent_train.best_path
    agent_at.best_turns  = agent_train.best_turns
    agent_at._tele_map   = dict(agent_train._tele_map)
    agent_at._blocked    = set(agent_train._blocked)
    agent_at.epsilon     = EPS_END   # minimal exploration — exploit learned policy

    test_a_stats, test_a_m, _ = run_maze_phase(
        MAZE_ALPHA, agent_at, TEST_EPISODES,
        train=False, label="MAZE-ALPHA  Test (no retraining)",
        train_stats_for_LE=train_stats)

    # ── 3. TEST maze-beta (fresh agent — zero-shot, no training) ─────────────
    agent_b          = QLearningAgent()
    agent_b.epsilon  = 0.80   # start with high exploration; decays each episode
    test_b_stats, test_b_m, _ = run_maze_phase(
        MAZE_BETA, agent_b, TEST_EPISODES,
        train=False, label="MAZE-BETA  Test (zero-shot, no training)")

    # ── 4. EXTRA CREDIT — maze-gamma ─────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  EXTRA CREDIT: Maze-Gamma (directional push-pad hazards ⬆️⬅️)")
    print(f"{'═'*55}")
    agent_g          = QLearningAgent()
    agent_g.epsilon  = 0.80
    test_g_stats, test_g_m, _ = run_maze_phase(
        MAZE_GAMMA, agent_g, TEST_EPISODES,
        train=False, label="MAZE-GAMMA  Extra Credit (zero-shot)",
        is_gamma=True)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═'*55}")
    print(f"  FINAL SUMMARY  ({elapsed:.1f} s total)")
    print(f"{'═'*55}")
    print(f"  {'Phase':<22} {'SR':>6}  {'AvgTurns':>9}  {'DeathRate':>10}")
    print(f"  {'-'*52}")
    for lbl, st in [
        ("Alpha Train",      train_stats),
        ("Alpha Test",       test_a_stats),
        ("Beta Test",        test_b_stats),
        ("Gamma (X-credit)", test_g_stats),
    ]:
        succ = [s for s in st if s.success]
        SR   = len(succ) / len(st) if st else 0
        AT   = sum(s.turns for s in succ) / len(succ) if succ else float("inf")
        DR   = sum(s.deaths for s in st) / max(1, sum(s.turns for s in st))
        at_s = f"{AT:.0f}" if AT < float("inf") else "N/A"
        print(f"  {lbl:<22} {SR:>6.1%}  {at_s:>9}  {DR:>10.4f}")
    print(f"{'═'*55}\n")
    print("  Output images saved to each maze's Contexts folder.")
    print("  Learning curve saved to: learning_curve_alpha.png\n")


if __name__ == "__main__":
    main()
