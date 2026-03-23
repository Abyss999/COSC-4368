from PIL import Image, ImageDraw # Added ImageDraw for hazard overlay.
import numpy as np
from collections import deque
from pathlib import Path # for renaming the file
from scipy import ndimage # for connected component analysis

# Image loaded here. change: Path("<file path>") to the maze image file.
input_path = Path("./MAZE_1.png")
img = Image.open(input_path).convert("RGB")
pixels = np.array(img)

height, width, _ = pixels.shape #gets the image dimensions
print("Image size:", width, "x", height)

# Constant values for hte grid so it can easily identify walls and free paths.
WALL = 1
FREE = 0

START_COLOR = (255, 200, 0) # Colors for the start and goal for the maze.
GOAL_COLOR  = (0, 255, 0)

# Creates a grid to represent the maze given.
grid = np.zeros((height, width), dtype=int)
start = None
goal = None

# Helps us determine the wall of the maze by checking the colors
# of the image from 0 to 255. A Threshold of 128 is used to determine
# what is a wall and what is a path to travel in. Like Curves/Levels in Photoshop
THRESHOLD = 128 

# Scans the image for the goal and start positions.
for x in range(width):
    if pixels[0, x][0] > THRESHOLD:
        start =(x, 0)
    if pixels[height-1, x][0] > THRESHOLD:
        goal = (x, height-1)
        
for y in range(height):
    for x in range(width):
        # If the Red channel is > 128, consider it a FREE path.
        if pixels[y, x][0] > THRESHOLD:
            grid[y, x] = FREE
        else:
            grid[y, x] = WALL
            
print(pixels[y, x])
print("Start:", start)
print("Goal:", goal)

if start is None or goal is None:
    raise ValueError("Start or Goal not found — check color values")

# BFS (Breadth First Search), used to search the whole grid for the
# fastest and best solution.
# This will also be the Movement Controller
def bfs(grid, start, goal):
    queue = deque([start])
    came_from = {start: None}
    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    while queue:
        x, y = queue.popleft()

        if (x, y) == goal:
            # For extra speed according ft. Gemini. Removed the "break" before "for dx, dy in directions:"
            # Reconstructs the path from the goal back to the start using the came_from dictionary.
            path = []
            current = goal
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if grid[ny, nx] == FREE and (nx, ny) not in came_from:
                    queue.append((nx, ny))
                    came_from[(nx, ny)] = (x, y)
    
    return []  # Returns an empty path if no solution is found.
    # And after testing the output speed, it is faster woah.
    
    # Used Claude to help fix the while loop since it stopped working. Had to switch the if and for loops.
            
    # Recontructing the path after applying BFS
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

# Drawing the line for the path
path = bfs(grid, start, goal)
print("Path found:", len(path))

solution_pixels = pixels.copy()
thickness = 2 # This will create a (2*2 + 1) = 5 pixel wide line (GPT help)

for (x, y) in path:
    for ty in range(-thickness, thickness + 1):
        for tx in range(-thickness, thickness + 1):
            nx, ny = x + tx, y + ty
            # Ensures we stay within the image's boundaries
            if 0 <= nx < width and 0 <= ny < height:
                solution_pixels[ny, nx] = [255, 0, 200] # Color for path
                
# Saving and outputting the Solution Image/Maze.
output_solved = input_path.with_name(input_path.stem + "_solved" + input_path.suffix)
solved_img = Image.fromarray(solution_pixels)
solved_img.save(output_solved)

print(f"Solved maze saved as {output_solved}")

# Adding the Hazard System

def find_hazard_centers(pixels, height, width, color_test_fn, min_cluster_size=20):
    # Scans image to match the pixels with the hazard color of the emojis/icons
    mask = np.zeros((height, width), dtype=bool)
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[y, x]
            if color_test_fn(r, g, b):
                mask[y, x] = True
                
    labeled, num_features = ndimage.label(mask)
    centers = []
    for i in range(1, num_features + 1):
        component = np.argwhere(labeled == i)
        if len(component) >= min_cluster_size:
            cy, cx = component.mean(axis=0).astype(int)
            centers.append((cx, cy))
    return centers

# Death Pits 🔥
DEATH_PIT_CELLS = find_hazard_centers(
    pixels, height, width,
    lambda r, g, b: r > 180 and 80 < g < 170 and b < 100
)

# Confusion Traps 😵‍💫
# Since this is a more complex emoji due to the face details,
# the coordinates will be hardcoded instead of using color detection.
CONFUSION_TRAP_CELLS = [
    (280, 39),
    (263, 297),
    (456, 632),
    (488, 121),
]


# Teleport Traps Section

# Green Teleport 🟢->✳️
GREEN_SOURCES = find_hazard_centers(
    pixels, height, width,
    lambda r, g, b: g > 180 and r < 150 and b < 180,
    min_cluster_size=30
) 

GREEN_SOURCES.sort() # Sorts by x coordinate

# Purple Teleport 🟣->🔯
PURPLE_SOURCES = find_hazard_centers(
    pixels, height, width,
    lambda r, g, b: r > 150 and r > 100 and r < 200 and g < 150,
    min_cluster_size=30
)

PURPLE_SOURCES.sort() # Sorts by x coordinate

# Yellow Teleport 🟡->✴️
YELLOW_SOURCES = find_hazard_centers(
    pixels, height, width,
    lambda r, g, b: r > 180 and g > 150 and b < 120,
    min_cluster_size=30
)
YELLOW_SOURCES.sort() # Sorts by x coordinate

# Builds teleport pairs
TELEPORT_PAIRS = {}
if len(GREEN_SOURCES) >= 2: TELEPORT_PAIRS[GREEN_SOURCES[0]] = GREEN_SOURCES[1]
if len(PURPLE_SOURCES) >= 2: TELEPORT_PAIRS[PURPLE_SOURCES[0]] = PURPLE_SOURCES[1]
if len(YELLOW_SOURCES) >= 2: TELEPORT_PAIRS[YELLOW_SOURCES[0]] = YELLOW_SOURCES[1]

# Prints the detected hazards for debugging purposes.
print("\nHazard Detection Summary:")
print(f"Death pits: 🔥:{len(DEATH_PIT_CELLS)}") # found -> {DEATH_PIT_CELLS}
print(f"Confusion traps: 😵:{len(CONFUSION_TRAP_CELLS)}")
print(f"Green teleport sources: 🟢:{len(GREEN_SOURCES)}")
print(f"Purple teleport sources: 🟣:{len(PURPLE_SOURCES)}")
print(f"Yellow teleport sources: 🟡:{len(YELLOW_SOURCES)}")
print(f"Teleport pairs: {TELEPORT_PAIRS}")

# Defines the path through BFS again
def near_cell(pos, cell_list, tolerance=8):
    px, py = pos
    for cx, cy in cell_list:
        if abs(px - cx) <= tolerance and abs(py - cy) <= tolerance:
            return True
    return False

def nearest_teleport_dest(pos, pairs, tolerance=8):
    px, py = pos
    for (sx, sy), (dx, dy) in pairs.items():
        if abs(px - sx) <= tolerance and abs(py - sy) <= tolerance:
            return (dx, dy)
    return None

# Agent States
agent_pos = start
death_count = 0
wall_hits = 0
confusion_log = 0
teleport_log = 0
confused_turns = 0
action_log = []

fire_rotations = {cell: 0 for cell in DEATH_PIT_CELLS}
rotation_labels = ["North", "East", "South", "West"]

print("\n" + "-"*50)
print("Hazard Maze")
print("-"*50)

for step_num, step in enumerate(path):
    for cell in DEATH_PIT_CELLS:
        fire_rotations[cell] = (fire_rotations[cell] + 1) % 4
        
    agent_pos = step
    
    if confused_turns > 0:
        confused_turns -= 1
        action_log.append(f"Step {step_num:>3}: Confused (inputs reversed)"
            f"Turns remaining: {confused_turns}"
        )
    
    if near_cell(agent_pos, DEATH_PIT_CELLS):
        death_count += 1
        rot_idx = 0
        for cell in DEATH_PIT_CELLS:
            if near_cell(agent_pos, [cell]):
                rot_idx = fire_rotations[cell]
                break
        msg = (f"Step {step_num:>3}: Fell into a Death Pit 🔥! Total deaths: {death_count}"
            f"Respawning at {start}")
        action_log.append(msg)
        agent_pos = start
        print(msg)
        continue
    
    dest = nearest_teleport_dest(agent_pos, TELEPORT_PAIRS)
    if dest is not None:
        teleport_log += 1
        msg = (f"Step {step_num:>3}: Teleported from {agent_pos} to {dest}!"
            f" Total teleports: {teleport_log}")
        action_log.append(msg)
        agent_pos = dest
        print(msg)
        chained = nearest_teleport_dest(agent_pos, TELEPORT_PAIRS)
        if chained is not None:
            teleport_log += 1
            msg = (f"Step {step_num:>3}: Chained teleport from {agent_pos} to {chained}!"
                f" Total teleports: {teleport_log}")
            action_log.append(msg)
            agent_pos = chained
            print(msg)
        continue
    
    if near_cell(agent_pos, CONFUSION_TRAP_CELLS):
        confusion_log += 1
        confused_turns = 2
        msg = (f"Step {step_num:>3}: Hit a Confusion Trap 😵! Total hits: {confusion_log}"
            f" Next 5 steps will be reversed.")
        action_log.append(msg)
        print(msg)
        continue

    if step_num % 50 == 0:
        action_log.append(f"Step {step_num:>3}: Agent at {agent_pos} (No Hazards)")

print("\n" + "-"*50)
print( "Maze Complete! Hazard maze Summary:")
print("\n")
print(f"Total Steps Taken: {len(path)}")
print(f"Total Deaths: {death_count}")
print(f"Total Teleports: {teleport_log}")
print(f"Total Confusion Hits: {confusion_log}")
print(f"Total Wall Hits: {wall_hits}")
print(f"Agent's Final Pos: {agent_pos}")
print("\n")

# Hazard Overlay image
hazard_img = Image.fromarray(solution_pixels).copy()
draw = ImageDraw.Draw(hazard_img)

for cx, cy in DEATH_PIT_CELLS:
    r = fire_rotations.get((cx, cy), 0)
    draw.ellipse([cx-6, cy-6, cx+6, cy+6],
    fill = (255,50,0),
    outline = (200,0,0),
    width =2)
    arrows = [(0, -4), (4,0), (0,4), (-4,0)]
    ax, ay = arrows[r]
    draw.line([cx, cy, cx+ax, cy+ay], fill=(255,255,0), width=2)
    
for cx, cy in CONFUSION_TRAP_CELLS:
    draw.rectangle([cx-6, cy-6, cx+6, cy+6],
    fill=(180,0,255),
    outline=(100,0,180),
    width=2)

pad_colors = [(0,200,0),(160,0,255),(230,200,0)]
for i, (src, dst) in enumerate(TELEPORT_PAIRS.items()):
    sx, sy = src
    dx, dy = dst
    c = pad_colors[i % len(pad_colors)]
    draw.ellipse([sx-5, sy-5, sx+5, sy+5], fill=c, outline=(255,255,255), width=1)
    draw.ellipse([dx-5, dy-5, dx+5, dy+5], fill=c, outline=(255,255,255), width=1)
    draw.line([sx, sy, dx, dy], fill=c, width=1)

output_hazard = input_path.with_name(input_path.stem + "_hazards" + input_path.suffix)
hazard_img.save(output_hazard)
print(f"Solved Hazard Maze saved as {output_hazard}")

# Log of Events
"""log_path = input_path.with_name(input_path.stem + "_hazard_log.txt")
with open(log_path, "w") as f:
    f.write("Hazard Maze Report:" + "\n")
    for line in action_log:
        f.write(line + "\n")
    f.write("\nSummary:" + "\n")
    f.write(f"DEATHS: {death_count}")
    f.write(f"TELEPORTS: {teleport_log}")
    f.write(f"CONFUSION HITS: {confusion_log}")
    f.write(f"WALL HITS: {wall_hits}\n")
    
print(f"\n" + "Hazard event log saved as {log_path}")"""""