from PIL import Image
import numpy as np
from collections import deque
from pathlib import Path # for renaming the file

# Image loaded here. change: Path("<file path>") to the maze image file.
input_path = Path("./Maze/MAZE_0.png")
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

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if grid[ny, nx] == FREE and (nx, ny) not in came_from:
                    queue.append((nx, ny))
                    came_from[(nx, ny)] = (x, y)
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
                return [] # Returns an empty path if no solution is found.
                # And after testing the output speed, it is faster woah.
            
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