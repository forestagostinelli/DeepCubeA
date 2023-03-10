# import environments, and cube3
import numpy as np
from typing import List, Dict, Tuple, Union
from cube3 import Cube3

cube_env = Cube3()

# Generate solved state
"""
a = cube_env.generate_goal_states(1)
print(a[0].colors)
print(cube_env.is_solved(a))
"""

# Generate tile representations after moving each face once
"""
moved_rep = []
for move_num in range(6):
    moved_rep.append(cube_env.next_state(a, move_num*2+1)[0][0])
    print(moved_rep[-1].colors)
    print(a[0].colors == moved_rep[-1].colors)
"""

# Get "rotate idxs", new and old. new idxs get set to values of old idxs.
"""
print(cube_env.rotate_idxs_new)
print(cube_env.rotate_idxs_old)
"""

num_to_colors = {
    0: "W",
    1: "Y",
    2: "B",
    3: "G",
    4: "O",
    5: "R",
}

colors = {
    "W": 0,
    "Y": 1,
    "B": 2,
    "G": 3,
    "O": 4,
    "R": 5,
}

moves = {
    0: "U",
    1: "D",
    2: "L",
    3: "R",
    4: "B",
    5: "F",
}

# Define corners as set of color tiles that lie in intersection of turning three faces
def get_corner_tile_rep(c: List[str]) -> List[int]:
    intersect_moves = [moves[colors[color]]+"1" for color in c]
    all_tiles = list(range(54))
    for move in intersect_moves:
        all_tiles = list(filter(lambda x : x in cube_env.rotate_idxs_old[move], all_tiles))
    return all_tiles

# True if tile is a corner
def check_corner(tile_num: int) -> bool:
    return tile_num % 9 in [0,2,6,8]

# Define edges as set of color tiles that lie in intersection of turning two faces and which fail the check_corner parity test
def get_edge_tile_rep(c: List[str]) -> List[int]:
    intersect_moves = [moves[colors[color]]+"1" for color in c]
    # Filter out all corner tiles
    all_tiles = list(filter(lambda x : not(check_corner(x)), list(range(54))))
    for move in intersect_moves:
        all_tiles = list(filter(lambda x : x in cube_env.rotate_idxs_old[move], all_tiles))
    return all_tiles

# Combinations of three colors that yield a valid corner
valid_corners = []
# Verify corner parity check
for color1 in range(len(colors)):
    for color2 in range(color1+1, len(colors)):
        for color3 in range(color2+1, len(colors)):
            color_combo = [num_to_colors[color1], num_to_colors[color2], num_to_colors[color3]]
            corner_tiles = get_corner_tile_rep(color_combo)
            # If there's any tiles in the intersection of these three operations, should be a (valid) corner
            if len(corner_tiles) > 0:
                valid_corners.append(color_combo)
                print(f"Color combo: {color_combo}, corner tile representation: {corner_tiles}")
            # Verify check_corner works as intended (should print nothing)
            for corner_tile in corner_tiles:
                if not(check_corner(corner_tile)):
                    print(f"Failed check_corner: {corner_tile}")

# Number of corners: 8
print(len(valid_corners))

# Combinations of two colors that yield a valid edge
valid_edges = []
for color1 in range(len(colors)):
    for color2 in range(color1+1, len(colors)):
        color_combo = [num_to_colors[color1], num_to_colors[color2]]
        edge_tiles = get_edge_tile_rep(color_combo)
        # If there's any tiles in the intersection of these two operations, should be a (valid) edge
        if len(edge_tiles) > 0:
            valid_edges.append(color_combo)
            print(f"Color combo: {color_combo}, edge tile representation: {edge_tiles}")
        # Verify check_corner works as intended (should print nothing)
        for edge_tile in edge_tiles:
            if check_corner(edge_tile):
                print(f"Failed check_corner: {edge_tile}")

# Number of edges: 12
print(len(valid_edges))
