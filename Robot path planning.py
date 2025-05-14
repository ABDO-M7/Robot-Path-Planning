import pygame
import random
from queue import Queue, PriorityQueue

# Constants
GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE

# Animation delays (ms)
EXPLORE_DELAY = 100
PATH_DELAY = 80
ROBOT_MOVE_DELAY = 120

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (180, 20, 20)
YELLOW = (255, 227, 26)
BLUE = (39, 84, 138)
GREEN = (95, 139, 76)
GRAY = (100, 100, 100)

# Initialize pygame
pygame.init()
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Robot Path Planning")

# Load and scale robot image
try:
    ROBOT_IMG = pygame.image.load("BMO.png")
    ROBOT_IMG = pygame.transform.scale(ROBOT_IMG, (CELL_SIZE, CELL_SIZE))
except pygame.error:
    # Fallback if image not found
    ROBOT_IMG = None

class Node:
    """Represents a single cell in the grid"""
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * CELL_SIZE
        self.y = row * CELL_SIZE
        self.color = WHITE
        self.neighbors = []

    def get_position(self):
        """Returns the grid position as (col, row)"""
        return self.col, self.row

    def is_empty(self):
        return self.color == WHITE

    def is_visited(self):
        return self.color == BLUE

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == GREEN

    def is_end(self):
        return self.color == RED

    def is_path(self):
        return self.color == YELLOW

    def reset(self):
        self.color = WHITE

    def set_start(self):
        self.color = GREEN

    def set_visited(self):
        self.color = BLUE

    def set_barrier(self):
        self.color = BLACK

    def set_end(self):
        self.color = RED

    def set_path(self):
        self.color = YELLOW

    def draw(self, window):
        """Draw this node on the window"""
        pygame.draw.rect(window, self.color, (self.x, self.y, CELL_SIZE, CELL_SIZE))

    def update_neighbors(self, grid):
        """Find all traversable neighboring nodes"""
        self.neighbors = []
        # Check all four directions (right, down, left, up)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            new_row, new_col = self.row + dr, self.col + dc
            # Check if the new position is valid and not a barrier
            if (0 <= new_row < GRID_SIZE and 
                0 <= new_col < GRID_SIZE and 
                not grid[new_row][new_col].is_barrier()):
                self.neighbors.append(grid[new_row][new_col])


class Grid:
    """Manages the grid of nodes"""
    def __init__(self):
        # Create the grid
        self.grid = [[Node(row, col) for col in range(GRID_SIZE)] for row in range(GRID_SIZE)]
        self.start_node = None
        self.end_node = None

    def draw(self, window):
        """Draw the grid and all nodes"""
        window.fill(WHITE)
        
        # Draw all nodes
        for row in self.grid:
            for node in row:
                node.draw(window)
                
        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            # Horizontal lines
            pygame.draw.line(window, GRAY, (0, i * CELL_SIZE), 
                            (WINDOW_WIDTH, i * CELL_SIZE))
            # Vertical lines
            pygame.draw.line(window, GRAY, (i * CELL_SIZE, 0), 
                            (i * CELL_SIZE, WINDOW_HEIGHT))
        
        pygame.display.update()

    def reset_path(self):
        """Reset all nodes except barriers, start, and end"""
        for row in self.grid:
            for node in row:
                if not (node.is_barrier() or node == self.start_node or node == self.end_node):
                    node.reset()

    def full_reset(self):
        """Reset the entire grid"""
        self.start_node = None
        self.end_node = None
        for row in self.grid:
            for node in row:
                node.reset()
        self.update_all_neighbors()

    def get_clicked_node(self, position):
        """Get the node at a given pixel position"""
        x, y = position
        row, col = y // CELL_SIZE, x // CELL_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return self.grid[row][col]
        return None

    def update_all_neighbors(self):
        """Update neighbors for all nodes"""
        for row in self.grid:
            for node in row:
                node.update_neighbors(self.grid)

    def reconstruct_path(self, came_from, current_node):
        """Reconstruct and visualize the path"""
        path = []
        while current_node in came_from:
            current_node = came_from[current_node]
            if current_node != self.start_node:
                current_node.set_path()
                path.append(current_node)
                
                # Visualize the path reconstruction
                current_node.draw(WINDOW)
                pygame.display.update(pygame.Rect(current_node.x, current_node.y, CELL_SIZE, CELL_SIZE))
                pygame.time.delay(PATH_DELAY)
                
        path.reverse()
        return path

    def generate_maze(self):
        """Generate a random maze using DFS with backtracking"""
        self.full_reset()
        
        # First make all cells walls
        for row in self.grid:
            for node in row:
                node.set_barrier()
        
        # Choose a random starting point
        start_row = random.randrange(GRID_SIZE)
        start_col = random.randrange(GRID_SIZE)
        current = self.grid[start_row][start_col]
        current.reset()
        
        # Start with the randomly chosen cell
        stack = [(start_row, start_col)]
        
        # Keep going until we've visited all cells
        while stack:
            current_row, current_col = stack[-1]
            
            # Find unvisited neighbors (with a wall of distance 2)
            neighbors = []
            for dr, dc in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
                new_row, new_col = current_row + dr, current_col + dc
                if (0 <= new_row < GRID_SIZE and 
                    0 <= new_col < GRID_SIZE and 
                    self.grid[new_row][new_col].is_barrier()):
                    neighbors.append((new_row, new_col, dr // 2, dc // 2))
            
            # If we have unvisited neighbors
            if neighbors:
                # Choose a random neighbor
                next_row, next_col, wall_row, wall_col = random.choice(neighbors)
                
                # Remove the walls
                self.grid[next_row][next_col].reset()
                self.grid[current_row + wall_row][current_col + wall_col].reset()
                
                # Push the new cell to the stack
                stack.append((next_row, next_col))
            else:
                # Backtrack
                stack.pop()
        
        # Ensure an entrance and exit
        self.grid[0][1].reset()
        self.grid[GRID_SIZE - 1][GRID_SIZE - 2].reset()
        
        # Randomly place start and end nodes
        empty_cells = [node for row in self.grid for node in row if node.is_empty()]
        self.start_node = random.choice(empty_cells)
        empty_cells.remove(self.start_node)
        self.end_node = random.choice(empty_cells)
        
        self.start_node.set_start()
        self.end_node.set_end()
        self.update_all_neighbors()


class PathFinder:
    """Implements different pathfinding algorithms"""
    def __init__(self, grid):
        self.grid = grid

    def show_message(self, text):
        """Display a popup message on the screen"""
        popup_width, popup_height = 300, 100
        popup_x = (WINDOW_WIDTH - popup_width) // 2
        popup_y = (WINDOW_HEIGHT - popup_height) // 2
        
        # Create the message
        font = pygame.font.SysFont('Arial', 24)
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        
        # Draw the popup
        pygame.draw.rect(WINDOW, WHITE, (popup_x, popup_y, popup_width, popup_height))
        pygame.draw.rect(WINDOW, BLACK, (popup_x, popup_y, popup_width, popup_height), 3)
        WINDOW.blit(text_surface, text_rect)
        
        pygame.display.update()
        pygame.time.delay(1000)

    def animate_robot(self, path):
        """Animate the robot moving along the path"""
        if not ROBOT_IMG:
            return
            
        prev_node = None
        for node in path:
            # Redraw the previous node to remove the robot
            if prev_node:
                prev_node.draw(WINDOW)
                pygame.display.update(pygame.Rect(prev_node.x, prev_node.y, CELL_SIZE, CELL_SIZE))
            
            # Draw the robot at the current node
            WINDOW.blit(ROBOT_IMG, (node.x, node.y))
            pygame.display.update(pygame.Rect(node.x, node.y, CELL_SIZE, CELL_SIZE))
            
            prev_node = node
            pygame.time.delay(ROBOT_MOVE_DELAY)

    def breadth_first_search(self):
        """Perform BFS algorithm to find a path from start to end"""
        start, end = self.grid.start_node, self.grid.end_node
        if not start or not end:
            self.show_message("Set start and end first!")
            return
        
        # Reset and redraw the grid
        self.grid.reset_path()
        self.grid.draw(WINDOW)
        
        # Initialize BFS
        queue = Queue()
        queue.put(start)
        visited = {start}
        came_from = {}
        
        # BFS loop
        while not queue.empty():
            current = queue.get()
            
            # Check if we reached the end
            if current == end:
                break
            
            # Process all neighbors
            for neighbor in current.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    
                    # Visualize exploration (except for end node)
                    if neighbor != end:
                        neighbor.set_visited()
                        neighbor.draw(WINDOW)
                        pygame.display.update(pygame.Rect(neighbor.x, neighbor.y, CELL_SIZE, CELL_SIZE))
                        pygame.time.delay(EXPLORE_DELAY)
                    
                    queue.put(neighbor)
        
        # Check if we found a path
        if end not in came_from:
            self.show_message("No Path Found!")
            return
        
        # Reconstruct and animate the path
        path = self.grid.reconstruct_path(came_from, end)
        self.animate_robot(path)
        self.show_message("Goal Reached!")

    def a_star(self):
        """Perform A* algorithm to find the optimal path from start to end"""
        start, end = self.grid.start_node, self.grid.end_node
        if not start or not end:
            self.show_message("Set start and end first!")
            return
        
        # Reset and redraw the grid
        self.grid.reset_path()
        self.grid.draw(WINDOW)
        
        # Heuristic function (Manhattan distance)
        def heuristic(node1, node2):
            x1, y1 = node1.get_position()
            x2, y2 = node2.get_position()
            return abs(x1 - x2) + abs(y1 - y2)
        
        # Initialize A*
        open_set = PriorityQueue()
        open_set.put((0, 0, start))  # (f_score, counter, node)
        open_set_hash = {start}  # For fast lookup
        
        came_from = {}
        
        # g_score: cost from start to current node
        g_score = {node: float('inf') for row in self.grid.grid for node in row}
        g_score[start] = 0
        
        # f_score: g_score + heuristic
        f_score = {node: float('inf') for row in self.grid.grid for node in row}
        f_score[start] = heuristic(start, end)
        
        counter = 0  # For breaking ties in the priority queue
        
        # A* main loop
        while not open_set.empty():
            # Get the node with the lowest f_score
            current = open_set.get()[2]
            open_set_hash.remove(current)
            
            # Check if we reached the end
            if current == end:
                path = self.grid.reconstruct_path(came_from, current)
                self.animate_robot(path)
                self.show_message("Goal Reached!")
                return
            
            # Process all neighbors
            for neighbor in current.neighbors:
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + 1
                
                # If this path is better
                if tentative_g_score < g_score[neighbor]:
                    # Update path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    
                    # Add to open set if not already there
                    if neighbor not in open_set_hash:
                        counter += 1
                        open_set.put((f_score[neighbor], counter, neighbor))
                        open_set_hash.add(neighbor)
                        
                        # Visualize (except for end node)
                        if neighbor != end:
                            neighbor.set_visited()
                            neighbor.draw(WINDOW)
                            pygame.display.update(pygame.Rect(neighbor.x, neighbor.y, CELL_SIZE, CELL_SIZE))
                            pygame.time.delay(EXPLORE_DELAY)
        
        # If we get here, no path was found
        self.show_message("No Path Found!")


class App:
    """Main application class"""
    def __init__(self):
        self.grid = Grid()
        self.pathfinder = PathFinder(self.grid)
        self.running = True
        
        # Display instructions
        self.show_instructions()

    def show_instructions(self):
        """Display instructions for using the application"""
        instructions = [
            "Left Click: Place start, end, barriers",
            "Right Click: Remove nodes",
            "A: Run A* algorithm",
            "B: Run BFS algorithm",
            "M: Generate maze",
            "R: Reset grid",
            "ESC/Q: Quit"
        ]
        
        # Semi-transparent background
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        WINDOW.blit(overlay, (0, 0))
        
        # Instructions text
        font = pygame.font.SysFont('Arial', 20)
        y_pos = 50
        for line in instructions:
            text = font.render(line, True, WHITE)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, y_pos))
            WINDOW.blit(text, text_rect)
            y_pos += 40
        
        # Start message
        start_font = pygame.font.SysFont('Arial', 26)
        start_text = start_font.render("Press any key to start", True, YELLOW)
        start_rect = start_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100))
        WINDOW.blit(start_text, start_rect)
        
        pygame.display.update()
        
        # Wait for a key press
        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False

    def run(self):
        """Main application loop"""
        while self.running:
            # Draw the grid
            self.grid.draw(WINDOW)
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                # Left mouse button: Place nodes
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    node = self.grid.get_clicked_node(pos)
                    if node:
                        if not self.grid.start_node and node != self.grid.end_node:
                            self.grid.start_node = node
                            node.set_start()
                        elif not self.grid.end_node and node != self.grid.start_node:
                            self.grid.end_node = node
                            node.set_end()
                        elif node != self.grid.start_node and node != self.grid.end_node:
                            node.set_barrier()
                
                # Right mouse button: Remove nodes
                elif pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    node = self.grid.get_clicked_node(pos)
                    if node:
                        node.reset()
                        if node == self.grid.start_node:
                            self.grid.start_node = None
                        elif node == self.grid.end_node:
                            self.grid.end_node = None
                
                # Keyboard controls
                if event.type == pygame.KEYDOWN:
                    # A: Run A* algorithm
                    if event.key == pygame.K_a:
                        self.grid.update_all_neighbors()
                        self.pathfinder.a_star()
                    
                    # B: Run BFS algorithm
                    elif event.key == pygame.K_b:
                        self.grid.update_all_neighbors()
                        self.pathfinder.breadth_first_search()
                    
                    # R: Reset grid
                    elif event.key == pygame.K_r:
                        self.grid.full_reset()
                    
                    # M: Generate maze
                    elif event.key == pygame.K_m:
                        self.grid.generate_maze()
                    
                    # Q/ESC: Quit
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        self.running = False
        
        pygame.quit()


if __name__ == "__main__":
    App().run()

