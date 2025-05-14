# Robot Path Planning Visualizer

A Python application that visualizes robot path planning algorithms in a grid-based environment.

![Robot Path Planning](robot.png)

## Description

This application demonstrates two popular pathfinding algorithms:

- **Breadth-First Search (BFS)**: Guarantees the shortest path in an unweighted graph but explores in all directions.
- **A\* Search**: Finds the optimal path more efficiently by using a heuristic to guide the search.

The application includes a graphical interface where you can set start and end points, create barriers, and watch as the robot navigates through the grid to find the best path.

## Features

- Interactive grid-based environment
- Two pathfinding algorithms: BFS and A\*
- Visualization of the exploration process
- Animated robot movement along the discovered path
- Random maze generation
- Easy-to-use interface with keyboard and mouse controls

## Requirements

- Python 3.x
- Pygame

## Installation

1. Make sure you have Python installed on your system
2. Install the required dependency:
   ```
   pip install pygame
   ```
3. Download the project files (robot.py and robot.png)

## Usage

Run the application:

```
python robot.py
```

### Controls

- **Left-click**: Place start point (first), end point (second), and barriers
- **Right-click**: Remove nodes
- **A key**: Run A\* algorithm
- **B key**: Run BFS algorithm
- **M key**: Generate a random maze
- **R key**: Reset the grid
- **ESC/Q key**: Quit the application

## How It Works

1. Set a start point (green) and an end point (red)
2. Create barriers (black) or generate a maze
3. Run either the A\* or BFS algorithm
4. Watch as the algorithm explores the grid (blue cells)
5. See the final path (yellow) and robot animation once the goal is reached

## License

This project is open-source and free to use for educational purposes.
