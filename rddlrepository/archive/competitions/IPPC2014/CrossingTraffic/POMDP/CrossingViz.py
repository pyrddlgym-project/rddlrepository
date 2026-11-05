import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

ROBOT_COLOR = '#1565c0'
GRID_COLOR = '#b0bec5'
ROAD_COLOR = '#cfd8dc'
SAFE_COLOR = '#fafafa'
GOAL_COLOR = '#c8e6c9'
GOAL_EDGE_COLOR = '#2e7d32'
CAR_COLOR = '#ffe0b2'
CAR_EDGE_COLOR = '#ef6c00'
WINDOW_COLOR = '#37474f'
HIT_COLOR = '#c62828'


class CrossingVisualizer(BaseViz):
    '''Draws the road with the obstacles moving left along it, the safe rows, the goal
    and the robot, which is marked when an obstacle hits it and is gone afterwards.

    The POMDP shows the true position of every obstacle, rather than only the arrivals
    that the robot observes. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_cell_size=2.0,
                 max_size=5.5,
                 min_width=5.6,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)

        # moving east and north follow the relations, whatever the objects are called
        self._xs = self._chain(model.type_to_objects['xpos'], 'EAST', 'WEST')
        self._ys = self._chain(model.type_to_objects['ypos'], 'NORTH', 'SOUTH')
        self._cells = [(col, row, x, y) for (col, x) in enumerate(self._xs)
                       for (row, y) in enumerate(self._ys)]
        self._goals = [(col, row) for (col, row, x, y) in self._cells
                       if self._nonfluent('GOAL', x, y)]

        # title above, then the grid, its coordinates, the legend and the caption
        cell_size = min(max_cell_size, max_size / max(len(self._xs), len(self._ys)))
        grid_width = len(self._xs) * cell_size
        grid_height = len(self._ys) * cell_size
        bottom = 0.95
        width = max(grid_width + 0.8, min_width)
        height = bottom + grid_height + 0.45
        self._figure_size = (width, height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width), bottom / height,
             grid_width / width, grid_height / height])
        self.ax.set_xlim(0, len(self._xs))
        self.ax.set_ylim(0, len(self._ys))
        self.ax.set_aspect('equal')
        self.ax.set_xticks(np.arange(len(self._xs)) + 0.5, self._xs)
        self.ax.set_yticks(np.arange(len(self._ys)) + 0.5, self._ys)
        self.ax.tick_params(length=0, labelsize=9, colors='#616161')
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        # the road and goal never change, so they are only drawn once
        for (col, row, x, y) in self._cells:
            safe = self._nonfluent('MIN-YPOS', y) or self._nonfluent('MAX-YPOS', y)
            self.ax.add_patch(patches.Rectangle(
                (col, row), 1, 1, facecolor=SAFE_COLOR if safe else ROAD_COLOR,
                edgecolor=GRID_COLOR, zorder=1))
        for (col, row) in self._goals:
            self.ax.add_patch(patches.Rectangle(
                (col, row), 1, 1, facecolor=GOAL_COLOR, edgecolor=GOAL_EDGE_COLOR,
                linewidth=2, zorder=2))
        self._star_size = 0.45 * cell_size * 72

        self.fig.legend(handles=[
            patches.Patch(facecolor=ROAD_COLOR, edgecolor=GRID_COLOR, label='Road'),
            patches.Patch(facecolor=SAFE_COLOR, edgecolor=GRID_COLOR, label='Safe row'),
            patches.Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_EDGE_COLOR, label='Goal'),
            patches.Patch(facecolor=CAR_COLOR, edgecolor=CAR_EDGE_COLOR, label='Obstacle'),
            Line2D([], [], marker='o', color='none', markerfacecolor=ROBOT_COLOR,
                   markeredgecolor='white', markersize=9, label='Robot')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / height), ncol=5,
           frameon=False, fontsize=8.5, columnspacing=1.0, handlelength=1.4)
        rate = self._nonfluents['INPUT-RATE']
        self.fig.text(0.5, 0.1 / height, f'Obstacles move left and arrive with probability {rate:g}',
                      ha='center', va='bottom', fontsize=9, color='#37474f')

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _chain(self, objects, forward, backward):

        # follow the relation from the objects nothing leads to, and any object left
        # over because of a cycle or missing relation keeps its declared position
        following = {}
        for a in objects:
            for b in objects:
                if a != b and (self._nonfluent(forward, a, b) or self._nonfluent(backward, b, a)):
                    following.setdefault(a, b)
        starts = [a for a in objects if a not in following.values()]
        order = []
        for start in starts + list(objects):
            current = start
            while current not in order:
                order.append(current)
                current = following.get(current, current)
        return order

    def _draw_obstacle(self, col, row):

        # the window is at the front of the car, which shows the direction of travel
        return [
            self.ax.add_patch(patches.FancyBboxPatch(
                (col + 0.12, row + 0.26), 0.76, 0.48,
                boxstyle='round,pad=0,rounding_size=0.1',
                facecolor=CAR_COLOR, edgecolor=CAR_EDGE_COLOR, linewidth=2, zorder=4)),
            self.ax.add_patch(patches.FancyBboxPatch(
                (col + 0.2, row + 0.34), 0.16, 0.32,
                boxstyle='round,pad=0,rounding_size=0.04',
                facecolor=WINDOW_COLOR, edgecolor='none', zorder=5))
        ]

    def _draw_robot(self, col, row):
        return [
            self.ax.add_patch(patches.Wedge(
                (col + 0.5, row + 0.14), 0.3, 0, 180, facecolor=ROBOT_COLOR, zorder=6)),
            self.ax.add_patch(patches.Circle(
                (col + 0.5, row + 0.62), 0.17, facecolor=ROBOT_COLOR, zorder=6))
        ]

    def _title(self, robots, hit):
        if any(robot in self._goals for robot in robots):
            return 'Goal reached', GOAL_EDGE_COLOR
        if not robots:
            return 'Robot lost', HIT_COLOR
        if hit:
            return 'Robot hit', HIT_COLOR
        distance = min((abs(col - goal_col) + abs(row - goal_row)
                        for (col, row) in robots for (goal_col, goal_row) in self._goals),
                       default=None)
        if distance is None:
            return 'Robot crossing', 'black'
        return f'{distance} cell{"s" if distance != 1 else ""} to goal', 'black'

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        artists = []
        robots, hit = [], False
        for (col, row, x, y) in self._cells:
            obstacle = state[RDDLPlanningModel.ground_var('obstacle-at', [x, y])]
            if obstacle:
                artists.extend(self._draw_obstacle(col, row))
            if state[RDDLPlanningModel.ground_var('robot-at', [x, y])]:
                robots.append((col, row))
                artists.extend(self._draw_robot(col, row))

                # a robot sharing a cell with an obstacle disappears on the next step
                if obstacle and (col, row) not in self._goals:
                    hit = True
                    artists.append(self.ax.add_patch(patches.Circle(
                        (col + 0.5, row + 0.5), 0.46, facecolor='none',
                        edgecolor=HIT_COLOR, linewidth=3, zorder=7)))

        # the star would only stick out from behind a robot standing on the goal
        for (col, row) in self._goals:
            if (col, row) not in robots:
                artists.extend(self.ax.plot(
                    [col + 0.5], [row + 0.5], marker='*', color=GOAL_EDGE_COLOR,
                    markeredgecolor='none', markersize=self._star_size, zorder=3))

        title, color = self._title(robots, hit)
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
