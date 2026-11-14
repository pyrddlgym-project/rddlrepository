import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

ROBOT_COLOR = '#1565c0'
TREAD_COLOR = '#37474f'
LIGHT_COLOR = '#ffb300'
GRID_COLOR = '#b0bec5'
GOAL_COLOR = '#c8e6c9'
GOAL_EDGE_COLOR = '#2e7d32'
LOST_COLOR = '#c62828'
FIELD_COLORMAP = 'YlOrRd'

# wide grids use cells at least this large where they can, so that their names fit
MIN_CELL_SIZE = 0.45


class NavigationVisualizer(BaseViz):
    '''Draws the grid with each cell coloured by the chance of the robot disappearing
    on entering it, along with the goal and the robot.

    The POMDP only places the robot on the second step, and shows where it really is
    rather than only what is observed. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_cell_size=2.0,
                 max_size=(9.0, 5.5),
                 preferred_size=5.5,
                 min_width=5.6,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._robot_is_placed_later = 'second-step' in model.state_fluents

        # moving east and north follow the relations, whatever the objects are called
        self._xs = self._chain(model.type_to_objects['xpos'], 'EAST', 'WEST')
        self._ys = self._chain(model.type_to_objects['ypos'], 'NORTH', 'SOUTH')
        self._cells = [(col, row, x, y) for (col, x) in enumerate(self._xs)
                       for (row, y) in enumerate(self._ys)]
        self._goals = [(col, row) for (col, row, x, y) in self._cells
                       if self._nonfluent('GOAL', x, y)]

        num_x, num_y = len(self._xs), len(self._ys)
        cell_size = min(max_cell_size, preferred_size / max(num_x, num_y))
        cell_size = max(cell_size, min(MIN_CELL_SIZE, max_size[0] / num_x, max_size[1] / num_y))
        self._star_size = 0.45 * cell_size * 72

        # title above, then the grid, its coordinates and the color scale
        grid_width, grid_height = num_x * cell_size, num_y * cell_size
        bottom = 0.9
        width = max(grid_width + 0.8, min_width)
        height = bottom + grid_height + 0.45
        self._figure_size = (width, height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width), bottom / height,
             grid_width / width, grid_height / height])
        self.ax.set_xlim(0, num_x)
        self.ax.set_ylim(0, num_y)
        self.ax.set_aspect('equal')
        self._set_names(self.ax.set_xticks, self._xs, cell_size,
                        0.075 * max(len(x) for x in self._xs) + 0.05)
        self._set_names(self.ax.set_yticks, self._ys, cell_size, 0.16)
        self.ax.tick_params(length=0, labelsize=9, colors='#616161')
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        # the chances and goal never change, so they are only drawn once
        norm = Normalize(vmin=0.0, vmax=1.0)
        colormap = plt.get_cmap(FIELD_COLORMAP)
        for (col, row, x, y) in self._cells:
            chance = self._nonfluent('P', x, y)
            goal = (col, row) in self._goals
            color = colormap(norm(chance))
            self.ax.add_patch(patches.Rectangle(
                (col, row), 1, 1, facecolor=GOAL_COLOR if goal else color,
                edgecolor=GOAL_EDGE_COLOR if goal else GRID_COLOR,
                linewidth=2 if goal else 1, zorder=2 if goal else 1))

        bar_width = min(3.6, grid_width)
        bar_axis = self.fig.add_axes(
            [(width - bar_width) / (2 * width), 0.42 / height, bar_width / width, 0.12 / height])
        bar = self.fig.colorbar(ScalarMappable(norm=norm, cmap=colormap), cax=bar_axis,
                                orientation='horizontal')
        bar.set_label('chance of disappearing', fontsize=9, labelpad=1)
        bar.ax.tick_params(labelsize=8, colors='#616161')
        bar.outline.set_edgecolor(GRID_COLOR)

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

    def _set_names(self, set_ticks, names, cell_size, name_size):

        # names that would overlap are thinned out, and the rest stay where they belong
        step = math.ceil(name_size / cell_size)
        positions = np.arange(0, len(names), step)
        set_ticks(positions + 0.5, [names[i] for i in positions])

    def _draw_robot(self, col, row):

        # a robot on treads, with a body, a head with two eyes and an antenna
        def box(left, bottom, right, top, radius, color):
            return patches.FancyBboxPatch(
                (col + left, row + bottom), right - left, top - bottom,
                boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=color,
                edgecolor='none', zorder=6)

        parts = [box(0.22, 0.08, 0.78, 0.2, 0.06, TREAD_COLOR),
                 box(0.27, 0.22, 0.73, 0.5, 0.05, ROBOT_COLOR),
                 box(0.31, 0.53, 0.69, 0.8, 0.06, ROBOT_COLOR),
                 box(0.485, 0.8, 0.515, 0.87, 0.0, TREAD_COLOR),
                 patches.Circle((col + 0.5, row + 0.88), 0.04, facecolor=LIGHT_COLOR, zorder=6)]
        parts.extend(patches.Circle((col + x, row + 0.67), 0.05, facecolor='white', zorder=7)
                     for x in (0.41, 0.59))
        return [self.ax.add_patch(part) for part in parts]

    def _title(self, state, robots):
        if any(robot in self._goals for robot in robots):
            return 'Goal reached', GOAL_EDGE_COLOR
        if not robots:
            if self._robot_is_placed_later and not state['second-step']:
                return 'Robot not placed yet', 'black'
            return 'Robot lost', LOST_COLOR
        distance = min((abs(col - goal_col) + abs(row - goal_row)
                        for (col, row) in robots for (goal_col, goal_row) in self._goals),
                       default=None)
        if distance is None:
            return 'Robot moving', 'black'
        return f'{distance} cell{"s" if distance != 1 else ""} to goal', 'black'

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        artists = []
        robots = []
        for (col, row, x, y) in self._cells:
            if state[RDDLPlanningModel.ground_var('robot-at', [x, y])]:
                robots.append((col, row))
                artists.extend(self._draw_robot(col, row))

        # the robot stands in front of any star
        for (col, row) in self._goals:
            if (col, row) not in robots:
                artists.extend(self.ax.plot(
                    [col + 0.5], [row + 0.5], marker='*', color=GOAL_EDGE_COLOR,
                    markeredgecolor='none', markersize=self._star_size, zorder=3))

        title, color = self._title(state, robots)
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
