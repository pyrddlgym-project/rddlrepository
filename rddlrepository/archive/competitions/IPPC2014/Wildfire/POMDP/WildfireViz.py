import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

FUEL_COLOR = '#c8e6c9'
NO_FUEL_COLOR = '#cfd8dc'
LOST_COLOR = '#ffcdd2'
GRID_COLOR = 'white'
TARGET_COLOR = '#2e7d32'
LOST_MARK_COLOR = '#c62828'
FLAME_COLOR = '#ef6c00'
FLAME_CORE_COLOR = '#ffca28'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class WildfireVisualizer(BaseViz):
    '''Draws the grid in the order the positions are declared, with the first y position
    at the top, showing which cells have fuel, which are burning and which are targets.
    Targets that have burned stay marked, as the reward penalizes them on every step.
    The POMDP shows the true fire rather than only the noisy observations of it. The drawing
    only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 cell_size=0.9,
                 max_size=9.0,
                 min_width=5.2,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._xs = model.type_to_objects['x_pos']
        self._ys = model.type_to_objects['y_pos']
        self._cells = [(col, len(self._ys) - 1 - row, x, y)
                       for (col, x) in enumerate(self._xs) for (row, y) in enumerate(self._ys)]
        self._targets = {(x, y) for (_, _, x, y) in self._cells if nonfluents[
            RDDLPlanningModel.ground_var('TARGET', [x, y])]}

        # large grids use smaller cells to fit on screen, and the figure leaves room for
        # the names of the rows and columns, the title and the legend
        num_x, num_y = len(self._xs), len(self._ys)
        cell_size = min(cell_size, max_size / max(num_x, num_y))
        self._cell_size = cell_size
        left, right, bottom, top = 0.65, 0.2, 0.8, 0.5
        fig_width = max(num_x * cell_size + left + right, min_width)
        fig_height = num_y * cell_size + bottom + top

        self.fig = plt.figure(figsize=(fig_width, fig_height))
        self.ax = self.fig.add_axes(
            [(fig_width - num_x * cell_size + left - right) / 2 / fig_width, bottom / fig_height,
             num_x * cell_size / fig_width, num_y * cell_size / fig_height])
        self.ax.set_xlim(0, num_x)
        self.ax.set_ylim(0, num_y)
        self.ax.set_aspect('equal')
        self.ax.tick_params(length=0, colors='#616161')
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')

        # the names, grid and flame shape never change, so they are only computed once
        self._set_names(self.ax.set_xticks, self.ax.set_xticklabels, self._xs, cell_size)
        self._set_names(self.ax.set_yticks, self.ax.set_yticklabels, self._ys[::-1], left - 0.1)
        for i in range(1, num_x):
            self.ax.axvline(i, color=GRID_COLOR, linewidth=1, zorder=2)
        for i in range(1, num_y):
            self.ax.axhline(i, color=GRID_COLOR, linewidth=1, zorder=2)
        angles = np.linspace(0, 2 * math.pi, 40)
        self._flame = np.column_stack(
            (np.sin(angles) * np.sin(angles / 2), np.cos(angles)))
        self._marker_size = min(12, 16 * cell_size)

        self.fig.legend(handles=[
            patches.Patch(facecolor=FUEL_COLOR, label='Has fuel'),
            patches.Patch(facecolor=NO_FUEL_COLOR, label='No fuel'),
            patches.Patch(facecolor=FLAME_COLOR, label='Burning'),
            Line2D([], [], marker='*', color='none', markerfacecolor=TARGET_COLOR,
                   markeredgecolor='none', markersize=11, label='Target'),
            Line2D([], [], marker='*', color='none', markerfacecolor=LOST_MARK_COLOR,
                   markeredgecolor='none', markersize=11, label='Target lost')
        ], loc='lower center', ncol=5, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _set_names(self, set_ticks, set_labels, names, width):

        # names shrink to the room they have, and are left out if unreadable
        font_size = min(9, 72 * width / (0.62 * max(len(n) for n in names)),
                        72 * 0.8 * self._cell_size)
        if font_size < MIN_FONT_SIZE:
            set_ticks([])
            return
        set_ticks([i + 0.5 for i in range(len(names))])
        set_labels(names, fontsize=font_size)

    def _draw_flame(self, col, row):
        artists = []
        for (scale, drop, color, zorder) in ((1.0, 0.0, FLAME_COLOR, 3),
                                             (0.55, 0.14, FLAME_CORE_COLOR, 4)):
            points = self._flame * [0.3 * scale, 0.36 * scale]
            points = points + [col + 0.5, row + 0.48 - drop]
            artists.append(self.ax.add_patch(plt.Polygon(
                points, closed=True, facecolor=color, edgecolor='none', zorder=zorder)))
        return artists

    def _draw_cell(self, col, row, x, y, burning, no_fuel):
        target = (x, y) in self._targets

        # same rule as the reward: a target that burned has no fuel, as it cannot be cut out
        lost = target and (burning or no_fuel)
        if lost:
            color = LOST_COLOR
        else:
            color = NO_FUEL_COLOR if no_fuel else FUEL_COLOR
        artists = [self.ax.add_patch(patches.Rectangle(
            (col, row), 1, 1, facecolor=color, edgecolor='none', zorder=0))]
        if burning:
            artists.extend(self._draw_flame(col, row))
        if target:
            artists.extend(self.ax.plot(
                col + 0.17, row + 0.83, marker='*', markersize=self._marker_size,
                color=LOST_MARK_COLOR if lost else TARGET_COLOR, markeredgecolor='white',
                markeredgewidth=0.5, linestyle='none', zorder=5))
        return artists, lost

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        artists = []
        num_burning = 0
        num_lost = 0
        for (col, row, x, y) in self._cells:
            burning = bool(state[RDDLPlanningModel.ground_var('burning', [x, y])])
            no_fuel = bool(state[RDDLPlanningModel.ground_var('out-of-fuel', [x, y])])
            cell_artists, lost = self._draw_cell(col, row, x, y, burning, no_fuel)
            artists.extend(cell_artists)
            num_burning += burning
            num_lost += lost

        self.ax.set_title(f'{num_burning} burning, {num_lost} / {len(self._targets)} targets lost',
                          fontsize=13, fontweight='bold',
                          color=LOST_MARK_COLOR if num_lost else (TARGET_COLOR if not num_burning else 'black'))

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
