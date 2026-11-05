import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

ZONE_COLOR = '#ffe0b2'
GRID_COLOR = '#cfd8dc'
CAMERA_COLOR = '#1565c0'
INTRUDER_COLOR = '#37474f'
UNDETECTED_COLOR = '#c62828'
DETECTED_COLOR = '#2e7d32'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class IntrudersVisualizer(BaseViz):
    '''Draws the grid with its danger zone, the cameras, and the intruders colored by
    how the reward treats them. Rows and columns follow the moves in TARGET-X and
    TARGET-Y. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 cell_size=0.8,
                 max_size=9.0,
                 min_width=5.2,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._cameras = model.type_to_objects['camera']
        self._intruders = model.type_to_objects['intruder']

        # moving right and down define the geometry of the grid
        self._xs = self._get_axis_order('TARGET-X', 'right', model.type_to_objects['x-pos'])
        self._ys = self._get_axis_order('TARGET-Y', 'down', model.type_to_objects['y-pos'])
        self._columns = {x: i for (i, x) in enumerate(self._xs)}
        self._rows = {y: len(self._ys) - 1 - i for (i, y) in enumerate(self._ys)}
        self._danger = {(x, y) for x in self._xs for y in self._ys if self._nonfluents[
            RDDLPlanningModel.ground_var('IS-DANGER-ZONE', [x, y])]}

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

        # the grid and its names never change, so they are only drawn once
        self._set_names(self.ax.set_xticks, self.ax.set_xticklabels, self._xs, cell_size)
        self._set_names(self.ax.set_yticks, self.ax.set_yticklabels, self._ys[::-1], left - 0.1)
        for i in range(1, num_x):
            self.ax.axvline(i, color=GRID_COLOR, linewidth=0.8, zorder=1)
        for i in range(1, num_y):
            self.ax.axhline(i, color=GRID_COLOR, linewidth=0.8, zorder=1)
        for (x, y) in self._danger:
            self.ax.add_patch(patches.Rectangle(
                (self._columns[x], self._rows[y]), 1, 1, facecolor=ZONE_COLOR,
                edgecolor='none', zorder=0))

        self.fig.legend(handles=[
            patches.Patch(facecolor=ZONE_COLOR, label='Danger zone'),
            patches.Patch(facecolor=CAMERA_COLOR, alpha=0.25, label='Camera'),
            Line2D([], [], marker='o', color='none', markerfacecolor=INTRUDER_COLOR,
                   markeredgecolor='none', label='Intruder'),
            Line2D([], [], marker='o', color='none', markerfacecolor=UNDETECTED_COLOR,
                   markeredgecolor='none', label='Undetected'),
            Line2D([], [], marker='o', color='none', markerfacecolor=DETECTED_COLOR,
                   markeredgecolor='none', label='Detected')
        ], loc='lower center', ncol=5, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _get_axis_order(self, fluent, direction, objects):

        # follow the moves from the object that no other move reaches: moves that do
        # not form a single line or loop have no clear order, so use the declared one
        following = {}
        for start in objects:
            targets = [end for end in objects if end != start and self._nonfluents[
                RDDLPlanningModel.ground_var(fluent, [direction, start, end])]]
            if len(targets) > 1:
                return objects
            if targets:
                following[start] = targets[0]
        first = [o for o in objects if o not in following.values()]
        order = [first[0] if len(first) == 1 else objects[0]]
        while order[-1] in following and following[order[-1]] not in order:
            order.append(following[order[-1]])
        return order if len(order) == len(objects) else objects

    def _set_names(self, set_ticks, set_labels, names, width):

        # names shrink to the room they have, and are left out if unreadable
        font_size = min(9, 72 * width / (0.62 * max(len(n) for n in names)),
                        72 * 0.8 * self._cell_size)
        if font_size < MIN_FONT_SIZE:
            set_ticks([])
            return
        set_ticks([i + 0.5 for i in range(len(names))])
        set_labels(names, fontsize=font_size)

    def _get_cells(self, state, fluent, objects):
        cells = {}
        for o in objects:
            for x in self._xs:
                for y in self._ys:
                    if state[RDDLPlanningModel.ground_var(fluent, [o, x, y])]:
                        cells.setdefault((x, y), []).append(o)
        return cells

    def _draw_cameras(self, cell, names):
        x, y = cell
        center = (self._columns[x] + 0.5, self._rows[y] + 0.5)
        artists = [self.ax.add_patch(plt.Circle(
            center, 0.44, facecolor=CAMERA_COLOR, alpha=0.25, edgecolor='none', zorder=2))]

        # several cameras in one cell cover the same area, so they are counted instead
        font_size = min(9, 72 * 0.3 * self._cell_size)
        if len(names) > 1 and font_size >= MIN_FONT_SIZE:
            artists.append(self.ax.text(
                center[0] + 0.3, center[1] + 0.3, f'x{len(names)}', ha='center', va='center',
                fontsize=font_size, color=CAMERA_COLOR, fontweight='bold', zorder=5))
        return artists

    def _draw_intruders(self, cell, names, covered):
        x, y = cell
        if cell not in self._danger:
            color = INTRUDER_COLOR
        else:
            color = DETECTED_COLOR if covered else UNDETECTED_COLOR

        # intruders in the same cell are spread around its center so all can be seen
        count = len(names)
        spread = 0.22 if count > 1 else 0.0
        radius = min(0.11, 0.8 * spread * math.sin(math.pi / count)) if count > 1 else 0.11
        artists = []
        for i in range(count):
            angle = 2 * math.pi * i / count
            artists.append(self.ax.add_patch(plt.Circle(
                (self._columns[x] + 0.5 + spread * math.sin(angle),
                 self._rows[y] + 0.5 + spread * math.cos(angle)),
                radius, facecolor=color, edgecolor='white', linewidth=0.8, zorder=4)))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        cameras = self._get_cells(state, 'camera-at', self._cameras)
        intruders = self._get_cells(state, 'intruder-at', self._intruders)

        artists = []
        for (cell, names) in cameras.items():
            artists.extend(self._draw_cameras(cell, names))
        for (cell, names) in intruders.items():
            artists.extend(self._draw_intruders(cell, names, cell in cameras))

        # same rule as the reward
        undetected = sum(len(names) for (cell, names) in intruders.items()
                         if cell in self._danger and cell not in cameras)
        self.ax.set_title(f'Undetected in danger zone: {undetected}', fontsize=13,
                          fontweight='bold', color=UNDETECTED_COLOR if undetected else DETECTED_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
