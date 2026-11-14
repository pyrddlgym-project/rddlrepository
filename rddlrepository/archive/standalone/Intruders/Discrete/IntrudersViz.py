import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

# the grid shows the same pictures as the continuous version
from ..Continuous.IntrudersViz import (
    FENCE_COLOR, NIGHT_COLOR, PAVING_COLOR, PAVING_LINE_COLOR, SAFE_COLOR, SAFE_DARK_COLOR,
    SAFE_TITLE_COLOR, STEALING_COLOR, _burglar, _draw_legend, _glow, _lantern
)

GRID_COLOR = '#3f5046'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class IntrudersVisualizer(BaseViz):
    '''Draws the grid as grounds at night: the danger zone is paved behind a fence with a
    safe in it, each cell with a camera is lit by a lantern, and each intruder is a
    burglar, who carries off a sack while in the danger zone out of the light and raises
    their hands when caught in it, as the reward treats them. Rows and columns follow the
    moves in TARGET-X and TARGET-Y. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 cell_size=0.8,
                 max_size=9.0,
                 min_width=5.6,
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
        self._figure_size = (fig_width, fig_height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(fig_width - num_x * cell_size + left - right) / 2 / fig_width, bottom / fig_height,
             num_x * cell_size / fig_width, num_y * cell_size / fig_height])
        self.ax.set_xlim(0, num_x)
        self.ax.set_ylim(0, num_y)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(NIGHT_COLOR)
        self.ax.tick_params(length=0, colors='#616161')
        for spine in self.ax.spines.values():
            spine.set_color(NIGHT_COLOR)

        # the grid, the danger zone and the names never change, so they are only drawn once
        self._set_names(self.ax.set_xticks, self.ax.set_xticklabels, self._xs, cell_size)
        self._set_names(self.ax.set_yticks, self.ax.set_yticklabels, self._ys[::-1], left - 0.1)
        for i in range(1, num_x):
            self.ax.axvline(i, color=GRID_COLOR, linewidth=0.8, zorder=0.4)
        for i in range(1, num_y):
            self.ax.axhline(i, color=GRID_COLOR, linewidth=0.8, zorder=0.4)
        self._draw_danger_zone()
        _draw_legend(self.fig, self._figure_size, lambda ax, x, y: _lantern(ax, x, y, 0.26))

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

    def _draw_danger_zone(self):

        # each cell of the zone is paved in quarters, a fence runs along the outside of the
        # zone, and a safe stands in the cell nearest its middle
        cells = {(self._columns[x], self._rows[y]) for (x, y) in self._danger}
        for (col, row) in cells:
            self.ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=PAVING_COLOR,
                                                edgecolor='none', zorder=0.5))
            for k in (0.25, 0.5, 0.75):
                self.ax.plot([col + k, col + k], [row, row + 1], color=PAVING_LINE_COLOR,
                             linewidth=0.6, zorder=0.6)
                self.ax.plot([col, col + 1], [row + k, row + k], color=PAVING_LINE_COLOR,
                             linewidth=0.6, zorder=0.6)
            for (dc, dr, xs, ys) in ((-1, 0, [0, 0], [0, 1]), (1, 0, [1, 1], [0, 1]),
                                     (0, -1, [0, 1], [0, 0]), (0, 1, [0, 1], [1, 1])):
                if (col + dc, row + dr) not in cells:
                    self.ax.plot([col + a for a in xs], [row + b for b in ys], color=FENCE_COLOR,
                                 linewidth=2, linestyle=(0, (3, 2)), zorder=0.7)
        if cells:
            (mx, my) = np.mean([(c + 0.5, r + 0.5) for (c, r) in cells], axis=0)
            (col, row) = min(cells, key=lambda cell: (cell[0] + 0.5 - mx) ** 2 + (cell[1] + 0.5 - my) ** 2)
            (x, y, s) = (col + 0.5, row + 0.5, 0.36)
            self.ax.add_patch(patches.FancyBboxPatch(
                (x - s / 2, y - s / 2), s, s, boxstyle=f'round,pad=0,rounding_size={0.08 * s}',
                facecolor=SAFE_COLOR, edgecolor=SAFE_DARK_COLOR, linewidth=1, zorder=0.8))
            self.ax.add_patch(plt.Circle((x, y), 0.18 * s, facecolor='#cfd8dc',
                                         edgecolor=SAFE_DARK_COLOR, linewidth=0.8, zorder=0.9))

    def _get_cells(self, state, fluent, objects):
        cells = {}
        for o in objects:
            for x in self._xs:
                for y in self._ys:
                    if state[RDDLPlanningModel.ground_var(fluent, [o, x, y])]:
                        cells.setdefault((x, y), []).append(o)
        return cells

    def _draw_cameras(self, cell, names):

        # the cell is lit, with one lantern for each camera in it hanging along its top
        x, y = cell
        (cx, cy) = (self._columns[x] + 0.5, self._rows[y] + 0.5)
        artists = _glow(self.ax, cx, cy, 0.46)
        count = len(names)
        size = min(0.26, 0.8 / count)
        for i in range(count):
            artists += _lantern(self.ax, cx + (i - (count - 1) / 2) * 0.55 * size,
                                cy + 0.5 - 0.7 * size, size)
        return artists

    def _draw_intruders(self, cell, names, lit):
        x, y = cell
        if cell not in self._danger:
            pose = 'walking'
        else:
            pose = 'caught' if lit else 'stealing'

        # intruders in the same cell stand side by side, below any lanterns
        count = len(names)
        pitch = min(0.3, 0.9 / count)
        size = min(0.56 if lit else 0.72, 1.7 * pitch)
        middle = self._rows[y] + (0.36 if lit else 0.5)
        artists = []
        for i in range(count):
            artists += _burglar(self.ax, self._columns[x] + 0.5 + (i - (count - 1) / 2) * pitch,
                                middle, size, pose)
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
        stealing = sum(len(names) for (cell, names) in intruders.items()
                       if cell in self._danger and cell not in cameras)
        self.ax.set_title('Nothing stolen' if stealing == 0 else
                          f'{stealing} burglar{"" if stealing == 1 else "s"} stealing',
                          fontsize=13, fontweight='bold',
                          color=STEALING_COLOR if stealing else SAFE_TITLE_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
