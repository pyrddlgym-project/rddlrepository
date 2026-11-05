import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class SokobanVisualizer(BaseViz):
    '''Draws the walls, storage areas, boxes and player, with the boxes on a storage
    area in green. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 cell_size=0.8,
                 max_size=9,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)

        # IX and IY define the geometry of the grid, and moving up increases IY
        self._xs = sorted(model.type_to_objects['x-pos'], key=lambda x:
                          self._nonfluents[RDDLPlanningModel.ground_var('IX', [x])])
        self._ys = sorted(model.type_to_objects['y-pos'], key=lambda y:
                          self._nonfluents[RDDLPlanningModel.ground_var('IY', [y])])
        self._cells = [(col, row, x, y) for (col, x) in enumerate(self._xs)
                       for (row, y) in enumerate(self._ys)]

        # large grids use smaller cells to fit on screen, and there is always room
        # above the grid for the title, even if the grid is narrow
        cell_size = min(cell_size, max_size / max(len(self._xs), len(self._ys)))
        grid_size = (len(self._xs) * cell_size, len(self._ys) * cell_size)
        self._figure_size = (max(grid_size[0], 4.8), grid_size[1] + 0.95)
        width, height = self._figure_size
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([(width - grid_size[0]) / (2 * width), 0.45 / height,
                                     grid_size[0] / width, grid_size[1] / height])
        self.ax.set_xlim(0, len(self._xs))
        self.ax.set_ylim(0, len(self._ys))
        self.ax.set_aspect('equal')
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_facecolor('#fafafa')
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')

        # walls and storage areas never change, so they are only drawn once
        for (col, row, x, y) in self._cells:
            if self._nonfluents[RDDLPlanningModel.ground_var('WALL', [x, y])]:
                self.ax.add_patch(patches.Rectangle(
                    (col, row), 1, 1, facecolor='#37474f', edgecolor='#37474f'))
            if self._is_storage(x, y):
                self.ax.add_patch(patches.Circle(
                    (col + 0.5, row + 0.5), 0.2, facecolor='none',
                    edgecolor='#2e7d32', linewidth=2, linestyle='--'))

        self.fig.legend(handles=[
            patches.Patch(facecolor='#37474f', label='Wall'),
            patches.Patch(facecolor='none', edgecolor='#2e7d32', linestyle='--', label='Storage'),
            patches.Patch(facecolor='#ffe0b2', edgecolor='#ef6c00', label='Box'),
            patches.Patch(facecolor='#c8e6c9', edgecolor='#2e7d32', label='Stored'),
            Line2D([], [], marker='o', color='none', markerfacecolor='#1565c0',
                   markeredgecolor='none', label='Player')
        ], loc='lower center', ncol=5, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _is_storage(self, x, y):
        return self._nonfluents[RDDLPlanningModel.ground_var('STORAGE', [x, y])]

    def _draw_box(self, col, row, stored):
        color = '#c8e6c9' if stored else '#ffe0b2'
        edge = '#2e7d32' if stored else '#ef6c00'
        artists = [self.ax.add_patch(patches.Rectangle(
            (col + 0.08, row + 0.08), 0.84, 0.84,
            facecolor=color, edgecolor=edge, linewidth=2))]
        artists.extend(self.ax.plot(
            [col + 0.08, col + 0.92], [row + 0.08, row + 0.92], color=edge, linewidth=1.5))
        artists.extend(self.ax.plot(
            [col + 0.08, col + 0.92], [row + 0.92, row + 0.08], color=edge, linewidth=1.5))
        return artists

    def _draw_man(self, col, row):
        return [
            self.ax.add_patch(patches.Wedge(
                (col + 0.5, row + 0.12), 0.3, 0, 180, facecolor='#1565c0', zorder=3)),
            self.ax.add_patch(patches.Circle(
                (col + 0.5, row + 0.62), 0.17, facecolor='#1565c0', zorder=3))
        ]

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):

        # draw the boxes and the man
        artists = []
        num_boxes, num_stored = 0, 0
        for (col, row, x, y) in self._cells:
            if state[RDDLPlanningModel.ground_var('box', [x, y])]:
                stored = bool(self._is_storage(x, y))
                num_boxes += 1
                num_stored += stored
                artists.extend(self._draw_box(col, row, stored))
            if state[RDDLPlanningModel.ground_var('man', [x, y])]:
                artists.extend(self._draw_man(col, row))

        # the problem is solved when every box is on a storage area
        solved = num_stored == num_boxes
        title = 'All stored' if solved else f'{num_stored} / {num_boxes} stored'
        self.ax.set_title(title, fontsize=13, fontweight='bold',
                          color='#2e7d32' if solved else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
