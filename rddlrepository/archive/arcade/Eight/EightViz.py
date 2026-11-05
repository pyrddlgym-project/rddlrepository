import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class EightVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(8, 4.5),
                 wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._xs = model.type_to_objects['x-pos']
        self._ys = model.type_to_objects['y-pos']
        self._squares = model.type_to_objects['square']
        self._font_size = 72 / max(len(self._xs), len(self._ys))

        # the location that each square must reach
        self._targets = {}
        for s in self._squares:
            for x in self._xs:
                for y in self._ys:
                    key = RDDLPlanningModel.ground_var('TARGET', [s, x, y])
                    if self._nonfluents[key]:
                        self._targets[s] = (x, y)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_subplot(1, 2, 1)
        self._setup_board(self.ax)

        # the target arrangement never changes, so it is only drawn once
        target_ax = self.fig.add_subplot(1, 2, 2)
        self._setup_board(target_ax)
        target_ax.set_title('Target', fontsize=14, fontweight='bold')
        for (s, cell) in self._targets.items():
            self._draw_tile(target_ax, s, cell, '#90a4ae')

    def _setup_board(self, ax):
        ax.set_xlim(0, len(self._xs))
        ax.set_ylim(0, len(self._ys))
        ax.set_aspect('equal')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_facecolor('#37474f')

    def _draw_tile(self, ax, square, cell, color):

        # the first y-pos object is the top row of the board
        col = self._xs.index(cell[0])
        row = len(self._ys) - 1 - self._ys.index(cell[1])
        tile = ax.add_patch(patches.FancyBboxPatch(
            (col + 0.05, row + 0.05), 0.9, 0.9,
            boxstyle='round,pad=0,rounding_size=0.08',
            facecolor=color, edgecolor='black', linewidth=1.5))
        label = ax.text(col + 0.5, row + 0.5, square, ha='center', va='center',
                        fontsize=self._font_size * min(1, 4 / len(square)),
                        fontweight='bold')
        return [tile, label]

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):

        # draw the squares, marking the ones already at their target
        artists = []
        num_correct = 0
        for s in self._squares:
            for x in self._xs:
                for y in self._ys:
                    if state[RDDLPlanningModel.ground_var('at', [s, x, y])]:
                        correct = self._targets.get(s) == (x, y)
                        num_correct += correct
                        artists.extend(self._draw_tile(
                            self.ax, s, (x, y), '#81c784' if correct else '#eceff1'))

        # squares without a target do not need to be in any particular place
        solved = all(
            state[RDDLPlanningModel.ground_var('at', [s, *cell])]
            for (s, cell) in self._targets.items())
        title = 'SOLVED' if solved else f'{num_correct} / {len(self._targets)} in place'
        self.ax.set_title(title, fontsize=14, fontweight='bold',
                          color='#2e7d32' if solved else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
