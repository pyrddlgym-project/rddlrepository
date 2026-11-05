from itertools import product

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class SudokuVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(5, 5.5),
                 wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._rows = model.type_to_objects['brow']
        self._cols = model.type_to_objects['bcol']
        self._cells = list(product(self._rows, self._cols))
        self._font_size = 170 / max(len(self._rows), len(self._cols))
        self._fixed = {(r, c) for (r, c) in self._cells if self._is_set('FIXED', r, c)}
        self._peers = self._conflict_pairs()

        self._last_board = None
        self._newest = None

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0.03, 0.03, 0.94, 0.87])
        self.ax.set_xlim(0, len(self._cols))
        self.ax.set_ylim(0, len(self._rows))
        self.ax.set_aspect('equal')
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self._draw_grid()

    def _is_set(self, var, *objects):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, objects)]

    def _conflict_pairs(self):

        # same rule as the reward: two cells share a box only if SHARED_R and
        # SHARED_C both hold in the order given, and so the board is marked as
        # solved exactly when the reward is paid
        pairs = set()
        for (r, c), (r2, c2) in product(self._cells, self._cells):
            if (r == r2) != (c == c2) or (r != r2 and c != c2
                                          and self._is_set('SHARED_R', r, r2)
                                          and self._is_set('SHARED_C', c, c2)):
                pairs.add(frozenset(((r, c), (r2, c2))))
        return [tuple(pair) for pair in pairs]

    def _block_edges(self, objects, var):

        # objects are in the same block if linked by a chain of shared pairs
        block = {obj: i for (i, obj) in enumerate(objects)}
        for (a, b) in product(objects, objects):
            if a != b and self._is_set(var, a, b):
                old = block[b]
                for obj in objects:
                    if block[obj] == old:
                        block[obj] = block[a]
        return [i for i in range(1, len(objects))
                if block[objects[i - 1]] != block[objects[i]]]

    def _cell_position(self, r, c):

        # the first row is drawn at the top of the board
        return self._cols.index(c), len(self._rows) - 1 - self._rows.index(r)

    def _draw_grid(self):
        nrows, ncols = len(self._rows), len(self._cols)

        # the grid and the fixed cells never change, so they are only drawn once
        for (r, c) in self._fixed:
            self.ax.add_patch(patches.Rectangle(
                self._cell_position(r, c), 1, 1, facecolor='#cfd8dc', zorder=0))
        for i in range(1, ncols):
            self.ax.axvline(i, color='#90a4ae', linewidth=1, zorder=2)
        for i in range(1, nrows):
            self.ax.axhline(i, color='#90a4ae', linewidth=1, zorder=2)
        for i in self._block_edges(self._cols, 'SHARED_C'):
            self.ax.axvline(i, color='black', linewidth=2.5, zorder=3)
        for i in self._block_edges(self._rows, 'SHARED_R'):
            self.ax.axhline(nrows - i, color='black', linewidth=2.5, zorder=3)
        for spine in self.ax.spines.values():
            spine.set_linewidth(2.5)

    def _update_newest(self, board):

        # rendering the same state again must not change the highlighted cell
        if board == self._last_board:
            return
        changed = []
        if self._last_board is not None:
            changed = [cell for cell in self._cells
                       if board[cell] != self._last_board[cell]]

        # several cells only change together when a new episode starts
        self._newest = None
        if len(changed) == 1 and board[changed[0]] > 0:
            self._newest = changed[0]
        self._last_board = board

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        board = {(r, c): int(state[RDDLPlanningModel.ground_var('board', [r, c])])
                 for (r, c) in self._cells}
        self._update_newest(board)

        # find cells in conflict
        conflict_pairs = [(a, b) for (a, b) in self._peers
                          if board[a] > 0 and board[a] == board[b]]
        conflicts = {cell for pair in conflict_pairs for cell in pair}

        # draw the numbers, conflicts and most recently filled cell
        artists = []
        for (cell, value) in board.items():
            x, y = self._cell_position(*cell)
            if cell in conflicts:
                artists.append(self.ax.add_patch(patches.Rectangle(
                    (x, y), 1, 1, facecolor='#ffcdd2', zorder=1)))
            if cell == self._newest:
                artists.append(self.ax.add_patch(patches.Rectangle(
                    (x + 0.04, y + 0.04), 0.92, 0.92, facecolor='none',
                    edgecolor='#ffa000', linewidth=2.5, zorder=4)))
            if value > 0:
                artists.append(self.ax.text(
                    x + 0.5, y + 0.5, str(value), ha='center', va='center',
                    fontsize=self._font_size, zorder=5, fontweight='bold',
                    color='black' if cell in self._fixed else '#1565c0'))

        # the board is solved when it is full and has no conflicts
        num_filled = sum(value > 0 for value in board.values())
        solved = num_filled == len(self._cells) and not conflict_pairs
        if solved:
            title = 'SOLVED'
        else:
            title = (f'{num_filled} / {len(self._cells)} filled, '
                     f'{len(conflict_pairs)} conflicts')
        self.ax.set_title(title, fontsize=13, fontweight='bold',
                          color='#2e7d32' if solved else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
