import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f5f7fa'
FRAME = '#9e9e9e'
HIDDEN = '#c0c0c0'
OPEN = '#e0e0e0'
LIGHT_EDGE = '#ffffff'
DARK_EDGE = '#7b7b7b'
GRID = '#a8a8a8'
EXPLODED = '#e53935'
MINE = '#212121'
TEXT = '#37474f'
WON = '#2e7d32'
LOST = '#c62828'

# colors of the numbers 1 to 8, as in the classic game
NUMBER_COLORS = ['#1976d2', '#388e3c', '#d32f2f', '#283593', '#8d2a1f', '#00838f', '#212121', '#757575']

# the largest size of the board on the screen and the smallest size of a cell (inches)
MAX_BOARD = (7.0, 5.5)
MIN_CELL = 0.06


class MinesweeperVisualizer(BaseViz):
    '''Draws the board as in the classic game: hidden cells are raised tiles, revealed cells are flat and show
    the number of mines next to them. The mines under the hidden tiles are drawn faintly, and a mine that was
    revealed is drawn on red.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nf = model.ground_vars_with_values(model.non_fluents)
        self._width = max(int(nf['WIDTH']), 1)
        self._height = max(int(nf['HEIGHT']), 1)
        self._mines = int(nf['MINES'])
        self._cells = []
        for x in model.type_to_objects['xpos']:
            for y in model.type_to_objects['ypos']:
                col = int(nf[RDDLPlanningModel.ground_var('X', [x])])
                row = int(nf[RDDLPlanningModel.ground_var('Y', [y])])
                if 0 <= col < self._width and 0 <= row < self._height:
                    self._cells.append((row, col, RDDLPlanningModel.ground_var('mine', [x, y]),
                                        RDDLPlanningModel.ground_var('revealed', [x, y])))

        # the same scale on both axes, with a frame of 0.15 cells
        cell = max(min(MAX_BOARD[0] / self._width, MAX_BOARD[1] / self._height), MIN_CELL)
        self._points = cell * 72                    # size of a cell in points
        frame = 0.15
        board_w, board_h = cell * (self._width + 2 * frame), cell * (self._height + 2 * frame)
        fig_w, fig_h = max(board_w, 4.5) + 0.4, board_h + 1.2
        self.fig = plt.figure(figsize=(fig_w, fig_h))
        self.fig.patch.set_facecolor(BACKGROUND)
        self.ax = self.fig.add_axes([(fig_w - board_w) / 2 / fig_w, 0.6 / fig_h, board_w / fig_w, board_h / fig_h])
        self.ax.set_xlim(-frame, self._width + frame)
        self.ax.set_ylim(self._height + frame, -frame)
        self.ax.axis('off')
        self.ax.add_patch(plt.Rectangle((-frame, -frame), self._width + 2 * frame, self._height + 2 * frame,
                                        facecolor=FRAME, edgecolor='none', zorder=0))
        self._title = self.fig.text(0.5, 1 - 0.12 / fig_h, '', ha='center', va='top',
                                    fontsize=13, fontweight='bold', color=TEXT)
        self._info = self.fig.text(0.5, 0.12 / fig_h, '', ha='center', va='bottom', fontsize=10, color=TEXT)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def _draw_mines(self, cells, faint):
        if not cells:
            return []
        rows = np.array([r for r, _ in cells]) + 0.5
        cols = np.array([c for _, c in cells]) + 0.5
        alpha = 0.3 if faint else 1.0
        size = (0.38 * self._points) ** 2
        artists = [self.ax.scatter(cols, rows, s=size, marker='o', color=MINE, alpha=alpha, linewidths=0, zorder=5)]
        spike = min(max(0.06 * self._points, 0.5), 2.5)
        for marker in ('+', 'x'):
            artists.append(self.ax.scatter(cols, rows, s=(0.6 * self._points) ** 2, marker=marker, color=MINE,
                                           alpha=alpha, linewidths=spike, zorder=5))
        if not faint and self._points >= 8:
            artists.append(self.ax.scatter(cols - 0.1, rows - 0.1, s=(0.1 * self._points) ** 2, marker='o',
                                           color='white', alpha=0.9, linewidths=0, zorder=6))
        return artists

    def render(self, state):
        h, w = self._height, self._width
        mine = np.zeros((h, w), bool)
        revealed = np.zeros((h, w), bool)
        for row, col, m, r in self._cells:
            mine[row, col] = bool(state[m])
            revealed[row, col] = bool(state[r])
        placed, lost = bool(state['placed']), bool(state['lost'])
        padded = np.pad(mine.astype(int), 1)
        near = sum(padded[1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w]
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
        exploded = revealed & mine

        artists = []

        # the cells
        colors = np.empty((h, w, 3))
        colors[:] = plt.matplotlib.colors.to_rgb(HIDDEN)
        colors[revealed] = plt.matplotlib.colors.to_rgb(OPEN)
        colors[exploded] = plt.matplotlib.colors.to_rgb(EXPLODED)
        artists.append(self.ax.imshow(colors, extent=(0, w, h, 0), interpolation='nearest', zorder=1))

        # thin lines between the cells, and a bevel on the hidden tiles
        grid = [[(c, 0), (c, h)] for c in range(w + 1)] + [[(0, r), (w, r)] for r in range(h + 1)]
        artists.append(self.ax.add_collection(LineCollection(
            grid, colors=GRID, linewidths=max(0.04 * self._points, 0.3), zorder=2)))
        bevel = min(max(0.08 * self._points, 0.4), 3.5)        # width of the bevel (points)
        e = 0.5 * bevel / self._points                          # its offset from the edge of the cell (cells)
        light, dark = [], []
        for row, col in zip(*np.nonzero(~revealed)):
            light += [[(col + e, row + 1 - e), (col + e, row + e), (col + 1 - e, row + e)]]
            dark += [[(col + e, row + 1 - e), (col + 1 - e, row + 1 - e), (col + 1 - e, row + e)]]
        if light:
            artists.append(self.ax.add_collection(LineCollection(light, colors=LIGHT_EDGE, linewidths=bevel,
                                                                 capstyle='butt', zorder=3)))
            artists.append(self.ax.add_collection(LineCollection(dark, colors=DARK_EDGE, linewidths=bevel,
                                                                 capstyle='butt', zorder=3)))

        # the numbers on the revealed safe cells
        if self._points >= 5:
            for row, col in zip(*np.nonzero(revealed & ~mine & (near > 0))):
                n = int(near[row, col])
                artists.append(self.ax.text(col + 0.5, row + 0.53, str(n), ha='center', va='center',
                                            fontsize=0.6 * self._points, fontweight='bold',
                                            color=NUMBER_COLORS[min(n, 8) - 1], zorder=4))

        # the mines
        artists += self._draw_mines(list(zip(*np.nonzero(mine & ~revealed))), faint=True)
        artists += self._draw_mines(list(zip(*np.nonzero(exploded))), faint=False)

        safe = w * h - self._mines
        found = int((revealed & ~mine).sum())
        if lost and exploded.any():
            title, color = 'Hit a mine', LOST
        elif lost:
            title, color = 'Revealed a cell twice', LOST
        elif placed and found >= safe:
            title, color = 'Board cleared', WON
        elif not placed:
            title, color = 'New board: the mines are placed on the first step', TEXT
        else:
            title, color = f'Revealed {found} of {safe} safe cells', TEXT
        self._title.set_text(title)
        self._title.set_color(color)
        self._info.set_text(f'{w} x {h} cells, {self._mines} mines')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
