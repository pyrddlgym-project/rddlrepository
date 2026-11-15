import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

TRAY_COLOR = '#37474f'
EMPTY_COLOR = '#546e7a'
DARK_TEXT_COLOR = '#263238'
OVER_COLOR = '#c62828'

# tiles by exponent, from 2 up: light teal to indigo to magenta, and black beyond
TILE_COLORS = ['#e0f2f1', '#b2dfdb', '#80cbc4', '#4db6ac', '#26a69a', '#00897b', '#5c6bc0',
               '#3f51b5', '#303f9f', '#7b1fa2', '#c2185b']
BEYOND_COLOR = '#212121'
DARK_TEXT_UP_TO = 3

# sizes in inches: a cell, which shrinks on large boards, and the room around the tray
MAX_CELL = 1.0
MAX_BOARD = 6.4
MARGIN = 0.25
TOP_MARGIN = 0.45


class Game2048Visualizer(BaseViz):
    '''Draws the board as a tray of cells, with each tile in a colour of its own and its
    value written on it. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._xs = {x: int(nonfluents[RDDLPlanningModel.ground_var('X', [x])])
                    for x in model.type_to_objects['xpos']}
        self._ys = {y: int(nonfluents[RDDLPlanningModel.ground_var('Y', [y])])
                    for y in model.type_to_objects['ypos']}
        (columns, rows) = (max(self._xs.values(), default=0) + 1, max(self._ys.values(), default=0) + 1)
        self._cell = cell = min(MAX_CELL, MAX_BOARD / max(columns, rows))

        (width, height) = (columns * cell + 2 * MARGIN, rows * cell + MARGIN + TOP_MARGIN)
        self.fig = plt.figure(figsize=(width, height))
        self.fig.patch.set_facecolor('white')
        self.ax = self.fig.add_axes([0, MARGIN / height, 1, rows * cell / height])
        self.ax.set_xlim(-MARGIN, columns * cell + MARGIN)
        self.ax.set_ylim(0, rows * cell)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # the tray and the empty cells never change, and each cell has a tile to show or hide
        pad = 0.06 * cell
        self.ax.add_patch(patches.FancyBboxPatch(
            (0, 0), columns * cell, rows * cell, boxstyle=f'round,pad=0,rounding_size={2 * pad}',
            facecolor=TRAY_COLOR, edgecolor='none', zorder=0))
        self._tiles = {}
        for (x, col) in self._xs.items():
            for (y, row) in self._ys.items():
                (left, bottom, size) = (col * cell + pad, (rows - 1 - row) * cell + pad, cell - 2 * pad)
                box = {'boxstyle': f'round,pad=0,rounding_size={1.2 * pad}', 'edgecolor': 'none'}
                self.ax.add_patch(patches.FancyBboxPatch((left, bottom), size, size, facecolor=EMPTY_COLOR,
                                                         zorder=1, **box))
                tile = self.ax.add_patch(patches.FancyBboxPatch((left, bottom), size, size, zorder=2, **box))
                label = self.ax.text(left + size / 2, bottom + size / 2, '', ha='center', va='center',
                                     fontweight='bold', zorder=3)
                self._tiles[(x, y)] = (tile, label)

    def _show(self, tile, label, exponent):
        if exponent <= 0:
            tile.set_visible(False)
            label.set_visible(False)
            return
        tile.set_visible(True)
        tile.set_facecolor(TILE_COLORS[exponent - 1] if exponent <= len(TILE_COLORS) else BEYOND_COLOR)

        # the value shrinks with its number of digits to fit on the tile
        text = str(2 ** exponent)
        size = min(72 * 0.4 * self._cell, 72 * 0.7 * self._cell / (0.62 * len(text)))
        label.set_visible(True)
        label.set_text(text)
        label.set_fontsize(size)
        label.set_color(DARK_TEXT_COLOR if exponent <= DARK_TEXT_UP_TO else 'white')

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        board = {}
        for ((x, y), (tile, label)) in self._tiles.items():
            board[(x, y)] = int(state[RDDLPlanningModel.ground_var('tile', [x, y])])
            self._show(tile, label, board[(x, y)])

        # same rule as the termination condition: a full board where no two equal tiles touch
        cells = {(self._xs[x], self._ys[y]): value for ((x, y), value) in board.items()}
        stuck = all(cells.values()) and not any(
            cells.get((col + 1, row)) == value or cells.get((col, row + 1)) == value
            for ((col, row), value) in cells.items())
        self.ax.set_title('No moves left' if stuck else '', fontsize=13, fontweight='bold', color=OVER_COLOR)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
