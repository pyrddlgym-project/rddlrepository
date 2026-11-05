import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class TetrisVisualizer(BaseViz):
    '''Draws the board and the piece that is about to be placed.

    Cells filled since the previous render are highlighted. The state does not
    say which row was cleared, so when rows are cleared the cells that moved down
    into an empty position are highlighted as well.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 cell_size=0.5,
                 max_size=9,
                 panel_width=2.2,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)

        # WIDTH and HEIGHT define the geometry of the board, with HEIGHT going up
        self._xs = sorted(model.type_to_objects['board-x'], key=lambda x:
                          self._nonfluents[RDDLPlanningModel.ground_var('WIDTH', [x])])
        self._ys = sorted(model.type_to_objects['board-y'], key=lambda y:
                          self._nonfluents[RDDLPlanningModel.ground_var('HEIGHT', [y])])
        self._cells = [(col, row, x, y) for (col, x) in enumerate(self._xs)
                       for (row, y) in enumerate(self._ys)]
        self._shape_cells = {s: self._get_shape_cells(s)
                             for s in model.type_to_objects['shape-type']}

        self._last_filled = set()
        self._newest = set()

        # large boards use smaller cells to fit on screen
        cell_size = min(cell_size, max_size / max(len(self._xs), len(self._ys)))

        # leave room above the board for the title, and beside it for the piece
        board_size = (len(self._xs) * cell_size, len(self._ys) * cell_size)
        self._figure_size = (board_size[0] + panel_width, max(board_size[1], 2.5) + 0.6)
        width, height = self._figure_size
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [0.1 / width, 0.1 / height, board_size[0] / width, board_size[1] / height])
        self.ax.set_xlim(0, len(self._xs))
        self.ax.set_ylim(0, len(self._ys))
        self.ax.set_aspect('equal')
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_facecolor('#fafafa')
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')
        for i in range(1, len(self._xs)):
            self.ax.axvline(i, color='#cfd8dc', linewidth=1, zorder=0)
        for i in range(1, len(self._ys)):
            self.ax.axhline(i, color='#cfd8dc', linewidth=1, zorder=0)

        # the panel showing the piece about to be placed
        self._panel = self.fig.add_axes(
            [(board_size[0] + 0.3) / width, 0.1 / height,
             (panel_width - 0.4) / width, (panel_width - 0.4) / height])

        # the panel fits a piece of the largest size the shape grid allows, plus padding
        self._panel_size = 1 + max(len(model.type_to_objects['shape-x']),
                                   len(model.type_to_objects['shape-y']))
        self._panel.set_xlim(0, self._panel_size)
        self._panel.set_ylim(0, self._panel_size)
        self._panel.set_aspect('equal')
        self._panel.get_xaxis().set_ticks([])
        self._panel.get_yaxis().set_ticks([])
        self._panel.set_facecolor('#eceff1')
        for spine in self._panel.spines.values():
            spine.set_color('#b0bec5')

    def _get_shape_cells(self, shape):

        # the piece is shown in its first rotation, as the action can choose any
        rotation = self._model.type_to_objects['shape-rot'][0]
        cells = []
        for i in self._model.type_to_objects['shape-x']:
            for j in self._model.type_to_objects['shape-y']:
                key = RDDLPlanningModel.ground_var('SHAPES', [shape, rotation, i, j])
                if self._nonfluents[key]:
                    cells.append(
                        (self._nonfluents[RDDLPlanningModel.ground_var('OFFSET-X', [i])],
                         self._nonfluents[RDDLPlanningModel.ground_var('OFFSET-Y', [j])]))
        return cells

    def _draw_cell(self, ax, x, y, color, size=1.0):
        return ax.add_patch(patches.Rectangle(
            (x + 0.03 * size, y + 0.03 * size), 0.94 * size, 0.94 * size,
            facecolor=color, edgecolor='white', linewidth=1))

    def _draw_piece(self, state):
        artists = []
        for (shape, cells) in self._shape_cells.items():
            if not state[RDDLPlanningModel.ground_var('current-shape-type', [shape])]:
                continue
            self._panel.set_title(f'Next piece: {shape}', fontsize=min(10, 200 / len(f'Next piece: {shape}')))
            if not cells:
                continue

            # center the piece, shrinking it if it does not fit in the panel
            min_x, max_x = min(c[0] for c in cells), max(c[0] for c in cells)
            min_y, max_y = min(c[1] for c in cells), max(c[1] for c in cells)
            size = min(1.0, self._panel_size / (max(max_x - min_x, max_y - min_y) + 1))
            left = (self._panel_size - size * (max_x - min_x + 1)) / 2
            bottom = (self._panel_size - size * (max_y - min_y + 1)) / 2
            for (x, y) in cells:
                artists.append(self._draw_cell(
                    self._panel, left + size * (x - min_x), bottom + size * (y - min_y),
                    '#1565c0', size))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        filled = {(col, row) for (col, row, x, y) in self._cells
                  if state[RDDLPlanningModel.ground_var('filled', [x, y])]}

        # rendering the same state again must not change the highlighted cells
        if filled != self._last_filled:
            self._newest = filled - self._last_filled
            self._last_filled = filled

        # draw the board and the piece about to be placed
        artists = [self._draw_cell(self.ax, col, row,
                                   '#ff8f00' if (col, row) in self._newest else '#1565c0')
                   for (col, row) in filled]
        artists.extend(self._draw_piece(state))

        game_over = state['game-over']
        self.fig.suptitle('Game over' if game_over else f'{len(filled)} / {len(self._cells)} cells filled',
                          fontsize=13, fontweight='bold', color='#c62828' if game_over else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
