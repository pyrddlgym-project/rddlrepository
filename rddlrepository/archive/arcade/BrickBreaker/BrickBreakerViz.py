import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f5f7fa'
COURT = '#1f2a36'
WALL = '#90a4ae'
WALL_EDGE = '#546e7a'
BRICK_COLORS = ['#c0603a', '#b5532f', '#cc7048', '#a94c2c']
MORTAR = '#e0d6c8'
BROKEN_EDGE = '#3a4a5c'
BALL = '#fff3c4'
BALL_EDGE = '#f9a825'
PADDLE = '#4fc3f7'
PADDLE_EDGE = '#0277bd'
LOST = '#e53935'
TEXT = '#37474f'

# the largest size of the court on the screen, the smallest size of a cell, and the smallest
# size of the ball and the paddle so that they can be seen on a large court (inches)
MAX_COURT = (6.5, 5.5)
MIN_CELL = 0.03
MIN_BALL = 0.06
MIN_PADDLE = 0.08


class BrickBreakerVisualizer(BaseViz):
    '''Draws the court from the front: the side walls and the ceiling, the bricks that are left
    (the broken ones as dark outlines), the paddle on the bottom row, and the ball with an arrow
    in the direction it is going. A lost ball is drawn in red below the paddle row.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nf = model.ground_vars_with_values(model.non_fluents)
        self._width = max(int(nf['WIDTH']), 1)
        self._height = max(int(nf['HEIGHT']), 1)
        self._bricks = list(model.type_to_objects['brick'])
        self._brick_at = [(int(nf[RDDLPlanningModel.ground_var('X', [b])]),
                           int(nf[RDDLPlanningModel.ground_var('Y', [b])])) for b in self._bricks]

        # the same scale on both axes, with a wall a third of a cell thick
        cell = min(MAX_COURT[0] / (self._width + 0.7), MAX_COURT[1] / (self._height + 1.2))
        cell = max(cell, MIN_CELL)
        self._ball_radius = min(max(0.28, MIN_BALL / cell), 1.5)
        self._paddle_height = min(max(0.36, MIN_PADDLE / cell), 1.0)
        court_w, court_h = cell * (self._width + 0.7), cell * (self._height + 1.2)
        fig_w, fig_h = max(court_w, 4.0) + 0.4, court_h + 1.2
        self.fig = plt.figure(figsize=(fig_w, fig_h))
        self.fig.patch.set_facecolor(BACKGROUND)
        self.ax = self.fig.add_axes([(fig_w - court_w) / 2 / fig_w, 0.65 / fig_h, court_w / fig_w, court_h / fig_h])
        self.ax.set_xlim(-0.85, self._width - 0.15 + 0.35)
        self.ax.set_ylim(-1.5 - 0.2, self._height - 0.5 + 0.5)
        self.ax.axis('off')
        self._draw_court()
        self._title = self.fig.text(0.5, 1 - 0.1 / fig_h, '', ha='center', va='top',
                                    fontsize=13, fontweight='bold', color=TEXT)
        self._info = self.fig.text(0.5, 0.1 / fig_h, '', ha='center', va='bottom', fontsize=10, color=TEXT,
                                   linespacing=1.5)

    def _cell(self, col, row):
        '''Center of a cell on the screen, with row 0 at the top.'''
        return col, self._height - 1 - row

    def _brick_patch(self, col, row, **kwargs):
        x, y = self._cell(col, row)
        return patches.FancyBboxPatch((x - 0.46, y - 0.4), 0.92, 0.8,
                                      boxstyle='round,pad=0,rounding_size=0.08', **kwargs)

    # ===== parts that never change =====

    def _draw_court(self):
        ax = self.ax
        w, h = self._width, self._height

        # the court, open at the bottom, with the walls and the ceiling around it
        ax.add_patch(patches.Rectangle((-0.5, -1.5), w, h + 1, facecolor=COURT, edgecolor='none', zorder=0))
        for (x, y, dx, dy) in ((-0.85, -1.5, 0.35, h + 1.35), (w - 0.5, -1.5, 0.35, h + 1.35),
                               (-0.85, h - 0.5, w + 0.7, 0.35)):
            ax.add_patch(patches.Rectangle((x, y), dx, dy, facecolor=WALL, edgecolor=WALL_EDGE, zorder=1))

        # where the bricks were, seen once they are broken
        ax.add_collection(PatchCollection(
            [self._brick_patch(c, r) for (c, r) in self._brick_at],
            facecolor='none', edgecolor=BROKEN_EDGE, linewidth=0.8, linestyle='--', zorder=1))

    # ===== parts that follow the state =====

    def _draw_state(self, state):
        ax = self.ax
        artists = []

        # the bricks that are left, in rows of slightly different clay
        left = [i for (i, b) in enumerate(self._bricks)
                if not bool(state[RDDLPlanningModel.ground_var('broken', [b])])]
        if left:
            artists.append(ax.add_collection(PatchCollection(
                [self._brick_patch(*self._brick_at[i]) for i in left],
                facecolor=[BRICK_COLORS[self._brick_at[i][1] % len(BRICK_COLORS)] for i in left],
                edgecolor=MORTAR, linewidth=1.0, zorder=2)))

        # the paddle on the bottom row
        x, y = self._cell(int(state['paddle']), self._height - 1)
        artists.append(ax.add_patch(patches.FancyBboxPatch(
            (x - 0.48, y - self._paddle_height / 2), 0.96, self._paddle_height,
            boxstyle=f'round,pad=0,rounding_size={0.4 * self._paddle_height}',
            facecolor=PADDLE, edgecolor=PADDLE_EDGE, linewidth=1.5, zorder=3)))

        # the ball, below the paddle row once it is lost, and the way it is going
        lost = bool(state['lost'])
        x, y = self._cell(int(state['ball-x']), int(state['ball-y']))
        if lost:
            y = -1.0
        dx, dy = int(state['dir-x']), -int(state['dir-y'])
        if not lost:
            reach = self._ball_radius + 0.2
            artists.append(ax.annotate('', xy=(x + reach * dx, y + reach * dy), xytext=(x, y), zorder=4,
                                       arrowprops=dict(arrowstyle='-|>', color=BALL_EDGE, lw=1.5)))
        artists.append(ax.add_patch(plt.Circle((x, y), self._ball_radius, facecolor=LOST if lost else BALL,
                                               edgecolor=BALL_EDGE, linewidth=1.5, zorder=5)))
        return artists, len(left), lost

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        artists, left, lost = self._draw_state(state)
        self._title.set_text('Ball lost' if lost else 'Brick breaker')
        self._info.set_text(f'bricks left: {left} of {len(self._bricks)}\n'
                            f'paddle at column {int(state["paddle"])}, '
                            f'ball at column {int(state["ball-x"])}, row {int(state["ball-y"])}')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
