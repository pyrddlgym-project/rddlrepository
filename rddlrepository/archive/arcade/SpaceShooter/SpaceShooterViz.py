import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f5f7fa'
SPACE = '#0d1b2a'
STAR = '#cfd8dc'
GROUND = '#37474f'
ALIEN = '#7cb342'
ALIEN_EDGE = '#33691e'
EYE = '#ffffff'
PUPIL = '#1b1b1b'
CANNON = '#90a4ae'
CANNON_EDGE = '#37474f'
HOT = '#ff7043'
BULLET = '#ffee58'
BOMB = '#ef5350'
BOMB_EDGE = '#b71c1c'
LOST = '#e53935'
TEXT = '#37474f'

# the largest size of the grid on the screen, the smallest size of a cell, and the smallest size
# of a bullet or bomb so that they can be seen on a large grid (inches)
MAX_GRID = (6.5, 5.5)
MIN_CELL = 0.03
MIN_SHOT = 0.15


class SpaceShooterVisualizer(BaseViz):
    '''Draws the grid as a patch of night sky over the ground: the aliens, with an arrow above the
    wave for the way it marches, the bullets (yellow) and bombs (red), and the cannon on the bottom
    row, whose barrel glows while it cools down. A destroyed cannon is drawn in red.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nf = model.ground_vars_with_values(model.non_fluents)
        self._width = max(int(nf['WIDTH']), 1)
        self._height = max(int(nf['HEIGHT']), 1)
        self._cool_down = max(int(nf['COOL-DOWN']), 1)
        xs = list(model.type_to_objects['xpos'])
        ys = list(model.type_to_objects['ypos'])
        index_x = {x: int(nf[RDDLPlanningModel.ground_var('X', [x])]) for x in xs}
        index_y = {y: int(nf[RDDLPlanningModel.ground_var('Y', [y])]) for y in ys}
        self._cells = [((x, y), index_x[x], index_y[y]) for x in xs for y in ys]
        self._wave_size = sum(bool(nf[RDDLPlanningModel.ground_var('WAVE', [x, y])]) for x in xs for y in ys)

        # the same scale on both axes, with room for the ground and the arrow of the wave
        cell = min(MAX_GRID[0] / self._width, MAX_GRID[1] / (self._height + 1.0))
        cell = max(cell, MIN_CELL)
        self._shot_size = min(max(1.0, MIN_SHOT / cell), 2.0)
        grid_w, grid_h = cell * self._width, cell * (self._height + 1.0)
        fig_w, fig_h = max(grid_w, 4.0) + 0.4, grid_h + 1.2
        self.fig = plt.figure(figsize=(fig_w, fig_h))
        self.fig.patch.set_facecolor(BACKGROUND)
        self.ax = self.fig.add_axes([(fig_w - grid_w) / 2 / fig_w, 0.65 / fig_h, grid_w / fig_w, grid_h / fig_h])
        self.ax.set_xlim(-0.5, self._width - 0.5)
        self.ax.set_ylim(-0.75, self._height + 0.25)
        self.ax.axis('off')
        self._draw_sky()
        self._title = self.fig.text(0.5, 1 - 0.1 / fig_h, '', ha='center', va='top',
                                    fontsize=13, fontweight='bold', color=TEXT)
        self._info = self.fig.text(0.5, 0.1 / fig_h, '', ha='center', va='bottom', fontsize=10, color=TEXT,
                                   linespacing=1.5)

    def _screen(self, x, y):
        '''Center of a cell on the screen, with row 0 at the top.'''
        return x, self._height - 1 - y

    # ===== parts that never change =====

    def _draw_sky(self):
        ax = self.ax
        w, h = self._width, self._height
        ax.add_patch(patches.Rectangle((-0.5, -0.5), w, h + 0.75, facecolor=SPACE, edgecolor='none', zorder=0))
        ax.add_patch(patches.Rectangle((-0.5, -0.75), w, 0.25, facecolor=GROUND, edgecolor='none', zorder=0))

        # the same stars on every render, about two for every three cells but not so many that
        # they hide the bullets on a large grid
        rng = np.random.default_rng(0)
        count = min(int(w * h * 0.7), 150)
        ax.scatter(rng.uniform(-0.5, w - 0.5, count), rng.uniform(-0.4, h + 0.2, count),
                   s=rng.uniform(0.3, 2.5, count), color=STAR, alpha=0.6, linewidths=0, zorder=0)

    # ===== parts that follow the state =====

    def _cells_where(self, state, name):
        return [self._screen(x, y) for (objects, x, y) in self._cells
                if bool(state[RDDLPlanningModel.ground_var(name, objects)])]

    def _draw_aliens(self, cells):
        bodies, eyes, pupils, antennae = [], [], [], []
        for (x, y) in cells:
            bodies.append(patches.FancyBboxPatch((x - 0.36, y - 0.26), 0.72, 0.48,
                                                 boxstyle='round,pad=0,rounding_size=0.22'))
            for dx in (-0.15, 0.15):
                eyes.append(patches.Circle((x + dx, y + 0.02), 0.1))
                pupils.append(patches.Circle((x + dx, y - 0.01), 0.045))
                antennae.append(patches.Polygon([(x + dx * 0.8, y + 0.2), (x + dx * 1.7, y + 0.42),
                                                 (x + dx * 1.7 + 0.03, y + 0.4)], closed=True))
            for dx in (-0.22, 0.0, 0.22):
                antennae.append(patches.Rectangle((x + dx - 0.04, y - 0.38), 0.08, 0.14))
        return [PatchCollection(antennae, facecolor=ALIEN_EDGE, edgecolor=ALIEN_EDGE, linewidth=0.8, zorder=2),
                PatchCollection(bodies, facecolor=ALIEN, edgecolor=ALIEN_EDGE, linewidth=1.0, zorder=3),
                PatchCollection(eyes, facecolor=EYE, edgecolor='none', zorder=4),
                PatchCollection(pupils, facecolor=PUPIL, edgecolor='none', zorder=5)]

    def _draw_cannon(self, x, heat, lost):
        y = self._screen(x, self._height - 1)[1]
        color = LOST if lost else CANNON
        glow = heat / self._cool_down
        barrel = HOT if heat > 0 and not lost else color
        return [patches.Polygon([(x - 0.44, y - 0.42), (x + 0.44, y - 0.42), (x + 0.3, y - 0.1), (x - 0.3, y - 0.1)],
                                closed=True, facecolor=color, edgecolor=CANNON_EDGE, linewidth=1.2, zorder=3),
                patches.Rectangle((x - 0.08, y - 0.12), 0.16, 0.5, facecolor=barrel, edgecolor=CANNON_EDGE,
                                  linewidth=1.0, alpha=1.0 if heat == 0 else 0.5 + 0.5 * glow, zorder=2)]

    def _draw_state(self, state):
        ax = self.ax
        artists = []
        lost = bool(state['lost'])

        aliens = self._cells_where(state, 'alien')
        for collection in self._draw_aliens(aliens):
            artists.append(ax.add_collection(collection))

        # the way the wave marches, above it
        if aliens:
            mid = float(np.mean([x for (x, _) in aliens]))
            top = max(y for (_, y) in aliens) + 0.55
            d = int(state['dir'])
            artists.append(ax.annotate('', xy=(mid + 0.8 * d, top), xytext=(mid - 0.8 * d, top), zorder=6,
                                       arrowprops=dict(arrowstyle='-|>', color=ALIEN, lw=1.5)))

        s = self._shot_size
        bullets = [patches.Rectangle((x - 0.08 * s, y - 0.22 * s), 0.16 * s, 0.44 * s)
                   for (x, y) in self._cells_where(state, 'bullet')]
        bombs = [patches.Polygon([(x, y - 0.3 * s), (x + 0.14 * s, y), (x, y + 0.2 * s), (x - 0.14 * s, y)], closed=True)
                 for (x, y) in self._cells_where(state, 'bomb')]
        if bullets:
            artists.append(ax.add_collection(PatchCollection(bullets, facecolor=BULLET, edgecolor='none', zorder=7)))
        if bombs:
            artists.append(ax.add_collection(PatchCollection(bombs, facecolor=BOMB, edgecolor=BOMB_EDGE,
                                                             linewidth=0.8, zorder=7)))

        for patch in self._draw_cannon(int(state['cannon']), int(state['heat']), lost):
            artists.append(ax.add_patch(patch))
        return artists, len(aliens), lost

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        artists, aliens, lost = self._draw_state(state)
        heat = int(state['heat'])
        self._title.set_text('Cannon destroyed' if lost else 'Space shooter')
        self._info.set_text(f'aliens left: {aliens} of {self._wave_size}   |   '
                            f'cannon {"ready" if heat == 0 else f"cooling, {heat} steps"}\n'
                            f'wave moves in {int(state["march-timer"]) + 1} steps (at most every '
                            f'{int(state["march-time"])}), next bomb in {int(state["bomb-timer"]) + 1}')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
