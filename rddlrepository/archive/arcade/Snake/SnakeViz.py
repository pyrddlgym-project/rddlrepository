import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f5f7fa'
GRASS = ('#aad751', '#a2d149')
WALL = '#6d4c41'
TAIL = '#9fa8da'
BODY = '#3949ab'
HEAD = '#283593'
CRASHED = '#e53935'
EYE = '#ffffff'
PUPIL = '#1b1b1b'
APPLE = '#e53935'
APPLE_EDGE = '#b71c1c'
STEM = '#5d4037'
LEAF = '#43a047'
TEXT = '#37474f'

# the largest size of the grid on the screen, the smallest size of a cell, and the smallest size
# of the head and the fruit so that they can be seen on a large grid (inches)
MAX_GRID = (6.0, 5.5)
MIN_CELL = 0.03
MIN_MARK = 0.15


class SnakeVisualizer(BaseViz):
    '''Draws the grid as a lawn inside a wall, with the fruit as an apple and the snake as a band
    from its tail (light) to its head, whose eyes look away from its neck. A snake that crashed has
    a red head.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nf = model.ground_vars_with_values(model.non_fluents)
        self._width = max(int(nf['WIDTH']), 1)
        self._height = max(int(nf['HEIGHT']), 1)
        xs = list(model.type_to_objects['xpos'])
        ys = list(model.type_to_objects['ypos'])
        index_x = {x: int(nf[RDDLPlanningModel.ground_var('X', [x])]) for x in xs}
        index_y = {y: int(nf[RDDLPlanningModel.ground_var('Y', [y])]) for y in ys}
        self._cells = [((x, y), index_x[x], index_y[y]) for x in xs for y in ys]

        # the same scale on both axes, with a wall of 0.05 to 0.12 inches (in cells)
        cell = min(MAX_GRID[0] / self._width, MAX_GRID[1] / self._height)
        cell = max(cell, MIN_CELL)
        self._cell = cell
        self._wall = min(max(0.3, 0.05 / cell), 0.12 / cell)
        self._mark = min(max(1.0, MIN_MARK / cell), 3.0)
        grid_w, grid_h = cell * (self._width + 2 * self._wall), cell * (self._height + 2 * self._wall)
        fig_w, fig_h = max(grid_w, 4.0) + 0.4, grid_h + 1.2
        self.fig = plt.figure(figsize=(fig_w, fig_h))
        self.fig.patch.set_facecolor(BACKGROUND)
        self.ax = self.fig.add_axes([(fig_w - grid_w) / 2 / fig_w, 0.65 / fig_h, grid_w / fig_w, grid_h / fig_h])
        self._draw_lawn()
        self.ax.set_xlim(-0.5 - self._wall, self._width - 0.5 + self._wall)
        self.ax.set_ylim(-0.5 - self._wall, self._height - 0.5 + self._wall)
        self.ax.axis('off')
        self._title = self.fig.text(0.5, 1 - 0.1 / fig_h, '', ha='center', va='top',
                                    fontsize=13, fontweight='bold', color=TEXT)
        self._info = self.fig.text(0.5, 0.1 / fig_h, '', ha='center', va='bottom', fontsize=10, color=TEXT)

    def _screen(self, x, y):
        '''Center of a cell on the screen, with row 0 at the top.'''
        return x, self._height - 1 - y

    # ===== parts that never change =====

    def _draw_lawn(self):
        w, h, t = self._width, self._height, self._wall
        self.ax.add_patch(patches.Rectangle((-0.5 - t, -0.5 - t), w + 2 * t, h + 2 * t, facecolor=WALL,
                                            edgecolor='none', zorder=0))
        checks = np.indices((h, w)).sum(axis=0) % 2
        self.ax.imshow(checks, cmap=ListedColormap(GRASS), extent=(-0.5, w - 0.5, -0.5, h - 0.5),
                       interpolation='nearest', aspect='auto', zorder=1)

    # ===== parts that follow the state =====

    def _draw_apple(self, x, y):
        s = self._mark
        return [patches.Rectangle((x - 0.03 * s, y + 0.18 * s), 0.06 * s, 0.18 * s, facecolor=STEM,
                                  edgecolor='none', zorder=3),
                patches.Ellipse((x + 0.13 * s, y + 0.3 * s), 0.22 * s, 0.11 * s, angle=25, facecolor=LEAF,
                                edgecolor='none', zorder=3),
                patches.Circle((x, y - 0.03 * s), 0.32 * s, facecolor=APPLE, edgecolor=APPLE_EDGE,
                               linewidth=1.0, zorder=4)]

    def _draw_body(self, body):
        '''The band through the body cells in order of age, from the tail to the head.'''
        body = sorted(body)
        segments, colors = [], []
        tail, body_color = np.array(to_rgb(TAIL)), np.array(to_rgb(BODY))
        for ((a1, x1, y1), (a2, x2, y2)) in zip(body, body[1:]):
            if a2 == a1 + 1 and abs(x2 - x1) + abs(y2 - y1) == 1:
                segments.append([self._screen(x1, y1), self._screen(x2, y2)])
                share = (len(segments) - 1) / max(len(body) - 2, 1)
                colors.append(tail + share * (body_color - tail))
        if not segments:
            return None
        width = max(0.62 * self._cell * 72, 1.5)
        return LineCollection(segments, colors=colors, linewidths=width, capstyle='round', zorder=5)

    def _draw_head(self, x, y, neck, crashed):
        s = self._mark
        cx, cy = self._screen(x, y)
        artists = [patches.Circle((cx, cy), 0.4 * s, facecolor=CRASHED if crashed else HEAD,
                                  edgecolor='none', zorder=6)]

        # the eyes look away from the neck, or out of the screen without one
        if neck is None:
            ahead, side = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        else:
            nx, ny = self._screen(*neck)
            ahead = np.array([cx - nx, cy - ny], dtype=float)
            side = np.array([-ahead[1], ahead[0]])
        for sign in (-1, 1):
            eye = np.array([cx, cy]) + s * (0.12 * ahead + 0.18 * sign * side)
            look = eye + s * 0.04 * ahead
            if crashed:
                for dx, dy in ((1, 1), (1, -1)):
                    artists.append(patches.FancyArrow(eye[0] - 0.07 * s * dx, eye[1] - 0.07 * s * dy,
                                                      0.14 * s * dx, 0.14 * s * dy, width=0.03 * s,
                                                      head_width=0, head_length=0, color=EYE, zorder=7))
            else:
                artists.append(patches.Circle(eye, 0.11 * s, facecolor=EYE, edgecolor='none', zorder=7))
                artists.append(patches.Circle(look, 0.05 * s, facecolor=PUPIL, edgecolor='none', zorder=8))
        return artists

    def _draw_state(self, state):
        ax = self.ax
        artists = []

        ages = {(x, y): int(state[RDDLPlanningModel.ground_var('age', objects)]) for (objects, x, y) in self._cells}
        body = [(age, x, y) for ((x, y), age) in ages.items() if age > 0]
        for (objects, x, y) in self._cells:
            if bool(state[RDDLPlanningModel.ground_var('fruit', objects)]):
                for patch in self._draw_apple(*self._screen(x, y)):
                    artists.append(ax.add_patch(patch))

        band = self._draw_body(body)
        if band is not None:
            artists.append(ax.add_collection(band))

        hx, hy = int(state['head-x']), int(state['head-y'])
        if 0 <= hx < self._width and 0 <= hy < self._height:
            age = ages.get((hx, hy), 0)
            necks = [(x, y) for (a, x, y) in body if a == age - 1 and abs(x - hx) + abs(y - hy) == 1]
            for patch in self._draw_head(hx, hy, necks[0] if necks else None, bool(state['lost'])):
                artists.append(ax.add_patch(patch))
        return artists, len(body)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        artists, cells = self._draw_state(state)
        size = self._width * self._height
        if bool(state['lost']):
            title = 'Snake crashed'
        elif int(state['length']) >= size:
            title = 'Grid filled'
        else:
            title = 'Snake'
        self._title.set_text(title)
        self._info.set_text(f'length {int(state["length"])}   |   {cells} of {size} cells taken')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
