import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BODY_COLOR = '#1c1c1c'
SOLVED_COLOR = '#2e7d32'
STICKER_COLORS = {'white': '#f5f5f5', 'green': '#009b48', 'red': '#b71234', 'blue': '#0046ad',
                  'orange': '#ff5800', 'yellow': '#ffd500'}
UNKNOWN_COLOR = '#9e9e9e'

# light on the faces seen from the corner: the top is brightest, the right side darkest
SHADES = {(0, 1, 0): 1.0, (0, 0, 1): 0.86, (1, 0, 0): 0.7}
STICKER = 0.84          # part of a small cube covered by its sticker

# sizes in inches
VIEW = 3.0
GAP = 0.5
TOP_MARGIN = 0.45
MARGIN = 0.15


def _project(p):

    # isometric view from the corner where x, y and z are largest
    (x, y, z) = p
    return ((x - z) * math.sqrt(3) / 2, y - (x + z) / 2)


def _flip(v):

    # the half turn that brings the opposite corner to the front, to show the hidden faces
    (x, y, z) = v
    return (-z, -y, -x)


def _square(centre, normal, half):
    axes = [tuple(1 if i == k else 0 for i in range(3)) for k in range(3) if normal[k] == 0]
    (e1, e2) = axes
    return [tuple(centre[i] + half * (a * e1[i] + b * e2[i]) for i in range(3))
            for (a, b) in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


class RubiksCubeVisualizer(BaseViz):
    '''Draws the cube twice in 3D: from the front, showing the up, front and right faces, and
    turned over, showing the down, back and left faces. The drawing only depends on the
    current state.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._size = n = max(1, int(nonfluents['SIZE']))
        self._stickers = list(model.type_to_objects.get('sticker', []))
        self._colours = {s: str(nonfluents[RDDLPlanningModel.ground_var('COLOR', [s])]).lstrip('@')
                         for s in self._stickers}

        width = 2 * VIEW + GAP + 2 * MARGIN
        height = VIEW + TOP_MARGIN + MARGIN
        self.fig = plt.figure(figsize=(width, height))
        self.fig.patch.set_facecolor('white')

        # one patch for each place a sticker can be, in each view, coloured on each frame
        self._places = {}
        extent = 2.1 * n
        for (k, flip) in enumerate((False, True)):
            ax = self.fig.add_axes([(MARGIN + k * (VIEW + GAP)) / width, MARGIN / height,
                                    VIEW / width, VIEW / height])
            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
            ax.set_aspect('equal')
            ax.axis('off')
            self._draw_view(ax, flip)
        self._title = self.fig.text(0.5, 1 - 0.3 / height, '', ha='center', va='center',
                                    fontsize=13, fontweight='bold', color=SOLVED_COLOR)

    def _draw_view(self, ax, flip):
        n = self._size
        view = _flip if flip else (lambda v: v)
        coords = range(-(n - 1), n, 2)
        for normal in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            seen = view(normal)
            if seen not in SHADES:
                continue

            # the black body of the face, then a sticker on each small cube
            body = _square(tuple(n * a for a in normal), normal, n)
            ax.add_patch(patches.Polygon([_project(view(p)) for p in body], closed=True,
                                         facecolor=BODY_COLOR, edgecolor=BODY_COLOR, linewidth=1.2,
                                         joinstyle='round', zorder=1))
            free = [k for k in range(3) if normal[k] == 0]
            for a in coords:
                for b in coords:
                    centre = [(n - 1) * normal[k] for k in range(3)]
                    (centre[free[0]], centre[free[1]]) = (a, b)
                    face_point = tuple(centre[k] + normal[k] for k in range(3))
                    corners = _square(face_point, normal, STICKER)
                    patch = ax.add_patch(patches.Polygon(
                        [_project(view(p)) for p in corners], closed=True, facecolor=UNKNOWN_COLOR,
                        edgecolor='none', joinstyle='round', zorder=2))
                    self._places.setdefault((normal, tuple(centre)), []).append((patch, SHADES[seen]))

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        def value(var, s):
            return int(state[RDDLPlanningModel.ground_var(var, [s])])

        facing = {}
        for s in self._stickers:
            normal = (value('nx', s), value('ny', s), value('nz', s))
            centre = (value('cx', s), value('cy', s), value('cz', s))
            colour = self._colours[s]
            facing.setdefault(normal, set()).add(colour)
            for (patch, shade) in self._places.get((normal, centre), []):
                rgb = to_rgb(STICKER_COLORS.get(colour, UNKNOWN_COLOR))
                patch.set_facecolor(tuple(shade * c for c in rgb))

        # same rule as the termination condition
        solved = all(len(colours) == 1 for colours in facing.values())
        self._title.set_text('Solved' if solved else '')

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
