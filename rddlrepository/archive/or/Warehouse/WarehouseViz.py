import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

FLOOR_COLOR = '#eceff1'
STORAGE_COLOR = '#dde2e6'
STORAGE_LINE_COLOR = '#f9a825'
STATION_COLOR = '#fff8e1'
HAZARD_COLOR = '#212121'
HAZARD_LIGHT_COLOR = '#fdd835'
RACK_COLOR = '#8d6e63'
RACK_EDGE_COLOR = '#4e342e'
BOX_COLOR = '#d7b98e'
REQUESTED_COLOR = '#e53935'
ROBOT_COLOR = '#ff8f00'
ROBOT_EDGE_COLOR = '#37474f'
POINTER_COLOR = 'white'
SHADOW_COLOR = '#00000040'
TEXT_COLOR = '#37474f'

# sizes in inches: a cell, which shrinks on large warehouses, and the room around the floor
MAX_CELL = 0.5
MAX_FLOOR = (6.4, 7.2)
MARGIN = 0.15
LEGEND_HEIGHT = 0.55

# parts of a cell
RACK = 0.74
ROBOT = 0.43
HEADINGS = {'up': (0, 1), 'right': (1, 0), 'down': (0, -1), 'left': (-1, 0)}


# ---------- pictures: each takes the axes and the centre and size of a cell, and returns
# its artists, which move with set_center

class _Shelf:
    '''A rack seen from above with four boxes on it, and a shadow when lifted.'''

    def __init__(self, ax, x, y, c):
        self.c = c
        self.shadow = ax.add_patch(patches.Rectangle((0, 0), RACK * c, RACK * c, facecolor=SHADOW_COLOR,
                                                     edgecolor='none', zorder=2))
        self.rack = ax.add_patch(patches.Rectangle((0, 0), RACK * c, RACK * c, facecolor=RACK_COLOR,
                                                   edgecolor=RACK_EDGE_COLOR, linewidth=0.6, zorder=2))
        self.boxes = [ax.add_patch(patches.Rectangle((0, 0), 0.27 * c, 0.27 * c, facecolor=BOX_COLOR,
                                                     edgecolor=RACK_EDGE_COLOR, linewidth=0.4, zorder=2))
                      for _ in range(4)]
        self.set(x, y, False, False)

    def set(self, x, y, requested, lifted):
        c = self.c
        (zorder, dx) = (5, 0.06 * c) if lifted else (2, 0)
        self.shadow.set_visible(lifted)
        self.shadow.set_xy((x - RACK * c / 2 + dx, y - RACK * c / 2 - dx))
        self.rack.set_xy((x - RACK * c / 2, y - RACK * c / 2))
        for (box, (bx, by)) in zip(self.boxes, ((-1, -1), (-1, 1), (1, -1), (1, 1))):
            box.set_xy((x + bx * 0.17 * c - 0.135 * c, y + by * 0.17 * c - 0.135 * c))
            box.set_facecolor(REQUESTED_COLOR if requested else BOX_COLOR)
            box.set_zorder(zorder + 0.2)
        self.shadow.set_zorder(zorder - 0.1)
        self.rack.set_zorder(zorder + 0.1)


class _Robot:
    '''A round warehouse robot seen from above, with a pointer at its front.'''

    def __init__(self, ax, x, y, c):
        self.c = c
        self.body = ax.add_patch(plt.Circle((x, y), ROBOT * c, facecolor=ROBOT_COLOR,
                                            edgecolor=ROBOT_EDGE_COLOR, linewidth=0.8, zorder=4))
        self.wheel = ax.add_patch(plt.Circle((x, y), 0.18 * c, facecolor='none',
                                             edgecolor=ROBOT_EDGE_COLOR, linewidth=0.6, zorder=4.1))
        self.pointer = ax.add_patch(patches.Polygon([(x, y)] * 3, closed=True, facecolor=POINTER_COLOR,
                                                    edgecolor=ROBOT_EDGE_COLOR, linewidth=0.5, zorder=6))
        self.set(x, y, 'up')

    def set(self, x, y, heading):
        c = self.c
        (hx, hy) = HEADINGS.get(heading, (0, 1))
        (px, py) = (-hy, hx)
        self.body.set_center((x, y))
        self.wheel.set_center((x, y))
        tip = (x + hx * 0.47 * c, y + hy * 0.47 * c)
        base = (x + hx * 0.3 * c, y + hy * 0.3 * c)
        self.pointer.set_xy([tip, (base[0] + px * 0.13 * c, base[1] + py * 0.13 * c),
                             (base[0] - px * 0.13 * c, base[1] - py * 0.13 * c)])


def _station(ax, x, y, c):

    # a work area on the floor inside a black and yellow hazard frame
    ax.add_patch(patches.Rectangle((x - 0.5 * c, y - 0.5 * c), c, c, facecolor=HAZARD_LIGHT_COLOR,
                                   hatch='////', edgecolor=HAZARD_COLOR, linewidth=0, zorder=1))
    ax.add_patch(patches.Rectangle((x - 0.36 * c, y - 0.36 * c), 0.72 * c, 0.72 * c, facecolor=STATION_COLOR,
                                   edgecolor=HAZARD_COLOR, linewidth=0.5, zorder=1.1))


class WarehouseVisualizer(BaseViz):
    '''Draws the warehouse floor from above: the storage cells marked out in yellow, the
    workstations in hazard frames, the shelves as racks of boxes with red boxes on the
    requested ones, and the robots as round orange robots with a pointer at their front.
    A robot carrying a shelf lifts it above itself.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        xs = {x: int(nonfluents[RDDLPlanningModel.ground_var('X', [x])])
              for x in model.type_to_objects.get('xpos', [])}
        ys = {y: int(nonfluents[RDDLPlanningModel.ground_var('Y', [y])])
              for y in model.type_to_objects.get('ypos', [])}
        self._columns = max(int(nonfluents['WIDTH']), max(xs.values(), default=0) + 1)
        self._rows = max(int(nonfluents['HEIGHT']), max(ys.values(), default=0) + 1)
        self._robots = list(model.type_to_objects.get('robot', []))
        self._shelves = list(model.type_to_objects.get('shelf', []))
        (columns, rows) = (self._columns, self._rows)
        self._cell = c = min(MAX_CELL, MAX_FLOOR[0] / columns, MAX_FLOOR[1] / rows)

        (floor_w, floor_h) = (columns * c, rows * c)
        width = max(floor_w, 5.8) + 2 * MARGIN
        height = LEGEND_HEIGHT + floor_h + 2 * MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.fig.patch.set_facecolor('white')
        self.ax = self.fig.add_axes([(width - floor_w) / 2 / width, (LEGEND_HEIGHT + MARGIN) / height,
                                     floor_w / width, floor_h / height])
        self.ax.set_xlim(0, floor_w)
        self.ax.set_ylim(0, floor_h)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # the floor never changes, so it is only drawn once
        self.ax.add_patch(patches.Rectangle((0, 0), floor_w, floor_h, facecolor=FLOOR_COLOR,
                                            edgecolor=ROBOT_EDGE_COLOR, linewidth=1.0, zorder=0))
        for (x, col) in xs.items():
            for (y, row) in ys.items():
                (cx, cy) = self._centre(col, row)
                if nonfluents[RDDLPlanningModel.ground_var('STORAGE', [x, y])]:
                    self.ax.add_patch(patches.Rectangle(
                        (cx - 0.46 * c, cy - 0.46 * c), 0.92 * c, 0.92 * c, facecolor=STORAGE_COLOR,
                        edgecolor=STORAGE_LINE_COLOR, linewidth=0.7, zorder=0.5))
                if nonfluents[RDDLPlanningModel.ground_var('STATION', [x, y])]:
                    _station(self.ax, cx, cy, c)

        # the shelves and robots move, so each keeps its pictures and they are moved each frame
        self._shelf_pictures = {s: _Shelf(self.ax, 0, 0, c) for s in self._shelves}
        self._robot_pictures = {r: _Robot(self.ax, 0, 0, c) for r in self._robots}
        self._draw_legend()

    def _centre(self, col, row):
        c = self._cell
        return ((col + 0.5) * c, (self._rows - 1 - row + 0.5) * c)

    def _draw_legend(self):
        (width, _) = self._figure_size
        height = self._figure_size[1]
        ax = self.fig.add_axes([0, 0.04 / height, 1, 0.46 / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, 0.46)
        ax.set_aspect('equal')
        ax.axis('off')
        entries = [('robot', 'Robot'), ('loaded', 'Loaded'), ('shelf', 'Shelf'),
                   ('requested', 'Requested'), ('station', 'Station')]
        widths = [0.42 + 0.066 * len(label) + 0.3 for (_, label) in entries]
        x = (width - sum(widths)) / 2
        for ((kind, label), entry_width) in zip(entries, widths):
            (cx, cy, s) = (x + 0.17, 0.23, 0.34)
            if kind in ('robot', 'loaded'):
                _Robot(ax, cx, cy, s).set(cx, cy, 'right')
            if kind in ('loaded', 'shelf', 'requested'):
                _Shelf(ax, cx, cy, s).set(cx, cy, kind == 'requested', kind == 'loaded')
            if kind == 'station':
                _station(ax, cx, cy, s)
            ax.text(x + 0.42, 0.21, label, ha='left', va='center', fontsize=8.5, color=TEXT_COLOR)
            x += entry_width

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        def value(var, *args):
            return state[RDDLPlanningModel.ground_var(var, list(args))]

        carried = {s for r in self._robots for s in self._shelves if value('carrying', r, s)}
        for (s, picture) in self._shelf_pictures.items():
            (x, y) = self._centre(int(value('shelf-x', s)), int(value('shelf-y', s)))
            picture.set(x, y, bool(value('requested', s)), s in carried)
        for (r, picture) in self._robot_pictures.items():
            (x, y) = self._centre(int(value('robot-x', r)), int(value('robot-y', r)))
            picture.set(x, y, str(value('facing', r)).lstrip('@'))

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
