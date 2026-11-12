import math

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f5f7fa'
GROUND = '#cfd8dc'
ASPHALT = '#546e7a'
ASPHALT_2 = '#37474f'
STALL_LINE = '#eceff1'
OFFICE = '#ffffff'
OFFICE_EDGE = '#90a4ae'
WINDOW = '#263238'
CAR_COLORS = ['#e53935', '#1e88e5', '#fdd835', '#43a047', '#fb8c00', '#8e24aa', '#f5f5f5', '#212121']
TEXT = '#37474f'

# the largest picture (inches), the largest stall (inches) and the most stalls in a row of a lot
MAX_PICTURE = (9.0, 6.5)
MAX_UNIT = 0.32
ROW = 10
STALL_H = 1.6     # height of a stall, in stall widths
SIGN_H = 1.3      # height of the office above a lot, in stall widths
GAP = 0.8         # space between lots, in stall widths
MIN_FONT = 4.5    # smallest text drawn (points)


class CarRentalVisualizer(BaseViz):
    '''Draws each location as a parking lot seen from above, with an office bearing its name and a stall
    for each car it can hold. The cars on hand fill the stalls in order. Stalls past the parking limit
    form a darker second lot, when that lot costs extra.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nf = model.ground_vars_with_values(model.non_fluents)
        self._locations = []
        for loc in model.type_to_objects['location']:
            size = max(int(nf[RDDLPlanningModel.ground_var('MAX-CARS', [loc])]), 0)
            limit = int(nf[RDDLPlanningModel.ground_var('PARKING-LIMIT', [loc])])
            cost = float(nf[RDDLPlanningModel.ground_var('PARKING-COST', [loc])])
            second = limit if (cost > 0 and 0 <= limit < size) else size
            self._locations.append(dict(name=loc, var=RDDLPlanningModel.ground_var('cars', [loc]),
                                        size=size, second=second))
        n = max(len(self._locations), 1)

        # lots on a grid, all with the same stall size
        self._row = max(min(ROW, max(loc['size'] for loc in self._locations) if self._locations else 1), 1)
        rows_per_lot = max([math.ceil(loc['size'] / self._row) for loc in self._locations] + [1])
        self._lot_w = max(self._row, 4) + 0.6
        self._lot_h = rows_per_lot * STALL_H + 0.6
        self._cell_w = self._lot_w + GAP
        self._cell_h = self._lot_h + SIGN_H + 0.6 + GAP
        self._cols = min(n, max(1, math.ceil(math.sqrt(n * self._cell_h / self._cell_w * 1.4))))
        self._grid_rows = math.ceil(n / self._cols)
        width, height = self._cols * self._cell_w + GAP, self._grid_rows * self._cell_h + GAP
        unit = min(MAX_UNIT, MAX_PICTURE[0] / width, MAX_PICTURE[1] / height)
        self._unit = unit

        fig_w, fig_h = max(width * unit, 4.0) + 0.4, height * unit + 0.9
        self.fig = plt.figure(figsize=(fig_w, fig_h))
        self.fig.patch.set_facecolor(BACKGROUND)
        self.ax = self.fig.add_axes([(fig_w - width * unit) / 2 / fig_w, 0.35 / fig_h,
                                     width * unit / fig_w, height * unit / fig_h])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis('off')
        self.ax.add_patch(patches.Rectangle((0, 0), width, height, facecolor=GROUND, edgecolor='none', zorder=0))
        self._counts = []
        for (k, loc) in enumerate(self._locations):
            self._draw_lot(k, loc)
        self._title = self.fig.text(0.5, 1 - 0.08 / fig_h, '', ha='center', va='top',
                                    fontsize=13, fontweight='bold', color=TEXT)

    def _origin(self, k):
        '''Lower left corner of the lot of location k.'''
        c, r = k % self._cols, k // self._cols
        return GAP + c * self._cell_w, GAP + (self._grid_rows - 1 - r) * self._cell_h

    def _stall(self, k, i):
        '''Lower left corner of stall i of location k: rows from the top of the lot.'''
        x0, y0 = self._origin(k)
        return x0 + 0.3 + i % self._row, y0 + self._lot_h - 0.3 - (i // self._row + 1) * STALL_H

    # ===== parts that never change =====

    def _draw_lot(self, k, loc):
        ax = self.ax
        x0, y0 = self._origin(k)
        lot_h = max(math.ceil(loc['size'] / self._row), 1) * STALL_H + 0.6   # under the office, as tall as its stalls
        ax.add_patch(patches.Rectangle((x0, y0 + self._lot_h - lot_h), self._lot_w, lot_h, facecolor=ASPHALT,
                                       edgecolor='none', zorder=1))
        lines = []
        for i in range(loc['size']):
            sx, sy = self._stall(k, i)
            if i >= loc['second']:
                ax.add_patch(patches.Rectangle((sx, sy), 1, STALL_H, facecolor=ASPHALT_2, edgecolor='none',
                                               zorder=1))
            lines.append([(sx, sy + 0.1), (sx, sy + STALL_H - 0.1)])
            if i % self._row == self._row - 1 or i == loc['size'] - 1:
                lines.append([(sx + 1, sy + 0.1), (sx + 1, sy + STALL_H - 0.1)])
        if lines:
            ax.add_collection(LineCollection(lines, colors=STALL_LINE, linewidths=max(0.03 * self._unit * 72, 0.3),
                                             zorder=2))

        # the office, with the name of the location on its front
        ox, oy = x0 + 0.15 * self._lot_w, y0 + self._lot_h
        ow, oh = 0.7 * self._lot_w, SIGN_H
        ax.add_patch(patches.Rectangle((ox, oy), ow, oh, facecolor=OFFICE, edgecolor=OFFICE_EDGE,
                                       linewidth=max(0.04 * self._unit * 72, 0.4), zorder=2))
        ax.add_patch(patches.Polygon([(ox - 0.15, oy + oh), (ox + ow + 0.15, oy + oh), (ox + ow / 2, oy + oh + 0.5)],
                                     closed=True, facecolor=OFFICE_EDGE, edgecolor='none', zorder=2))
        # names that fit on the office, and no text at all where it would be too small to read
        size = min(9.0, self._unit * 72 * 0.55, ow * self._unit * 72 / (0.65 * max(len(loc['name']), 8)))
        if size < MIN_FONT:
            self._counts.append(None)
            return
        ax.text(ox + ow / 2, oy + 0.62 * oh, loc['name'], ha='center', va='center', fontsize=size,
                color=TEXT, fontweight='bold', zorder=3)
        self._counts.append(ax.text(ox + ow / 2, oy + 0.24 * oh, '', ha='center', va='center',
                                    fontsize=size * 0.9, color=TEXT, zorder=3))

    # ===== parts that follow the state =====

    def _cars(self, k, count):
        bodies, glass, colors = [], [], []
        for i in range(count):
            sx, sy = self._stall(k, i)
            bodies.append(patches.FancyBboxPatch((sx + 0.2, sy + 0.18), 0.6, STALL_H - 0.36,
                                                 boxstyle='round,pad=0,rounding_size=0.15'))
            glass.append(patches.Rectangle((sx + 0.28, sy + STALL_H - 0.62), 0.44, 0.22))
            glass.append(patches.Rectangle((sx + 0.28, sy + 0.32), 0.44, 0.14))
            colors.append(CAR_COLORS[(7 * i + 3 * k) % len(CAR_COLORS)])
        return [PatchCollection(bodies, facecolor=colors, edgecolor='#263238',
                                linewidth=max(0.02 * self._unit * 72, 0.2), zorder=3),
                PatchCollection(glass, facecolor=WINDOW, edgecolor='none', zorder=4)]

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        artists, total = [], 0
        for (k, loc) in enumerate(self._locations):
            cars = int(state[loc['var']])
            total += cars
            for collection in self._cars(k, max(min(cars, loc['size']), 0)):
                artists.append(self.ax.add_collection(collection))
            if self._counts[k] is not None:
                self._counts[k].set_text(f'{cars} of {loc["size"]} car{"" if loc["size"] == 1 else "s"}')
        self._title.set_text(f'Car rental: {total} car{"" if total == 1 else "s"} on hand')
        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
