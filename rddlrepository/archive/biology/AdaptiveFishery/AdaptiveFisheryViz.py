import math

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f2f4f6'
LAND = '#dfe8d0'
SEA = '#3d7fb8'
RIVER = '#7fb8e0'
GRAVEL = '#cdb592'
GRAVEL_DOT = '#a88d66'
NEST = '#8d6e4f'
RUN = '#cfd8dc'          # silver salmon coming back from the sea
SPAWNER = '#d84315'      # red spawning sockeye
FISH_EDGE = '#37474f'
EYE = '#1b1b1b'
HULL = '#8d4b2a'
CARD = '#ffffff'
CARD_EDGE = '#90a4ae'
CURVE = '#1565c0'
BAND = '#90caf9'
REPLACE = '#9e9e9e'
DOT = '#37474f'
NEW_DOT = '#ff8f00'
TEXT = '#37474f'

CELL = (2.3, 1.55)       # a stock's corner of the scene: its river and its card
RUN_GRID = (5, 3)        # fish in the sea, across and up
CATCH_GRID = (5, 2)      # fish in the boat
SPAWN_GRID = (5, 3)      # fish on the gravel


class AdaptiveFisheryVisualizer(BaseViz):
    '''Draws a river for each stock, and beside it the card on which the manager keeps its belief.

    In the river, the fish returning this generation swim in the sea at its mouth; the boat holds last
    generation's catch, and the gravel upstream the fish that spawned (outlined nests show how many it
    holds). Each fish stands for the same number of fish in every picture of a stock (a fixed number for
    each stock, from its prior). An arrow above a group shows that it overflows.

    The card plots the recruits (up) the manager expects from a number of spawners (across, up to what
    the gravel holds): the curve of its current estimates, the band of 80% of its uncertainty about the
    curve, the dashed line on which fish just replace themselves, and a dot for each generation seen in
    this episode (the latest in orange).
    '''

    def __init__(self, model: RDDLPlanningModel, figure_size=(10.5, 6.6), wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._stocks = model.type_to_objects['stock']
        nf = model.ground_vars_with_values(model.non_fluents)
        init = model.ground_vars_with_values(model.state_fluents)
        g = lambda d, name, s: float(d[RDDLPlanningModel.ground_var(name, [s])])
        self._cap = {s: max(g(nf, 'SPAWNING-CAPACITY', s), 1e-12) for s in self._stocks}

        # scales from the prior: a fish of each stock, and the axes of its card, which hold the band of
        # the prior (up to three times the peak of its curve)
        self._unit, self._xmax, self._ymax = {}, {}, {}
        for s in self._stocks:
            a, b, r0, cap = g(init, 'alpha-hat', s), g(init, 'beta-hat', s), g(init, 'recruits', s), self._cap[s]
            va, vb, cab = g(init, 'var-alpha', s), g(init, 'var-beta', s), g(init, 'cov-alpha-beta', s)
            sp = np.linspace(0, cap, 101)
            sd = np.sqrt(np.maximum(va - 2 * sp * cab + sp * sp * vb, 0))
            with np.errstate(over='ignore', invalid='ignore'):
                peak = float(np.max(sp * np.exp(np.minimum(a - b * sp, 50))))
                band = float(np.max(sp * np.exp(np.minimum(a - b * sp + 1.2816 * sd, 50))))
            peak = peak if math.isfinite(peak) else cap
            band = min(band, 3 * peak) if math.isfinite(band) else peak
            high = max(r0, cap, peak)
            self._unit[s] = self._nice(1.2 * high / (RUN_GRID[0] * RUN_GRID[1]))
            self._xmax[s] = cap
            self._ymax[s] = 1.1 * max(high, band)
        self._history = {s: [] for s in self._stocks}

        # the grid of stocks, as large as fits the scene
        self.fig = plt.figure(figsize=figure_size)
        self.fig.patch.set_facecolor(BACKGROUND)
        w, h = figure_size
        box_w, box_h = 0.70, 0.94
        aspect = (box_h * h) / (box_w * w)
        num = len(self._stocks)
        best = None
        for cols in range(1, num + 1):
            rows = math.ceil(num / cols)
            scale = min(1.0 / (cols * CELL[0]), aspect / (rows * CELL[1]))
            if best is None or scale > best[0]:
                best = (scale, cols, rows)
        _, self._cols, self._rows = best
        span_w, span_h = self._cols * CELL[0], self._rows * CELL[1]
        half_w = max(span_w / 2, span_h / 2 / aspect) * 1.03
        self.ax = self.fig.add_axes([0.01, 0.03, box_w, box_h])
        self.ax.set_xlim(span_w / 2 - half_w, span_w / 2 + half_w)
        self.ax.set_ylim(span_h / 2 - half_w * aspect, span_h / 2 + half_w * aspect)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.add_patch(patches.FancyBboxPatch((span_w / 2 - half_w, span_h / 2 - half_w * aspect), 2 * half_w,
                                                 2 * half_w * aspect, boxstyle='round,pad=0,rounding_size=0.3',
                                                 facecolor=LAND, edgecolor='none', zorder=0))
        points = box_w * w * 72 / (2 * half_w)          # points in a unit of the scene
        self._name_size = min(11, 0.16 * points)
        self._line = max(0.3, min(1.5, 0.012 * points))
        self._origin = {}
        for (i, s) in enumerate(self._stocks):
            x, y = (i % self._cols) * CELL[0], (self._rows - 1 - i // self._cols) * CELL[1]
            self._origin[s] = (x, y)
            self._river(self.ax, x + 0.08, y + 0.2, 1.1, self._nests(s))
            self._card(self.ax, *self._card_box(x, y))
            if self._name_size >= 6:
                self.ax.text(x + 0.63, y + 0.16, s, ha='center', va='top', fontsize=self._name_size, color=TEXT)

        pw = 0.27
        self.panel = self.fig.add_axes([0.725, 0.03, pw, 0.94])
        self._ph = 0.94 * h / (pw * w)
        self.panel.set_xlim(0, 1)
        self.panel.set_ylim(0, self._ph)
        self.panel.set_aspect('equal')
        self.panel.axis('off')
        self._draw_panel()

    # ===== scales =====

    @staticmethod
    def _nice(v):
        '''The smallest of ..., 0.1, 0.2, 0.5, 1, 2, 5, 10, ... at least v.'''
        if not (v > 0 and math.isfinite(v)):
            return 1.0
        k = 10 ** math.floor(math.log10(v))
        for m in (1, 2, 5, 10):
            if m * k >= v * (1 - 1e-9):
                return m * k
        return 10 * k

    def _places(self, value, s):
        return int(math.ceil(max(value, 0.0) / self._unit[s] - 1e-9))

    def _nests(self, s):
        return min(self._places(self._cap[s], s), SPAWN_GRID[0] * SPAWN_GRID[1])

    @staticmethod
    def _card_box(x, y):
        return x + 1.3, y + 0.32, 0.92, 1.0

    # ===== pictures =====

    @staticmethod
    def _bands(size):
        '''The heights of the sea, the river and the gravel of a river picture of width size.'''
        return 0.42 * size, 0.3 * size, 0.42 * size

    def _river(self, ax, x, y, size, nests, zorder=1):
        '''The sea at the bottom, the river running up to the gravel at the top, and its nests.'''
        sea, river, gravel = self._bands(size)
        ax.add_patch(patches.FancyBboxPatch((x, y), size, sea, boxstyle='round,pad=0,rounding_size=%g' % (0.05 * size),
                                            facecolor=SEA, edgecolor='none', zorder=zorder))
        ax.add_patch(patches.Polygon([(x + 0.05 * size, y + sea - 0.01), (x + 0.95 * size, y + sea - 0.01),
                                      (x + 0.72 * size, y + sea + river + 0.01), (x + 0.28 * size, y + sea + river + 0.01)],
                                     closed=True, facecolor=RIVER, edgecolor='none', zorder=zorder))
        gy = y + sea + river
        ax.add_patch(patches.FancyBboxPatch((x, gy), size, gravel, boxstyle='round,pad=0,rounding_size=%g' % (0.05 * size),
                                            facecolor=GRAVEL, edgecolor='none', zorder=zorder))
        rng = np.random.default_rng(0)
        pebbles = [patches.Circle((x + size * rng.uniform(0.03, 0.97), gy + gravel * rng.uniform(0.05, 0.95)),
                                  0.008 * size) for _ in range(40)]
        ax.add_collection(PatchCollection(pebbles, facecolor=GRAVEL_DOT, edgecolor='none', zorder=zorder))
        nx, ny = SPAWN_GRID
        cw, ch = size / nx, gravel / ny
        holes = [patches.Ellipse((x + (k % nx + 0.5) * cw, gy + (k // nx + 0.5) * ch), 0.8 * cw, 0.75 * ch)
                 for k in range(nests)]
        ax.add_collection(PatchCollection(holes, facecolor='none', edgecolor=NEST, linewidth=0.5,
                                          linestyle=(0, (2, 1.5)), zorder=zorder + 0.1))

    @staticmethod
    def _fish(x, y, length, facing=1):
        '''The body and the tail of a fish centred at (x, y), and its eye.'''
        h = 0.42 * length
        body = patches.Ellipse((x + 0.08 * facing * length, y), 0.78 * length, h)
        t = x - 0.28 * facing * length
        tail = patches.Polygon([(t, y), (t - 0.22 * facing * length, y + 0.45 * h), (t - 0.22 * facing * length, y - 0.45 * h)],
                               closed=True)
        eye = patches.Circle((x + 0.3 * facing * length, y + 0.08 * h), 0.06 * h)
        return [body, tail], eye

    def _school(self, ax, x, y, w, h, grid, count, color, facing=1, zorder=3):
        '''count fish in a grid of places filling the box (x, y, w, h), and an arrow if they overflow.'''
        nx, ny = grid
        cw, ch = w / nx, h / ny
        shapes, eyes = [], []
        for k in range(min(count, nx * ny)):
            i, j = k % nx, k // nx
            body, eye = self._fish(x + (i + 0.5) * cw, y + (j + 0.5) * ch, min(0.85 * cw, 1.9 * ch), facing)
            shapes.extend(body)
            eyes.append(eye)
        artists = [ax.add_collection(PatchCollection(shapes, facecolor=color, edgecolor=FISH_EDGE,
                                                     linewidth=0.4 * self._line, zorder=zorder)),
                   ax.add_collection(PatchCollection(eyes, facecolor=EYE, edgecolor='none', zorder=zorder + 0.1))]
        if count > nx * ny:
            a = 0.5 * cw
            artists.append(ax.add_patch(patches.Polygon([(x + w / 2 - a / 2, y + h + 0.02), (x + w / 2 + a / 2, y + h + 0.02),
                                                         (x + w / 2, y + h + 0.02 + 0.6 * a)], closed=True,
                                                        facecolor=FISH_EDGE, edgecolor='none', zorder=zorder + 0.2)))
        return artists

    def _boat(self, ax, x, y, size, count, zorder=4):
        '''A boat in the river holding the catch.'''
        sea, river, _ = self._bands(size)
        cx, base = x + 0.5 * size, y + sea + 0.04 * size
        hw, hh = 0.3 * size, 0.08 * size
        artists = [ax.add_patch(patches.Polygon([(cx - hw, base + hh), (cx + hw, base + hh), (cx + 0.75 * hw, base),
                                                 (cx - 0.75 * hw, base)], closed=True, facecolor=HULL,
                                                edgecolor='none', zorder=zorder + 0.3))]
        artists.extend(self._school(ax, cx - 0.9 * hw, base + 0.6 * hh, 1.8 * hw, river - 0.1 * size, CATCH_GRID,
                                    count, RUN, zorder=zorder))
        return artists

    def _card(self, ax, x, y, w, h, zorder=2):
        '''The card of a belief: axes for spawners (across) and recruits (up).'''
        ax.add_patch(patches.FancyBboxPatch((x - 0.05, y - 0.05), w + 0.1, h + 0.1,
                                            boxstyle='round,pad=0,rounding_size=0.06', facecolor=CARD,
                                            edgecolor=CARD_EDGE, linewidth=0.6 * self._line, zorder=zorder))
        ax.plot([x, x, x + w], [y + h, y, y], color=TEXT, linewidth=0.8 * self._line, zorder=zorder + 0.1)

        # a spawner at the end of the axis of spawners, a fish coming back at the top of the other
        for (fx, fy, color, facing) in ((x + w - 0.07, y - 0.035, SPAWNER, 1), (x + 0.07, y + h - 0.01, RUN, -1)):
            body, eye = self._fish(fx, fy, 0.11, facing)
            ax.add_collection(PatchCollection(body, facecolor=color, edgecolor=FISH_EDGE, linewidth=0.3 * self._line,
                                              zorder=zorder + 0.5))

    def _belief(self, ax, x, y, w, h, s, alpha, beta, va, vb, cab, zorder=3):
        '''The expected recruits of the belief, the band of its uncertainty, the dots seen so far.'''
        X, Y = self._xmax[s], self._ymax[s]
        sp = np.linspace(0, X, 60)
        sd = np.sqrt(np.maximum(va - 2 * sp * cab + sp * sp * vb, 0))
        with np.errstate(over='ignore', invalid='ignore'):
            mid = sp * np.exp(np.minimum(alpha - beta * sp, 50))
            lo = sp * np.exp(np.minimum(alpha - beta * sp - 1.2816 * sd, 50))
            hi = sp * np.exp(np.minimum(alpha - beta * sp + 1.2816 * sd, 50))
        px = x + w * sp / X
        py = lambda r: y + h * np.clip(np.nan_to_num(r, nan=0.0, posinf=Y) / Y, 0, 1)
        artists = [ax.fill_between(px, py(lo), py(hi), facecolor=BAND, edgecolor='none', alpha=0.8, zorder=zorder),
                   ax.plot(px, py(mid), color=CURVE, linewidth=1.4 * self._line, zorder=zorder + 0.2)[0],
                   ax.plot([x, x + w * min(1.0, Y / X)], [y, y + h * min(1.0, X / Y)], color=REPLACE,
                           linewidth=0.8 * self._line, linestyle=(0, (3, 2)), zorder=zorder + 0.1)[0]]
        seen = self._history[s]
        if seen:
            dx = [x + w * min(a / X, 1.0) for (a, _) in seen]
            dy = [y + h * min(b / Y, 1.0) for (_, b) in seen]
            size = (2.2 * self._line) ** 2
            artists.append(ax.scatter(dx[:-1], dy[:-1], s=size, color=DOT, edgecolor='none', zorder=zorder + 0.3))
            artists.append(ax.scatter(dx[-1:], dy[-1:], s=1.8 * size, color=NEW_DOT, edgecolor=DOT,
                                      linewidth=0.4 * self._line, zorder=zorder + 0.4))
        return artists

    def _draw_panel(self):
        p, top = self.panel, self._ph
        p.text(0.0, top - 0.02, 'Adaptive fishery', fontsize=15, fontweight='bold', color=TEXT, va='top')
        p.text(0.0, top - 0.17, 'Catch the most fish over the years,\nlearning how each stock renews itself',
               fontsize=9.5, color=TEXT, va='top')
        y = top - 0.5
        old_line, self._line = self._line, 1.0
        for kind, label in (('run', 'Fish coming back\nfrom the sea'), ('catch', 'Caught last time'),
                            ('spawn', 'Spawned last time,\nin nests of the gravel'),
                            ('curve', 'Recruits from spawners,\nby the estimates'), ('band', 'Uncertainty of the\nestimates'),
                            ('dots', 'Spawners and their\nrecruits seen'), ('line', 'Fish just replace\nthemselves')):
            if kind == 'run':
                p.add_patch(patches.Rectangle((0.02, y - 0.07), 0.22, 0.14, facecolor=SEA, edgecolor='none'))
                self._school(p, 0.02, y - 0.07, 0.22, 0.14, (2, 1), 2, RUN)
            elif kind == 'catch':
                p.add_patch(patches.Rectangle((0.02, y - 0.07), 0.22, 0.14, facecolor=RIVER, edgecolor='none'))
                p.add_patch(patches.Polygon([(0.04, y - 0.01), (0.22, y - 0.01), (0.19, y - 0.06), (0.07, y - 0.06)],
                                            closed=True, facecolor=HULL, edgecolor='none', zorder=4))
                self._school(p, 0.05, y - 0.02, 0.16, 0.08, (2, 1), 2, RUN)
            elif kind == 'spawn':
                p.add_patch(patches.Rectangle((0.02, y - 0.07), 0.22, 0.14, facecolor=GRAVEL, edgecolor='none'))
                p.add_patch(patches.Ellipse((0.185, y), 0.08, 0.1, facecolor='none', edgecolor=NEST, linewidth=0.5,
                                            linestyle=(0, (2, 1.5))))
                self._school(p, 0.02, y - 0.07, 0.22, 0.14, (2, 1), 1, SPAWNER)
            elif kind == 'curve':
                xs = np.linspace(0, 1, 30)
                p.plot(0.03 + 0.2 * xs, y - 0.06 + 0.14 * xs * np.exp(1.6 * (1 - xs)) / np.exp(0.6) * 0.75,
                       color=CURVE, linewidth=1.4)
            elif kind == 'band':
                p.add_patch(patches.Rectangle((0.03, y - 0.05), 0.2, 0.1, facecolor=BAND, edgecolor='none', alpha=0.8))
            elif kind == 'dots':
                p.scatter([0.06, 0.12], [y - 0.02, y + 0.03], s=5, color=DOT, edgecolor='none')
                p.scatter([0.19], [y + 0.0], s=9, color=NEW_DOT, edgecolor=DOT, linewidth=0.4)
            else:
                p.plot([0.03, 0.23], [y - 0.06, y + 0.06], color=REPLACE, linewidth=0.8, linestyle=(0, (3, 2)))
            p.text(0.3, y, label, fontsize=9.5, color=TEXT, va='center')
            y -= 0.215
        self._line = old_line
        p.text(0.0, y + 0.02, 'A fish counts the same number\nof fish in all pictures of a river', fontsize=8.5,
               color='#78909c', va='top', style='italic')

    # ===== each frame =====

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        get = lambda name, s: float(state[RDDLPlanningModel.ground_var(name, [s])])

        # before the first generation nothing was caught or spawned: a new episode
        started = any(get('last-caught', s) > 0 or get('last-spawners', s) > 0 for s in self._stocks)
        artists = []
        for s in self._stocks:
            x, y = self._origin[s]
            run, caught, spawned = get('recruits', s), get('last-caught', s), get('last-spawners', s)
            if not started:
                self._history[s] = []
            elif spawned > 0 and (not self._history[s] or self._history[s][-1] != (spawned, run)):
                self._history[s].append((spawned, run))
            rx, ry, size = x + 0.08, y + 0.2, 1.1
            sea, river, gravel = self._bands(size)
            artists.extend(self._school(self.ax, rx + 0.03, ry + 0.02, size - 0.06, sea - 0.06, RUN_GRID,
                                        self._places(run, s), RUN))
            artists.extend(self._boat(self.ax, rx, ry, size, self._places(caught, s)))
            artists.extend(self._school(self.ax, rx, ry + sea + river, size, gravel, SPAWN_GRID,
                                        self._places(spawned, s), SPAWNER, facing=-1))
            artists.extend(self._belief(self.ax, *self._card_box(x, y), s, get('alpha-hat', s), get('beta-hat', s),
                                        get('var-alpha', s), get('var-beta', s), get('cov-alpha-beta', s)))

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
