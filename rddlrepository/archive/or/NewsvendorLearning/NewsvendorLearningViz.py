import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from scipy.special import gammaincinv
from scipy.stats import nbinom

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f2f4f6'
FLOOR = '#e8e2d6'
SHOP = '#fff8e1'
SHOP_EDGE = '#5d4037'
AWNING = ['#1e88e5', '#ffffff']
WINDOW = '#e3f2fd'
EMPTY = '#ffcdd2'
GOODS = '#c58b4c'
GOODS_EDGE = '#7b5126'
SOLD_EDGE = '#90a4ae'
PERSON = '#37474f'
MAYBE = '#b0bec5'
HOURS = '#fff3c4'
RED = '#d62828'
TEXT = '#37474f'

GRID = (5, 4)           # boxes on a shelf and people in a crowd, across and up
OPEN, CLOSE = 9, 17     # a day runs from 9 to 5 on the clock
CELL = (2.3, 1.55)      # a product's corner of the shop: its stall and its crowd


class NewsvendorLearningVisualizer(BaseViz):
    '''Draws the shop at closing time, a stall for each product with the crowd of customers it expects.

    The window of a stall shows its shelf as a grid of 20 places, each for the same number of units (a
    fixed number for each product, enough for 20 places to hold most of the demand its prior allows). A
    box still standing was left over and is thrown away; the outline of a box was sold. A stall that sold
    out has a red window and a clock showing when it sold out, the day running from 9 to 5. Beside each
    stall, on the same scale, stand the customers the shop expects a day from its belief (the mean), and
    faded the ones it may also get (up to the 90% point of its predictive distribution), so the crowd is
    sure and small once the shop has learned the demand. An arrow above a grid shows that it overflows.
    '''

    def __init__(self, model: RDDLPlanningModel, figure_size=(10.5, 6.6), wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._products = model.type_to_objects['product']
        nf = model.ground_vars_with_values(model.non_fluents)
        g = lambda name, p: float(nf[RDDLPlanningModel.ground_var(name, [p])])
        self._a0 = {p: g('PRIOR-SHAPE', p) for p in self._products}
        self._b0 = {p: g('PRIOR-RATE', p) for p in self._products}
        self._shelf = float(nf['SHELF-SPACE'])

        # units for a place on the shelf or in the crowd: 1, 2, 5, 10, ... so that 20 hold the demand
        # up to the 97.5% point of the prior predictive
        places = GRID[0] * GRID[1]
        self._unit = {}
        for p in self._products:
            high = float(nbinom(self._a0[p], self._b0[p] / (self._b0[p] + 1)).ppf(0.975))
            self._unit[p] = self._nice(max(high, 1.0) / places)

        # the shelf space is shown only if it could run out: if it is smaller than plausible demand
        prior_high = sum(float(gammaincinv(self._a0[p], 0.995)) / self._b0[p] for p in self._products)
        self._shelf_shown = self._shelf < prior_high

        # the grid of products, as large as fits the scene
        self.fig = plt.figure(figsize=figure_size)
        self.fig.patch.set_facecolor(BACKGROUND)
        w, h = figure_size
        box_w, box_h = 0.70, 0.94
        aspect = (box_h * h) / (box_w * w)
        num = len(self._products)
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
                                                 facecolor=FLOOR, edgecolor='none', zorder=0))
        points = box_w * w * 72 / (2 * half_w)          # points in a unit of the scene
        self._name_size = min(11, 0.16 * points)
        self._origin = {}
        for (i, p) in enumerate(self._products):
            x, y = (i % self._cols) * CELL[0], (self._rows - 1 - i // self._cols) * CELL[1]
            self._origin[p] = (x, y)
            self._stall(self.ax, x + 0.15, y + 0.2, 1.1, p)

        pw = 0.27
        self.panel = self.fig.add_axes([0.725, 0.03, pw, 0.94])
        self._ph = 0.94 * h / (pw * w)
        self.panel.set_xlim(0, 1)
        self.panel.set_ylim(0, self._ph)
        self.panel.set_aspect('equal')
        self.panel.axis('off')
        self._draw_panel()

    # ===== pictures =====

    @staticmethod
    def _nice(v):
        k = 10 ** math.floor(math.log10(max(v, 1.0)))
        for m in (1, 2, 5, 10):
            if m * k >= v - 1e-9:
                return m * k
        return 10 * k

    @staticmethod
    def _window(x, y, size):
        '''The window of a stall of width size whose lower left corner is (x, y).'''
        return x + 0.08 * size, y + 0.1 * size, 0.84 * size, 0.62 * size

    def _stall(self, ax, x, y, size, name=None, zorder=2):
        '''A market stall: a counter under a striped awning; its window shows the shelf.'''
        w, h = size, 0.82 * size
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=SHOP, edgecolor=SHOP_EDGE, linewidth=0.8,
                                       zorder=zorder))
        n = 5
        for k in range(n):
            ax.add_patch(patches.Polygon([(x + k * w / n, y + h), (x + (k + 1) * w / n, y + h),
                                          (x + (k + 1) * w / n - 0.02 * w, y + h + 0.2 * h),
                                          (x + k * w / n + 0.02 * w, y + h + 0.2 * h)], closed=True,
                                         facecolor=AWNING[k % 2], edgecolor=SHOP_EDGE, linewidth=0.4,
                                         zorder=zorder + 0.1))
        if name is not None and self._name_size >= 6:
            ax.text(x + w / 2, y - 0.06, name, ha='center', va='top', fontsize=self._name_size, color=TEXT,
                    zorder=zorder)

    def _shelf_goods(self, ax, x, y, size, left, sold, sold_out, zorder=3):
        '''The window of a stall: standing boxes left over, outlines of boxes sold, red when sold out.'''
        wx, wy, ww, wh = self._window(x, y, size)
        parts = [patches.Rectangle((wx, wy), ww, wh, facecolor=EMPTY if sold_out else WINDOW,
                                   edgecolor=RED if sold_out else SHOP_EDGE, linewidth=1.6 if sold_out else 0.5)]
        nx, ny = GRID
        cw, ch = ww / nx, wh / ny
        for k in range(min(left + sold, nx * ny)):
            i, j = k % nx, k // nx
            box = (wx + i * cw + 0.14 * cw, wy + j * ch + 0.12 * ch)
            if k < left:
                parts.append(patches.Rectangle(box, 0.72 * cw, 0.76 * ch, facecolor=GOODS, edgecolor=GOODS_EDGE,
                                               linewidth=0.4))
            else:
                parts.append(patches.Rectangle(box, 0.72 * cw, 0.76 * ch, facecolor='none', edgecolor=SOLD_EDGE,
                                               linewidth=0.7, linestyle=(0, (2, 1.5))))
        if left + sold > nx * ny:
            parts.append(self._overflow(wx + ww / 2, wy + wh + 0.04, 0.6 * cw))
        for k, part in enumerate(parts):
            part.set_zorder(zorder + 0.01 * k)
            ax.add_patch(part)
        return parts

    @staticmethod
    def _overflow(cx, base, w):
        return patches.Polygon([(cx - w / 2, base), (cx + w / 2, base), (cx, base + 0.7 * w)], closed=True,
                               facecolor=SHOP_EDGE, edgecolor='none')

    @staticmethod
    def _person(ax, cx, base, h, color, zorder=4):
        '''A person standing on base, of height h.'''
        parts = [patches.FancyBboxPatch((cx - 0.22 * h, base), 0.44 * h, 0.55 * h,
                                        boxstyle='round,pad=0,rounding_size=%g' % (0.18 * h), facecolor=color,
                                        edgecolor='none'),
                 patches.Circle((cx, base + 0.75 * h), 0.18 * h, facecolor=color, edgecolor='none')]
        for part in parts:
            part.set_zorder(zorder)
            ax.add_patch(part)
        return parts

    def _crowd(self, ax, x, y, size, sure, maybe, zorder=4):
        '''The customers expected a day, in a grid as tall as a stall: sure ones dark, possible ones faded.'''
        wx, wy, ww, wh = x, y, 0.9 * size, 0.95 * size
        nx, ny = GRID
        cw, ch = ww / nx, wh / ny
        parts = []
        for k in range(min(sure + maybe, nx * ny)):
            i, j = k % nx, k // nx
            parts.extend(self._person(ax, wx + (i + 0.5) * cw, wy + j * ch + 0.06 * ch, 0.9 * ch,
                                      PERSON if k < sure else MAYBE, zorder))
        if sure + maybe > nx * ny:
            arrow = self._overflow(wx + ww / 2, wy + wh + 0.04, 0.6 * cw)
            arrow.set_facecolor(PERSON)
            ax.add_patch(arrow)
            parts.append(arrow)
        return parts

    @staticmethod
    def _clock(ax, cx, cy, rad, t, zorder=6):
        '''A wall clock at the time of day t (0 at opening, 1 at closing), opening hours shaded.'''
        angle = lambda hour: 90 - 30 * (hour % 12)
        parts = [ax.add_patch(patches.Circle((cx, cy), rad, facecolor='white', edgecolor='#455a64',
                                             linewidth=1.0, zorder=zorder)),
                 ax.add_patch(patches.Wedge((cx, cy), rad * 0.92, angle(CLOSE), angle(OPEN), facecolor=HOURS,
                                            edgecolor='none', zorder=zorder))]
        for hour in range(12):
            a = math.radians(angle(hour))
            parts.append(ax.plot([cx + 0.78 * rad * math.cos(a), cx + 0.94 * rad * math.cos(a)],
                                 [cy + 0.78 * rad * math.sin(a), cy + 0.94 * rad * math.sin(a)],
                                 color='#607d8b', linewidth=0.7, zorder=zorder)[0])
        a = math.radians(angle(OPEN + (CLOSE - OPEN) * min(max(t, 0.0), 1.0)))
        parts.append(ax.plot([cx, cx + 0.7 * rad * math.cos(a)], [cy, cy + 0.7 * rad * math.sin(a)],
                             color='#263238', linewidth=1.6, solid_capstyle='round', zorder=zorder)[0])
        parts.append(ax.add_patch(patches.Circle((cx, cy), 0.09 * rad, facecolor='#263238', zorder=zorder)))
        return parts

    def _draw_panel(self):
        p, top = self.panel, self._ph
        p.text(0.0, top - 0.02, 'Newsvendor learning', fontsize=15, fontweight='bold', color=TEXT, va='top')
        y = top - 0.3
        rows = [
            ('left', 'Left over at closing,\nthrown away'),
            ('sold', 'Sold today'),
            ('out', 'Sold out, at the time\non the clock'),
            ('sure', 'Customers expected\na day'),
            ('maybe', 'Customers possible\non a busy day'),
        ]
        if self._shelf_shown:
            rows.append(('space', 'Shelf space used\ntoday'))
        for kind, label in rows:
            if kind in ('left', 'sold', 'out'):
                self._stall(p, 0.03, y - 0.09, 0.2)
                self._shelf_goods(p, 0.03, y - 0.09, 0.2, 7 if kind == 'left' else 0, 7 if kind == 'sold' else 0,
                                  kind == 'out')
                if kind == 'out':
                    self._clock(p, 0.22, y + 0.1, 0.055, 0.55)
            elif kind in ('sure', 'maybe'):
                for k in range(3):
                    self._person(p, 0.06 + 0.07 * k, y - 0.07, 0.15, PERSON if kind == 'sure' else MAYBE)
            else:
                # filled each frame by the space the day's stock took
                p.add_patch(patches.Rectangle((0.02, y - 0.04), 0.22, 0.08, facecolor='white', edgecolor=SHOP_EDGE,
                                              linewidth=0.8, zorder=2))
            p.text(0.3, y, label, fontsize=9.5, color=TEXT, va='center')
            y -= 0.24
        p.text(0.0, y + 0.02, 'A box and a person count\nthe same number of units', fontsize=8.5,
               color='#78909c', va='top', style='italic')
        if self._shelf_shown:
            self._space_y = top - 0.3 - 0.24 * 5
            self._space_artist = None

    # ===== each frame =====

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        get = lambda name, p: float(state[RDDLPlanningModel.ground_var(name, [p])])

        # before the first day nothing has been stocked or sold
        started = any(get('selling-time', p) > 0 or get('last-stock', p) > 0 for p in self._products)
        artists, used = [], 0.0
        for p in self._products:
            x, y = self._origin[p]
            u = self._unit[p]
            stock, sold, on_sale = get('last-stock', p), get('last-sold', p), get('last-on-sale', p)
            used += stock
            left = stock - sold
            left_places = int(math.ceil(left / u - 1e-9))
            sold_places = int(math.ceil(stock / u - 1e-9)) - left_places
            sold_out = started and on_sale < 1
            artists.extend(self._shelf_goods(self.ax, x + 0.15, y + 0.2, 1.1, left_places, sold_places, sold_out))
            if sold_out:
                artists.extend(self._clock(self.ax, x + 1.18, y + 1.28, 0.17, on_sale))

            # the crowd the shop expects a day, from its belief
            a, b = self._a0[p] + get('sales', p), self._b0[p] + get('selling-time', p)
            sure = int(round(a / b / u))
            high = float(nbinom(a, b / (b + 1)).ppf(0.9))
            maybe = max(int(round(high / u)) - sure, 0)
            artists.extend(self._crowd(self.ax, x + 1.33, y + 0.2, 1.0, sure, maybe))

        if self._shelf_shown:
            y = self._space_y
            frac = min(used / max(self._shelf, 1.0), 1.0)
            artists.append(self.panel.add_patch(patches.Rectangle((0.02, y - 0.04), 0.22 * frac, 0.08,
                                                                  facecolor=GOODS, edgecolor='none', zorder=3)))

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
