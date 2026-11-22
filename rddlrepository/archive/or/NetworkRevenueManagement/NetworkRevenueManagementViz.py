import math

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from scipy.stats import beta as beta_dist, gamma as gamma_dist

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f2f4f6'
FLOOR = '#e8e2d6'
CARD = '#fffdf7'
CARD_EDGE = '#8d6e63'
LIT = '#ffc107'
GLOW = '#ffe082'
UNLIT = '#cfd8dc'
TAG = '#fff3c4'
TAG_EDGE = '#a1887f'
PERSON = '#37474f'
MAYBE = '#b0bec5'
ROOM = '#efebe9'
ROOF = '#6d4c41'
CRATE = '#c58b4c'
CRATE_EDGE = '#7b5126'
EMPTY_SLOT = '#d7ccc8'
TEXT = '#37474f'
PRODUCT_COLORS = ['#1e88e5', '#e53935', '#43a047', '#8e24aa', '#fb8c00', '#00897b', '#6d4c41', '#3949ab']

CROWD = (5, 2)          # people in the crowd of a product on a card, across and up
CRATES = (5, 4)         # crates in a stockroom, across and up
CARD_W, ROW_H, HEAD_H = 2.2, 0.7, 0.42
ROOM_W, ROOM_H = 1.6, 1.45
GAP = 0.25


class NetworkRevenueManagementVisualizer(BaseViz):
    '''Draws the price vectors as cards on a counter, and the resources as stockrooms below them.

    A card lists the price of each product under its price vector, beside the crowd of customers a
    period the retailer expects for the product at those prices: dark for its current estimate (the
    posterior mean), and faded up to the 90% point of its belief, so the crowd is sure once the retailer
    has learned the demand. A person stands for the same number of customers a period on every card of a
    product (a fixed number for each product, from its prior). The lamp of the card offered last period
    is lit; no lamp is lit after a period with nothing offered.

    A stockroom shows what is left of a resource as crates, each the same number of units (a fixed
    number for each resource, from its starting inventory), and above it the units of the resource that
    a unit of each product uses, by the colour of the product.
    '''

    def __init__(self, model: RDDLPlanningModel, figure_size=(10.5, 6.6), wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._products = model.type_to_objects['product']
        self._resources = model.type_to_objects['resource']
        self._prices = model.type_to_objects['price']
        nf = model.ground_vars_with_values(model.non_fluents)
        init = model.ground_vars_with_values(model.state_fluents)
        g2 = lambda name, a, b: float(nf[RDDLPlanningModel.ground_var(name, [a, b])])
        self._unit_demand = bool(nf['UNIT-DEMAND'])
        self._price = {(i, k): g2('PRICE', i, k) for i in self._products for k in self._prices}
        self._a0 = {(i, k): max(g2('PRIOR-A', i, k), 1e-12) for i in self._products for k in self._prices}
        self._b0 = {(i, k): max(g2('PRIOR-B', i, k), 1e-12) for i in self._products for k in self._prices}
        self._uses = {(i, j): g2('USES', i, j) for i in self._products for j in self._resources}
        self._color = {i: PRODUCT_COLORS[n % len(PRODUCT_COLORS)] for (n, i) in enumerate(self._products)}

        # scales: a person of each product (its 90% point under the prior, over the price vectors, fills
        # a crowd), and a crate of each resource (its starting inventory fills a stockroom)
        places = CROWD[0] * CROWD[1]
        self._person_unit = {}
        for i in self._products:
            high = max(self._quantiles(self._a0[(i, k)], self._b0[(i, k)], 0, 0)[1] for k in self._prices)
            self._person_unit[i] = self._nice(high / places)
        self._crate_unit = {}
        for j in self._resources:
            start = float(init[RDDLPlanningModel.ground_var('inventory', [j])])
            self._crate_unit[j] = self._nice(max(start, 1.0) / (CRATES[0] * CRATES[1]))
        self._digits = 2 if any(abs(p - round(p)) > 1e-9 for p in self._price.values()) else 0

        # the layout, as large as fits the scene
        self.fig = plt.figure(figsize=figure_size)
        self.fig.patch.set_facecolor(BACKGROUND)
        w, h = figure_size
        box_w, box_h = 0.70, 0.94
        aspect = (box_h * h) / (box_w * w)
        card_h = HEAD_H + ROW_H * len(self._products)
        K, M = len(self._prices), len(self._resources)
        best = None
        for cols in range(1, K + 1):
            rows = math.ceil(K / cols)
            span_w = max(cols * (CARD_W + GAP), ROOM_W + GAP)
            room_cols = min(max(int(span_w // (ROOM_W + GAP)), 1), M)
            room_rows = math.ceil(M / room_cols)
            span_h = rows * (card_h + GAP) + 0.3 + room_rows * (ROOM_H + GAP)
            scale = min(1.0 / span_w, aspect / span_h)
            if best is None or scale > best[0]:
                best = (scale, cols, rows, room_cols, span_w, span_h)
        _, cols, rows, room_cols, span_w, span_h = best
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
        self._font = min(10, 0.14 * points)
        self._line = max(0.3, min(1.2, 0.012 * points))

        x0 = (span_w - cols * (CARD_W + GAP) + GAP) / 2
        self._card_at = {}
        for (n, k) in enumerate(self._prices):
            x = x0 + (n % cols) * (CARD_W + GAP)
            y = span_h - (n // cols + 1) * (card_h + GAP) + GAP
            self._card_at[k] = (x, y)
            self._card(x, y, card_h, k)
        rx0 = (span_w - min(M, room_cols) * (ROOM_W + GAP) + GAP) / 2
        self._room_at = {}
        for (n, j) in enumerate(self._resources):
            x = rx0 + (n % room_cols) * (ROOM_W + GAP)
            y = (math.ceil(M / room_cols) - 1 - n // room_cols) * (ROOM_H + GAP)
            self._room_at[j] = (x, y)
            self._room(x, y, j)

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

    def _quantiles(self, a, b, demanded, offered):
        '''The posterior mean of a mean demand and its 90% point.'''
        if self._unit_demand:
            pa, pb = a + demanded, max(b + offered - demanded, 1e-12)
            return pa / (pa + pb), float(beta_dist.ppf(0.9, pa, pb))
        shape, rate = a + demanded, b + offered
        return shape / rate, float(gamma_dist.ppf(0.9, shape, scale=1.0 / rate))

    # ===== pictures =====

    def _text(self, ax, x, y, s, size, **kw):
        if size >= 5:
            ax.text(x, y, s, fontsize=size, color=TEXT, **kw)

    def _card(self, x, y, h, k):
        ax = self.ax
        ax.add_patch(patches.FancyBboxPatch((x, y), CARD_W, h, boxstyle='round,pad=0,rounding_size=0.08',
                                            facecolor=CARD, edgecolor=CARD_EDGE, linewidth=self._line, zorder=2))
        self._text(ax, x + 0.12, y + h - HEAD_H / 2, k, self._font, va='center', ha='left', zorder=3)
        for (n, i) in enumerate(self._products):
            ry = y + h - HEAD_H - (n + 1) * ROW_H
            self._product_box(ax, x + 0.12, ry + 0.2, 0.28, self._color[i])
            self._text(ax, x + 0.28, ry + 0.16, i, self._font * 0.75, ha='center', va='top', zorder=3)
            # the price tag
            tx, ty = x + 0.46, ry + 0.2
            ax.add_patch(patches.Polygon([(tx, ty + 0.13), (tx + 0.08, ty + 0.26), (tx + 0.5, ty + 0.26),
                                          (tx + 0.5, ty), (tx + 0.08, ty)], closed=True, facecolor=TAG,
                                         edgecolor=TAG_EDGE, linewidth=0.6 * self._line, zorder=3))
            p = self._price[(i, k)]
            label = ('%.' + str(self._digits) + 'f') % p
            self._text(ax, tx + 0.29, ty + 0.13, label, min(self._font * 0.85, self._font * 4.5 / max(len(label), 4)),
                       ha='center', va='center', zorder=4)

    @staticmethod
    def _product_box(ax, x, y, s, color, zorder=3):
        parts = [patches.Rectangle((x, y), s, 0.8 * s, facecolor=color, edgecolor='#263238', linewidth=0.5),
                 patches.Polygon([(x, y + 0.8 * s), (x + 0.2 * s, y + s), (x + 1.2 * s, y + s), (x + s, y + 0.8 * s)],
                                 closed=True, facecolor=color, edgecolor='#263238', linewidth=0.5, alpha=0.7)]
        for part in parts:
            part.set_zorder(zorder)
            ax.add_patch(part)

    def _lamp(self, ax, cx, cy, r, lit, zorder=4):
        parts = []
        if lit:
            parts.append(ax.add_patch(patches.Circle((cx, cy), 1.8 * r, facecolor=GLOW, edgecolor='none',
                                                     alpha=0.8, zorder=zorder)))
        parts.append(ax.add_patch(patches.Circle((cx, cy), r, facecolor=LIT if lit else UNLIT, edgecolor='#455a64',
                                                 linewidth=0.6 * self._line, zorder=zorder + 0.1)))
        return parts

    @staticmethod
    def _people(x, y, w, h, grid, sure, maybe):
        '''People in a grid filling (x, y, w, h): sure ones dark, possible ones faded; and the overflow.'''
        nx, ny = grid
        cw, ch = w / nx, h / ny
        dark, light = [], []
        for n in range(min(sure + maybe, nx * ny)):
            cx, base, s = x + (n % nx + 0.5) * cw, y + (n // nx) * ch + 0.04 * ch, 0.9 * ch
            body = [patches.FancyBboxPatch((cx - 0.22 * s, base), 0.44 * s, 0.55 * s,
                                           boxstyle='round,pad=0,rounding_size=%g' % (0.18 * s)),
                    patches.Circle((cx, base + 0.75 * s), 0.18 * s)]
            (dark if n < sure else light).extend(body)
        return dark, light, sure + maybe > nx * ny

    def _crowd(self, ax, x, y, w, h, sure, maybe, zorder=4):
        dark, light, over = self._people(x, y, w, h, CROWD, sure, maybe)
        artists = [ax.add_collection(PatchCollection(dark, facecolor=PERSON, edgecolor='none', zorder=zorder)),
                   ax.add_collection(PatchCollection(light, facecolor=MAYBE, edgecolor='none', zorder=zorder))]
        if over:
            a = 0.12
            artists.append(ax.add_patch(patches.Polygon([(x + w - a, y + h), (x + w, y + h), (x + w - a / 2, y + h + 0.1)],
                                                        closed=True, facecolor=PERSON, edgecolor='none', zorder=zorder)))
        return artists

    def _room(self, x, y, j):
        ax = self.ax
        body_h = ROOM_H - 0.55
        ax.add_patch(patches.Rectangle((x, y), ROOM_W, body_h, facecolor=ROOM, edgecolor=ROOF,
                                       linewidth=self._line, zorder=2))
        ax.add_patch(patches.Polygon([(x - 0.06, y + body_h), (x + ROOM_W + 0.06, y + body_h),
                                      (x + ROOM_W / 2, y + body_h + 0.22)], closed=True, facecolor=ROOF,
                                     edgecolor='none', zorder=2))
        self._text(ax, x + ROOM_W / 2, y - 0.04, j, self._font, ha='center', va='top', zorder=3)
        # the units of this resource in a unit of each product that uses it
        users = [i for i in self._products if self._uses[(i, j)] > 0]
        for (n, i) in enumerate(users[:6]):
            cx = x + (n + 0.5) * ROOM_W / max(len(users[:6]), 1) - 0.12
            self._product_box(ax, cx - 0.08, y + body_h + 0.27, 0.13, self._color[i])
            u = self._uses[(i, j)]
            self._text(ax, cx + 0.12, y + body_h + 0.33, '%g' % u, self._font * 0.85, ha='left', va='center',
                       zorder=3)
        # the empty places of the crates
        nx, ny = CRATES
        cw, ch = (ROOM_W - 0.1) / nx, (body_h - 0.08) / ny
        slots = [patches.Rectangle((x + 0.05 + (n % nx) * cw + 0.08 * cw, y + 0.04 + (n // nx) * ch + 0.08 * ch),
                                   0.84 * cw, 0.84 * ch) for n in range(nx * ny)]
        ax.add_collection(PatchCollection(slots, facecolor='none', edgecolor=EMPTY_SLOT, linewidth=0.5 * self._line,
                                          zorder=2.5))

    def _crates(self, x, y, count, zorder=3):
        nx, ny = CRATES
        body_h = ROOM_H - 0.55
        cw, ch = (ROOM_W - 0.1) / nx, (body_h - 0.08) / ny
        boxes = [patches.Rectangle((x + 0.05 + (n % nx) * cw + 0.08 * cw, y + 0.04 + (n // nx) * ch + 0.08 * ch),
                                   0.84 * cw, 0.84 * ch) for n in range(min(count, nx * ny))]
        return [self.ax.add_collection(PatchCollection(boxes, facecolor=CRATE, edgecolor=CRATE_EDGE,
                                                       linewidth=0.5 * self._line, zorder=zorder))]

    def _draw_panel(self):
        p, top = self.panel, self._ph
        p.text(0.0, top - 0.02, 'Network revenue\nmanagement', fontsize=15, fontweight='bold', color=TEXT, va='top')
        p.text(0.0, top - 0.42, 'Earn the most from the stock over\nthe season, learning the demand',
               fontsize=9.5, color=TEXT, va='top')
        y = top - 0.78
        old_line, self._line = self._line, 1.0
        for kind, label in (('lamp', 'Prices offered\nlast period'), ('tag', 'Price of a product'),
                            ('sure', 'Customers a period,\nby the estimates'),
                            ('maybe', 'More, if demand is\nhigher than estimated'),
                            ('crate', 'Stock left of\na resource'), ('uses', 'Units of the resource\nin a product')):
            if kind == 'lamp':
                self._lamp(p, 0.13, y, 0.045, True)
            elif kind == 'tag':
                p.add_patch(patches.Polygon([(0.03, y), (0.07, y + 0.06), (0.23, y + 0.06), (0.23, y - 0.06),
                                             (0.07, y - 0.06)], closed=True, facecolor=TAG, edgecolor=TAG_EDGE,
                                            linewidth=0.6))
            elif kind in ('sure', 'maybe'):
                dark, light, _ = self._people(0.03, y - 0.08, 0.21, 0.16, (3, 1), 3 if kind == 'sure' else 0,
                                              0 if kind == 'sure' else 3)
                p.add_collection(PatchCollection(dark or light, facecolor=PERSON if kind == 'sure' else MAYBE,
                                                 edgecolor='none'))
            elif kind == 'crate':
                for n in range(3):
                    p.add_patch(patches.Rectangle((0.04 + 0.065 * n, y - 0.04), 0.055, 0.08, facecolor=CRATE,
                                                  edgecolor=CRATE_EDGE, linewidth=0.5))
            else:
                self._product_box(p, 0.06, y - 0.05, 0.08, PRODUCT_COLORS[0])
                p.text(0.17, y - 0.01, '3', fontsize=9, color=TEXT, va='center')
            p.text(0.3, y, label, fontsize=9.5, color=TEXT, va='center')
            y -= 0.23
        self._line = old_line
        p.text(0.0, y + 0.02, 'A person counts the same customers\non every card of a product',
               fontsize=8.5, color='#78909c', va='top', style='italic')

    # ===== each frame =====

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        get = lambda name, *objs: float(state[RDDLPlanningModel.ground_var(name, list(objs))])
        artists = []
        card_h = HEAD_H + ROW_H * len(self._products)
        for k in self._prices:
            x, y = self._card_at[k]
            artists.extend(self._lamp(self.ax, x + CARD_W - 0.2, y + card_h - HEAD_H / 2, 0.09,
                                      bool(get('last-offer', k))))
            n_k = get('offered', k)
            for (n, i) in enumerate(self._products):
                ry = y + card_h - HEAD_H - (n + 1) * ROW_H
                mean, high = self._quantiles(self._a0[(i, k)], self._b0[(i, k)], get('demanded', i, k), n_k)
                u = self._person_unit[i]
                sure = int(round(mean / u))
                maybe = max(int(round(high / u)) - sure, 0)
                artists.extend(self._crowd(self.ax, x + 1.05, ry + 0.06, 1.05, ROW_H - 0.14, sure, maybe))
        for j in self._resources:
            x, y = self._room_at[j]
            left = get('inventory', j)
            artists.extend(self._crates(x, y, int(math.ceil(max(left, 0.0) / self._crate_unit[j] - 1e-9))))

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
