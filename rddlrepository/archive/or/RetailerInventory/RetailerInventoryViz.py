import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f2f4f6'
LAND = '#e6ebdf'
ROAD = '#9aa5ad'
ROAD_EDGE = '#7d8890'
ROAD_LINE = '#f5f5f5'
WALL = '#d7ccc8'
WALL_EDGE = '#6d4c41'
ROOF = '#8d6e63'
FACTORY = '#b0bec5'
FACTORY_EDGE = '#455a64'
SMOKE = '#cfd8dc'
SHOP = '#fff8e1'
SHOP_EDGE = '#5d4037'
AWNING = ['#e53935', '#ffffff']
WINDOW = '#e3f2fd'
EMPTY = '#ffcdd2'
GOODS = '#c58b4c'
GOODS_EDGE = '#7b5126'
PALLET = '#d9a066'
TRUCK = '#1e88e5'
TRUCK_EDGE = '#0d47a1'
CARGO = '#ffffff'
RED = '#d62828'
TEXT = '#37474f'
MUTED = '#78909c'

BASE = 2.0          # road length for a delay of zero, in days
SHELF_BOXES = (5, 4)
RACK = (10, 5)
CRATES = 6


class RetailerInventoryVisualizer(BaseViz):
    '''Draws the supply network from above: the factory of the manufacturer on the left, the
    warehouse in the middle and the stores around it, each at the end of a road whose length is
    its delay in days (one tick mark a day). The warehouse is drawn with its racks, one pallet
    for every 50th of its capacity in stock, and each store with its shop window, one box for every
    20th of its capacity on the shelf; a store with an empty shelf is shown in red. A truck on a
    road carries goods on their way, as far along as the days they have travelled, loaded with
    one crate for every sixth of a day's production (to the warehouse) or of the store's capacity.
    '''

    def __init__(self, model: RDDLPlanningModel, figure_size=(10.5, 6.6), wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        objs = model.type_to_objects
        self._stores = list(objs.get('store', []))
        self._days = list(objs.get('day', []))
        nf = model.ground_vars_with_values(model.non_fluents)
        g = lambda name, *args: nf[RDDLPlanningModel.ground_var(name, list(args))]
        self._day_of = {d: int(g('DAYS', d)) for d in self._days}
        self._wdelay = int(g('WAREHOUSE-DELAY'))
        self._prod = max(int(g('PRODUCTION-CAPACITY')), 1)
        self._wcap = max(int(g('WAREHOUSE-CAPACITY')), 1)
        self._sdelay = [int(g('STORE-DELAY', s)) for s in self._stores]
        self._scap = [max(int(g('STORE-CAPACITY', s)), 1) for s in self._stores]

        # layout: the warehouse at the origin, the factory to the left, the stores around on the right
        K = len(self._stores)
        spread = 0.0 if K <= 1 else min(math.radians(150), math.radians(22) * (K - 1))
        self._angle = [0.0] if K == 1 else list(np.linspace(-spread, spread, K)) if K else []
        step = spread * 2 / (K - 1) if K > 1 else math.pi
        self._step = step
        self._store_r = [BASE + d for d in self._sdelay]
        self._store_xy = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(self._store_r, self._angle)]
        self._fac_xy = (-(BASE + self._wdelay), 0.0)
        r_min = min(self._store_r) if K else BASE
        self._shop_size = min(0.95, 0.8 * r_min * step) if K else 0.95
        self._wh = 2.0
        self._truck = min(0.5, 0.85 * self._shop_size + 0.1)

        xs = [self._fac_xy[0] - 0.8, 0.6 * self._wh] + [x + self._shop_size for x, _ in self._store_xy] + \
             [x - self._shop_size for x, _ in self._store_xy]
        ys = [-0.4 * self._wh, 0.6 * self._wh] + [y + self._shop_size for _, y in self._store_xy] + [y - self._shop_size for _, y in self._store_xy]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

        self.fig = plt.figure(figsize=figure_size)
        self.fig.patch.set_facecolor(BACKGROUND)
        w, h = figure_size
        box_w, box_h = 0.70, 0.94
        self.ax = self.fig.add_axes([0.01, 0.03, box_w, box_h])
        aspect = (box_h * h) / (box_w * w)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w = max((x1 - x0) / 2, (y1 - y0) / 2 / aspect) * 1.04
        self.ax.set_xlim(cx - half_w, cx + half_w)
        self.ax.set_ylim(cy - half_w * aspect, cy + half_w * aspect)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.add_patch(patches.FancyBboxPatch((cx - half_w, cy - half_w * aspect), 2 * half_w,
                                                 2 * half_w * aspect, boxstyle='round,pad=0,rounding_size=0.3',
                                                 facecolor=LAND, edgecolor='none', zorder=0))
        pw = 0.27
        self.panel = self.fig.add_axes([0.725, 0.03, pw, 0.94])
        self._ph = 0.94 * h / (pw * w)
        self.panel.set_xlim(0, 1)
        self.panel.set_ylim(0, self._ph)
        self.panel.set_aspect('equal')
        self.panel.axis('off')

        self._draw_map()
        self._draw_panel()

    # ===== pictures =====

    def _road(self, ax, start, end, days, width, tick=0.12):
        (xa, ya), (xb, yb) = start, end
        ax.plot([xa, xb], [ya, yb], color=ROAD_EDGE, linewidth=width + 2.0, solid_capstyle='round', zorder=1)
        ax.plot([xa, xb], [ya, yb], color=ROAD, linewidth=width, solid_capstyle='round', zorder=1.1)
        ax.plot([xa, xb], [ya, yb], color=ROAD_LINE, linewidth=0.8, linestyle=(0, (4, 4)), zorder=1.2)
        # a mark at the end of each day of travel
        L = math.hypot(xb - xa, yb - ya)
        if days > 0 and L > 0:
            nx, ny = -(yb - ya) / L, (xb - xa) / L
            for k in range(1, days):
                t = k / days
                px, py = xa + t * (xb - xa), ya + t * (yb - ya)
                ax.plot([px - tick * nx, px + tick * nx], [py - tick * ny, py + tick * ny], color=ROAD_EDGE,
                        linewidth=1.2, zorder=1.3)

    def _factory(self, ax, x, y, size, zorder=3):
        '''The manufacturer: a factory with a sawtooth roof and a smoking chimney.'''
        w, h = size, 0.55 * size
        bx, by = x - w / 2, y - h / 2
        ax.add_patch(patches.Rectangle((x + 0.22 * w, by + h), 0.12 * w, 0.45 * h, facecolor=FACTORY_EDGE,
                                       edgecolor='none', zorder=zorder))
        for k, (dx, dy, r) in enumerate(((0.3, 1.05, 0.09), (0.36, 1.22, 0.12), (0.46, 1.42, 0.15))):
            ax.add_patch(patches.Circle((x + dx * w, by + h + dy * h * 0.62), r * w, facecolor=SMOKE,
                                        edgecolor='none', zorder=zorder))
        teeth = 3
        pts = [(bx, by), (bx + w, by), (bx + w, by + h)]
        for k in range(teeth, 0, -1):
            pts += [(bx + k * w / teeth - 0.0, by + h), (bx + (k - 1) * w / teeth, by + h + 0.3 * h),
                    (bx + (k - 1) * w / teeth, by + h)]
        ax.add_patch(patches.Polygon(pts, closed=True, facecolor=FACTORY, edgecolor=FACTORY_EDGE, linewidth=1.0,
                                     zorder=zorder + 0.1))
        ax.add_patch(patches.Rectangle((x - 0.12 * w, by), 0.24 * w, 0.5 * h, facecolor=FACTORY_EDGE,
                                       edgecolor='none', zorder=zorder + 0.2))
        for k in (-1, 1):
            ax.add_patch(patches.Rectangle((x + k * 0.33 * w - 0.08 * w, by + 0.45 * h), 0.16 * w, 0.22 * h,
                                           facecolor=WINDOW, edgecolor=FACTORY_EDGE, linewidth=0.5,
                                           zorder=zorder + 0.2))

    def _warehouse_shell(self, ax, x, y, size, zorder=3):
        '''The warehouse: a wide building under a gabled roof, its racks seen through the front.'''
        w, h = size, 0.62 * size
        bx, by = x - w / 2, y - h / 2
        ax.add_patch(patches.Polygon([(bx - 0.05 * w, by + h), (x, by + h + 0.28 * h), (bx + 1.05 * w, by + h)],
                                     closed=True, facecolor=ROOF, edgecolor=WALL_EDGE, linewidth=1.0, zorder=zorder))
        ax.add_patch(patches.Rectangle((bx, by), w, h, facecolor=WALL, edgecolor=WALL_EDGE, linewidth=1.0,
                                       zorder=zorder))
        rx, ry, rw, rh = box = (bx + 0.08 * w, by + 0.1 * h, 0.84 * w, 0.8 * h)
        ax.add_patch(patches.Rectangle((rx, ry), rw, rh, facecolor='#efebe9', edgecolor=WALL_EDGE, linewidth=0.6,
                                       zorder=zorder + 0.1))
        nx, ny = RACK
        for j in range(1, ny):
            ax.plot([rx, rx + rw], [ry + j * rh / ny, ry + j * rh / ny], color='#a1887f', linewidth=0.6,
                    zorder=zorder + 0.2)
        return box

    def _pallets(self, ax, box, n, zorder=3.5):
        rx, ry, rw, rh = box
        nx, ny = RACK
        cw, ch = rw / nx, rh / ny
        parts = []
        for k in range(min(n, nx * ny)):
            i, j = k % nx, k // nx
            p = patches.Rectangle((rx + i * cw + 0.1 * cw, ry + j * ch + 0.1 * ch), 0.8 * cw, 0.75 * ch,
                                  facecolor=PALLET, edgecolor=GOODS_EDGE, linewidth=0.4, zorder=zorder)
            ax.add_patch(p)
            parts.append(p)
        return parts

    def _shop(self, ax, x, y, size, boxes, total_boxes=SHELF_BOXES, zorder=3, static=True):
        '''A store: a shop front with a striped awning; its window shows the goods on the shelf,
        and turns red when the shelf is empty.'''
        w, h = size, 0.8 * size
        bx, by = x - w / 2, y - h / 2
        parts = []
        if static:
            parts.append(patches.Rectangle((bx, by), w, h, facecolor=SHOP, edgecolor=SHOP_EDGE, linewidth=0.8,
                                           zorder=zorder))
            n = 5
            for k in range(n):
                parts.append(patches.Polygon([(bx + k * w / n, by + h), (bx + (k + 1) * w / n, by + h),
                                              (bx + (k + 1) * w / n - 0.02 * w, by + h + 0.22 * h),
                                              (bx + k * w / n + 0.02 * w, by + h + 0.22 * h)], closed=True,
                                             facecolor=AWNING[k % 2], edgecolor=SHOP_EDGE, linewidth=0.4,
                                             zorder=zorder + 0.1))
        wx, wy, ww, wh = bx + 0.1 * w, by + 0.12 * h, 0.8 * w, 0.7 * h
        empty = boxes == 0
        parts.append(patches.Rectangle((wx, wy), ww, wh, facecolor=EMPTY if empty else WINDOW,
                                       edgecolor=RED if empty else SHOP_EDGE, linewidth=1.4 if empty else 0.5,
                                       zorder=zorder + 0.2))
        nx, ny = total_boxes
        cw, ch = ww / nx, wh / ny
        for k in range(min(boxes, nx * ny)):
            i, j = k % nx, k // nx
            parts.append(patches.Rectangle((wx + i * cw + 0.12 * cw, wy + j * ch + 0.1 * ch), 0.76 * cw, 0.78 * ch,
                                           facecolor=GOODS, edgecolor=GOODS_EDGE, linewidth=0.3,
                                           zorder=zorder + 0.3))
        for p in parts:
            ax.add_patch(p)
        return parts

    def _truck_icon(self, ax, x, y, heading, length, crates, zorder=6):
        '''A delivery truck seen from above, facing heading, with its load of crates.'''
        L, W = length, 0.45 * length
        tr = Affine2D().rotate(heading).translate(x, y) + ax.transData
        parts = [
            patches.Rectangle((-L / 2, -W / 2), 0.72 * L, W, facecolor=CARGO, edgecolor=TRUCK_EDGE, linewidth=0.8),
            patches.FancyBboxPatch((0.25 * L, -0.44 * W), 0.25 * L, 0.88 * W,
                                   boxstyle=f'round,pad=0,rounding_size={0.12 * W}', facecolor=TRUCK,
                                   edgecolor=TRUCK_EDGE, linewidth=0.8),
            patches.Rectangle((0.38 * L, -0.36 * W), 0.07 * L, 0.72 * W, facecolor='#bbdefb', edgecolor='none'),
        ]
        per_row = CRATES // 2
        cw, ch = 0.72 * L / per_row, W / 2
        for k in range(min(crates, CRATES)):
            i, j = k // 2, k % 2
            parts.append(patches.Rectangle((-L / 2 + i * cw + 0.1 * cw, -W / 2 + j * ch + 0.12 * ch), 0.8 * cw,
                                           0.76 * ch, facecolor=GOODS, edgecolor=GOODS_EDGE, linewidth=0.3))
        for k, p in enumerate(parts):
            p.set_transform(tr)
            p.set_zorder(zorder + 0.01 * k)
            ax.add_patch(p)
        return parts

    # ===== the scene =====

    def _edge(self, center, toward, half):
        '''The point where the road from center toward a target leaves a building of half size.'''
        (cx, cy), (tx, ty) = center, toward
        L = math.hypot(tx - cx, ty - cy)
        if L == 0:
            return center
        return cx + half * (tx - cx) / L, cy + half * (ty - cy) / L

    def _draw_map(self):
        ax = self.ax
        # roads as wide as the room between them where they leave the warehouse allows
        lo, hi = ax.get_xlim()
        points = ax.get_position().width * self.fig.get_figwidth() * 72 / (hi - lo)
        w = max(1.5, min(9.0, 0.55 * points * 0.5 * self._wh * self._step - 2.0))
        self._legs = []
        start = self._edge((0, 0), self._fac_xy, 0.5 * self._wh)
        end = self._edge(self._fac_xy, (0, 0), 0.45)
        self._road(ax, end, start, self._wdelay, max(w + 2.0, 5.0))
        self._wleg = (end, start)
        for (sx, sy), dl in zip(self._store_xy, self._sdelay):
            a = self._edge((0, 0), (sx, sy), 0.5 * self._wh)
            b = self._edge((sx, sy), (0, 0), 0.5 * self._shop_size)
            self._road(ax, a, b, dl, w)
            self._legs.append((a, b))
        self._factory(ax, self._fac_xy[0], self._fac_xy[1], 1.1)
        self._rack_box = self._warehouse_shell(ax, 0.0, 0.0, self._wh)
        for (sx, sy) in self._store_xy:
            self._shop(ax, sx, sy, self._shop_size, 1)

    def _draw_panel(self):
        p, top = self.panel, self._ph
        p.text(0.0, top - 0.02, 'Retailer inventory', fontsize=15, fontweight='bold', color=TEXT, va='top')
        y = top - 0.26
        rows = [
            ('factory', 'Manufacturer'),
            ('warehouse', 'Warehouse, a pallet for\neach 50th of its capacity'),
            ('shop', 'Store, a box for each 20th\nof its capacity on the shelf'),
            ('empty', 'Store with an empty shelf'),
            ('truck', 'Goods on the road'),
            ('road', 'Road, a mark for each day\nof travel'),
        ]
        for kind, label in rows:
            if kind == 'factory':
                self._factory(p, 0.12, y - 0.01, 0.17)
            elif kind == 'warehouse':
                self._pallets(p, self._warehouse_shell(p, 0.12, y, 0.2), 23)
            elif kind == 'shop':
                self._shop(p, 0.12, y - 0.01, 0.16, 7)
            elif kind == 'empty':
                self._shop(p, 0.12, y - 0.01, 0.16, 0)
            elif kind == 'truck':
                self._truck_icon(p, 0.12, y, 0.0, 0.2, 4, zorder=3)
            elif kind == 'road':
                self._road(p, (0.02, y), (0.22, y), 2, 6.0, tick=0.035)
            p.text(0.3, y, label, fontsize=9.5, color=TEXT, va='center')
            y -= 0.2

    # ===== each frame =====

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def _on_leg(self, leg, delay, T):
        '''Where goods T days from arriving are along a road taking delay days.'''
        (xa, ya), (xb, yb) = leg
        f = (delay - T + 0.5) / delay if delay > 0 else 1.0
        f = min(max(f, 0.0), 1.0)
        return xa + f * (xb - xa), ya + f * (yb - ya), math.atan2(yb - ya, xb - xa)

    def render(self, state):
        ax = self.ax
        st = lambda name, *args: int(round(float(state[RDDLPlanningModel.ground_var(name, list(args))])))
        artists = []

        # the warehouse racks
        stock = st('stock')
        nx, ny = RACK
        artists += self._pallets(ax, self._rack_box, int(math.ceil(stock * nx * ny / self._wcap)) if stock > 0 else 0)
        if stock <= 0:
            rx, ry, rw, rh = self._rack_box
            p = patches.Rectangle((rx, ry), rw, rh, facecolor=EMPTY, edgecolor=RED, linewidth=1.6, zorder=3.4)
            ax.add_patch(p)
            artists.append(p)

        # the store shelves
        bx, by = SHELF_BOXES
        for k, s in enumerate(self._stores):
            shelf = st('shelf', s)
            boxes = int(math.ceil(shelf * bx * by / self._scap[k])) if shelf > 0 else 0
            sx, sy = self._store_xy[k]
            artists += self._shop(ax, sx, sy, self._shop_size, boxes, static=False, zorder=3.2)

        # trucks on the roads
        for d in self._days:
            T = self._day_of[d]
            q = st('inbound', d)
            if q > 0 and T >= 1:
                x, y, hd = self._on_leg(self._wleg, max(self._wdelay, T), T)
                crates = int(math.ceil(q * CRATES / self._prod))
                artists += self._truck_icon(ax, x, y, hd, 0.55, crates)
            for k, s in enumerate(self._stores):
                q = st('transit', s, d)
                if q > 0 and T >= 1:
                    x, y, hd = self._on_leg(self._legs[k], max(self._sdelay[k], T), T)
                    crates = int(math.ceil(q * CRATES / self._scap[k]))
                    artists += self._truck_icon(ax, x, y, hd, self._truck, crates)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
