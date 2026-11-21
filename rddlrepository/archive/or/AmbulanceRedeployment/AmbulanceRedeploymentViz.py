import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f2f4f6'
LAND = '#e8ebe4'
FIELDS = ['#d5e3c1', '#cfdfb8', '#dde6c6', '#e2e2c0']
LAWN = '#dfe8d2'
HOUSE_ROOFS = ['#b5654a', '#9c5a46', '#c98b5e', '#8d6e63']
TREE = '#8fbf78'
STREET = '#ffffff'
STREET_EDGE = '#c3cad1'
BUILDING_LIGHT = np.array([0.93, 0.89, 0.82])
BUILDING_DARK = np.array([0.62, 0.55, 0.50])
ROOF_EDGE = '#8d8178'
STATION = '#eceff1'
STATION_DOOR = '#546e7a'
HOSPITAL = '#ffffff'
HOSPITAL_EDGE = '#607d8b'
RED = '#d62828'
BLUE = '#1565c0'
PURPLE = '#7b1fa2'
COVER = (0.12, 0.45, 0.78)
VAN = '#ffffff'
VAN_EDGE = '#263238'
GLASS = '#4f6d8a'
TEXT = '#37474f'
MUTED = '#90a4ae'
BAR_BACK = '#dfe4ea'
DECIDE = '#f4a100'


class AmbulanceRedeploymentVisualizer(BaseViz):
    '''Draws the city from above: streets, and blocks whose buildings are denser where calls
    come from more often (quiet blocks are parks). Ambulance stations are small garages, the
    hospitals are large buildings with a red cross. Ambulances are drawn as vehicles at the
    point they have reached, facing where they drive: with their lights flashing while they
    answer a call or carry a patient, parked beside the emergency at a scene, parked at the
    hospital while handing over a patient. Each emergency is a red warning sign; a dashed ring
    marks a call still waiting for an ambulance. The green area is the part of the city that a
    free ambulance can reach in time. An orange ring marks an ambulance just freed, whose base
    is being chosen.
    '''

    def __init__(self, model: RDDLPlanningModel, figure_size=(10.5, 6.6), wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        objs = model.type_to_objects
        self._ambulances = list(objs.get('ambulance', []))
        self._bases = list(objs.get('base', []))
        self._hospitals = list(objs.get('hospital', []))
        self._cells = list(objs.get('cell', []))
        self._slots = list(objs.get('slot', []))
        nf = model.ground_vars_with_values(model.non_fluents)
        g = lambda name, *args: float(nf[RDDLPlanningModel.ground_var(name, list(args))])
        self._rs, self._ts = g('RESPONSE-SPEED'), g('TRAVEL-SPEED')
        self._turnout, self._threshold, self._end = g('TURNOUT-TIME'), g('THRESHOLD'), g('END-TIME')
        self._cell_rects = [(g('CELL-X-MIN', c), g('CELL-X-MAX', c), g('CELL-Y-MIN', c), g('CELL-Y-MAX', c))
                            for c in self._cells]
        self._demand = [g('DEMAND', c) for c in self._cells]
        self._base_xy = [(g('BASE-X', b), g('BASE-Y', b)) for b in self._bases]
        self._hosp_xy = [(g('HOSPITAL-X', h), g('HOSPITAL-Y', h)) for h in self._hospitals]
        self._slot_order = self._order_slots(nf)

        xs = [r[0] for r in self._cell_rects] + [r[1] for r in self._cell_rects] + \
             [p[0] for p in self._base_xy + self._hosp_xy]
        ys = [r[2] for r in self._cell_rects] + [r[3] for r in self._cell_rects] + \
             [p[1] for p in self._base_xy + self._hosp_xy]
        if not xs:
            xs, ys = [0.0, 1.0], [0.0, 1.0]
        self._x0, self._x1, self._y0, self._y1 = min(xs), max(xs), min(ys), max(ys)
        self._span = max(self._x1 - self._x0, self._y1 - self._y0, 1e-6)
        self._van_len = 0.07 * self._span

        self.fig = plt.figure(figsize=figure_size)
        self.fig.patch.set_facecolor(BACKGROUND)
        w, h = figure_size
        side = 0.88 * h / w
        self.ax = self.fig.add_axes([0.015, 0.06, side, 0.88])
        pad = 0.03 * self._span
        cx, cy = (self._x0 + self._x1) / 2, (self._y0 + self._y1) / 2
        self.ax.set_xlim(cx - self._span / 2 - pad, cx + self._span / 2 + pad)
        self.ax.set_ylim(cy - self._span / 2 - pad, cy + self._span / 2 + pad)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        pw = 0.955 - side - 0.03
        self.panel = self.fig.add_axes([0.03 + side + 0.015, 0.06, pw, 0.88])
        self._ph = 0.88 * h / (pw * w)        # panel height in panel widths
        self.panel.set_xlim(0, 1)
        self.panel.set_ylim(0, self._ph)
        self.panel.set_aspect('equal')
        self.panel.axis('off')

        # coverage is computed on a grid over the city
        n = 220
        self._gx, self._gy = np.meshgrid(np.linspace(self._x0, self._x1, n), np.linspace(self._y0, self._y1, n))

        self._draw_city()
        self._draw_panel()

    def _order_slots(self, nf):
        first = [s for s in self._slots if bool(nf.get(RDDLPlanningModel.ground_var('FIRST-SLOT', [s]), False))]
        order, seen = [], set()
        cur = first[0] if first else (self._slots[0] if self._slots else None)
        while cur is not None and cur not in seen:
            order.append(cur)
            seen.add(cur)
            nxt = [s for s in self._slots
                   if bool(nf.get(RDDLPlanningModel.ground_var('NEXT-SLOT', [cur, s]), False))]
            cur = nxt[0] if nxt else None
        return order + [s for s in self._slots if s not in seen]

    def _density(self, x, y):
        for d, (xl, xh, yl, yh) in zip(self._demand, self._cell_rects):
            if xl <= x <= xh and yl <= y <= yh:
                return d / max((xh - xl) * (yh - yl), 1e-12)
        return 0.0

    # ===== the city, which never changes =====

    def _draw_city(self):
        ax = self.ax
        x0, x1, y0, y1 = self._x0, self._x1, self._y0, self._y1
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=LAND, edgecolor='none', zorder=0))

        # a street grid at a round spacing (about 15 across), main roads every third street
        step = 10 ** math.floor(math.log10(self._span / 15))
        for mult in (1, 2, 5, 10):
            if self._span / (step * mult) <= 22:
                step *= mult
                break
        xs = sorted(set([x0] + list(np.arange(math.ceil(x0 / step) * step, x1 + 1e-9, step)) + [x1]))
        ys = sorted(set([y0] + list(np.arange(math.ceil(y0 / step) * step, y1 + 1e-9, step)) + [y1]))
        nx, ny = len(xs) - 1, len(ys) - 1
        dens = np.array([[self._density((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2) for j in range(ny)]
                         for i in range(nx)])
        busy = np.sqrt(dens / max(dens.max(), 1e-12))      # 0 to 1 by block
        urban = busy >= 0.1
        rng = np.random.default_rng(7)

        # farmland where calls are rare: fields of slightly different crops
        fields, field_colors, trees = [], [], []
        for i in range(nx):
            for j in range(ny):
                if urban[i, j]:
                    continue
                fields.append(patches.Rectangle((xs[i], ys[j]), xs[i + 1] - xs[i], ys[j + 1] - ys[j]))
                field_colors.append(FIELDS[rng.integers(len(FIELDS))])
                for _ in range(rng.integers(0, 3)):
                    trees.append(patches.Circle((rng.uniform(xs[i], xs[i + 1]), rng.uniform(ys[j], ys[j + 1])),
                                                0.07 * step))
        ax.add_collection(PatchCollection(fields, facecolor=field_colors, edgecolor='none', zorder=0.5))
        ax.add_collection(PatchCollection(trees, facecolor=TREE, edgecolor='none', zorder=2.1))

        # streets in town, main roads everywhere
        unit = self._points_per_unit()
        minor, major = [], []
        major_x = [i for i in range(len(xs)) if round(xs[i] / step) % 3 == 0 or i in (0, nx)]
        major_y = [j for j in range(len(ys)) if round(ys[j] / step) % 3 == 0 or j in (0, ny)]
        for i in range(len(xs)):
            for j in range(ny):
                near = (i > 0 and urban[i - 1, j]) or (i < nx and urban[i, j])
                seg = [(xs[i], ys[j]), (xs[i], ys[j + 1])]
                if i in major_x:
                    major.append(seg)
                elif near:
                    minor.append(seg)
        for j in range(len(ys)):
            for i in range(nx):
                near = (j > 0 and urban[i, j - 1]) or (j < ny and urban[i, j])
                seg = [(xs[i], ys[j]), (xs[i + 1], ys[j])]
                if j in major_y:
                    major.append(seg)
                elif near:
                    minor.append(seg)
        for lines, w in ((minor, 0.12), (major, 0.22)):
            width = max(w * step * unit, 1.0)
            ax.add_collection(LineCollection(lines, colors=STREET_EDGE, linewidths=width + 1.2, zorder=1,
                                             capstyle='projecting'))
            ax.add_collection(LineCollection(lines, colors=STREET, linewidths=width, zorder=1.1,
                                             capstyle='projecting'))

        # town blocks: houses on lawns, then offices, then towers where calls come most often
        lawns, shadows, buildings, colors = [], [], [], []
        for i in range(nx):
            for j in range(ny):
                f = busy[i, j]
                if not urban[i, j]:
                    continue
                m = 0.11 * step
                bx, by = xs[i] + m, ys[j] + m
                bw, bh = xs[i + 1] - xs[i] - 2 * m, ys[j + 1] - ys[j] - 2 * m
                if bw <= 0 or bh <= 0:
                    continue
                if f < 0.3:
                    lawns.append(patches.Rectangle((bx, by), bw, bh))
                    n = 4
                    for a_ in range(n):
                        for b_ in range(n):
                            if rng.random() > 0.3 + 1.5 * f:
                                continue
                            hx, hy = bx + (a_ + 0.5) * bw / n, by + (b_ + 0.5) * bh / n
                            hw, hh = 0.5 * bw / n, 0.42 * bh / n
                            buildings.append(patches.Rectangle((hx - hw / 2, hy - hh / 2), hw, hh))
                            colors.append(HOUSE_ROOFS[rng.integers(len(HOUSE_ROOFS))])
                    continue
                lots = 3 if f < 0.6 else 2
                lw, lh = bw / lots, bh / lots
                for a_ in range(lots):
                    for b_ in range(lots):
                        if rng.random() > 0.55 + 0.45 * f:
                            continue
                        shrink = rng.uniform(0.06, 0.16)
                        rx, ry = bx + a_ * lw + shrink * lw, by + b_ * lh + shrink * lh
                        rw, rh = lw * (1 - 2 * shrink), lh * (1 - 2 * shrink)
                        off = (0.04 + 0.12 * f) * step
                        shadows.append(patches.Rectangle((rx + off, ry - off), rw, rh))
                        buildings.append(patches.Rectangle((rx, ry), rw, rh))
                        tone = min(max(f + rng.uniform(-0.1, 0.1), 0.0), 1.0)
                        colors.append(BUILDING_LIGHT + tone * (BUILDING_DARK - BUILDING_LIGHT))
        ax.add_collection(PatchCollection(lawns, facecolor=LAWN, edgecolor='none', zorder=2))
        ax.add_collection(PatchCollection(shadows, facecolor=(0, 0, 0, 0.18), edgecolor='none', zorder=2.1))
        if buildings:
            ax.add_collection(PatchCollection(buildings, facecolor=colors, edgecolor=ROOF_EDGE, linewidth=0.3,
                                              zorder=2.2))

        for (bx, by) in self._base_xy:
            self._station(ax, bx, by, 0.045 * self._span, zorder=4)
        for (hx, hy) in self._hosp_xy:
            self._hospital(ax, hx, hy, 0.1 * self._span, zorder=4)

    def _points_per_unit(self):
        bbox = self.ax.get_position()
        width_in = bbox.width * self.fig.get_figwidth()
        lo, hi = self.ax.get_xlim()
        return width_in * 72 / (hi - lo)

    def _station(self, ax, x, y, size, zorder):
        '''An ambulance station: a garage with two doors under a red band.'''
        w, h = size, 0.7 * size
        ax.add_patch(patches.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=STATION, edgecolor=STATION_DOOR,
                                       linewidth=0.8, zorder=zorder))
        for k in (-1, 1):
            ax.add_patch(patches.Rectangle((x + k * 0.22 * w - 0.17 * w, y - h / 2), 0.34 * w, 0.55 * h,
                                           facecolor=STATION_DOOR, edgecolor='none', zorder=zorder + 0.1))
        ax.add_patch(patches.Rectangle((x - w / 2, y + 0.25 * h), w, 0.25 * h, facecolor=RED, edgecolor='none',
                                       zorder=zorder + 0.1))

    def _hospital(self, ax, x, y, size, zorder):
        '''A hospital: a large building with rows of windows and a red cross.'''
        w, h = size, 0.75 * size
        ax.add_patch(patches.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=HOSPITAL, edgecolor=HOSPITAL_EDGE,
                                       linewidth=1.0, zorder=zorder))
        ax.add_patch(patches.Rectangle((x - 0.18 * w, y - h / 2 - 0.12 * h), 0.36 * w, 0.14 * h,
                                       facecolor='#b0bec5', edgecolor='none', zorder=zorder))
        for k in range(4):
            for side in (-1, 1):
                ax.add_patch(patches.Rectangle((x + side * 0.36 * w - 0.05 * w, y - 0.32 * h + k * 0.2 * h),
                                               0.1 * w, 0.1 * h, facecolor='#b3d4f0', edgecolor='none',
                                               zorder=zorder + 0.1))
        self._cross(ax, x, y, 0.2 * w, zorder + 0.1)

    def _cross(self, ax, x, y, arm, zorder, color=RED):
        thick = 0.36 * arm
        ax.add_patch(patches.Rectangle((x - thick, y - arm), 2 * thick, 2 * arm, facecolor=color, edgecolor='none',
                                       zorder=zorder))
        ax.add_patch(patches.Rectangle((x - arm, y - thick), 2 * arm, 2 * thick, facecolor=color, edgecolor='none',
                                       zorder=zorder))

    def _ambulance(self, ax, x, y, heading, length, sirens, zorder=8):
        '''An ambulance seen from above, facing heading, lights flashing when sirens is true.'''
        L, W = length, 0.46 * length
        tr = Affine2D().rotate(heading).translate(x, y) + ax.transData
        parts = []
        if sirens:
            parts.append(patches.Circle((0.1 * L, 0), 0.75 * L, facecolor=RED, alpha=0.18, edgecolor='none'))
        parts += [
            patches.FancyBboxPatch((-L / 2, -W / 2), L, W, boxstyle=f'round,pad=0,rounding_size={0.15 * W}',
                                   facecolor=VAN, edgecolor=VAN_EDGE, linewidth=0.9),
            patches.Rectangle((0.22 * L, -0.4 * W), 0.12 * L, 0.8 * W, facecolor=GLASS, edgecolor='none'),
            patches.Rectangle((-L / 2, -W / 2), 0.7 * L, 0.14 * W, facecolor=RED, edgecolor='none'),
            patches.Rectangle((-L / 2, 0.36 * W), 0.7 * L, 0.14 * W, facecolor=RED, edgecolor='none'),
            patches.Rectangle((-0.24 * L, -0.24 * W), 0.08 * L, 0.48 * W, facecolor=RED, edgecolor='none'),
            patches.Rectangle((-0.33 * L, -0.06 * W), 0.26 * L, 0.12 * W, facecolor=RED, edgecolor='none'),
            patches.Rectangle((0.1 * L, -0.4 * W), 0.08 * L, 0.4 * W,
                              facecolor=(RED if sirens else '#b0bec5'), edgecolor='none'),
            patches.Rectangle((0.1 * L, 0.0), 0.08 * L, 0.4 * W,
                              facecolor=(BLUE if sirens else '#b0bec5'), edgecolor='none'),
        ]
        for k, p in enumerate(parts):
            p.set_transform(tr)
            p.set_zorder(zorder + 0.01 * k)
            ax.add_patch(p)
        return parts

    def _decision(self, ax, x, y, length):
        self._ambulance(ax, x, y, 0.0, length, False, zorder=3)
        ax.add_patch(patches.Circle((x, y), 0.62 * length, facecolor='none', edgecolor=DECIDE, linewidth=2.0,
                                    zorder=4))

    def _warning(self, ax, x, y, size, zorder=7, waiting=False):
        '''An emergency: a red warning sign; a dashed ring when no ambulance is on its way.'''
        parts = []
        if waiting:
            parts.append(patches.Circle((x, y + 0.3 * size), 0.95 * size, facecolor='none', edgecolor=RED,
                                        linewidth=1.4, linestyle=(0, (3, 2)), zorder=zorder))
        parts.append(patches.Polygon([(x - 0.6 * size, y - 0.15 * size), (x + 0.6 * size, y - 0.15 * size),
                                      (x, y + 0.9 * size)], closed=True, facecolor=RED, edgecolor='white',
                                     linewidth=1.0, zorder=zorder))
        parts.append(patches.Rectangle((x - 0.06 * size, y + 0.18 * size), 0.12 * size, 0.42 * size,
                                       facecolor='white', edgecolor='none', zorder=zorder + 0.1))
        parts.append(patches.Circle((x, y + 0.04 * size), 0.07 * size, facecolor='white', edgecolor='none',
                                    zorder=zorder + 0.1))
        for p in parts:
            ax.add_patch(p)
        return parts

    # ===== the side panel =====

    def _draw_panel(self):
        p, top = self.panel, self._ph
        p.text(0.0, top - 0.03, 'Ambulance dispatch', fontsize=15, fontweight='bold', color=TEXT, va='top')
        y = top - 0.16
        rows = [
            ('ambulance on an emergency', lambda yy: self._ambulance(p, 0.09, yy, 0.0, 0.1, True, zorder=3)),
            ('free ambulance', lambda yy: self._ambulance(p, 0.09, yy, 0.0, 0.1, False, zorder=3)),
            ('freed, its base being chosen', lambda yy: self._decision(p, 0.09, yy, 0.1)),
            ('emergency call', lambda yy: self._warning(p, 0.09, yy - 0.025, 0.06, zorder=3)),
            ('call waiting for an ambulance', lambda yy: self._warning(p, 0.09, yy - 0.025, 0.06, zorder=3,
                                                                       waiting=True)),
            ('ambulance station', lambda yy: self._station(p, 0.09, yy, 0.085, zorder=3)),
            ('hospital', lambda yy: self._hospital(p, 0.09, yy, 0.1, zorder=3)),
        ]
        for label, draw in rows:
            draw(y)
            p.text(0.22, y, label, fontsize=10, color=TEXT, va='center')
            y -= 0.09
        p.add_patch(patches.Rectangle((0.02, y - 0.04), 0.14, 0.08, facecolor=COVER + (0.25,),
                                      edgecolor=COVER, linewidth=1.2))
        p.text(0.22, y, f'reached within {self._threshold:g} min', fontsize=10, color=TEXT, va='center')
        y -= 0.06
        p.text(0.22, y, 'by a free ambulance', fontsize=10, color=TEXT, va='center')
        y -= 0.11

        # calls waiting, head of the list first
        p.text(0.0, y, 'Calls waiting', fontsize=11, fontweight='bold', color=TEXT, va='center')
        y -= 0.08
        n = max(len(self._slot_order), 1)
        size = min(0.08, 0.98 / n / 1.25)
        self._slot_y, self._slot_size = y, size
        self._slot_x = [0.6 * size + k * 1.25 * size for k in range(len(self._slot_order))]
        for x in self._slot_x:
            p.add_patch(patches.Polygon([(x - 0.6 * size, y - 0.4 * size), (x + 0.6 * size, y - 0.4 * size),
                                         (x, y + 0.6 * size)], closed=True, facecolor='none', edgecolor=MUTED,
                                        linewidth=0.8, zorder=2))
        y -= 0.12

        # time and late calls
        self._bar_w = 0.98
        p.text(0.0, y, 'Time of the episode', fontsize=11, fontweight='bold', color=TEXT, va='center')
        y -= 0.07
        self._time_y = y
        p.add_patch(patches.Rectangle((0.0, y - 0.025), self._bar_w, 0.05, facecolor=BAR_BACK, edgecolor='none'))
        hours = self._end / 60.0
        tick = 24.0 if hours > 48 else (6.0 if hours > 12 else 1.0)
        t = tick
        while t < hours - 1e-9:
            x = self._bar_w * t / hours
            p.plot([x, x], [y - 0.025, y + 0.025], color='white', linewidth=0.8, zorder=3)
            t += tick
        span = f'{hours / 24:g} days' if hours > 48 else f'{hours:g} hours'
        p.text(self._bar_w, y - 0.06, span, fontsize=8, color=TEXT, ha='right', va='center')
        y -= 0.13
        p.text(0.0, y, 'Calls reached late', fontsize=11, fontweight='bold', color=TEXT, va='center')
        y -= 0.07
        self._late_y = y
        p.add_patch(patches.Rectangle((0.0, y - 0.025), self._bar_w, 0.05, facecolor=BAR_BACK, edgecolor='none'))
        for frac in (0.25, 0.5, 0.75):
            p.plot([self._bar_w * frac] * 2, [y - 0.025, y + 0.025], color='white', linewidth=0.8, zorder=3)
        p.text(0.0, y - 0.06, '0%', fontsize=8, color=TEXT, va='center')
        p.text(self._bar_w, y - 0.06, '100% of calls', fontsize=8, color=TEXT, ha='right', va='center')

    # ===== parts that follow the state =====

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        ax, p = self.ax, self.panel
        st = lambda name, *args: state[RDDLPlanningModel.ground_var(name, list(args))]
        now = float(st('time'))
        L = self._van_len
        artists = []

        # where each ambulance is now
        vans = []
        for a in self._ambulances:
            duty = str(st('duty', a)).lstrip('@')
            fx, fy, tx, ty = (float(st(v, a)) for v in ('from-x', 'from-y', 'to-x', 'to-y'))
            start = float(st('start', a))
            speed = self._rs if duty == 'to-scene' else self._ts
            trip = abs(tx - fx) + abs(ty - fy)
            driven = max(0.0, now - start) * speed
            f = 1.0 if driven >= trip else driven / trip
            x, y = fx + f * (tx - fx), fy + f * (ty - fy)
            moving = f < 1.0 and trip > 0
            heading = math.atan2(ty - fy, tx - fx) if trip > 0 else 0.0
            vans.append(dict(duty=duty, x=x, y=y, tx=tx, ty=ty, moving=moving, heading=heading,
                             pending=bool(st('pending', a)), at_base=(duty == 'available' and not moving)))

        # the area free ambulances reach in time
        reach = np.full(self._gx.shape, np.inf)
        for v in vans:
            if v['duty'] == 'available':
                delay = self._turnout if (v['at_base'] and not v['pending']) else 0.0
                r = max(self._threshold - delay, 0.0) * self._rs
                d = np.abs(self._gx - v['x']) + np.abs(self._gy - v['y'])
                reach = np.minimum(reach, d - r)
        covered = reach <= 0
        if covered.any():
            rgba = np.zeros(covered.shape + (4,))
            rgba[covered] = COVER + (0.2,)
            artists.append(ax.imshow(rgba, origin='lower', extent=(self._x0, self._x1, self._y0, self._y1),
                                     interpolation='nearest', zorder=3))
            if not covered.all():
                artists.append(ax.contour(self._gx, self._gy, covered.astype(float), levels=[0.5],
                                          colors=[COVER], linewidths=1.4, zorder=3.1))

        # emergencies and routes
        for v in vans:
            if v['duty'] in ('to-scene', 'at-scene'):
                artists.extend(self._warning(ax, v['tx'], v['ty'], 0.04 * self._span))
            if v['moving'] and v['duty'] in ('to-scene', 'to-hospital'):
                arrow = patches.FancyArrowPatch((v['x'], v['y']), (v['tx'], v['ty']), arrowstyle='-|>',
                                                mutation_scale=12, linestyle=(0, (4, 2)), linewidth=1.6,
                                                color=RED if v['duty'] == 'to-scene' else PURPLE, zorder=6,
                                                shrinkA=0, shrinkB=6)
                ax.add_patch(arrow)
                artists.append(arrow)
            elif v['moving'] and v['duty'] == 'available':
                line, = ax.plot([v['x'], v['tx']], [v['y'], v['ty']], linestyle=(0, (2, 2)), linewidth=1.0,
                                color=TEXT, alpha=0.6, zorder=6)
                artists.append(line)
        waiting = [k for k, s in enumerate(self._slot_order) if bool(st('waiting', s))]
        for k in waiting:
            s = self._slot_order[k]
            artists.extend(self._warning(ax, float(st('wait-x', s)), float(st('wait-y', s)), 0.04 * self._span,
                                         waiting=True))

        # the ambulances, parked ones side by side
        parked = {}
        for v in vans:
            if v['moving']:
                sx, sy, heading = v['x'], v['y'], v['heading']
            else:
                key = (round(v['x'], 3), round(v['y'], 3))
                k = parked.get(key, 0)
                parked[key] = k + 1
                if v['duty'] == 'at-scene':
                    sx, sy = v['x'] + 0.75 * L, v['y'] - 0.2 * L - 0.55 * L * k
                elif v['duty'] == 'at-hospital' or v['pending']:
                    sx, sy = v['x'] - 0.3 * L, v['y'] - 0.95 * L - 0.55 * L * k
                else:
                    sx, sy = v['x'], v['y'] - 0.7 * L - 0.55 * L * k
                heading = 0.0
            sirens = v['duty'] in ('to-scene', 'to-hospital')
            artists.extend(self._ambulance(ax, sx, sy, heading, L, sirens))
            if v['pending']:
                ring = patches.Circle((sx, sy), 0.75 * L, facecolor='none', edgecolor=DECIDE, linewidth=2.0,
                                      zorder=9)
                ax.add_patch(ring)
                artists.append(ring)

        # panel: waiting list, time and late calls
        size, y = self._slot_size, self._slot_y
        for k in waiting:
            if k < len(self._slot_x):
                x = self._slot_x[k]
                tri = patches.Polygon([(x - 0.6 * size, y - 0.4 * size), (x + 0.6 * size, y - 0.4 * size),
                                       (x, y + 0.6 * size)], closed=True, facecolor=RED, edgecolor='none', zorder=3)
                p.add_patch(tri)
                artists.append(tri)
        frac_t = min(max(now / self._end, 0.0), 1.0) if self._end > 0 else 0.0
        calls, late = int(st('calls')), int(st('late-calls'))
        frac_l = late / calls if calls > 0 else 0.0
        for bar in (patches.Rectangle((0.0, self._time_y - 0.025), self._bar_w * frac_t, 0.05, facecolor='#457b9d',
                                      edgecolor='none', zorder=2),
                    patches.Rectangle((0.0, self._late_y - 0.025), self._bar_w * frac_l, 0.05, facecolor=RED,
                                      edgecolor='none', zorder=2)):
            p.add_patch(bar)
            artists.append(bar)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
