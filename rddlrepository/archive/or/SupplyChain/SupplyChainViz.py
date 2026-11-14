import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

GROUND_COLOR = '#eef1e6'
ROAD_COLOR = '#78909c'
ROAD_LINE_COLOR = '#eceff1'
FACTORY_COLOR = '#90a4ae'
FACTORY_DARK_COLOR = '#546e7a'
WINDOW_COLOR = '#ffe082'
SMOKE_COLOR = '#b0bec5'
WALL_COLOR = '#d7ccc8'
ROOF_COLOR = '#8d6e63'
DOOR_COLOR = '#a1887f'
DOOR_LINE_COLOR = '#795548'
RACK_COLOR = '#607d8b'
CRATE_COLOR = '#d4a35f'
CRATE_EDGE_COLOR = '#8d6e63'
EMPTY_SLOT_COLOR = '#e0e4e6'
OWED_COLOR = '#c62828'
OWED_FILL_COLOR = '#ffebee'
LAST_COLOR = '#455a64'
BEFORE_COLOR = '#b0bec5'
BAR_TRACK_COLOR = '#e3e7ea'
TEXT_COLOR = '#37474f'
LOSING_COLOR = '#c62828'

# sizes in inches: the factory, the road between it and the warehouses, and a row of a
# warehouse at full size, which shrinks with many warehouses
MAX_SIZE = (9.0, 5.6)
TOP_MARGIN = 0.45
LEGEND_HEIGHT = 0.5
FACTORY_WIDTH = 2.0
ROAD_WIDTH = 0.9
MAX_ROW = 1.1
WAREHOUSE_WIDTH = 1.0
RACK_WIDTH = 2.0
DEMAND_WIDTH = 1.3
GAP = 0.2

# crates on a rack and in the factory yard
RACK_SLOTS = (10, 2)
YARD_SLOTS = (6, 4)


# ---------- pictures: each takes the axes and a place and size in inches

def _factory(ax, left, bottom, width, height):

    # a hall with a sawtooth roof, lit windows and a smoking chimney
    (w, h) = (width, height)
    points = [(left, bottom), (left, bottom + 0.55 * h)]
    for k in range(3):
        points += [(left + (k + 1) * 0.25 * w, bottom + 0.8 * h), (left + (k + 1) * 0.25 * w, bottom + 0.55 * h)]
    points = points[:-1] + [(left + 0.75 * w, bottom)]
    ax.add_patch(patches.Rectangle((left + 0.8 * w, bottom), 0.1 * w, 0.95 * h, facecolor=FACTORY_DARK_COLOR,
                                   edgecolor='none', zorder=2))
    ax.add_patch(patches.Polygon(points, closed=True, facecolor=FACTORY_COLOR, edgecolor=FACTORY_DARK_COLOR,
                                 linewidth=1, zorder=2.1))
    for k in range(3):
        ax.add_patch(patches.Rectangle((left + (0.05 + 0.24 * k) * w, bottom + 0.15 * h), 0.14 * w, 0.18 * h,
                                       facecolor=WINDOW_COLOR, edgecolor='none', zorder=2.2))
    for (dx, dy, r) in ((0.85, 1.02, 0.05), (0.92, 1.1, 0.065), (1.0, 1.17, 0.08)):
        ax.add_patch(plt.Circle((left + dx * w, bottom + dy * h), r * h, facecolor=SMOKE_COLOR,
                                edgecolor='none', zorder=2))


def _warehouse(ax, left, bottom, width, height):

    # a warehouse with a gabled roof and a roll-up door
    (w, h) = (width, height)
    ax.add_patch(patches.Rectangle((left, bottom), w, 0.62 * h, facecolor=WALL_COLOR,
                                   edgecolor=ROOF_COLOR, linewidth=1, zorder=2))
    ax.add_patch(patches.Polygon([(left - 0.06 * w, bottom + 0.6 * h), (left + 0.5 * w, bottom + 0.95 * h),
                                  (left + 1.06 * w, bottom + 0.6 * h)], closed=True, facecolor=ROOF_COLOR,
                                 edgecolor='none', zorder=2.1))
    (dl, dw, dh) = (left + 0.25 * w, 0.5 * w, 0.45 * h)
    ax.add_patch(patches.Rectangle((dl, bottom), dw, dh, facecolor=DOOR_COLOR, edgecolor=DOOR_LINE_COLOR,
                                   linewidth=0.8, zorder=2.2))
    for k in range(1, 5):
        ax.plot([dl, dl + dw], [bottom + k * dh / 5] * 2, color=DOOR_LINE_COLOR, linewidth=0.6, zorder=2.3)


def _crates(ax, left, bottom, width, height, slots, rack):

    # a grid of crates, all hidden, on the shelves of a rack or stacked in a yard: the
    # bottom row fills first, from the left
    (cols, rows) = slots
    (cw, ch) = (width / cols, height / rows)
    if rack:
        for r in range(rows + 1):
            ax.plot([left, left + width], [bottom + r * ch] * 2, color=RACK_COLOR, linewidth=1.5, zorder=2)
        for x in (left, left + width):
            ax.plot([x, x], [bottom, bottom + height], color=RACK_COLOR, linewidth=1.5, zorder=2)
    crates = []
    for r in range(rows):
        for c in range(cols):
            (x, y, s) = (left + c * cw + 0.08 * cw, bottom + r * ch + 0.06 * ch, min(0.84 * cw, 0.86 * ch))
            crate = ax.add_patch(patches.Rectangle((x, y), 0.84 * cw, s, facecolor=CRATE_COLOR,
                                                   edgecolor=CRATE_EDGE_COLOR, linewidth=0.6, zorder=2.5))
            lid = ax.plot([x, x + 0.84 * cw], [y + 0.5 * s] * 2, color=CRATE_EDGE_COLOR, linewidth=0.5,
                          zorder=2.6)[0]
            owed = ax.add_patch(patches.Rectangle((x, y), 0.84 * cw, s, facecolor=OWED_FILL_COLOR,
                                                  edgecolor=OWED_COLOR, linewidth=0.9, linestyle=(0, (2, 1.5)),
                                                  zorder=2.5))
            for artist in (crate, lid, owed):
                artist.set_visible(False)
            crates.append(((crate, lid), (owed,)))
    return crates


def _count(share, slots):

    # as many slots as the share of them, and one for any amount at all
    return max(round(share * slots), 1) if share > 0 else 0


def _fill(crates, share, owed_share=0.0):

    # crates for the stock, or the outlines of crates owed in their place
    stocked, owed = _count(share, len(crates)), _count(owed_share, len(crates))
    for (k, (crate, missing)) in enumerate(crates):
        for artist in crate:
            artist.set_visible(k < stocked)
        for artist in missing:
            artist.set_visible(k < owed)


def _cart(ax, left, bottom, size):

    # a shopping cart: a basket on a frame with a handle and two wheels
    s = size
    ax.add_patch(patches.Polygon([(left + 0.1 * s, bottom + 0.75 * s), (left + 0.95 * s, bottom + 0.75 * s),
                                  (left + 0.82 * s, bottom + 0.3 * s), (left + 0.22 * s, bottom + 0.3 * s)],
                                 closed=True, facecolor='none', edgecolor=LAST_COLOR, linewidth=1.2, zorder=2))
    ax.plot([left - 0.05 * s, left + 0.1 * s, left + 0.25 * s, left + 0.85 * s],
            [bottom + 0.88 * s, bottom + 0.88 * s, bottom + 0.2 * s, bottom + 0.2 * s],
            color=LAST_COLOR, linewidth=1.2, solid_joinstyle='round', zorder=2)
    for dx in (0.3, 0.78):
        ax.add_patch(plt.Circle((left + dx * s, bottom + 0.08 * s), 0.07 * s, facecolor=LAST_COLOR,
                                edgecolor='none', zorder=2))


class SupplyChainVisualizer(BaseViz):
    '''Draws the factory with the crates in its yard on the left, and a road from it to each
    warehouse, which keeps its crates on a rack. A warehouse that owes items shows them as
    the red outlines of missing crates on its rack instead, and two bars next to a shopping
    cart show its demand in the last two steps.

    The action is not part of the state, so what was produced and shipped only shows in the
    stock. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._warehouses = list(model.type_to_objects['warehouse'])

        # everything is laid out in inches, with rows that shrink to fit many warehouses
        count = max(len(self._warehouses), 1)
        room = MAX_SIZE[1] - TOP_MARGIN - LEGEND_HEIGHT - 0.1
        self._row = min(MAX_ROW, room / count)
        scale = min(1.0, self._row / MAX_ROW)
        rows_height = max(count * self._row, 2.2)
        row_width = ((WAREHOUSE_WIDTH + RACK_WIDTH + DEMAND_WIDTH + 2 * GAP) * max(scale, 0.75)
                     + 0.32 * 0.8 * self._row + 0.08)
        width = FACTORY_WIDTH + ROAD_WIDTH + row_width + 0.4
        height = LEGEND_HEIGHT + rows_height + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, LEGEND_HEIGHT / height, 1, rows_height / height])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, rows_height)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.fig.patch.set_facecolor('white')

        self._centers = [rows_height / 2 + (count / 2 - row - 0.5) * self._row
                         for row in range(len(self._warehouses))]
        self._draw_factory(rows_height / 2)
        self._draw_roads(rows_height / 2, scale)
        self._draw_warehouses(scale)
        self._draw_legend()

    def _value(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    # ---------- the scene

    def _draw_factory(self, middle):
        _factory(self.ax, 0.25, middle - 0.1, 1.55, 1.0)
        self._yard = _crates(self.ax, 0.3, middle - 0.95, 1.4, 0.75, YARD_SLOTS, rack=False)

    def _draw_roads(self, middle, scale):

        # one road leaves the factory and branches to the door of each warehouse
        if not self._warehouses:
            return
        road = {'color': ROAD_COLOR, 'linewidth': 7 * max(scale, 0.5), 'solid_capstyle': 'round',
                'solid_joinstyle': 'round', 'zorder': 1}
        line = {'color': ROAD_LINE_COLOR, 'linewidth': 0.8, 'dashes': (3, 3), 'zorder': 1.1}
        (start, bus, end) = (FACTORY_WIDTH - 0.2, FACTORY_WIDTH + ROAD_WIDTH / 2, FACTORY_WIDTH + ROAD_WIDTH)
        door = 0.15 * self._row
        ys = [y - 0.4 * self._row + door for y in self._centers]
        low, high = min(ys + [middle - 0.1]), max(ys + [middle - 0.1])
        for (xs, yy) in (([start, bus], [middle - 0.1] * 2), ([bus, bus], [low, high])):
            self.ax.plot(xs, yy, **road)
            self.ax.plot(xs, yy, **line)
        for y in ys:
            self.ax.plot([bus, end + 0.1], [y, y], **road)
            self.ax.plot([bus, end + 0.1], [y, y], **line)

    def _draw_warehouses(self, scale):
        s = max(scale, 0.75)
        h = 0.8 * self._row
        self._racks, self._demand = {}, {}
        rack_x = FACTORY_WIDTH + ROAD_WIDTH + (WAREHOUSE_WIDTH + GAP) * s
        cart = 0.32 * h
        demand_x = rack_x + (RACK_WIDTH + GAP) * s + cart + 0.08
        for (warehouse, y) in zip(self._warehouses, self._centers):
            bottom = y - 0.4 * self._row
            _warehouse(self.ax, FACTORY_WIDTH + ROAD_WIDTH, bottom, WAREHOUSE_WIDTH * s, h)
            self._racks[warehouse] = _crates(self.ax, rack_x, bottom + 0.05 * h, RACK_WIDTH * s, 0.75 * h,
                                             RACK_SLOTS, rack=True)
            _cart(self.ax, demand_x - cart - 0.08, bottom + 0.22 * h, cart)
            bars = []
            for (k, color) in enumerate((LAST_COLOR, BEFORE_COLOR)):
                bar_y = bottom + (0.42 - 0.25 * k) * h
                self.ax.add_patch(patches.Rectangle((demand_x, bar_y), DEMAND_WIDTH * s, 0.18 * h,
                                                    facecolor=BAR_TRACK_COLOR, edgecolor='none', zorder=2))
                bars.append(self.ax.add_patch(patches.Rectangle((demand_x, bar_y), 0, 0.18 * h,
                                                                facecolor=color, edgecolor='none', zorder=2.1)))
            self._demand[warehouse] = (bars, DEMAND_WIDTH * s)

            # the name sits on the roof, where it fits
            size = min(8.5, 72 * 0.18 * h)
            if size >= 5:
                self.ax.text(FACTORY_WIDTH + ROAD_WIDTH + WAREHOUSE_WIDTH * s / 2, bottom + 0.69 * h,
                             warehouse, ha='center', va='center', fontsize=size, color='white',
                             fontweight='bold', zorder=3)

    def _draw_legend(self):
        (width, height) = self._figure_size
        ax = self.fig.add_axes([0, 0.04 / height, 1, 0.42 / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, 0.42)
        ax.set_aspect('equal')
        ax.axis('off')
        entries = [('crate', 'Stock'), ('owed', 'Items owed'), ('last', 'Demand last step'),
                   ('before', 'Demand the step before')]
        widths = [0.45 + 0.066 * len(label) + 0.35 for (_, label) in entries]
        x = (width - sum(widths)) / 2
        for ((kind, label), entry_width) in zip(entries, widths):
            if kind in ('crate', 'owed'):
                crates = _crates(ax, x + 0.05, 0.08, 0.3, 0.26, (2, 1), rack=False)
                _fill(crates, 1.0 if kind == 'crate' else 0.0, 1.0 if kind == 'owed' else 0.0)
            else:
                ax.add_patch(patches.Rectangle((x, 0.13), 0.38, 0.14, edgecolor='none',
                                               facecolor=LAST_COLOR if kind == 'last' else BEFORE_COLOR))
            ax.text(x + 0.45, 0.2, label, ha='left', va='center', fontsize=8.5, color=TEXT_COLOR)
            x += entry_width

    # ---------- the state

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var

        capacity = self._value('CAPACITY-FACTORY')
        _fill(self._yard, min(max(int(state['stock-factory']), 0) / capacity, 1) if capacity > 0 else 0)

        short = 0
        most = self._value('MAX-DEMAND')
        for warehouse in self._warehouses:
            stock = int(state[fluent('stock-warehouse', [warehouse])])
            capacity = self._value('CAPACITY-WAREHOUSE', warehouse)

            # what is owed has no limit, so the rack is full of it at one full warehouse owed
            owed = max(-stock, 0)
            _fill(self._racks[warehouse], min(max(stock, 0) / capacity, 1) if capacity > 0 else 0,
                  min(owed / capacity, 1) if capacity > 0 else float(owed > 0))
            short += owed > 0

            (bars, full) = self._demand[warehouse]
            for (bar, var) in zip(bars, ('demand-new', 'demand-old')):
                demand = int(state[fluent(var, [warehouse])])
                bar.set_width(full * min(demand / most, 1) if most > 0 else 0)

        total = len(self._warehouses)
        title = (f'{short} / {total} warehouse{"" if total == 1 else "s"} short' if short
                 else 'No warehouse short')
        self.ax.set_title(title, fontsize=13, fontweight='bold',
                          color=LOSING_COLOR if short else 'black')

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
