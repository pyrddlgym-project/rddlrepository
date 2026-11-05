import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

STOCK_COLOR = '#42a5f5'
OWED_COLOR = '#e57373'
OWED_ZONE_COLOR = '#ffebee'
EMPTY_COLOR = '#eceff1'
LAST_COLOR = '#455a64'
BEFORE_COLOR = '#b0bec5'
LINE_COLOR = '#90a4ae'
CARD_COLOR = '#f5f7f8'
LOSING_COLOR = '#c62828'
DETAIL_COLOR = '#37474f'

# sizes in inches
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
MAX_SIZE = (9.0, 5.5)
MIN_WIDTH = 8.6
HEADER_HEIGHT = 0.26
MAX_ROW_HEIGHT = 1.0
FACTORY_WIDTH = 1.75
FACTORY_HEIGHT = 1.6
BUS_WIDTH = 0.45
NAME_WIDTH = 0.7
GAUGE_WIDTH = 2.95
COLUMN_GAP = 0.25
DEMAND_LABEL_WIDTH = 0.5
DEMAND_BAR_WIDTH = 1.6
DEMAND_VALUE_WIDTH = 0.35
GAUGE_HEIGHT = 0.14
MEDIUM_GAUGE_HEIGHT = 0.22
DEMAND_BAR_HEIGHT = 0.15

# the part of a warehouse's gauge left of zero, which fills up with the items owed
OWED_SHARE = 0.25

# what fits in a row of each height
FULL_ROW = 0.8
MEDIUM_ROW = 0.42
ARROW_ROW = 0.3
MIN_TEXT_ROW = 0.16
CHAR_WIDTH = 0.8


class SupplyChainVisualizer(BaseViz):
    '''Draws the factory on the left, supplying one row per warehouse. A row shows the stock
    against the capacity with the items owed in red, and the demand of the last two steps.

    The action is not part of the state, so what was produced and shipped only shows in the
    stock. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._warehouses = list(model.type_to_objects['warehouse'])

        # everything is laid out in inches
        count = max(len(self._warehouses), 1)
        room = MAX_SIZE[1] - BOTTOM_MARGIN - TOP_MARGIN - HEADER_HEIGHT
        self._row_height = min(MAX_ROW_HEIGHT, room / count)
        rows_height = max(count * self._row_height, FACTORY_HEIGHT)
        self._demand_width = DEMAND_LABEL_WIDTH + DEMAND_BAR_WIDTH + DEMAND_VALUE_WIDTH
        grid_width = (FACTORY_WIDTH + BUS_WIDTH + NAME_WIDTH + GAUGE_WIDTH + COLUMN_GAP
                      + self._demand_width)
        grid_height = HEADER_HEIGHT + rows_height

        width = max(grid_width + 0.4, MIN_WIDTH)
        height = grid_height + BOTTOM_MARGIN + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, BOTTOM_MARGIN / height, 1, grid_height / height])
        self.ax.set_xlim(-(width - grid_width) / 2, (width + grid_width) / 2)
        self.ax.set_ylim(0, grid_height)
        self.ax.axis('off')

        self._rows_center = rows_height / 2
        self._first_row = rows_height / 2 + count * self._row_height / 2
        self._name_x = FACTORY_WIDTH + BUS_WIDTH
        self._gauge_x = self._name_x + NAME_WIDTH
        self._demand_x = self._gauge_x + GAUGE_WIDTH + COLUMN_GAP

        self._draw_header(grid_height)
        self._draw_factory()
        self._draw_routes()
        self._draw_warehouses()

        handles = [
            patches.Patch(facecolor=STOCK_COLOR, edgecolor='none', label='Stock'),
            patches.Patch(facecolor=OWED_COLOR, edgecolor='none', label='Items owed'),
            patches.Patch(facecolor=LAST_COLOR, edgecolor='none', label='Demand last step'),
            patches.Patch(facecolor=BEFORE_COLOR, edgecolor='none', label='Demand the step before')
        ]
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.25 / height),
                        ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                        handlelength=1.4)
        caption = (f'Items sell for {self._value("PRICE"):g} and cost {self._value("PRODUCTION-COST"):g} to make, '
                   f'each owed item costs {self._value("PENALTY-COST"):g} per step. '
                   'Nothing ships if the factory cannot supply all orders.')
        self.fig.text(0.5, 0.1 / height, caption, ha='center', va='bottom', fontsize=9,
                      color=DETAIL_COLOR)

    def _value(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _draw_header(self, top):
        y = top - HEADER_HEIGHT / 2
        for (x, column_width, text) in ((self._gauge_x, GAUGE_WIDTH, 'Stock'),
                                        (self._demand_x, self._demand_width, 'Demand')):
            self.ax.text(x + column_width / 2, y, text, ha='center', va='center', fontsize=8,
                         fontweight='bold', color=DETAIL_COLOR)

    def _draw_factory(self):
        bottom = self._rows_center - FACTORY_HEIGHT / 2
        top = bottom + FACTORY_HEIGHT
        self.ax.add_patch(patches.FancyBboxPatch(
            (0.02, bottom + 0.02), FACTORY_WIDTH - 0.04, FACTORY_HEIGHT - 0.04,
            boxstyle='round,pad=0,rounding_size=0.06', facecolor=CARD_COLOR, edgecolor=LINE_COLOR,
            linewidth=1.2, zorder=1))
        center = FACTORY_WIDTH / 2
        self.ax.text(center, top - 0.24, 'Factory', ha='center', va='center', fontsize=10.5,
                     fontweight='bold', zorder=2)
        self._factory_stock = self.ax.text(center, top - 0.52, '', ha='center', va='center',
                                           fontsize=8.5, zorder=2)

        self._factory_width = FACTORY_WIDTH - 0.3
        self.ax.add_patch(patches.Rectangle((0.15, top - 0.8), self._factory_width, GAUGE_HEIGHT,
                                            facecolor=EMPTY_COLOR, edgecolor='none', zorder=2))
        self._factory_bar = self.ax.add_patch(patches.Rectangle(
            (0.15, top - 0.8), 0, GAUGE_HEIGHT, facecolor=STOCK_COLOR, edgecolor='none', zorder=3))

        for (y, text) in ((top - 1.06, f'Makes up to {self._value("PRODUCE-MAX"):g}'),
                          (top - 1.28, f'Storage {self._value("STORAGE-COST-FACTORY"):g} per item')):
            self.ax.text(center, y, text, ha='center', va='center', fontsize=7.5,
                         color=DETAIL_COLOR, zorder=2)

    def _row_center(self, row):
        return self._first_row - (row + 0.5) * self._row_height

    def _draw_routes(self):
        if not self._warehouses:
            return
        bus = FACTORY_WIDTH + BUS_WIDTH * 0.4
        end = self._name_x - 0.04
        centers = [self._row_center(row) for row in range(len(self._warehouses))]
        line = {'color': LINE_COLOR, 'linewidth': 1.2, 'zorder': 0}
        self.ax.plot([FACTORY_WIDTH, bus], [self._rows_center] * 2, **line)
        self.ax.plot([bus, bus], [min(centers + [self._rows_center]), max(centers + [self._rows_center])], **line)
        arrows = self._row_height >= ARROW_ROW
        for y in centers:
            if arrows:
                self.ax.add_patch(patches.FancyArrowPatch(
                    (bus, y), (end, y), arrowstyle='-|>', mutation_scale=9, shrinkA=0, shrinkB=0,
                    color=LINE_COLOR, linewidth=1.2, zorder=0))
            else:
                self.ax.plot([bus, end], [y, y], **line)

    def _draw_warehouses(self):
        height = self._row_height
        full = height >= FULL_ROW
        has_text = height >= ARROW_ROW
        self._gauges, self._stock_texts = {}, {}
        self._last, self._before = {}, {}
        limit = int((NAME_WIDTH - 0.08) * 72 / (CHAR_WIDTH * 8.5))
        for (row, warehouse) in enumerate(self._warehouses):
            y = self._row_center(row)
            if height >= MIN_TEXT_ROW:
                self.ax.text(self._name_x + 0.04, y, self._shorten(warehouse, limit), ha='left',
                             va='center', fontsize=8.5, fontweight='bold')

            # the gauge sits between its two lines of text, or holds its text when rows are low
            bar_height = GAUGE_HEIGHT if full else min(MEDIUM_GAUGE_HEIGHT, height * 0.7)
            bar_y = y - bar_height / 2
            zero = self._gauge_x + GAUGE_WIDTH * OWED_SHARE
            self.ax.add_patch(patches.Rectangle((self._gauge_x, bar_y), GAUGE_WIDTH * OWED_SHARE, bar_height,
                                                facecolor=OWED_ZONE_COLOR, edgecolor='none', zorder=1))
            self.ax.add_patch(patches.Rectangle((zero, bar_y), GAUGE_WIDTH * (1 - OWED_SHARE), bar_height,
                                                facecolor=EMPTY_COLOR, edgecolor='none', zorder=1))
            self._gauges[warehouse] = self.ax.add_patch(patches.Rectangle(
                (zero, bar_y), 0, bar_height, edgecolor='none', zorder=2))
            self.ax.plot([zero, zero], [bar_y - 0.02, bar_y + bar_height + 0.02], color=DETAIL_COLOR,
                         linewidth=1.0, zorder=3)

            if full:
                self._stock_texts[warehouse] = self.ax.text(
                    self._gauge_x + GAUGE_WIDTH / 2, y + 0.2, '', ha='center', va='center', fontsize=8.5)
                self.ax.text(self._gauge_x + GAUGE_WIDTH / 2, y - 0.2,
                             f'Truck carries {self._value("TRUCK-CAPACITY", warehouse):g} for '
                             f'{self._value("TRUCK-COST", warehouse):g}, '
                             f'storage {self._value("STORAGE-COST-WAREHOUSE", warehouse):g}',
                             ha='center', va='center', fontsize=7.5, color=DETAIL_COLOR)
            elif has_text:
                self._stock_texts[warehouse] = self.ax.text(
                    self._gauge_x + GAUGE_WIDTH / 2, y, '', ha='center', va='center', fontsize=7.5, zorder=4)

            self._draw_demand(warehouse, y)

    def _draw_demand(self, warehouse, y):
        height = self._row_height
        has_text = height >= MEDIUM_ROW
        bar_height = min(DEMAND_BAR_HEIGHT, height * 0.4)
        offset = max(bar_height / 2 + 0.02, 0.11) if has_text else bar_height / 2 + height * 0.03
        bar_x = self._demand_x + DEMAND_LABEL_WIDTH
        for (shift, label, color, store) in ((offset, 'Last', LAST_COLOR, self._last),
                                             (-offset, 'Before', BEFORE_COLOR, self._before)):
            bar_y = y + shift - bar_height / 2
            self.ax.add_patch(patches.Rectangle((bar_x, bar_y), DEMAND_BAR_WIDTH, bar_height,
                                                facecolor=EMPTY_COLOR, edgecolor='none', zorder=1))
            bar = self.ax.add_patch(patches.Rectangle((bar_x, bar_y), 0, bar_height, facecolor=color,
                                                      edgecolor='none', zorder=2))
            value = None
            if has_text:
                self.ax.text(self._demand_x, y + shift, label, ha='left', va='center', fontsize=7.5,
                             color=DETAIL_COLOR)
                value = self.ax.text(bar_x + DEMAND_BAR_WIDTH + 0.06, y + shift, '', ha='left',
                                     va='center', fontsize=8)
            store[warehouse] = (bar, value)

    def _set_gauge(self, warehouse, stock):
        capacity = self._value('CAPACITY-WAREHOUSE', warehouse)
        zero = self._gauge_x + GAUGE_WIDTH * OWED_SHARE
        bar = self._gauges[warehouse]
        if stock >= 0:
            share = min(stock / capacity, 1) if capacity > 0 else 0
            bar.set(x=zero, width=GAUGE_WIDTH * (1 - OWED_SHARE) * share, facecolor=STOCK_COLOR)
        else:

            # what is owed has no limit, so it fills its zone at one full capacity
            share = min(-stock / capacity, 1) if capacity > 0 else 1
            width = GAUGE_WIDTH * OWED_SHARE * share
            bar.set(x=zero - width, width=width, facecolor=OWED_COLOR)

    def _set_demand(self, cell, demand):
        most = self._value('MAX-DEMAND')
        (bar, value) = cell
        bar.set_width(DEMAND_BAR_WIDTH * min(demand / most, 1) if most > 0 else 0)
        if value is not None:
            value.set_text(str(demand))

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var

        stock = int(state['stock-factory'])
        capacity = self._value('CAPACITY-FACTORY')
        self._factory_stock.set_text(f'Stock {stock} / {capacity:g}')
        self._factory_bar.set_width(self._factory_width * min(max(stock, 0) / capacity, 1) if capacity > 0 else 0)

        owed = {}
        for warehouse in self._warehouses:
            stock = int(state[fluent('stock-warehouse', [warehouse])])
            self._set_gauge(warehouse, stock)
            if stock < 0:
                owed[warehouse] = -stock

            if warehouse in self._stock_texts:
                capacity = self._value('CAPACITY-WAREHOUSE', warehouse)
                if stock < 0:
                    self._stock_texts[warehouse].set(text=f'Short {-stock}', color=LOSING_COLOR,
                                                     fontweight='bold')
                else:
                    self._stock_texts[warehouse].set(text=f'{stock} / {capacity:g}', color='black',
                                                     fontweight='normal')

            self._set_demand(self._last[warehouse], int(state[fluent('demand-new', [warehouse])]))
            self._set_demand(self._before[warehouse], int(state[fluent('demand-old', [warehouse])]))

        total = len(self._warehouses)
        title = f'Step {int(state["epoch"])}, '
        if owed:
            items = sum(owed.values())
            title += (f'{len(owed)} / {total} warehouse{"" if total == 1 else "s"} '
                      f'short by {items} item{"" if items == 1 else "s"}')
        else:
            title += 'no warehouse short'
        self.ax.set_title(title, fontsize=13, fontweight='bold',
                          color=LOSING_COLOR if owed else 'black')

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
