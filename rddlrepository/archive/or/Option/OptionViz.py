import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BELOW_COLOR = '#90caf9'
ABOVE_COLOR = '#66bb6a'
SPENT_COLOR = '#cfd8dc'
EMPTY_COLOR = '#eceff1'
TIME_COLOR = '#78909c'
STRIKE_COLOR = '#c62828'
WINNING_COLOR = '#2e7d32'
DETAIL_COLOR = '#37474f'
SPENT_TEXT_COLOR = '#90a4ae'

# sizes in inches
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
MAX_SIZE = (9.0, 5.5)
MIN_WIDTH = 8.6
MIN_HEIGHT = 2.8
TIME_HEIGHT = 0.42
HEADER_HEIGHT = 0.28
MAX_ROW_HEIGHT = 0.62
NAME_WIDTH = 1.6
BAR_WIDTH = 4.6
PRICE_WIDTH = 0.85
OVER_WIDTH = 0.85
BAR_HEIGHT = 0.24
TIME_BAR_HEIGHT = 0.14

# what fits in a row of each height
DETAIL_ROW = 0.42
MIN_TEXT_ROW = 0.16
CHAR_WIDTH = 0.75


class OptionVisualizer(BaseViz):
    '''Draws one bar per asset with its price against the strike, and the time against the
    maturity. The title shows what exercising the option pays in the current state.

    Once exercised the option cannot pay again, so the prices are greyed out, although they
    keep moving. What it paid is not part of the state. The drawing only depends on the
    current state.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._assets = list(model.type_to_objects['asset'])
        self._strike = float(self._nonfluents['STRIKE'])
        self._maturity = float(self._nonfluents['MATURITY'])

        # the scale keeps still unless a price leaves it, and does not depend on past frames
        initial = model.ground_vars_with_values(model.state_fluents)
        prices = [float(initial[self._key('price', asset)]) for asset in self._assets]
        self._base_scale = max(2 * max(prices + [self._strike]), 1.0)

        # everything is laid out in inches
        count = max(len(self._assets), 1)
        room = MAX_SIZE[1] - BOTTOM_MARGIN - TOP_MARGIN - TIME_HEIGHT - HEADER_HEIGHT
        self._row_height = min(MAX_ROW_HEIGHT, room / count)
        grid_width = NAME_WIDTH + BAR_WIDTH + PRICE_WIDTH + OVER_WIDTH
        grid_height = TIME_HEIGHT + HEADER_HEIGHT + count * self._row_height

        width = max(grid_width + 0.4, MIN_WIDTH)
        height = max(grid_height + BOTTOM_MARGIN + TOP_MARGIN, MIN_HEIGHT)
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, BOTTOM_MARGIN / height, 1,
                                     (height - BOTTOM_MARGIN - TOP_MARGIN) / height])
        extra = height - BOTTOM_MARGIN - TOP_MARGIN - grid_height
        self.ax.set_xlim(-(width - grid_width) / 2, (width + grid_width) / 2)
        self.ax.set_ylim(-extra / 2, grid_height + extra / 2)
        self.ax.axis('off')

        self._grid_height = grid_height
        self._bars_top = grid_height - TIME_HEIGHT - HEADER_HEIGHT
        self._draw_time()
        self._draw_header()
        self._draw_assets()

        handles = [
            patches.Patch(facecolor=BELOW_COLOR, edgecolor='none', label='Price up to the strike'),
            patches.Patch(facecolor=ABOVE_COLOR, edgecolor='none', label='Price above the strike'),
            patches.Patch(facecolor=SPENT_COLOR, edgecolor='none', label='Option exercised')
        ]
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.25 / height),
                        ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                        handlelength=1.4)
        caption = ('Exercising pays the highest price minus the strike, once only. '
                   'The option exercises itself at maturity.')
        self.fig.text(0.5, 0.1 / height, caption, ha='center', va='bottom', fontsize=9,
                      color=DETAIL_COLOR)

    def _key(self, var, *args):
        return RDDLPlanningModel.ground_var(var, list(args))

    def _value(self, var, *args):
        return self._nonfluents[self._key(var, *args)]

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _draw_time(self):
        y = self._grid_height - TIME_HEIGHT / 2
        self.ax.text(0.04, y, 'Time', ha='left', va='center', fontsize=8.5, fontweight='bold')
        bar_y = y - TIME_BAR_HEIGHT / 2
        self.ax.add_patch(patches.Rectangle((NAME_WIDTH, bar_y), BAR_WIDTH, TIME_BAR_HEIGHT,
                                            facecolor=EMPTY_COLOR, edgecolor='none', zorder=1))
        self._time_bar = self.ax.add_patch(patches.Rectangle(
            (NAME_WIDTH, bar_y), 0, TIME_BAR_HEIGHT, facecolor=TIME_COLOR, edgecolor='none', zorder=2))
        self._time_text = self.ax.text(NAME_WIDTH + BAR_WIDTH + 0.08, y, '', ha='left', va='center',
                                       fontsize=8.5)

    def _draw_header(self):
        y = self._bars_top + HEADER_HEIGHT / 2
        for (x, text) in ((NAME_WIDTH + BAR_WIDTH + PRICE_WIDTH / 2, 'Price'),
                          (NAME_WIDTH + BAR_WIDTH + PRICE_WIDTH + OVER_WIDTH / 2, 'Over strike')):
            self.ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold',
                         color=DETAIL_COLOR)
        self._strike_label = self.ax.text(NAME_WIDTH, y, '', ha='center', va='center', fontsize=8,
                                          fontweight='bold', color=STRIKE_COLOR)
        self._strike_line = self.ax.plot([NAME_WIDTH, NAME_WIDTH], [0, self._bars_top + 0.04],
                                         color=STRIKE_COLOR, linewidth=1.4, zorder=4)[0]

    def _draw_assets(self):
        height = self._row_height
        has_text = height >= MIN_TEXT_ROW
        has_detail = height >= DETAIL_ROW
        bar_height = min(BAR_HEIGHT, height * 0.65)
        limit = int((NAME_WIDTH - 0.1) * 72 / (CHAR_WIDTH * 8.5))
        self._below, self._above, self._names, self._prices, self._overs = {}, {}, {}, {}, {}
        self._details = []
        for (row, asset) in enumerate(self._assets):
            y = self._bars_top - (row + 0.5) * height
            bar_y = y - bar_height / 2
            self.ax.add_patch(patches.Rectangle((NAME_WIDTH, bar_y), BAR_WIDTH, bar_height,
                                                facecolor=EMPTY_COLOR, edgecolor='none', zorder=1))
            self._below[asset] = self.ax.add_patch(patches.Rectangle(
                (NAME_WIDTH, bar_y), 0, bar_height, edgecolor='none', zorder=2))
            self._above[asset] = self.ax.add_patch(patches.Rectangle(
                (NAME_WIDTH, bar_y), 0, bar_height, edgecolor='none', zorder=2))
            if not has_text:
                continue

            name_y = y + 0.08 if has_detail else y
            self._names[asset] = self.ax.text(0.04, name_y, self._shorten(asset, limit), ha='left',
                                              va='center', fontsize=8.5)
            if has_detail:
                self._details.append(self.ax.text(
                    0.04, y - 0.1, f'Mean {self._value("MEAN", asset):g}, var {self._value("VAR", asset):g}',
                    ha='left', va='center', fontsize=7, color=DETAIL_COLOR))
            right = NAME_WIDTH + BAR_WIDTH
            self._prices[asset] = self.ax.text(right + PRICE_WIDTH / 2, y, '', ha='center', va='center',
                                               fontsize=8.5)
            self._overs[asset] = self.ax.text(right + PRICE_WIDTH + OVER_WIDTH / 2, y, '', ha='center',
                                              va='center', fontsize=8.5)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        time = float(state['time'])
        exercised = bool(state['exercised'])
        prices = {asset: float(state[self._key('price', asset)]) for asset in self._assets}
        highest = max(prices.values(), default=0.0)
        payoff = max(0.0, highest - self._strike)

        # time
        share = min(time / self._maturity, 1) if self._maturity > 0 else 1
        self._time_bar.set_width(BAR_WIDTH * share)
        self._time_text.set_text(f'{time:.3f} / {self._maturity:g} years')

        # the strike line moves with the scale when a price goes beyond it
        scale = max(self._base_scale, 1.1 * highest)
        inches = BAR_WIDTH / scale
        strike_x = NAME_WIDTH + self._strike * inches
        self._strike_line.set_xdata([strike_x, strike_x])
        self._strike_label.set(text=f'Strike {self._strike:g}', x=strike_x)

        for asset in self._assets:
            price = prices[asset]
            below = min(price, self._strike) * inches
            above = max(price - self._strike, 0) * inches
            self._below[asset].set(width=below, facecolor=SPENT_COLOR if exercised else BELOW_COLOR)
            self._above[asset].set(x=NAME_WIDTH + below, width=above,
                                   facecolor=SPENT_COLOR if exercised else ABOVE_COLOR)
            if asset in self._names:
                best = price == highest and not exercised
                base = SPENT_TEXT_COLOR if exercised else 'black'
                self._names[asset].set(fontweight='bold' if best else 'normal', color=base)
                self._prices[asset].set(text=f'{price:.2f}', color=base,
                                        fontweight='bold' if best else 'normal')
                over = price - self._strike
                good = over > 0 and not exercised
                self._overs[asset].set(text=f'{over:+.2f}', color=WINNING_COLOR if good else
                                       (base if exercised else DETAIL_COLOR),
                                       fontweight='bold' if best else 'normal')

        if exercised:
            title = 'Option exercised, nothing more to pay'
        else:
            if time >= self._maturity:
                title = f'Expires this step, pays {payoff:.2f}'
            elif payoff > 0:
                title = f'In the money, exercising pays {payoff:.2f}'
            else:
                title = 'Out of the money, exercising pays nothing'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color='black')

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
