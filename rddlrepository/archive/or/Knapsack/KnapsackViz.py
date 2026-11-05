import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

NEWEST_COLOR = '#ffa000'
NO_ROOM_COLOR = '#c62828'
INITIAL_COLOR = '#9e9e9e'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5

# the part of viridis that keeps white labels readable
ITEM_COLORMAP = ListedColormap(plt.cm.viridis(np.linspace(0, 0.7, 256)))


class KnapsackVisualizer(BaseViz):
    '''Draws each knapsack with the items inside it, and the item on offer, all on
    the same scale. The height of an item is its weight, its number is its value,
    and its color is its value per unit of weight.

    The state only holds the totals of each knapsack, so the items are worked out
    from how the totals change: this requires render to be called on every step.
    Items placed between two renders appear as a single item, and whatever a
    knapsack holds at the start of an episode is drawn in grey. An offered item
    that was not placed leaves no trace in the state, so no item is highlighted.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 column_width=0.9,
                 plot_height=5.0,
                 max_width=14.0,
                 min_width=4.2,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._knapsacks = model.type_to_objects['knapsack']
        self._capacity = {k: self._nonfluents[RDDLPlanningModel.ground_var('CAPACITY', [k])]
                          for k in self._knapsacks}
        items = model.type_to_objects['item']
        weights = [self._nonfluents[RDDLPlanningModel.ground_var('ITEM-WEIGHT', [i])]
                   for i in items]
        values = [self._nonfluents[RDDLPlanningModel.ground_var('ITEM-VALUE', [i])]
                  for i in items]
        self._base_scale = max(list(self._capacity.values()) + weights)

        # items without weight have no value per unit of weight, so they take the
        # top of the scale
        ratios = [v / w for (v, w) in zip(values, weights) if w > 0] or [1.0]
        self._norm = Normalize(min(ratios), max(ratios), clip=True)

        init_state = model.ground_vars_with_values(model.state_fluents)
        self._init_state = self._get_state(init_state)

        self._items = {k: [] for k in self._knapsacks}
        self._newest = set()
        self._last_state = None

        # the columns are the knapsacks, a gap and the item on offer: the left margin
        # holds the weight axis, and the bottom has the color scale and a note
        num_knapsacks = len(self._knapsacks)
        self._next_x = num_knapsacks + 0.6
        num_columns = num_knapsacks + 1.5
        left, right, bottom, top = 0.7, 0.2, 1.2, 0.45
        fig_width = min(max(column_width * num_columns + left + right, min_width), max_width)
        fig_height = plot_height + bottom + top
        self._plot_height = plot_height
        self._column_width = (fig_width - left - right) / num_columns

        self.fig = plt.figure(figsize=(fig_width, fig_height))
        self.ax = self.fig.add_axes(
            [left / fig_width, bottom / fig_height,
             (fig_width - left - right) / fig_width, plot_height / fig_height])
        self.ax.set_xlim(0, num_columns)
        self.ax.set_xticks([])
        self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.set_ylabel('weight')
        for side in ('top', 'right', 'bottom'):
            self.ax.spines[side].set_visible(False)

        # the knapsacks and their names never change, so they are only drawn once
        for (i, k) in enumerate(self._knapsacks):
            self.ax.add_patch(patches.Rectangle(
                (i + 0.1, 0), 0.8, self._capacity[k],
                facecolor='none', edgecolor='black', linewidth=1.2, zorder=3))
            self._add_label(i + 0.5, 0, k, -5, 'top', 10)
        self._add_label(self._next_x + 0.4, 0, 'offer', -5, 'top', 10, fontweight='bold')
        self.ax.axvline(num_knapsacks + 0.25, color='#bdbdbd', linestyle='--', linewidth=1)

        # the color scale is only useful if the items differ
        if self._norm.vmax > self._norm.vmin:
            bar_width = min(2.5, fig_width - 1)
            bar_axis = self.fig.add_axes(
                [(fig_width - bar_width) / 2 / fig_width, 0.68 / fig_height,
                 bar_width / fig_width, 0.1 / fig_height])
            bar = self.fig.colorbar(ScalarMappable(self._norm, ITEM_COLORMAP),
                                    cax=bar_axis, orientation='horizontal')
            bar.ax.tick_params(labelsize=7)
            bar.set_label('value per unit of weight', fontsize=8, labelpad=1)
        self.fig.text(
            0.5, 0.06 / fig_height,
            'Height: weight. Number: value. Red count: the offer does not fit.',
            ha='center', va='bottom', fontsize=8)

    def _get_state(self, state):
        return (tuple(int(state[RDDLPlanningModel.ground_var('total-weight', [k])])
                      for k in self._knapsacks),
                tuple(int(state[RDDLPlanningModel.ground_var('total-value', [k])])
                      for k in self._knapsacks),
                int(state['current-weight']),
                int(state['current-value']))

    def _add_label(self, x, y, text, offset, va, font_size, **kwargs):

        # labels shrink to the width of a column and are left out if unreadable
        length = max(len(line) for line in text.split('\n'))
        font_size = min(font_size, 72 * self._column_width / (0.62 * length))
        if font_size < MIN_FONT_SIZE:
            return None
        return self.ax.annotate(
            text, (x, y), xytext=(0, offset), textcoords='offset points',
            ha='center', va=va, fontsize=font_size, zorder=5,
            annotation_clip=False, **kwargs)

    def _update_items(self, state):

        # rendering the same state again must not change the items or highlights
        if state == self._last_state:
            return
        last_state, self._last_state = self._last_state, state
        weights, values = state[0], state[1]

        # weights only ever grow during an episode, so a smaller one means a new one
        if last_state is None or state == self._init_state or \
                any(w < last for (w, last) in zip(weights, last_state[0])):
            self._items = {k: [(w, v, True)] if w > 0 else []
                           for (k, w, v) in zip(self._knapsacks, weights, values)}
            self._newest = set()
            return

        self._newest = set()
        for (i, k) in enumerate(self._knapsacks):
            weight = weights[i] - last_state[0][i]
            value = values[i] - last_state[1][i]
            if weight != 0 or value != 0:
                self._items[k].append((weight, value, False))
                self._newest.add(k)

    def _item_color(self, weight, value):
        return ITEM_COLORMAP(self._norm(value / weight if weight > 0 else self._norm.vmax))

    def _draw_item(self, x, bottom, weight, value, color, inches_per_unit, label=True):
        height = weight * inches_per_unit
        artists = [self.ax.add_patch(patches.Rectangle(
            (x, bottom), 0.8, weight, facecolor=color, zorder=2,
            edgecolor='white', linewidth=0.8 if height >= 0.06 else 0))]

        # only label items tall enough to hold the text
        font_size = min(9, 72 * 0.8 * self._column_width / (0.62 * len(str(value))))
        if label and height >= 0.18 and font_size >= MIN_FONT_SIZE:
            artists.append(self.ax.text(
                x + 0.4, bottom + weight / 2, str(value), ha='center', va='center',
                color='white', fontsize=font_size, fontweight='bold', zorder=4))
        return artists

    def _draw_knapsack(self, index, knapsack, total_weight, total_value, offer_weight,
                       inches_per_unit):
        artists = []
        bottom = 0
        items = self._items[knapsack]
        for (position, (weight, value, initial)) in enumerate(items):
            color = INITIAL_COLOR if initial else self._item_color(weight, value)
            artists.extend(self._draw_item(
                index + 0.1, bottom, weight, value, color, inches_per_unit))

            # an item without weight has no height, so this shows it as a line
            if knapsack in self._newest and position == len(items) - 1:
                artists.append(self.ax.add_patch(patches.Rectangle(
                    (index + 0.1, bottom), 0.8, weight, facecolor='none',
                    edgecolor=NEWEST_COLOR, linewidth=3, zorder=4)))
            bottom += weight

        no_room = total_weight + offer_weight > self._capacity[knapsack]
        label = self._add_label(
            index + 0.5, max(total_weight, self._capacity[knapsack]),
            f'{total_weight} / {self._capacity[knapsack]}\nvalue {total_value}', 3, 'bottom', 9,
            color=NO_ROOM_COLOR if no_room else 'black',
            fontweight='bold' if no_room else 'normal')
        if label is not None:
            artists.append(label)
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        state = self._get_state(state)
        self._update_items(state)
        weights, values, offer_weight, offer_value = state

        # the scale has to grow if invalid states are allowed, and leaves room for
        # the labels above the tallest column
        headroom = 1 + 0.5 / self._plot_height
        top = (max(self._base_scale, offer_weight, *weights) or 1) * headroom
        self.ax.set_ylim(0, top)
        inches_per_unit = self._plot_height / top

        artists = []
        for (i, k) in enumerate(self._knapsacks):
            artists.extend(self._draw_knapsack(
                i, k, weights[i], values[i], offer_weight, inches_per_unit))

        artists.extend(self._draw_item(
            self._next_x, 0, offer_weight, offer_value,
            self._item_color(offer_weight, offer_value), inches_per_unit, label=False))
        offer_label = self._add_label(
            self._next_x + 0.4, offer_weight,
            f'weight {offer_weight}\nvalue {offer_value}', 3, 'bottom', 9)
        if offer_label is not None:
            artists.append(offer_label)

        self.ax.set_title(f'Total value: {sum(values)}', fontsize=13, fontweight='bold')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
