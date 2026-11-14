import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

CANVAS_COLOR = '#6d8b74'
CANVAS_EDGE_COLOR = '#3e5245'
INSIDE_COLOR = '#f4f1e8'
STRAP_COLOR = '#3e5245'
BUCKLE_COLOR = '#cfd8dc'
NO_ROOM_COLOR = '#c62828'
NEWEST_COLOR = '#ff6f00'
INITIAL_COLOR = '#b0bec5'
STAND_COLOR = '#90a4ae'
TEXT_COLOR = '#263238'

# items run from slate to gold as their value per unit of weight grows
ITEM_COLORMAP = LinearSegmentedColormap.from_list(
    'loot', ['#546e7a', '#26a69a', '#ffc107'])

# a knapsack is at most this wide, in inches, or this fraction of its column; its walls
# are this thick and its flap this tall, in inches; items are this far apart
MAX_PACK_WIDTH, PACK_FRACTION = 1.3, 0.76
WALL, FLAP = 0.05, 0.2
ITEM_GAP = 0.025

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class KnapsackVisualizer(BaseViz):
    '''Draws each knapsack as a backpack, cut away to show the items inside it, and the
    item on offer on a stand, all on the same scale. The height of an item is its
    weight, its number is its value, and its color runs from slate to gold as its value
    per unit of weight grows. The flap of a knapsack turns red when the offer does not
    fit in it.

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
                 min_width=4.8,
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

        # the columns are the knapsacks, a gap and the item on offer: the bottom has the
        # color scale and a note
        num_knapsacks = len(self._knapsacks)
        self._next_x = num_knapsacks + 0.6
        num_columns = num_knapsacks + 1.5
        left, right, bottom, top = 0.2, 0.2, 1.2, 0.45
        fig_width = min(max(column_width * num_columns + left + right, min_width), max_width)
        fig_height = plot_height + bottom + top
        self._plot_height = plot_height
        self._column_width = (fig_width - left - right) / num_columns
        self._pack_width = min(PACK_FRACTION, MAX_PACK_WIDTH / self._column_width)

        self.fig = plt.figure(figsize=(fig_width, fig_height))
        self.ax = self.fig.add_axes(
            [left / fig_width, bottom / fig_height,
             (fig_width - left - right) / fig_width, plot_height / fig_height])
        self.ax.set_xlim(0, num_columns)
        self.ax.axis('off')

        # the knapsacks, the stand and their names never change, so they are only drawn
        # once, on the usual scale; only the flaps change color
        self._set_scale(self._base_scale)
        self._flaps = {}
        for (i, k) in enumerate(self._knapsacks):
            self._flaps[k] = self._draw_pack(i, self._capacity[k])
            self._add_label(i + 0.5, self._floor, k, -5, 'top', 10)
        self._draw_stand()
        self._add_label(self._next_x + 0.4, self._floor, 'offer', -5, 'top', 10,
                        fontweight='bold')

        # the color scale is only useful if the items differ
        if self._norm.vmax > self._norm.vmin:
            bar_width = min(2.5, fig_width - 1)
            bar_axis = self.fig.add_axes(
                [(fig_width - bar_width) / 2 / fig_width, 0.68 / fig_height,
                 bar_width / fig_width, 0.1 / fig_height])
            bar = self.fig.colorbar(ScalarMappable(self._norm, ITEM_COLORMAP),
                                    cax=bar_axis, orientation='horizontal')
            bar.ax.tick_params(labelsize=8, colors='#616161')
            bar.outline.set_edgecolor('#b0bec5')
            bar.set_label('value per unit of weight', fontsize=9, labelpad=1)
        self.fig.text(
            0.5, 0.1 / fig_height,
            'Height: weight. Number: value. A red flap: the offer does not fit.',
            ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)

    def _get_state(self, state):
        return (tuple(int(state[RDDLPlanningModel.ground_var('total-weight', [k])])
                      for k in self._knapsacks),
                tuple(int(state[RDDLPlanningModel.ground_var('total-value', [k])])
                      for k in self._knapsacks),
                int(state['current-weight']),
                int(state['current-value']))

    def _set_scale(self, top):

        # weights go up, with room below for the bottoms of the packs and the stand, and
        # above for the flaps, handles and labels: rounded corners need the shape of a
        # unit in inches
        top = top or 1
        self._floor = -0.05 * top
        headroom = (FLAP + 0.15 + 0.5) * top / self._plot_height
        self.ax.set_ylim(self._floor, top + headroom)
        self._inches_per_unit = self._plot_height / (top + headroom - self._floor)
        self._aspect = self._column_width / self._inches_per_unit

    def _units(self, inches):
        return inches / self._inches_per_unit

    def _add_label(self, x, y, text, offset, va, font_size, **kwargs):

        # labels shrink to the width of a column and are left out if unreadable
        length = max(len(line) for line in text.split('\n'))
        font_size = min(font_size, 72 * self._column_width / (0.62 * length))
        if font_size < MIN_FONT_SIZE:
            return None
        return self.ax.annotate(
            text, (x, y), xytext=(0, offset), textcoords='offset points',
            ha='center', va=va, fontsize=font_size, zorder=7,
            color=kwargs.pop('color', TEXT_COLOR), annotation_clip=False, **kwargs)

    def _rounded(self, x, y, width, height, radius, **kwargs):

        # a rectangle with corners rounded by the given number of inches
        return patches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f'round,pad=0,rounding_size={radius / self._column_width}',
            mutation_aspect=self._aspect, **kwargs)

    def _draw_pack(self, index, capacity):

        # side pockets, the canvas body cut away to show its inside, a handle, and a flap
        # on top with a strap and buckle, which is returned
        center = index + 0.5
        half = self._pack_width / 2
        wall = WALL / self._column_width
        bottom = self._floor * 0.6
        pocket = 0.2 * self._pack_width
        for left in (center - half - 0.4 * pocket, center + half - 0.6 * pocket):
            self.ax.add_patch(self._rounded(
                left, bottom, pocket, 0.3 * capacity, 0.05,
                facecolor=CANVAS_COLOR, edgecolor=CANVAS_EDGE_COLOR, linewidth=1.2, zorder=1))
        self.ax.add_patch(self._rounded(
            center - half, bottom, 2 * half, capacity - bottom, 0.14,
            facecolor=CANVAS_COLOR, edgecolor=CANVAS_EDGE_COLOR, linewidth=1.5, zorder=1.5))
        self.ax.add_patch(self._rounded(
            center - half + wall, 0, 2 * (half - wall), capacity, 0.08,
            facecolor=INSIDE_COLOR, edgecolor='none', zorder=1.6))

        # the handle is a loop above the flap
        t = np.linspace(0, np.pi, 30)
        flap_top = capacity + self._units(FLAP)
        self.ax.plot(center + 0.18 * self._pack_width * np.cos(t),
                     flap_top + self._units(0.12) * np.sin(t),
                     color=STRAP_COLOR, linewidth=2.5, solid_capstyle='round', zorder=1.4)
        flap = self.ax.add_patch(self._rounded(
            center - half - 0.02, capacity - self._units(0.02), 2 * half + 0.04,
            self._units(FLAP) + self._units(0.02), 0.12,
            facecolor=CANVAS_COLOR, edgecolor=CANVAS_EDGE_COLOR, linewidth=1.5, zorder=5))
        strap = 0.12 * self._pack_width
        self.ax.add_patch(patches.Rectangle(
            (center - strap / 2, capacity - self._units(0.12)), strap,
            self._units(FLAP + 0.1), facecolor=STRAP_COLOR, edgecolor='none', zorder=5.5))
        self.ax.add_patch(self._rounded(
            center - 0.7 * strap, capacity - self._units(0.09), 1.4 * strap, self._units(0.07),
            0.015, facecolor=BUCKLE_COLOR, edgecolor=CANVAS_EDGE_COLOR, linewidth=0.8,
            zorder=5.6))
        return flap

    def _draw_stand(self):

        # a low stand under the item on offer
        self.ax.add_patch(self._rounded(
            self._next_x - 0.05, self._floor * 0.8, 0.9, -self._floor * 0.6, 0.03,
            facecolor=STAND_COLOR, edgecolor='none', zorder=1))

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

    def _draw_item(self, center, bottom, weight, value, color, newest=False, label=True):

        # an item is a package with a gap around it and its value on it: one without
        # weight has no height, so it shows as a line
        gap = self._units(ITEM_GAP)
        width = self._pack_width - 2 * (WALL + 0.04) / self._column_width
        (left, low, height) = (center - width / 2, bottom + gap, max(weight - 2 * gap, 0.0))
        artists = [self.ax.add_patch(self._rounded(
            left, low, width, height,
            min(0.04, 0.4 * height * self._inches_per_unit),
            facecolor=color, edgecolor=NEWEST_COLOR if newest else 'none',
            linewidth=2.5 if newest else 0, zorder=2))]

        # only label items tall enough to hold the text, in a color that stands out
        font_size = min(9, 72 * width * self._column_width / (0.62 * len(str(value))))
        if label and height * self._inches_per_unit >= 0.18 and font_size >= MIN_FONT_SIZE:
            light = np.dot(to_rgb(color), (0.299, 0.587, 0.114)) > 0.6
            artists.append(self.ax.text(
                center, low + height / 2, str(value), ha='center', va='center',
                color=TEXT_COLOR if light else 'white', fontsize=font_size,
                fontweight='bold', zorder=4))
        return artists

    def _draw_knapsack(self, index, knapsack, total_weight, total_value, offer_weight):
        artists = []
        bottom = 0
        items = self._items[knapsack]
        for (position, (weight, value, initial)) in enumerate(items):
            color = INITIAL_COLOR if initial else self._item_color(weight, value)
            newest = knapsack in self._newest and position == len(items) - 1
            artists.extend(self._draw_item(index + 0.5, bottom, weight, value, color, newest))
            bottom += weight

        no_room = total_weight + offer_weight > self._capacity[knapsack]
        self._flaps[knapsack].set_facecolor(NO_ROOM_COLOR if no_room else CANVAS_COLOR)
        label = self._add_label(
            index + 0.5, max(total_weight, self._capacity[knapsack]) + self._units(FLAP + 0.12),
            f'{total_weight} / {self._capacity[knapsack]}\nvalue {total_value}', 3, 'bottom', 9,
            color=NO_ROOM_COLOR if no_room else TEXT_COLOR,
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

        # the scale has to grow if invalid states are allowed
        self._set_scale(max(self._base_scale, offer_weight, *weights))

        artists = []
        for (i, k) in enumerate(self._knapsacks):
            artists.extend(self._draw_knapsack(i, k, weights[i], values[i], offer_weight))

        artists.extend(self._draw_item(
            self._next_x + 0.4, 0, offer_weight, offer_value,
            self._item_color(offer_weight, offer_value)))
        offer_label = self._add_label(
            self._next_x + 0.4, offer_weight,
            f'weight {offer_weight}\nvalue {offer_value}', 3, 'bottom', 9)
        if offer_label is not None:
            artists.append(offer_label)

        self.ax.set_title(f'Total value: {sum(values)}', fontsize=13, fontweight='bold',
                          color=TEXT_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
