import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


class BinPackingVisualizer(BaseViz):
    '''Draws each bin with the items inside it, and the item to be placed next,
    all on the same scale.

    The state only holds the total size of each bin, so the items are worked out
    from how the totals change: this requires render to be called on every step.
    Items placed between two renders appear as a single item, and whatever a bin
    holds at the start of an episode is drawn in grey.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 column_width=0.9,
                 plot_height=5.0,
                 max_width=14.0,
                 min_width=3.5,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._bins = model.type_to_objects['bin']
        self._capacity = {b: self._nonfluents[RDDLPlanningModel.ground_var('CAPACITY', [b])]
                          for b in self._bins}
        self._item_sizes = [self._nonfluents[RDDLPlanningModel.ground_var('ITEM-SIZE', [i])]
                            for i in model.type_to_objects['item']]
        self._base_scale = max(list(self._capacity.values()) + self._item_sizes)

        self._items = {b: [] for b in self._bins}
        self._newest = set()
        self._last_state = None

        # the columns are the bins, a gap and the next item: the left margin holds
        # the size axis, and the rest leaves room for the title and labels
        num_bins = len(self._bins)
        self._next_x = num_bins + 0.6
        num_columns = num_bins + 1.5
        margins = (0.7, 0.2, 0.75, 0.45)
        fig_width = min(max(column_width * num_columns + margins[0] + margins[1], min_width),
                        max_width)
        fig_height = plot_height + margins[2] + margins[3]
        self._plot_height = plot_height
        self._column_width = (fig_width - margins[0] - margins[1]) / num_columns

        self.fig = plt.figure(figsize=(fig_width, fig_height))
        self.ax = self.fig.add_axes(
            [margins[0] / fig_width, margins[2] / fig_height,
             1 - (margins[0] + margins[1]) / fig_width, plot_height / fig_height])
        self.ax.set_xlim(0, num_columns)
        self.ax.set_xticks([])
        self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.set_ylabel('size')
        for side in ('top', 'right', 'bottom'):
            self.ax.spines[side].set_visible(False)

        # the bins and their names never change, so they are only drawn once
        for (i, b) in enumerate(self._bins):
            self.ax.add_patch(patches.Rectangle(
                (i + 0.1, 0), 0.8, self._capacity[b],
                facecolor='none', edgecolor='black', linewidth=1.2, zorder=3))
            self._add_label(i + 0.5, 0, b, -5, 'top', 10)
        self._add_label(self._next_x + 0.4, 0, 'next', -5, 'top', 10, fontweight='bold')
        self.ax.axvline(num_bins + 0.25, color='#bdbdbd', linestyle='--', linewidth=1)
        self.fig.text(0.5, 0.15 / fig_height, 'Red count: the next item does not fit',
                      ha='center', va='bottom', fontsize=8, color=NO_ROOM_COLOR)

    def _add_label(self, x, y, text, offset, va, font_size, **kwargs):

        # labels shrink to the width of a column and are left out if unreadable
        font_size = min(font_size, 72 * self._column_width / (0.62 * len(text)))
        if font_size < MIN_FONT_SIZE:
            return None
        return self.ax.annotate(
            text, (x, y), xytext=(0, offset), textcoords='offset points',
            ha='center', va=va, fontsize=font_size, zorder=5,
            annotation_clip=False, **kwargs)

    def _update_items(self, totals, next_size):

        # rendering the same state again must not change the items or highlights
        current_state = (totals, next_size)
        if current_state == self._last_state:
            return
        last_totals = None if self._last_state is None else self._last_state[0]
        self._last_state = current_state

        # bins only ever fill up during an episode, so a smaller total means a new one
        if last_totals is None or any(totals[b] < last_totals[b] for b in self._bins):
            self._items = {b: [(totals[b], True)] if totals[b] > 0 else []
                           for b in self._bins}
            self._newest = set()
            return

        self._newest = {b for b in self._bins if totals[b] > last_totals[b]}
        for b in self._newest:
            self._items[b].append((totals[b] - last_totals[b], False))

    def _item_color(self, size):
        low, high = min(self._item_sizes), max(self._item_sizes)
        scale = (size - low) / (high - low) if high > low else 1.0
        return plt.cm.viridis(0.7 * min(max(scale, 0.0), 1.0))

    def _draw_item(self, x, bottom, size, color, inches_per_unit, label=True):
        height = size * inches_per_unit
        artists = [self.ax.add_patch(patches.Rectangle(
            (x, bottom), 0.8, size, facecolor=color, zorder=2,
            edgecolor='white', linewidth=0.8 if height >= 0.06 else 0))]

        # only label items tall enough to hold the text
        font_size = min(9, 72 * 0.8 * self._column_width / (0.62 * len(str(size))))
        if label and height >= 0.18 and font_size >= MIN_FONT_SIZE:
            artists.append(self.ax.text(
                x + 0.4, bottom + size / 2, str(size), ha='center', va='center',
                color='white', fontsize=font_size, fontweight='bold', zorder=4))
        return artists

    def _draw_bin(self, index, bin_name, total, next_size, inches_per_unit):
        artists = []
        bottom = 0
        for (size, initial) in self._items[bin_name]:
            color = INITIAL_COLOR if initial else self._item_color(size)
            artists.extend(self._draw_item(
                index + 0.1, bottom, size, color, inches_per_unit))
            if bin_name in self._newest and not initial and \
                    bottom + size == total:
                artists.append(self.ax.add_patch(patches.Rectangle(
                    (index + 0.1, bottom), 0.8, size, facecolor='none',
                    edgecolor=NEWEST_COLOR, linewidth=3, zorder=4)))
            bottom += size

        no_room = total + next_size > self._capacity[bin_name]
        label = self._add_label(
            index + 0.5, max(total, self._capacity[bin_name]),
            f'{total} / {self._capacity[bin_name]}', 3, 'bottom', 9,
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
        totals = {b: int(state[RDDLPlanningModel.ground_var('total-size', [b])])
                  for b in self._bins}
        next_size = int(state['current-size'])
        self._update_items(totals, next_size)

        # bins can be overfilled if invalid actions are allowed: the scale has to
        # grow to show them, and leaves room for the labels above the tallest column
        top = max(self._base_scale, next_size, *totals.values()) or 1
        self.ax.set_ylim(0, top * (1 + 0.3 / self._plot_height))
        inches_per_unit = self._plot_height / (top * (1 + 0.3 / self._plot_height))

        artists = []
        for (i, b) in enumerate(self._bins):
            artists.extend(self._draw_bin(i, b, totals[b], next_size, inches_per_unit))

        if next_size > 0:
            artists.extend(self._draw_item(
                self._next_x, 0, next_size, self._item_color(next_size), inches_per_unit,
                label=False))
        size_label = self._add_label(
            self._next_x + 0.4, max(next_size, 0), f'size {next_size}', 3, 'bottom', 9)
        if size_label is not None:
            artists.append(size_label)

        used = sum(total > 0 for total in totals.values())
        self.ax.set_title(f'{used} / {len(self._bins)} bins used',
                          fontsize=13, fontweight='bold')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
