import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

TOTE_COLOR = '#eceff1'
WALL_COLOR = '#455a64'
NO_ROOM_COLOR = '#c62828'
NEWEST_COLOR = '#ff8f00'
INITIAL_COLOR = '#b0bec5'
OTHER_COLOR = '#a1887f'
TAPE_COLOR = '#ffffff'
BELT_COLOR = '#37474f'
ROLLER_COLOR = '#90a4ae'
TEXT_COLOR = '#263238'

# boxes take colors in the order of their sizes, smallest first
# ponytail: colors repeat beyond ten sizes, where only the numbers tell boxes apart
BOX_COLORS = ('#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#76b7b2', '#edc948', '#b07aa1',
              '#9c755f', '#ff9da7', '#2f4b7c')

# the walls of a bin are at these fractions of its column, and boxes are this far apart,
# in inches
WALL_LEFT, WALL_RIGHT = 0.1, 0.9
BOX_GAP = 0.025

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class BinPackingVisualizer(BaseViz):
    '''Draws each bin as a tote with the boxes inside it, and the box to be placed next
    on a conveyor belt, all on the same scale. Boxes of each size have their own color,
    and the rim of a bin turns red when the next box does not fit in it.

    The state only holds the total size of each bin, so the boxes are worked out
    from how the totals change: this requires render to be called on every step.
    Boxes placed between two renders appear as a single box, and whatever a bin
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
        self._size_rank = {size: k for (k, size) in enumerate(sorted(set(self._item_sizes)))}
        self._base_scale = max(list(self._capacity.values()) + self._item_sizes)

        self._items = {b: [] for b in self._bins}
        self._newest = set()
        self._last_state = None

        # the columns are the bins, a gap and the conveyor with the next box: the margins
        # leave room for the title and labels
        num_bins = len(self._bins)
        self._next_x = num_bins + 0.6
        num_columns = num_bins + 1.5
        margins = (0.2, 0.2, 0.75, 0.45)
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
        self.ax.axis('off')

        # the totes, the conveyor and their names never change, so they are only drawn
        # once, on the usual scale; only the rims change color
        self._set_scale(self._base_scale)
        self._rims = {}
        for (i, b) in enumerate(self._bins):
            self._rims[b] = self._draw_tote(i, self._capacity[b])
            self._add_label(i + 0.5, self._floor, b, -5, 'top', 10)
        self._draw_conveyor()
        self._add_label(self._next_x + 0.4, self._floor, 'next', -5, 'top', 10, fontweight='bold')
        self.fig.text(0.5, 0.1 / fig_height, 'A red rim: the next box does not fit',
                      ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)

    def _set_scale(self, top):

        # sizes go up, with room below for the floors of the totes and the conveyor and
        # above for the labels: rounded corners need the shape of a unit in inches
        top = top or 1
        self._floor = -0.05 * top
        self.ax.set_ylim(self._floor, top * (1 + 0.3 / self._plot_height))
        self._inches_per_unit = self._plot_height / (top * (1 + 0.3 / self._plot_height) - self._floor)
        self._aspect = self._column_width / self._inches_per_unit

    def _add_label(self, x, y, text, offset, va, font_size, **kwargs):

        # labels shrink to the width of a column and are left out if unreadable
        font_size = min(font_size, 72 * self._column_width / (0.62 * len(text)))
        if font_size < MIN_FONT_SIZE:
            return None
        return self.ax.annotate(
            text, (x, y), xytext=(0, offset), textcoords='offset points',
            ha='center', va=va, fontsize=font_size, zorder=6, color=kwargs.pop('color', TEXT_COLOR),
            annotation_clip=False, **kwargs)

    def _rounded(self, x, y, width, height, radius, **kwargs):

        # a rectangle with corners rounded by the given number of inches
        return patches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f'round,pad=0,rounding_size={radius / self._column_width}',
            mutation_aspect=self._aspect, **kwargs)

    def _draw_tote(self, index, capacity):

        # the inside of the tote, its walls and floor, handles on its sides and a rim on
        # top, which is returned
        (left, right) = (index + WALL_LEFT, index + WALL_RIGHT)
        self.ax.add_patch(self._rounded(
            left, self._floor * 0.6, right - left, capacity - self._floor * 0.6, 0.06,
            facecolor=TOTE_COLOR, edgecolor=WALL_COLOR, linewidth=2.5, zorder=1))
        handle = 0.12 * capacity
        for (x, sign) in ((left, -1), (right, 1)):
            self.ax.add_patch(self._rounded(
                x + (sign - 1) * 0.03, capacity - 1.6 * handle, 0.06, handle, 0.02,
                facecolor=WALL_COLOR, edgecolor='none', zorder=1))
        rim = self._rounded(left - 0.04, capacity - 0.004 * capacity, right - left + 0.08,
                            0.025 * capacity, 0.02, facecolor=WALL_COLOR, edgecolor='none',
                            zorder=5)
        return self.ax.add_patch(rim)

    def _draw_conveyor(self):

        # a belt with rollers under the next box
        (left, right) = (self._next_x - 0.05, self._next_x + 0.85)
        self.ax.add_patch(self._rounded(
            left, self._floor * 0.9, right - left, -self._floor * 0.8, 0.04,
            facecolor=BELT_COLOR, edgecolor='none', zorder=1))
        rollers = np.linspace(left + 0.1, right - 0.1, 4)
        self.ax.plot(rollers, [self._floor * 0.5] * len(rollers), linestyle='none',
                     marker='o', markersize=max(3.0, 72 * 0.6 * -self._floor * self._inches_per_unit),
                     color=ROLLER_COLOR, markeredgecolor='white', markeredgewidth=0.6, zorder=2)

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

        # boxes placed together between two renders have a size of no single item
        if size not in self._size_rank:
            return OTHER_COLOR
        return BOX_COLORS[self._size_rank[size] % len(BOX_COLORS)]

    def _draw_item(self, x, bottom, size, color, newest=False, label=True):

        # a box with a gap around it, a strip of tape across its top and its size on it
        gap = BOX_GAP / self._inches_per_unit
        (width, height) = (0.8 - 2 * 0.06, size - 2 * gap)
        (left, low) = (x + 0.06, bottom + gap)
        if height <= 0:
            return []
        artists = [self.ax.add_patch(self._rounded(
            left, low, width, height, min(0.04, 0.4 * height * self._inches_per_unit),
            facecolor=color, edgecolor=NEWEST_COLOR if newest else 'none',
            linewidth=2.5 if newest else 0, zorder=2))]
        tape = 0.035 / self._inches_per_unit
        if height * self._inches_per_unit >= 0.12:
            artists.append(self.ax.add_patch(patches.Rectangle(
                (left + 0.38 * width, low + height - 2.2 * tape), 0.24 * width, 2.2 * tape,
                facecolor=TAPE_COLOR, edgecolor='none', alpha=0.55, zorder=3)))

        # only label boxes tall enough to hold the text
        font_size = min(9, 72 * width * self._column_width / (0.62 * len(str(size))))
        if label and height * self._inches_per_unit >= 0.18 and font_size >= MIN_FONT_SIZE:
            artists.append(self.ax.text(
                left + width / 2, low + height / 2 - tape, str(size), ha='center', va='center',
                color='white', fontsize=font_size, fontweight='bold', zorder=4))
        return artists

    def _draw_bin(self, index, bin_name, total, next_size):
        artists = []
        bottom = 0
        for (size, initial) in self._items[bin_name]:
            newest = bin_name in self._newest and not initial and bottom + size == total
            color = INITIAL_COLOR if initial else self._item_color(size)
            artists.extend(self._draw_item(index + 0.1, bottom, size, color, newest))
            bottom += size

        no_room = total + next_size > self._capacity[bin_name]
        self._rims[bin_name].set_facecolor(NO_ROOM_COLOR if no_room else WALL_COLOR)
        label = self._add_label(
            index + 0.5, max(total, self._capacity[bin_name]),
            f'{total} / {self._capacity[bin_name]}', 5, 'bottom', 9,
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
        totals = {b: int(state[RDDLPlanningModel.ground_var('total-size', [b])])
                  for b in self._bins}
        next_size = int(state['current-size'])
        self._update_items(totals, next_size)

        # bins can be overfilled if invalid actions are allowed: the scale has to grow to
        # show them
        self._set_scale(max(self._base_scale, next_size, *totals.values()))

        artists = []
        for (i, b) in enumerate(self._bins):
            artists.extend(self._draw_bin(i, b, totals[b], next_size))

        if next_size > 0:
            artists.extend(self._draw_item(
                self._next_x, 0, next_size, self._item_color(next_size)))
        size_label = self._add_label(
            self._next_x + 0.4, max(next_size, 0), f'size {next_size}', 5, 'bottom', 9)
        if size_label is not None:
            artists.append(size_label)

        used = sum(total > 0 for total in totals.values())
        self.ax.set_title(f'{used} / {len(self._bins)} bins used',
                          fontsize=13, fontweight='bold', color=TEXT_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
