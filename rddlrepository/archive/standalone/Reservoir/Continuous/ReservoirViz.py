import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

TANK_COLOR = '#eceff1'
TANK_EDGE_COLOR = '#37474f'
RANGE_COLOR = '#c8e6c9'
RANGE_EDGE_COLOR = '#2e7d32'
WATER_COLOR = '#42a5f5'
LOW_COLOR = '#ff8f00'
HIGH_COLOR = '#c62828'
FLOW_COLOR = '#546e7a'
BACK_FLOW_COLOR = '#ad1457'
SEA_COLOR = '#bbdefb'
SEA_EDGE_COLOR = '#1565c0'
OK_COLOR = '#2e7d32'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5

# the width of the sea along the right edge of the figure, in inches
SEA_WIDTH = 0.55


class ReservoirVisualizer(BaseViz):
    '''Draws the reservoirs in a row, sorted so that water flows from left to right along
    the arcs above them, ending at the sea along the right edge. Each tank is scaled to its own
    capacity, with its water and the range of levels that are not penalized. Flows cannot
    all go left to right if they form a cycle, and those that do not are drawn below the
    row. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 column_width=1.1,
                 max_width=14.0,
                 min_width=6.8,
                 tank_height=1.25,
                 arc_height=2.4,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        tanks = model.type_to_objects['reservoir']
        self._top = self._get_values(nonfluents, 'TOP_RES', tanks)
        self._high = self._get_values(nonfluents, 'MAX_LEVEL', tanks)
        self._low = self._get_values(nonfluents, 'MIN_LEVEL', tanks)
        self._rain = self._get_values(nonfluents, 'RAIN_VAR', tanks)
        self._to_sea = [r for r in tanks if nonfluents[
            RDDLPlanningModel.ground_var('CONNECTED_TO_SEA', [r])]]
        flows = [(r1, r2) for r1 in tanks for r2 in tanks if r1 != r2 and nonfluents[
            RDDLPlanningModel.ground_var('RES_CONNECT', [r1, r2])]]

        self._tanks = self._get_flow_order(tanks, flows)
        position = {r: i for (i, r) in enumerate(self._tanks)}
        self._forward = [(a, b) for (a, b) in flows if position[a] < position[b]]
        self._backward = [(a, b) for (a, b) in flows if position[a] > position[b]]

        # one column for each reservoir, smaller to fit on screen, and the sea along the
        # right edge of the figure
        num_columns = len(self._tanks)
        margin = 0.2
        sea_width = SEA_WIDTH if self._to_sea else margin
        width = max(min(column_width * num_columns + margin + sea_width, max_width), min_width)
        self._column = min((width - margin - sea_width) / num_columns, column_width)
        self._tank_width = 0.82 * self._column

        # from the bottom: the legend, flows that go backwards, names, tanks and the arcs
        self._back_height = 0.5 if self._backward else 0.0
        self._tank_y = 0.45 + self._back_height + 0.5
        self._tank_height = tank_height
        self._arc_y = self._tank_y + tank_height
        self._arc_height = arc_height if self._forward or self._to_sea else 0.0
        height = self._arc_y + self._arc_height + 0.6

        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1 - 0.5 / height])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height - 0.5)
        self.ax.axis('off')

        self._sea_x = width - sea_width
        start = (self._sea_x - (num_columns - 1) * self._column) / 2
        self._x = {r: start + i * self._column for (i, r) in enumerate(self._tanks)}
        self._num_columns = num_columns

        # the flows, tanks and sea never change, so they are only drawn once
        self._draw_flows()
        for r in self._tanks:
            self._draw_tank(r)
        if self._to_sea:
            self._draw_sea(width, height)

        # the legend and title are centered on the land, as the sea would cover them
        self._title_x = self._sea_x / 2 / width
        handles = [
            patches.Patch(facecolor=WATER_COLOR, label='In range'),
            patches.Patch(facecolor=LOW_COLOR, label='Too low'),
            patches.Patch(facecolor=HIGH_COLOR, label='Too high'),
            patches.Patch(facecolor=RANGE_COLOR, edgecolor=RANGE_EDGE_COLOR, label='Desired range'),
            Line2D([], [], color=FLOW_COLOR, label='Flow, left to right')
        ]
        if self._backward:
            handles.append(Line2D([], [], color=BACK_FLOW_COLOR, label='Right to left'))
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(self._title_x, 0),
                        ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                        handlelength=1.4)

    def _get_values(self, nonfluents, var, tanks):
        return {r: nonfluents[RDDLPlanningModel.ground_var(var, [r])] for r in tanks}

    def _get_flow_order(self, tanks, flows):

        # repeatedly take the first reservoir with nothing left flowing into it, which also
        # keeps the declared order as far as possible: with a cycle of flows no reservoir
        # may qualify, so take the one with the fewest left instead
        remaining = list(tanks)
        order = []
        while remaining:
            inflows = {r: sum(b == r and a in remaining for (a, b) in flows) for r in remaining}
            chosen = min(remaining, key=lambda r: inflows[r])
            order.append(chosen)
            remaining.remove(chosen)
        return order

    def _add_text(self, x, y, text, va, font_size, width, **kwargs):

        # text shrinks to the given width, and is left out if unreadable
        size = min(font_size, 72 * width / (0.62 * len(text)))
        if size < MIN_FONT_SIZE:
            return None
        return self.ax.text(x, y, text, ha='center', va=va, fontsize=size, **kwargs)

    def _draw_arc(self, x1, x2, base, height, color, width, alpha):
        angles = np.linspace(0, np.pi, 60)
        self.ax.plot((x1 + x2) / 2 + (x2 - x1) / 2 * np.cos(angles), base + height * np.sin(angles),
                     color=color, linewidth=width, alpha=alpha, zorder=1)

    def _draw_flows(self):

        # lines get thinner and fainter as there are more of them, to stay readable
        count = max(len(self._forward) + len(self._backward) + len(self._to_sea), 1)
        width = max(0.6, min(1.5, 40 / count))
        alpha = max(0.3, min(0.8, 30 / count))

        # the height of an arc grows with its length, so that arcs nest instead of overlapping
        columns = max(self._num_columns - 1, 1)
        for (a, b) in self._forward:
            span = abs(self._x[b] - self._x[a]) / (columns * self._column)
            self._draw_arc(self._x[a], self._x[b], self._arc_y,
                           (0.12 + 0.88 * span) * self._arc_height, FLOW_COLOR, width, alpha)
        for r in self._to_sea:
            span = (self._sea_x - self._x[r]) / (self._sea_x - self._x[self._tanks[0]])
            self._draw_arc(self._x[r], self._sea_x, self._arc_y,
                           (0.12 + 0.88 * span) * self._arc_height, SEA_EDGE_COLOR, 1.5, 0.8)
        for (a, b) in self._backward:
            span = abs(self._x[b] - self._x[a]) / (columns * self._column)
            self._draw_arc(self._x[a], self._x[b], self._tank_y - 0.42,
                           -(0.2 + 0.8 * span) * self._back_height, BACK_FLOW_COLOR, width, alpha)

    def _level_y(self, reservoir, level):
        fraction = level / self._top[reservoir] if self._top[reservoir] > 0 else 0.0
        return self._tank_y + min(max(fraction, 0.0), 1.0) * self._tank_height

    def _draw_tank(self, reservoir):
        x = self._x[reservoir]
        left = x - self._tank_width / 2
        self.ax.add_patch(patches.Rectangle(
            (left, self._tank_y), self._tank_width, self._tank_height, facecolor=TANK_COLOR,
            edgecolor='none', zorder=2))

        # the range of levels that are not penalized
        low, high = self._level_y(reservoir, self._low[reservoir]), self._level_y(reservoir, self._high[reservoir])
        self.ax.add_patch(patches.Rectangle(
            (left, low), self._tank_width, high - low, facecolor=RANGE_COLOR, edgecolor='none', zorder=2))
        for y in (low, high):
            self.ax.plot([left, left + self._tank_width], [y, y], color=RANGE_EDGE_COLOR,
                         linestyle='--', linewidth=1, zorder=5)
        self.ax.add_patch(patches.Rectangle(
            (left, self._tank_y), self._tank_width, self._tank_height, facecolor='none',
            edgecolor=TANK_EDGE_COLOR, linewidth=1.2, zorder=6))
        self._add_text(x, self._tank_y - 0.05, reservoir, 'top', 10, self._column, zorder=7)
        self._add_text(x, self._tank_y - 0.24, f'rain {self._rain[reservoir]:g}', 'top', 8,
                       self._column, color='#616161', zorder=7)

    def _draw_sea(self, right, top):

        # the sea has no edge of its own except the shore, facing the reservoirs
        self.ax.add_patch(patches.Rectangle(
            (self._sea_x, 0), right - self._sea_x, top, facecolor=SEA_COLOR, edgecolor='none',
            zorder=0, clip_on=False))
        self.ax.plot([self._sea_x, self._sea_x], [0, top], color=SEA_EDGE_COLOR, linewidth=1.5,
                     zorder=1, clip_on=False)
        self.ax.text((self._sea_x + right) / 2, self._tank_y + self._tank_height / 2, 'sea', rotation=90,
                     ha='center', va='center', fontsize=10, color=SEA_EDGE_COLOR, fontweight='bold',
                     zorder=7)

    def _draw_water(self, reservoir, level):
        x = self._x[reservoir]
        if level < self._low[reservoir]:
            color = LOW_COLOR
        else:
            color = HIGH_COLOR if level > self._high[reservoir] else WATER_COLOR
        artists = [self.ax.add_patch(patches.Rectangle(
            (x - self._tank_width / 2, self._tank_y), self._tank_width,
            self._level_y(reservoir, level) - self._tank_y,
            facecolor=color, edgecolor='none', alpha=0.85, zorder=3))]

        # the fill shows how full a tank is against its own capacity, and the number the level
        label = self._add_text(
            x, self._tank_y + self._tank_height / 2, f'{level:.3g}', 'center', 10, self._tank_width,
            fontweight='bold', zorder=7,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.0, 'alpha': 0.8})
        if label is not None:
            artists.append(label)
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        levels = {r: float(state[RDDLPlanningModel.ground_var('rlevel', [r])]) for r in self._tanks}

        artists = []
        for r in self._tanks:
            artists.extend(self._draw_water(r, levels[r]))

        # same rule as the reward
        in_range = sum(self._low[r] <= levels[r] <= self._high[r] for r in self._tanks)
        all_in_range = in_range == len(self._tanks)
        self.ax.set_title(f'{in_range} / {len(self._tanks)} reservoirs in range', fontsize=13,
                          fontweight='bold', x=self._title_x,
                          color=OK_COLOR if all_in_range else HIGH_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
