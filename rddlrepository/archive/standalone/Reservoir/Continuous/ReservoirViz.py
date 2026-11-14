import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

GROUND_COLOR = '#d7ccc8'
GROUND_EDGE_COLOR = '#8d6e63'
VALLEY_COLOR = '#f5f5f5'
DAM_COLOR = '#90a4ae'
DAM_EDGE_COLOR = '#455a64'
RANGE_EDGE_COLOR = '#2e7d32'
WATER_COLOR = '#42a5f5'
LOW_COLOR = '#ff8f00'
HIGH_COLOR = '#c62828'
RIVER_COLOR = '#1e88e5'
SAND_COLOR = '#e8d5a3'
SAND_EDGE_COLOR = '#c8a96a'
SEA_COLOR = '#bbdefb'
SEA_EDGE_COLOR = '#1565c0'
OK_COLOR = '#2e7d32'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5

# the width of the sea along the right edge of the figure, in inches
SEA_WIDTH = 0.55

# the shape of a reservoir, as fractions of its width: the bottom of the valley starts this
# far from its left bank, and the dam stands on the right, this thick at the bottom and top
BANK = 0.3
DAM_BOTTOM, DAM_TOP = 0.25, 0.08

# rivers run in lanes this far apart at most, in inches, down to this depth below the
# reservoirs, and are this wide with their sandbanks, as a fraction of their lane or of the
# space between them where they meet a reservoir, with water in the middle this fraction
LANE = 0.18
MAX_RIVER_DEPTH = 3.6
RIVER_WIDTH = 0.75
WATER_WIDTH = 0.5


class ReservoirVisualizer(BaseViz):
    '''Draws the reservoirs as valleys held back by dams, in a row sorted so that water
    flows from left to right, ending at the sea along the right edge. Rivers run under the
    reservoirs: each enters a reservoir from below on its upstream side and leaves on its
    downstream side, at a place of its own spread across the reservoir, and runs across at
    a depth of its own, so that no two rivers run along each other. Each valley is scaled
    to its own capacity, with its water and dashed lines around the range of levels that
    are not penalized. Flows cannot all go left to right if they form a cycle, and those
    that do not are drawn as tunnels. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 column_width=1.1,
                 max_width=14.0,
                 min_width=6.8,
                 tank_height=1.25,
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
        self._tank_width = 0.9 * self._column

        self._sea_x = width - sea_width
        start = (self._sea_x - (num_columns - 1) * self._column) / 2
        self._x = {r: start + i * self._column for (i, r) in enumerate(self._tanks)}
        self._num_columns = num_columns

        # each reservoir spreads the rivers it meets across its flat bottom, from the foot of
        # its bank to the end of its dam: those coming in on its upstream side, nearest
        # sources first, then those going out, farthest first, so that rivers cross as
        # little as possible; the sea is the farthest of all
        ends = [(a, b) for (a, b) in self._forward + self._backward]
        ends += [(r, None) for r in self._to_sea]
        distance = {(a, b): num_columns if b is None else abs(position[b] - position[a])
                    for (a, b) in ends}
        self._ports = {}
        bottom = (1 - BANK) * self._tank_width
        spacing = bottom
        for r in self._tanks:
            inflows = sorted((e for e in ends if e[1] == r), key=lambda e: distance[e])
            outflows = sorted((e for e in ends if e[0] == r), key=lambda e: -distance[e])
            places = [(e, 1) for e in inflows] + [(e, 0) for e in outflows]
            for (k, (e, end)) in enumerate(places):
                self._ports[(e, end)] = (self._x[r] + self._tank_width / 2 - bottom
                                         + (k + 0.5) * bottom / len(places))
            if places:
                spacing = min(spacing, bottom / len(places))

        # each river runs across at a depth of its own
        self._rivers = [(self._ports[(e, 0)], self._sea_x if e[1] is None else self._ports[(e, 1)],
                         e) for e in ends]
        (self._lanes, num_lanes) = self._get_lanes([(x1, x2) for (x1, x2, _) in self._rivers])
        self._lane = min(LANE, MAX_RIVER_DEPTH / (num_lanes + 0.5)) if num_lanes else 0.0
        self._river_width = min(RIVER_WIDTH * self._lane, 0.8 * spacing)

        # from the bottom: the legend, names, the ground with the rivers in it, and the
        # valleys
        self._ground_y = 0.95
        self._tank_y = self._ground_y + 0.1 + (num_lanes + 0.5) * self._lane
        self._tank_height = tank_height
        self._arc_y = self._tank_y + tank_height
        height = self._arc_y + 0.06 + 0.75

        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1 - 0.5 / height])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height - 0.5)
        self.ax.axis('off')

        # the land, rivers and sea never change, so they are only drawn once
        self.ax.add_patch(patches.Rectangle(
            (0, self._ground_y), self._sea_x, self._arc_y - self._ground_y,
            facecolor=GROUND_COLOR, edgecolor='none', zorder=1))
        self.ax.plot([0, self._sea_x], [self._arc_y] * 2, color=GROUND_EDGE_COLOR,
                     linewidth=1.5, zorder=1)
        for r in self._tanks:
            self._draw_valley(r)
        self._draw_rivers()
        if self._to_sea:
            self._draw_sea(width, height)

        # the legend and title are centered on the land, as the sea would cover them
        self._title_x = self._sea_x / 2 / width
        handles = [
            patches.Patch(facecolor=WATER_COLOR, label='In range'),
            patches.Patch(facecolor=LOW_COLOR, label='Too low'),
            patches.Patch(facecolor=HIGH_COLOR, label='Too high'),
            Line2D([], [], color=RANGE_EDGE_COLOR, linestyle='--', label='Desired range'),
            patches.Patch(facecolor=DAM_COLOR, edgecolor=DAM_EDGE_COLOR, label='Dam'),
            (Line2D([], [], color=SAND_COLOR, linewidth=7),
             Line2D([], [], color=RIVER_COLOR, linewidth=3))
        ]
        labels = ['In range', 'Too low', 'Too high', 'Desired range', 'Dam', 'River, downstream →']
        if self._backward:
            handles.append((Line2D([], [], color=SAND_COLOR, linewidth=7),
                            Line2D([], [], color=RIVER_COLOR, linewidth=3, linestyle=':')))
            labels.append('Upstream ←')
        self.fig.legend(handles=handles, labels=labels, loc='lower center',
                        bbox_to_anchor=(self._title_x, 0),
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

    # the valley of a reservoir: its left bank slopes down to the bottom, and its right side
    # is the face of the dam
    def _bank_x(self, reservoir, y):
        left = self._x[reservoir] - self._tank_width / 2
        depth = 1 - (y - self._tank_y) / self._tank_height
        return left + BANK * self._tank_width * depth

    def _dam_x(self, reservoir):
        return self._x[reservoir] + self._tank_width / 2 - DAM_BOTTOM * self._tank_width

    def _floor_y(self, reservoir, x):

        # the bottom of a reservoir: its left bank, its valley floor, or its dam
        left = self._x[reservoir] - self._tank_width / 2
        slope = (x - left) / (BANK * self._tank_width)
        return self._tank_y + max(1 - slope, 0.0) * self._tank_height

    @staticmethod
    def _get_lanes(spans):

        # the shortest rivers take the lanes nearest the reservoirs, so that longer ones run
        # beneath them, and a river takes the nearest lane where it runs along no other
        order = sorted(range(len(spans)), key=lambda i: (abs(spans[i][1] - spans[i][0]), spans[i]))
        lanes, taken = [0] * len(spans), []
        for i in order:
            (left, right) = sorted(spans[i])
            lane = 0
            while lane < len(taken) and any(
                    left < r + 0.05 and l < right + 0.05 for (l, r) in taken[lane]):
                lane += 1
            if lane == len(taken):
                taken.append([])
            taken[lane].append((left, right))
            lanes[i] = lane
        return lanes, len(taken)

    def _draw_rivers(self):

        # a river is a channel of sand with water down its middle, with square corners: it
        # leaves the bottom of one reservoir, runs across at its depth and rises into the
        # bottom of the next, or runs on into the sea; the deepest are drawn first, so
        # those above them cross over them
        width = 72 * self._river_width
        order = sorted(range(len(self._rivers)), key=lambda i: -self._lanes[i])
        for (rank, i) in enumerate(order):
            (x1, x2, (a, b)) = self._rivers[i]
            backward = (a, b) in self._backward
            y = self._tank_y - (self._lanes[i] + 1) * self._lane
            (xs, ys) = ([x1, x1, x2], [self._floor_y(a, x1), y, y])
            if b is not None:
                (xs, ys) = (xs + [x2], ys + [self._floor_y(b, x2)])
            for (k, color, w, style) in ((0, SAND_EDGE_COLOR, width + 1.0, '-'),
                                         (1, SAND_COLOR, width, '-'),
                                         (2, RIVER_COLOR, WATER_WIDTH * width, ':' if backward else '-')):
                self.ax.plot(xs, ys, color=color, linewidth=w, linestyle=style,
                             solid_joinstyle='miter', solid_capstyle='butt',
                             zorder=2 + (3 * rank + k) / (3 * len(order) + 3))

    def _level_y(self, reservoir, level):
        fraction = level / self._top[reservoir] if self._top[reservoir] > 0 else 0.0
        return self._tank_y + min(max(fraction, 0.0), 1.0) * self._tank_height

    def _draw_valley(self, reservoir):
        x = self._x[reservoir]
        (bottom, top) = (self._tank_y, self._arc_y)
        dam = self._dam_x(reservoir)
        self.ax.add_patch(patches.Polygon(
            [(self._bank_x(reservoir, top), top), (self._bank_x(reservoir, bottom), bottom),
             (dam, bottom), (dam, top)], facecolor=VALLEY_COLOR, edgecolor=GROUND_EDGE_COLOR,
            linewidth=1.0, zorder=1.5))

        # the dam is thicker at the bottom, and stands a little above the valley
        crest = top + 0.06
        self.ax.add_patch(patches.Polygon(
            [(dam, bottom), (dam + DAM_BOTTOM * self._tank_width, bottom),
             (dam + DAM_TOP * self._tank_width, crest), (dam, crest)],
            facecolor=DAM_COLOR, edgecolor=DAM_EDGE_COLOR, linewidth=1.0, zorder=5))

        # the range of levels that are not penalized
        for level in (self._low[reservoir], self._high[reservoir]):
            y = self._level_y(reservoir, level)
            self.ax.plot([self._bank_x(reservoir, y), dam], [y, y], color=RANGE_EDGE_COLOR,
                         linestyle='--', linewidth=1, zorder=4)
        self._add_text(x, self._ground_y - 0.05, reservoir, 'top', 10, self._column, zorder=7)
        self._add_text(x, self._ground_y - 0.24, f'rain {self._rain[reservoir]:g}', 'top', 8,
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

        # the water fills the valley up to its level, against its own capacity
        (bottom, surface) = (self._tank_y, self._level_y(reservoir, level))
        dam = self._dam_x(reservoir)
        artists = [self.ax.add_patch(patches.Polygon(
            [(self._bank_x(reservoir, surface), surface), (self._bank_x(reservoir, bottom), bottom),
             (dam, bottom), (dam, surface)], facecolor=color, edgecolor='none', alpha=0.85,
            zorder=3))]
        label = self._add_text(
            x - DAM_BOTTOM * self._tank_width / 2, self._tank_y + self._tank_height / 2,
            f'{level:.3g}', 'center', 10, (1 - DAM_BOTTOM) * self._tank_width,
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
