import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

COMFORT_COLOR = '#66bb6a'
COLD_COLOR = '#42a5f5'
HOT_COLOR = '#c62828'
RANGE_COLOR = '#c8e6c9'
RANGE_EDGE_COLOR = '#2e7d32'
EDGE_COLOR = '#37474f'
OTHER_COLOR = '#b0bec5'
SOURCE_COLOR = '#ef6c00'
BACKGROUND_COLOR = '#eceff1'
LINK_COLOR = '#546e7a'
OK_COLOR = '#2e7d32'
BAD_COLOR = '#c62828'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class HVACVisualizer(BaseViz):
    '''Draws the spaces in a row in the order they are declared, with what they exchange
    heat with beyond each other, the outside and the hallway, in a row below them. Each
    tile fills up with its temperature, and dashed lines mark the comfort range of each
    room. Arcs above the spaces join adjacent spaces, and lines connect them to the outside
    and hallway, which show the mean temperature of the random values each space sees.

    All tiles use one temperature scale, from the coldest to the hottest temperature the
    domain can bring a space towards. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 column_width=1.1,
                 max_width=14.0,
                 min_width=6.6,
                 max_space_size=0.95,
                 max_source_size=0.75,
                 arc_height=1.6,
                 gap_height=1.1,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._spaces = model.type_to_objects['space']
        self._rooms = [s for s in self._spaces if self._get_nonfluent('IS_ROOM', s)]
        self._low = {s: self._get_nonfluent('TEMP_LOW', s) for s in self._spaces}
        self._up = {s: self._get_nonfluent('TEMP_UP', s) for s in self._spaces}
        self._links = [
            (s1, s2) for (i, s1) in enumerate(self._spaces) for s2 in self._spaces[i + 1:]
            if self._nonfluents[RDDLPlanningModel.ground_var('ADJ', [s1, s2])]
            or self._nonfluents[RDDLPlanningModel.ground_var('ADJ', [s2, s1])]]
        self._sources = {}
        for (name, flag, mean) in (('outside', 'ADJ_OUTSIDE', 'TEMP_OUTSIDE_MEAN'),
                                   ('hall', 'ADJ_HALL', 'TEMP_HALL_MEAN')):
            spaces = [s for s in self._spaces if self._get_nonfluent(flag, s)]
            if spaces:
                self._sources[name] = {s: self._get_nonfluent(mean, s) for s in spaces}
        self._scale = self._get_scale(model)

        # both rows use the same spacing, and wide rows use smaller tiles to fit on screen
        num_columns = max(len(self._spaces), len(self._sources), 1)
        margin, left_over = 0.2, 0.75
        width = max(min(column_width * num_columns + 2 * margin, max_width), min_width)
        self._column = min((width - 2 * margin) / num_columns, column_width)
        self._space_size = min(max_space_size, 0.8 * self._column)
        self._source_size = min(max_source_size, 0.68 * self._column)

        # from the bottom: names and tiles of sources, a gap for the lines, names and tiles
        # of spaces, and the arcs above them: rows and arcs with nothing to show are left out
        name_height = 0.28
        self._source_y = left_over + name_height
        self._space_y = self._source_y + name_height
        if self._sources:
            self._space_y += self._source_size + gap_height
        self._arc_y = self._space_y + self._space_size
        self._arc_height = arc_height if self._links else 0.0
        height = self._arc_y + self._arc_height + 0.6

        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1 - 0.5 / height])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height - 0.5)
        self.ax.axis('off')

        center = width / 2
        self._x = {}
        for row in (self._spaces, list(self._sources)):
            for (i, name) in enumerate(row):
                self._x[name] = center + (i - (len(row) - 1) / 2) * self._column

        # the links and tiles never change, so they are only drawn once
        self._draw_links()
        for s in self._spaces:
            self._draw_space(s)
        for (name, means) in self._sources.items():
            self._draw_source(name, means)

        self.fig.legend(handles=[
            patches.Patch(facecolor=COMFORT_COLOR, label='In range'),
            patches.Patch(facecolor=COLD_COLOR, label='Too cold'),
            patches.Patch(facecolor=HOT_COLOR, label='Too hot'),
            patches.Patch(facecolor=OTHER_COLOR, label='Not a room'),
            patches.Patch(facecolor=SOURCE_COLOR, label='Outside or hall'),
            patches.Patch(facecolor=RANGE_COLOR, edgecolor=RANGE_EDGE_COLOR, label='Comfort range')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / height), ncol=6, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)
        self.fig.text(0.5, 0.1 / height,
                      f'Tiles fill from {self._scale[0]:.1f} to {self._scale[1]:.1f} degrees',
                      ha='center', va='bottom', fontsize=9, color='#37474f')

    def _get_nonfluent(self, var, space):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, [space])]

    def _get_scale(self, model):

        # temperature moves towards the outside, the hallway, the air and other spaces,
        # so these values and the initial temperatures bound where it normally stays
        init_state = model.ground_vars_with_values(model.state_fluents)
        temps = [self._nonfluents['TEMP_AIR']] if self._rooms else []
        temps.extend(init_state[RDDLPlanningModel.ground_var('temp', [s])] for s in self._spaces)
        for means in self._sources.values():
            temps.extend(means.values())
        for s in self._rooms:
            temps.extend((self._low[s], self._up[s]))
        low, high = min(temps), max(temps)
        return (low, high) if high > low else (low - 1.0, low + 1.0)

    def _fraction(self, temp):
        low, high = self._scale
        return min(max((temp - low) / (high - low), 0.0), 1.0)

    def _add_text(self, x, y, text, va, font_size, width, **kwargs):

        # text shrinks to the given width, and is left out if unreadable
        size = min(font_size, 72 * width / (0.62 * len(text)))
        if size < MIN_FONT_SIZE:
            return None
        return self.ax.text(x, y, text, ha='center', va=va, fontsize=size, zorder=6, **kwargs)

    def _draw_links(self):

        # lines get thinner and fainter as there are more of them, to stay readable
        count = max(len(self._links) + sum(map(len, self._sources.values())), 1)
        width = max(0.7, min(1.5, 40 / count))
        alpha = max(0.35, min(0.8, 30 / count))

        # heat flows between two spaces if either one is declared adjacent to the other: the
        # height of an arc grows with its length, so that arcs nest instead of overlapping
        num_spaces = max(len(self._spaces) - 1, 1)
        angles = np.linspace(0, np.pi, 60)
        position = {s: i for (i, s) in enumerate(self._spaces)}
        for (s1, s2) in self._links:
            x1, x2 = self._x[s1], self._x[s2]
            span = abs(position[s2] - position[s1]) / num_spaces
            height = (0.15 + 0.85 * span) * self._arc_height
            self.ax.plot((x1 + x2) / 2 + (x2 - x1) / 2 * np.cos(angles),
                         self._arc_y + height * np.sin(angles),
                         color=LINK_COLOR, linewidth=width, alpha=alpha, zorder=1)

        # the lines to the outside and hallway run straight across the gap between the rows
        for (name, means) in self._sources.items():
            for s in means:
                self.ax.plot([self._x[name], self._x[s]],
                             [self._source_y + self._source_size, self._space_y],
                             color=SOURCE_COLOR, linewidth=width, alpha=alpha, zorder=1)

    def _draw_tile(self, name, y, size):
        x = self._x[name]
        self.ax.add_patch(patches.Rectangle(
            (x - size / 2, y), size, size, facecolor=BACKGROUND_COLOR,
            edgecolor='none', zorder=2))
        self.ax.add_patch(patches.Rectangle(
            (x - size / 2, y), size, size, facecolor='none',
            edgecolor=EDGE_COLOR, linewidth=1.2, zorder=5))
        self._add_text(x, y - 0.05, name, 'top', 10, self._column,
                       bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.8, 'alpha': 0.85})

    def _draw_space(self, space):
        self._draw_tile(space, self._space_y, self._space_size)
        if space in self._rooms:
            x = self._x[space]
            size = self._space_size
            low = self._space_y + size * self._fraction(self._low[space])
            high = self._space_y + size * self._fraction(self._up[space])
            self.ax.add_patch(patches.Rectangle(
                (x - size / 2, low), size, high - low, facecolor=RANGE_COLOR, edgecolor='none', zorder=2))
            for level in (low, high):
                self.ax.plot([x - size / 2, x + size / 2], [level, level], color=RANGE_EDGE_COLOR,
                             linestyle='--', linewidth=1, zorder=4)

    def _draw_source(self, name, means):
        self._draw_tile(name, self._source_y, self._source_size)
        mean = sum(means.values()) / len(means)
        self._draw_temperature(name, mean, self._source_y, self._source_size, SOURCE_COLOR)

    def _color(self, space, temp):
        if space not in self._rooms:
            return OTHER_COLOR
        if temp < self._low[space]:
            return COLD_COLOR
        return HOT_COLOR if temp > self._up[space] else COMFORT_COLOR

    def _draw_temperature(self, name, temp, y, size, color):
        x = self._x[name]
        artists = [self.ax.add_patch(patches.Rectangle(
            (x - size / 2, y), size, size * self._fraction(temp),
            facecolor=color, edgecolor='none', alpha=0.85, zorder=3))]
        label = self._add_text(
            x, y + size / 2, f'{temp:.1f}', 'center', 10, size, fontweight='bold',
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
        temps = {s: float(state[RDDLPlanningModel.ground_var('temp', [s])]) for s in self._spaces}

        artists = []
        for s in self._spaces:
            artists.extend(self._draw_temperature(
                s, temps[s], self._space_y, self._space_size, self._color(s, temps[s])))

        # same rule as the penalty in the reward
        comfortable = sum(self._low[s] <= temps[s] <= self._up[s] for s in self._rooms)
        self.ax.set_title(f'{comfortable} / {len(self._rooms)} rooms in range',
                          fontsize=13, fontweight='bold',
                          color=OK_COLOR if comfortable == len(self._rooms) else BAD_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
