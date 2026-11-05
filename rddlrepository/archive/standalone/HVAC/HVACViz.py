import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

COMFORT_COLOR = '#66bb6a'
COLD_COLOR = '#42a5f5'
HOT_COLOR = '#ef5350'
OTHER_COLOR = '#9e9e9e'
BACKGROUND_COLOR = '#eceff1'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class HVACVisualizer(BaseViz):
    '''Draws each space as a tile that fills up with its temperature, with lines
    joining adjacent spaces and dashed lines marking the comfort range of each room.

    The domain has no coordinates, so the spaces are placed evenly in the order they
    are declared. All tiles use one temperature scale, from the coldest to the hottest
    temperature the domain can bring a space towards. The drawing only depends on the
    current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(7.0, 7.6),
                 max_tile_size=1.1,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._spaces = model.type_to_objects['space']
        self._rooms = [s for s in self._spaces if self._get_nonfluent('IS_ROOM', s)]
        self._low = {s: self._get_nonfluent('TEMP_LOW', s) for s in self._spaces}
        self._up = {s: self._get_nonfluent('TEMP_UP', s) for s in self._spaces}
        self._outside = {s for s in self._spaces if self._get_nonfluent('ADJ_OUTSIDE', s)}
        self._hall = {s for s in self._spaces if self._get_nonfluent('ADJ_HALL', s)}
        self._scale = self._get_scale(model)

        # the spaces go clockwise from the top
        num_spaces = len(self._spaces)
        fig_width, fig_height = figure_size
        radius_inches = 2.3 if num_spaces > 1 else 0.0
        size = fig_width - 0.4
        inches_per_unit = size / 6.6
        radius = radius_inches / inches_per_unit
        self._positions = {
            s: (radius * math.sin(2 * math.pi * i / num_spaces),
                radius * math.cos(2 * math.pi * i / num_spaces))
            for (i, s) in enumerate(self._spaces)}

        # large numbers of spaces use smaller tiles and text to fit around the circle
        spacing = 2 * radius_inches * math.sin(math.pi / num_spaces) if num_spaces > 1 else 2.0
        tile_inches = min(max_tile_size, 0.7 * spacing)
        self._tile = tile_inches / inches_per_unit
        self._tile_inches = tile_inches

        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_axes(
            [0.2 / fig_width, 0.75 / fig_height, size / fig_width, size / fig_height])
        self.ax.set_xlim(-3.3, 3.3)
        self.ax.set_ylim(-3.3, 3.3)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        self._draw_adjacency()
        for s in self._spaces:
            self._draw_space(s)

        self.ax.legend(handles=[
            patches.Patch(facecolor=COMFORT_COLOR, label='In range'),
            patches.Patch(facecolor=COLD_COLOR, label='Too cold'),
            patches.Patch(facecolor=HOT_COLOR, label='Too hot'),
            patches.Patch(facecolor=OTHER_COLOR, label='Not a room'),
            Line2D([], [], color='black', linestyle='--', linewidth=1, label='Comfort range')
        ], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=5, frameon=False, fontsize=9,
           columnspacing=1.2, handlelength=1.5)
        self.fig.text(0.5, 0.1 / fig_height,
                      f'Tiles fill from {self._scale[0]:g} to {self._scale[1]:g} degrees',
                      ha='center', va='bottom', fontsize=8)

    def _get_nonfluent(self, var, space):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, [space])]

    def _get_scale(self, model):

        # temperature moves towards the outside, the hallway, the air and other spaces,
        # so these values and the initial temperatures bound where it normally stays
        init_state = model.ground_vars_with_values(model.state_fluents)
        temps = [self._nonfluents['TEMP_AIR']] if self._rooms else []
        for s in self._spaces:
            temps.append(init_state[RDDLPlanningModel.ground_var('temp', [s])])
            if s in self._outside:
                temps.append(self._get_nonfluent('TEMP_OUTSIDE_MEAN', s))
            if s in self._hall:
                temps.append(self._get_nonfluent('TEMP_HALL_MEAN', s))
        for s in self._rooms:
            temps.extend((self._low[s], self._up[s]))
        low, high = min(temps), max(temps)
        return (low, high) if high > low else (low - 1.0, low + 1.0)

    def _fraction(self, temp):
        low, high = self._scale
        return min(max((temp - low) / (high - low), 0.0), 1.0)

    def _font_size(self, text, font_size):

        # text shrinks to the width of a tile, and is left out if unreadable
        size = min(font_size, 72 * self._tile_inches / (0.62 * len(text)))
        return size if size >= MIN_FONT_SIZE else None

    def _add_text(self, x, y, text, va, font_size, **kwargs):
        size = self._font_size(text, font_size)
        if size is None:
            return None
        return self.ax.text(x, y, text, ha='center', va=va, fontsize=size, zorder=6, **kwargs)

    def _draw_adjacency(self):

        # heat flows between two spaces if either one is declared adjacent to the other
        for (i, s1) in enumerate(self._spaces):
            for s2 in self._spaces[i + 1:]:
                if self._nonfluents[RDDLPlanningModel.ground_var('ADJ', [s1, s2])] or \
                        self._nonfluents[RDDLPlanningModel.ground_var('ADJ', [s2, s1])]:
                    (x1, y1), (x2, y2) = self._positions[s1], self._positions[s2]
                    self.ax.plot([x1, x2], [y1, y2], color='#78909c', linewidth=1.5, zorder=1)

    def _draw_space(self, space):

        # the parts of a space that never change
        x, y = self._positions[space]
        half = self._tile / 2
        self.ax.add_patch(patches.Rectangle(
            (x - half, y - half), self._tile, self._tile,
            facecolor=BACKGROUND_COLOR, edgecolor='black', linewidth=1, zorder=2))
        self._add_text(x, y + half * 1.1, space, 'bottom', 10, fontweight='bold')

        if space in self._rooms:
            for temp in (self._low[space], self._up[space]):
                level = y - half + self._tile * self._fraction(temp)
                self.ax.plot([x - half, x + half], [level, level], color='black',
                             linestyle='--', linewidth=1, zorder=4)

        neighbors = [name for (name, spaces) in (('outside', self._outside), ('hall', self._hall))
                     if space in spaces]
        if neighbors:
            self._add_text(x, y - half * 1.1, ' + '.join(neighbors), 'top', 8, color='#546e7a')

    def _color(self, space, temp):
        if space not in self._rooms:
            return OTHER_COLOR
        if temp < self._low[space]:
            return COLD_COLOR
        return HOT_COLOR if temp > self._up[space] else COMFORT_COLOR

    def _draw_temperature(self, space, temp):
        x, y = self._positions[space]
        half = self._tile / 2
        artists = [self.ax.add_patch(patches.Rectangle(
            (x - half, y - half), self._tile, self._tile * self._fraction(temp),
            facecolor=self._color(space, temp), edgecolor='none', zorder=3))]
        label = self._add_text(
            x, y, f'{temp:.1f}', 'center', 11, fontweight='bold',
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
        temps = {s: float(state[RDDLPlanningModel.ground_var('temp', [s])])
                 for s in self._spaces}

        artists = []
        for s in self._spaces:
            artists.extend(self._draw_temperature(s, temps[s]))

        # same rule as the penalty in the reward
        comfortable = sum(self._low[s] <= temps[s] <= self._up[s] for s in self._rooms)
        all_comfortable = comfortable == len(self._rooms)
        self.ax.set_title(f'{comfortable} / {len(self._rooms)} rooms in range',
                          fontsize=13, fontweight='bold',
                          color='#2e7d32' if all_comfortable else '#c62828')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
