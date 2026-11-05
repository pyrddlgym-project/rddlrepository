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
EMPTY_COLOR = '#b0bec5'
HEATER_COLOR = '#ef6c00'
BACKGROUND_COLOR = '#eceff1'
ZONE_LINK_COLOR = '#546e7a'
OK_COLOR = '#2e7d32'
BAD_COLOR = '#c62828'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class HVACVisualizer(BaseViz):
    '''Draws the zones in a row in the order they are declared, with the heaters in a row
    below them, in the order that keeps them closest to the zones they serve. Each tile
    fills up with its temperature, and dashed lines mark the comfort range. Arcs above the
    zones join adjacent zones, and lines connect heaters to the zones they serve.

    All tiles use one temperature scale, so a tile hotter than the scale stays full and
    its number gives the exact value. Discomfort is only penalized in occupied zones. The
    drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 column_width=1.1,
                 max_width=14.0,
                 min_width=6.6,
                 max_zone_size=0.95,
                 max_heater_size=0.75,
                 arc_height=1.6,
                 gap_height=1.1,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._zones = model.type_to_objects['zone']
        self._low = self._nonfluents['TEMP-ZONE-MIN']
        self._high = self._nonfluents['TEMP-ZONE-MAX']
        self._heater_zones = {h: [z for z in self._zones if self._nonfluents[
            RDDLPlanningModel.ground_var('ADJ-HEATER', [h, z])]]
            for h in model.type_to_objects['heater']}
        self._zone_links = [
            (z1, z2) for (i, z1) in enumerate(self._zones) for z2 in self._zones[i + 1:]
            if self._nonfluents[RDDLPlanningModel.ground_var('ADJ-ZONES', [z1, z2])]
            or self._nonfluents[RDDLPlanningModel.ground_var('ADJ-ZONES', [z2, z1])]]
        self._heaters = self._get_heater_order(model.type_to_objects['heater'])
        self._scale = self._get_scale(model)

        # both rows use the same spacing, and wide rows use smaller tiles to fit on screen
        num_columns = max(len(self._zones), len(self._heaters), 1)
        margin, left_over = 0.2, 0.75
        width = max(min(column_width * num_columns + 2 * margin, max_width), min_width)
        self._column = min((width - 2 * margin) / num_columns, column_width)
        self._zone_size = min(max_zone_size, 0.8 * self._column)
        self._heater_size = min(max_heater_size, 0.68 * self._column)

        # from the bottom: names and tiles of heaters, a gap for the lines, names and tiles
        # of zones, and the arcs above them
        name_height = 0.28
        self._heater_y = left_over + name_height
        self._zone_y = self._heater_y + self._heater_size + gap_height + name_height
        self._arc_y = self._zone_y + self._zone_size
        self._arc_height = arc_height if self._zone_links else 0.0
        height = self._arc_y + self._arc_height + 0.6

        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1 - 0.5 / height])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height - 0.5)
        self.ax.axis('off')

        center = width / 2
        self._x = {}
        for row in (self._zones, self._heaters):
            for (i, name) in enumerate(row):
                self._x[name] = center + (i - (len(row) - 1) / 2) * self._column

        # the links and tiles never change, so they are only drawn once
        self._draw_links()
        for z in self._zones:
            self._draw_tile(z, self._zone_y, self._zone_size, comfort=True)
        for h in self._heaters:
            self._draw_tile(h, self._heater_y, self._heater_size, comfort=False)

        self.fig.legend(handles=[
            patches.Patch(facecolor=COMFORT_COLOR, label='In range'),
            patches.Patch(facecolor=COLD_COLOR, label='Too cold'),
            patches.Patch(facecolor=HOT_COLOR, label='Too hot'),
            patches.Patch(facecolor=EMPTY_COLOR, label='Unoccupied'),
            patches.Patch(facecolor=HEATER_COLOR, label='Heater'),
            patches.Patch(facecolor=RANGE_COLOR, edgecolor=RANGE_EDGE_COLOR, label='Comfort range')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / height), ncol=6, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)
        self.fig.text(0.5, 0.1 / height,
                      f'Tiles fill from {self._scale[0]:.1f} to {self._scale[1]:.1f} degrees',
                      ha='center', va='bottom', fontsize=9, color='#37474f')

    def _get_heater_order(self, heaters):

        # each heater goes below the middle of the zones it serves, which keeps the lines
        # short: heaters that serve no zones go last
        position = {z: i for (i, z) in enumerate(self._zones)}

        def key(heater):
            zones = self._heater_zones[heater]
            if not zones:
                return len(self._zones)
            return sum(position[z] for z in zones) / len(zones)

        return sorted(heaters, key=key)

    def _get_scale(self, model):

        # the outside, the comfort range and the initial temperatures bound where zones
        # normally stay: heaters can be hotter, and then their tiles stay full
        init_state = model.ground_vars_with_values(model.state_fluents)
        temps = [self._nonfluents['TEMP-OUT'], self._low, self._high]
        temps.extend(init_state[RDDLPlanningModel.ground_var('temp-zone', [z])] for z in self._zones)
        temps.extend(init_state[RDDLPlanningModel.ground_var('temp-heater', [h])] for h in self._heaters)
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
        count = max(len(self._zone_links) + sum(map(len, self._heater_zones.values())), 1)
        width = max(0.7, min(1.5, 40 / count))
        alpha = max(0.35, min(0.8, 30 / count))

        # heat flows between two zones if either one is declared adjacent to the other: the
        # height of an arc grows with its length, so that arcs nest instead of overlapping
        num_zones = max(len(self._zones) - 1, 1)
        angles = np.linspace(0, np.pi, 60)
        position = {z: i for (i, z) in enumerate(self._zones)}
        for (z1, z2) in self._zone_links:
            x1, x2 = self._x[z1], self._x[z2]
            span = abs(position[z2] - position[z1]) / num_zones
            height = (0.15 + 0.85 * span) * self._arc_height
            self.ax.plot((x1 + x2) / 2 + (x2 - x1) / 2 * np.cos(angles),
                         self._arc_y + height * np.sin(angles),
                         color=ZONE_LINK_COLOR, linewidth=width, alpha=alpha, zorder=1)

        # the lines from heaters to zones run straight across the gap between the rows
        for (h, zones) in self._heater_zones.items():
            for z in zones:
                self.ax.plot([self._x[h], self._x[z]],
                             [self._heater_y + self._heater_size, self._zone_y],
                             color=HEATER_COLOR, linewidth=width, alpha=alpha, zorder=1)

    def _draw_tile(self, name, y, size, comfort):
        x = self._x[name]
        self.ax.add_patch(patches.Rectangle(
            (x - size / 2, y), size, size, facecolor=BACKGROUND_COLOR,
            edgecolor='none', zorder=2))
        self.ax.add_patch(patches.Rectangle(
            (x - size / 2, y), size, size, facecolor='none',
            edgecolor=EDGE_COLOR, linewidth=1.2, zorder=5))
        self._add_text(x, y - 0.05, name, 'top', 10, self._column,
                       bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.8, 'alpha': 0.85})
        if comfort:
            low, high = y + size * self._fraction(self._low), y + size * self._fraction(self._high)
            self.ax.add_patch(patches.Rectangle(
                (x - size / 2, low), size, high - low, facecolor=RANGE_COLOR, edgecolor='none', zorder=2))
            for level in (low, high):
                self.ax.plot([x - size / 2, x + size / 2], [level, level], color=RANGE_EDGE_COLOR,
                             linestyle='--', linewidth=1, zorder=4)

    def _zone_color(self, temp, occupied):
        if not occupied:
            return EMPTY_COLOR
        if temp < self._low:
            return COLD_COLOR
        return HOT_COLOR if temp > self._high else COMFORT_COLOR

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
        artists = []
        occupied_zones = 0
        comfortable = 0
        for z in self._zones:
            temp = float(state[RDDLPlanningModel.ground_var('temp-zone', [z])])
            occupied = bool(state[RDDLPlanningModel.ground_var('occupied', [z])])
            artists.extend(self._draw_temperature(
                z, temp, self._zone_y, self._zone_size, self._zone_color(temp, occupied)))

            # same rule as the penalty in the reward
            if occupied:
                occupied_zones += 1
                comfortable += self._low <= temp <= self._high
        for h in self._heaters:
            temp = float(state[RDDLPlanningModel.ground_var('temp-heater', [h])])
            artists.extend(self._draw_temperature(
                h, temp, self._heater_y, self._heater_size, HEATER_COLOR))

        self.ax.set_title(f'{comfortable} / {occupied_zones} occupied zones in range',
                          fontsize=13, fontweight='bold',
                          color=OK_COLOR if comfortable == occupied_zones else BAD_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
