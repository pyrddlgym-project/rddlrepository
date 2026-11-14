import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

PRODUCTION_COLOR = '#1565c0'
LIMIT_COLOR = '#37474f'
GUIDE_COLOR = '#b0bec5'
CURVE_COLOR = '#37474f'
CAPACITY_COLOR = '#c62828'
MET_COLOR = '#2e7d32'
SHORT_COLOR = '#c62828'
BUILDING_COLOR = '#b0bec5'
BUILDING_EDGE_COLOR = '#455a64'
WINDOW_DARK_COLOR = '#546e7a'
WINDOW_LIT_COLOR = '#ffd54f'
GENERATOR_ON_COLOR = '#ffb300'
GENERATOR_OFF_COLOR = '#cfd8dc'
TUBE_COLOR = '#fafafa'
MERCURY_COLOR = '#e53935'

# a power station is at most this tall, in inches, with this many rows and columns of
# windows, which light up with the share of its capacity that it produces
STATION_HEIGHT = 0.85
WINDOW_ROWS, WINDOW_COLUMNS = 2, 3

# the thermometer stands this far to the right of the curve, in inches, this wide, with
# room for its numbers to its right
THERMOMETER_GAP, THERMOMETER_WIDTH, THERMOMETER_LABELS = 0.3, 0.3, 0.35

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class PowerGenVisualizer(BaseViz):
    '''Draws the production of each plant from the previous step within its limits,
    above its power station, whose windows light up with the share of its capacity that it
    produces and whose generator is amber while it runs, next to the demand at the current
    temperature and a thermometer.

    The state holds the production of the previous step and the temperature of the
    current one, so the comparison shows whether keeping the same production would
    meet the current demand. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 column_width=0.9,
                 plot_height=3.8,
                 curve_width=3.4,
                 max_width=14.0,
                 min_plants_width=3.0,
                 num_points=100,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._num_points = num_points

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._plants = model.type_to_objects['plant']
        self._min = self._get_values(nonfluents, 'PROD-UNITS-MIN')
        self._max = self._get_values(nonfluents, 'PROD-UNITS-MAX')
        self._cost = self._get_values(nonfluents, 'COST-PER-UNIT')
        self._penalty = self._get_values(nonfluents, 'PROD-CHANGE-PENALTY')
        self._min_consumption = nonfluents['MIN-CONSUMPTION']
        self._coefficient = nonfluents['DEMAND-EXP-COEF']
        self._min_demand_temp = nonfluents['MIN-DEMAND-TEMP']
        self._capacity = sum(self._max.values())

        # the curve reaches the total capacity, unless demand does not depend on the
        # temperature, and otherwise grows only if the temperature leaves this range
        excess = max(self._capacity - self._min_consumption, 1.0)
        self._half_range = math.sqrt(excess / self._coefficient) if self._coefficient > 0 else 1.0

        # the left margin of each panel holds its axis, and the bottom the labels
        left, gap, right, bottom, top = 0.65, 0.75, 0.2, 1.0, 0.85
        thermometer = THERMOMETER_GAP + THERMOMETER_WIDTH + THERMOMETER_LABELS
        plants_width = max(column_width * len(self._plants), min_plants_width)
        plants_width = min(plants_width,
                           max_width - curve_width - left - gap - right - thermometer)
        fig_width = left + plants_width + gap + curve_width + thermometer + right
        self._column_inches = plants_width / len(self._plants)

        # each plant has a power station under its production, at most as tall as it is
        # wide, so that one unit is as long across as up
        station = min(STATION_HEIGHT, self._column_inches)
        fig_height = bottom + station + plot_height + top
        plot_bottom = (bottom + station) / fig_height

        self.fig = plt.figure(figsize=(fig_width, fig_height))
        self._plant_ax = self.fig.add_axes(
            [left / fig_width, plot_bottom, plants_width / fig_width, plot_height / fig_height])
        self._station_ax = self.fig.add_axes(
            [left / fig_width, bottom / fig_height, plants_width / fig_width,
             station / fig_height])
        self._station_ax.set_xlim(0, len(self._plants))
        self._station_ax.set_ylim(0, station / self._column_inches)
        self._station_ax.axis('off')
        self._stations = {}
        self._curve_ax = self.fig.add_axes(
            [(left + plants_width + gap) / fig_width, plot_bottom,
             curve_width / fig_width, plot_height / fig_height])
        self._thermometer_ax = self.fig.add_axes(
            [(left + plants_width + gap + curve_width + THERMOMETER_GAP) / fig_width,
             plot_bottom, THERMOMETER_WIDTH / fig_width, plot_height / fig_height])
        self._setup_plants(plot_height)
        self._setup_curve()
        self._setup_thermometer()

    def _get_values(self, nonfluents, var):
        return {p: nonfluents[RDDLPlanningModel.ground_var(var, [p])] for p in self._plants}

    def _font_size(self, lines, font_size):

        # text shrinks to the width of a column, and is left out if unreadable
        length = max(len(line) for line in lines)
        size = min(font_size, 72 * 0.95 * self._column_inches / (0.62 * length))
        return size if size >= MIN_FONT_SIZE else None

    def _setup_plants(self, plot_height):
        ax = self._plant_ax
        ax.set_xlim(0, len(self._plants))
        ax.set_xticks([])
        ax.set_ylabel('production', fontsize=9)
        ax.tick_params(labelsize=8, colors='#616161')
        ax.set_title('Production at the previous step', fontsize=10)
        for side in ('top', 'right', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.spines['left'].set_color('#b0bec5')
        self._plot_height = plot_height

        # the limits and names of the plants never change, so they are only drawn once
        self._low = min(0.0, *self._min.values())
        self._high = max(*self._max.values(), 0.0)
        for (i, p) in enumerate(self._plants):
            ax.add_patch(patches.Rectangle(
                (i + 0.12, self._low), 0.76, self._max[p] - self._low, facecolor='none',
                edgecolor=LIMIT_COLOR, linewidth=1.2, zorder=3))
            if self._min[p] > self._low:
                ax.plot([i + 0.12, i + 0.88], [self._min[p]] * 2, color=LIMIT_COLOR,
                        linestyle='--', linewidth=1, zorder=4)
            self._draw_station(i, p)
            size = self._font_size([p], 10)
            if size is not None:
                self._station_ax.annotate(p, (i + 0.5, 0), xytext=(0, -6), textcoords='offset points',
                            ha='center', va='top', fontsize=size, annotation_clip=False)
            lines = [f'cost {self._cost[p]:g}', f'change {self._penalty[p]:g}']
            size = self._font_size(lines, 8)
            if size is not None:
                self._station_ax.annotate('\n'.join(lines), (i + 0.5, 0), xytext=(0, -22),
                            textcoords='offset points', ha='center', va='top', fontsize=size,
                            color='#616161', annotation_clip=False)

    def _setup_curve(self):
        ax = self._curve_ax
        ax.set_xlabel('temperature', fontsize=9, labelpad=2)
        ax.set_ylabel('power', fontsize=9)
        ax.tick_params(labelsize=8, colors='#616161')
        ax.set_title('Demand at the current temperature', fontsize=10)
        ax.set_autoscale_on(False)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#b0bec5')
        self._curve_line, = ax.plot([], [], color=CURVE_COLOR, linewidth=2, zorder=2)
        self._production_line = ax.axhline(0, color=PRODUCTION_COLOR, linewidth=1.5, zorder=1)
        ax.axhline(self._capacity, color=CAPACITY_COLOR, linestyle='--', linewidth=1.2, zorder=1)
        self.fig.legend(handles=[
            Line2D([], [], color=CURVE_COLOR, linewidth=2, label='Demand'),
            Line2D([], [], color=PRODUCTION_COLOR, linewidth=1.5, label='Production'),
            Line2D([], [], color=CAPACITY_COLOR, linestyle='--', linewidth=1.2, label='Capacity'),
            Line2D([], [], marker='o', color='none', markerfacecolor=MET_COLOR,
                   markeredgecolor='none', label='Would be met'),
            Line2D([], [], marker='o', color='none', markerfacecolor=SHORT_COLOR,
                   markeredgecolor='none', label='Would fall short'),
            patches.Patch(facecolor=WINDOW_LIT_COLOR, edgecolor=BUILDING_EDGE_COLOR,
                          label='Output'),
            Line2D([], [], marker='o', color='none', markerfacecolor=GENERATOR_ON_COLOR,
                   markeredgecolor=BUILDING_EDGE_COLOR, label='Running')
        ], loc='lower center', ncol=7, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _draw_station(self, i, plant):

        # a building with a sawtooth roof and a chimney, with its windows, and its
        # generator beside it, wired up to the production above
        ax = self._station_ax
        (x, width, height, roof) = (i + 0.1, 0.52, 0.4, 0.1)
        ax.add_patch(patches.Rectangle(
            (x + 0.36, height), 0.08, 0.3, facecolor=BUILDING_COLOR,
            edgecolor=BUILDING_EDGE_COLOR, linewidth=0.8, zorder=1))
        outline = [(x, 0), (x, height)]
        for k in range(3):
            outline += [(x + k * width / 3, height + roof), (x + (k + 1) * width / 3, height)]
        outline.append((x + width, 0))
        ax.add_patch(patches.Polygon(
            outline, facecolor=BUILDING_COLOR, edgecolor=BUILDING_EDGE_COLOR, linewidth=0.8,
            zorder=2))
        windows = []
        size = 0.08
        for row in range(WINDOW_ROWS):
            for column in range(WINDOW_COLUMNS):
                windows.append(ax.add_patch(patches.Rectangle(
                    (x + width * (column + 0.5) / WINDOW_COLUMNS - size / 2, 0.08 + 0.15 * row),
                    size, size, facecolor=WINDOW_DARK_COLOR, edgecolor='none', zorder=3)))
        (gx, gy, radius) = (i + 0.8, 0.16, 0.12)
        ax.plot([gx, gx], [gy + radius, ax.get_ylim()[1]], color=BUILDING_EDGE_COLOR,
                linewidth=1, zorder=1)
        generator = ax.add_patch(patches.Circle(
            (gx, gy), radius, facecolor=GENERATOR_OFF_COLOR, edgecolor=BUILDING_EDGE_COLOR,
            linewidth=1, zorder=2))
        t = np.linspace(-1, 1, 30)
        ax.plot(gx + 0.6 * radius * t, gy + 0.3 * radius * np.sin(np.pi * t),
                color=BUILDING_EDGE_COLOR, linewidth=1, zorder=3)
        self._stations[plant] = (windows, generator)

    def _update_stations(self, production, on):

        # any production lights at least one window, and full capacity all of them
        for p in self._plants:
            (windows, generator) = self._stations[p]
            share = production[p] / self._max[p] if self._max[p] > 0 else 0.0
            lit = int(np.ceil(min(max(share, 0.0), 1.0) * len(windows)))
            for (k, window) in enumerate(windows):
                window.set_facecolor(WINDOW_LIT_COLOR if k < lit else WINDOW_DARK_COLOR)
            generator.set_facecolor(GENERATOR_ON_COLOR if on[p] else GENERATOR_OFF_COLOR)

    def _setup_thermometer(self):
        ax = self._thermometer_ax
        ax.set_xlim(-1, 1)
        ax.set_xticks([])
        ax.yaxis.tick_right()
        ax.tick_params(labelsize=8, colors='#616161')
        ax.set_title('\u00b0C', fontsize=10)
        for side in ('top', 'bottom', 'left', 'right'):
            ax.spines[side].set_visible(False)
        self._tube = [ax.plot([0, 0], [0, 1], color=color, linewidth=width,
                              solid_capstyle='round', zorder=zorder)[0]
                      for (color, width, zorder) in ((BUILDING_EDGE_COLOR, 12, 1),
                                                     (TUBE_COLOR, 9, 2))]
        self._bulb = ax.plot([0], [0], marker='o', markersize=18, color=MERCURY_COLOR,
                             markeredgecolor=BUILDING_EDGE_COLOR, markeredgewidth=1.5,
                             zorder=3)[0]
        self._mercury = ax.plot([0, 0], [0, 0], color=MERCURY_COLOR, linewidth=5,
                                solid_capstyle='butt', zorder=4)[0]

    def _draw_thermometer(self, low, high, temperature):

        # the tube spans the same temperatures as the curve, above its bulb
        span = high - low
        bulb = low - 0.1 * span
        ax = self._thermometer_ax
        ax.set_ylim(bulb - 0.05 * span, high + 0.02 * span)
        ax.set_yticks([t for t in MaxNLocator(5).tick_values(low, high) if low <= t <= high])
        for line in self._tube:
            line.set_ydata([bulb, high])
        self._bulb.set_ydata([bulb])
        self._mercury.set_ydata([bulb, min(max(temperature, low), high)])

    def _demand(self, temperature):

        # same rule as the domain
        return self._min_consumption + self._coefficient * (
            temperature - self._min_demand_temp) ** 2

    def _draw_plants(self, production):
        ax = self._plant_ax

        # production outside of the limits is only possible if invalid actions are allowed,
        # and the scale must then grow, with room for the labels above the tallest column
        high = max(self._high, *production.values())
        low = min(self._low, *production.values())
        span = (high - low) or 1.0
        ax.set_ylim(low, high + 0.08 * span)

        artists = []
        for (i, p) in enumerate(self._plants):
            amount = production[p]
            artists.append(ax.add_patch(patches.Rectangle(
                (i + 0.12, min(amount, 0.0)), 0.76, abs(amount), facecolor=PRODUCTION_COLOR,
                edgecolor='none', alpha=0.85, zorder=2)))
            text = f'{amount:.3g} / {self._max[p]:g}'
            size = self._font_size([text], 9)
            if size is not None:
                artists.append(ax.annotate(
                    text, (i + 0.5, max(amount, self._max[p])), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom', fontsize=size,
                    annotation_clip=False, zorder=5))
        return artists

    def _draw_curve(self, temperature, demand, total):
        ax = self._curve_ax
        half = max(self._half_range, 1.1 * abs(temperature - self._min_demand_temp))
        xs = np.linspace(self._min_demand_temp - half, self._min_demand_temp + half,
                         self._num_points)
        self._curve_line.set_data(xs, self._demand(xs))
        self._production_line.set_ydata([total, total])
        self._draw_thermometer(xs[0], xs[-1], temperature)

        ax.set_xlim(xs[0], xs[-1])
        top = max(self._capacity, demand, total, self._min_consumption) or 1.0
        ax.set_ylim(0, 1.1 * top)

        met = total >= demand
        artists = ax.plot([temperature], [demand], marker='o', markersize=10, zorder=4,
                          color=MET_COLOR if met else SHORT_COLOR, markeredgecolor='white')
        artists.extend(ax.plot([temperature, temperature], [demand, total], zorder=3,
                               color=MET_COLOR if met else SHORT_COLOR, linestyle=':', linewidth=1.5))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        production = {p: float(state[RDDLPlanningModel.ground_var('prevProd', [p])])
                      for p in self._plants}
        temperature = float(state['temperature'])
        demand = float(self._demand(temperature))
        total = sum(production.values())

        self._update_stations(production, {p: production[p] > 0 for p in self._plants})
        artists = self._draw_plants(production)
        artists.extend(self._draw_curve(temperature, demand, total))

        self.fig.suptitle(
            f'Demand {demand:.1f} at temperature {temperature:.1f}, previous production {total:.1f}',
            fontsize=13, fontweight='bold', color=MET_COLOR if total >= demand else SHORT_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
