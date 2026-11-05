import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
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

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class PowerGenVisualizer(BaseViz):
    '''Draws the production of each plant from the previous step within its limits,
    next to the demand at the current temperature.

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
        plants_width = max(column_width * len(self._plants), min_plants_width)
        plants_width = min(plants_width, max_width - curve_width - left - gap - right)
        fig_width = left + plants_width + gap + curve_width + right
        fig_height = bottom + plot_height + top
        self._column_inches = plants_width / len(self._plants)

        self.fig = plt.figure(figsize=(fig_width, fig_height))
        self._plant_ax = self.fig.add_axes(
            [left / fig_width, bottom / fig_height,
             plants_width / fig_width, plot_height / fig_height])
        self._curve_ax = self.fig.add_axes(
            [(left + plants_width + gap) / fig_width, bottom / fig_height,
             curve_width / fig_width, plot_height / fig_height])
        self._setup_plants(plot_height)
        self._setup_curve()

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
            size = self._font_size([p], 10)
            if size is not None:
                ax.annotate(p, (i + 0.5, self._low), xytext=(0, -6), textcoords='offset points',
                            ha='center', va='top', fontsize=size, annotation_clip=False)
            lines = [f'cost {self._cost[p]:g}', f'change {self._penalty[p]:g}']
            size = self._font_size(lines, 8)
            if size is not None:
                ax.annotate('\n'.join(lines), (i + 0.5, self._low), xytext=(0, -22),
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
                   markeredgecolor='none', label='Would fall short')
        ], loc='lower center', ncol=5, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

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
