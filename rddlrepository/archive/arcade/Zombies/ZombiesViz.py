from collections import deque

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

HUMAN_COLOR = '#1565c0'
ZOMBIE_COLOR = '#2e7d32'
STUNNED_COLOR = '#90a4ae'
DANGER_COLOR = '#c62828'
INFECTED_COLOR = '#ff8f00'

# a human can only be infected by a zombie closer than this along both axes
INFECTION_RANGE = 1.0


class ZombiesVisualizer(BaseViz):
    '''Draws the field, the agents, the areas where an infection can occur next,
    and a short trail behind each agent.

    The view only covers the region the agents have visited during the episode,
    growing as they spread out, as they may use any part of the field.

    The view, trails and highlight on newly infected humans are built from the
    states seen so far, so they require render to be called on every step.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(6.7, 7.2),
                 trail_length=5,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._init_state = model.ground_vars_with_values(model.state_fluents)
        self._agents = model.type_to_objects['name']

        self._trails = {a: deque(maxlen=trail_length + 1) for a in self._agents}
        self._infected = set()
        self._visited = None
        self._last_state = None

        x_max, y_max = self._nonfluents['X_MAX'], self._nonfluents['Y_MAX']
        self._field = (x_max, y_max)

        # leave room for the tick labels, the title above and the legend below
        self._figure_size = figure_size
        fig_width, fig_height = self._figure_size
        width, height = fig_width - 0.7, fig_height - 1.2
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [0.5 / fig_width, 0.75 / fig_height, width / fig_width, height / fig_height])

        self.ax.tick_params(labelsize=8, colors='#616161')
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')
        self.ax.set_facecolor('#eceff1')
        self.ax.add_patch(patches.Rectangle(
            (0, 0), x_max, y_max, facecolor='#fafafa', edgecolor='#b0bec5', zorder=0))

        self.fig.legend(handles=[
            Line2D([], [], marker='o', color='none', markerfacecolor=HUMAN_COLOR, label='Human'),
            Line2D([], [], marker='o', color='none', markerfacecolor=ZOMBIE_COLOR, label='Zombie'),
            Line2D([], [], marker='o', color='none', markerfacecolor=STUNNED_COLOR, label='Stunned'),
            patches.Patch(facecolor=DANGER_COLOR, alpha=0.2, label='Infection zone')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / fig_height), ncol=4,
           frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)
        self.fig.text(0.5, 0.1 / fig_height, f'Field: {x_max:g} x {y_max:g}', ha='center', va='bottom', fontsize=9, color='#37474f')

    def _get(self, state, var, agent):
        return state[RDDLPlanningModel.ground_var(var, [agent])]

    def _update_history(self, state):

        # rendering the same state again must not change the trails or highlights
        if state == self._last_state:
            return

        # zombies move randomly, so only a new episode returns to the initial state
        new_episode = self._last_state is None or state == self._init_state
        if new_episode:
            for trail in self._trails.values():
                trail.clear()
            self._infected = set()
            self._visited = None
        else:
            self._infected = {
                a for a in self._agents if self._get(state, 'zombie', a)
                and not self._get(self._last_state, 'zombie', a)}

        for a in self._agents:
            self._trails[a].append(
                (self._get(state, 'x_loc', a), self._get(state, 'y_loc', a)))
        self._last_state = dict(state)

        # the range of positions visited so far along each axis
        xs = [self._get(state, 'x_loc', a) for a in self._agents]
        ys = [self._get(state, 'y_loc', a) for a in self._agents]
        if self._visited is None:
            self._visited = ((min(xs), max(xs)), (min(ys), max(ys)))
        else:
            (x_low, x_high), (y_low, y_high) = self._visited
            self._visited = ((min(x_low, *xs), max(x_high, *xs)),
                             (min(y_low, *ys), max(y_high, *ys)))

    def _update_view(self):

        # both axes use the same span while the field allows it, so that the scale
        # stays the same along each axis, with room for the zones around the agents:
        # a field narrower than the span stretches along that axis, as the labels show
        span = max(high - low + 2 * INFECTION_RANGE for (low, high) in self._visited)
        limits = []
        for ((low, high), field) in zip(self._visited, self._field):
            size = min(span, field)
            start = min(max((low + high - size) / 2, 0), field - size)

            # agents on the boundary must still be fully visible
            margin = 0.03 * (size or 1.0)
            limits.append((min(start, low) - margin, max(start + size, high) + margin))
        self.ax.set_xlim(*limits[0])
        self.ax.set_ylim(*limits[1])

    def _draw_trail(self, agent, color):
        artists = []
        trail = list(self._trails[agent])
        for i in range(1, len(trail)):
            (x1, y1), (x2, y2) = trail[i - 1], trail[i]
            artists.extend(self.ax.plot(
                [x1, x2], [y1, y2], color=color, linewidth=1.5, zorder=2,
                alpha=0.6 * i / (len(trail) - 1)))
        return artists

    def _draw_agent(self, state, agent):
        x, y = self._get(state, 'x_loc', agent), self._get(state, 'y_loc', agent)
        zombie = self._get(state, 'zombie', agent)
        stunned = zombie and self._get(state, 'stunned', agent)
        color = STUNNED_COLOR if stunned else (ZOMBIE_COLOR if zombie else HUMAN_COLOR)

        artists = self._draw_trail(agent, color)
        if zombie and not stunned:
            artists.append(self.ax.add_patch(patches.Rectangle(
                (x - INFECTION_RANGE, y - INFECTION_RANGE),
                2 * INFECTION_RANGE, 2 * INFECTION_RANGE,
                facecolor=to_rgba(DANGER_COLOR, 0.12),
                edgecolor=to_rgba(DANGER_COLOR, 0.5), zorder=1)))
        if agent in self._infected:
            artists.extend(self.ax.plot(
                [x], [y], marker='o', markersize=17, color='none', zorder=4,
                markeredgecolor=INFECTED_COLOR, markeredgewidth=3))
        artists.extend(self.ax.plot(
            [x], [y], marker='o', markersize=11, color=color,
            markeredgecolor='white', zorder=3))
        artists.append(self.ax.annotate(
            agent, (x, y), xytext=(0, 11), textcoords='offset points',
            ha='center', va='bottom', fontsize=9, zorder=5))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        self._update_history(state)
        self._update_view()

        artists = []
        for a in self._agents:
            artists.extend(self._draw_agent(state, a))

        num_zombies = sum(bool(self._get(state, 'zombie', a)) for a in self._agents)
        num_humans = len(self._agents) - num_zombies
        if num_humans == 0:
            title = 'All infected'
        else:
            title = (f'{num_humans} human{"s" if num_humans != 1 else ""}, '
                     f'{num_zombies} zombie{"s" if num_zombies != 1 else ""}')
        self.ax.set_title(title, fontsize=13, fontweight='bold',
                          color=DANGER_COLOR if num_humans == 0 else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
