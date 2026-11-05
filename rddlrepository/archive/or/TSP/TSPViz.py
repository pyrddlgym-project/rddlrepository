import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

ROUTE_COLOR = '#1565c0'
NEWEST_COLOR = '#ffa000'
OPTION_COLOR = '#9e9e9e'
VISITED_COLOR = '#90caf9'
DONE_COLOR = '#2e7d32'

# the curve of the route, which keeps moves in opposite directions apart
ROUTE_CURVE = 0.12

# text smaller than this is not readable, and costs need this much room in inches
MIN_FONT_SIZE = 5
MIN_COST_SPACING = 0.35


class TSPVisualizer(BaseViz):
    '''Draws the nodes on a circle, the route so far and the moves available next.

    The domain only defines the cost between each pair of nodes, which may not
    correspond to any map, so the nodes are placed evenly in the order they are
    declared: the costs are written on the moves, and distance has no meaning.

    The route is built from the states seen so far, so it requires render to be
    called on every step. Nodes visited between two renders are not on the route.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(7.0, 7.6),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._nodes = model.type_to_objects['node']
        self._origins = {n for n in self._nodes if self._nonfluents[
            RDDLPlanningModel.ground_var('ORIGIN', [n])]}
        self._init_state = self._get_state(model.ground_vars_with_values(model.state_fluents))

        self._route = []
        self._last_state = None

        # the nodes go clockwise from the top, with room outside the circle for names
        num_nodes = len(self._nodes)
        self._positions = {
            n: (math.sin(2 * math.pi * i / num_nodes), math.cos(2 * math.pi * i / num_nodes))
            for (i, n) in enumerate(self._nodes)}

        fig_width, fig_height = figure_size
        size = fig_width - 0.4
        limit = 1.35
        self._inches_per_unit = size / (2 * limit)

        # large numbers of nodes use smaller markers and text to fit around the circle
        spacing = self._inches_per_unit * (
            2 * math.sin(math.pi / num_nodes) if num_nodes > 1 else 2)
        self._show_costs = spacing >= MIN_COST_SPACING
        diameter = min(0.4, 0.8 * spacing)
        self._marker_size = 72 * diameter

        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_axes(
            [0.2 / fig_width, 0.55 / fig_height, size / fig_width, size / fig_height])
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # the names of the nodes never change, so they are only drawn once
        gap = (diameter / 2 + 0.08) / self._inches_per_unit
        for n in self._nodes:
            x, y = self._positions[n]
            font_size = min(10, 72 * 0.9 / (0.62 * len(n)), 72 * 0.8 * spacing)
            if font_size >= MIN_FONT_SIZE:
                self.ax.text(x * (1 + gap), y * (1 + gap), n, fontsize=font_size,
                             ha='center' if abs(x) < 0.1 else ('left' if x > 0 else 'right'),
                             va='center' if abs(y) < 0.9 else ('bottom' if y > 0 else 'top'))

        self.ax.legend(handles=[
            Line2D([], [], marker='s', color='none', markerfacecolor='white',
                   markeredgecolor='black', label='Origin'),
            Line2D([], [], marker='o', color='none', markerfacecolor=VISITED_COLOR,
                   markeredgecolor='black', label='Visited'),
            Line2D([], [], marker='o', color='none', markerfacecolor='white',
                   markeredgecolor=NEWEST_COLOR, markeredgewidth=2, label='Current'),
            Line2D([], [], color=ROUTE_COLOR, linewidth=2, label='Route'),
            Line2D([], [], color=OPTION_COLOR, linestyle='--', label='Next moves')
        ], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=5, frameon=False, fontsize=9)

    def _get_state(self, state):
        return (frozenset(n for n in self._nodes
                          if state[RDDLPlanningModel.ground_var('current', [n])]),
                frozenset(n for n in self._nodes
                          if state[RDDLPlanningModel.ground_var('visited', [n])]))

    def _cost(self, start, end):
        return self._nonfluents[RDDLPlanningModel.ground_var('COST', [start, end])]

    def _update_route(self, state):

        # rendering the same state again must not change the route
        if state == self._last_state:
            return
        last_state, self._last_state = self._last_state, state

        # visited nodes are never forgotten during an episode
        if last_state is None or state == self._init_state or not state[1] >= last_state[1]:
            self._route = []
            return
        self._route.extend((start, end) for start in last_state[0] for end in state[0]
                           if start != end)

    def _get_options(self, state):
        current, visited = state

        # same rule as the action preconditions
        unvisited = [n for n in self._nodes if n not in visited]
        if unvisited:
            targets = unvisited
        else:
            targets = [n for n in self._origins if n not in current]
        return [(start, end) for start in current for end in targets if start != end]

    def _draw_cost(self, x, y, cost, color):
        return self.ax.text(
            x, y, f'{cost:g}', ha='center', va='center', fontsize=8, color=color, zorder=6,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.8, 'alpha': 0.85})

    def _draw_option(self, start, end):
        (x1, y1), (x2, y2) = self._positions[start], self._positions[end]
        artists = [self.ax.add_patch(patches.FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=10,
            color=OPTION_COLOR, linestyle='--', linewidth=1, zorder=1,
            shrinkA=self._marker_size / 2, shrinkB=self._marker_size / 2))]

        # near the end of the move, as all of the moves share the same start
        if self._show_costs:
            artists.append(self._draw_cost(
                x1 + 0.72 * (x2 - x1), y1 + 0.72 * (y2 - y1), self._cost(start, end), '#616161'))
        return artists

    def _draw_move(self, start, end, newest):
        (x1, y1), (x2, y2) = self._positions[start], self._positions[end]
        color = NEWEST_COLOR if newest else ROUTE_COLOR
        artists = [self.ax.add_patch(patches.FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14,
            connectionstyle=f'arc3,rad={ROUTE_CURVE}', color=color, linewidth=2, zorder=2,
            shrinkA=self._marker_size / 2, shrinkB=self._marker_size / 2))]

        # the middle of the curve
        if self._show_costs:
            artists.append(self._draw_cost(
                (x1 + x2) / 2 + ROUTE_CURVE / 2 * (y2 - y1),
                (y1 + y2) / 2 - ROUTE_CURVE / 2 * (x2 - x1),
                self._cost(start, end), color))
        return artists

    def _draw_node(self, node, current, visited):
        x, y = self._positions[node]
        return self.ax.plot(
            [x], [y], marker='s' if node in self._origins else 'o',
            markersize=self._marker_size, zorder=3,
            markerfacecolor=VISITED_COLOR if node in visited else 'white',
            markeredgecolor=NEWEST_COLOR if node in current else 'black',
            markeredgewidth=3 if node in current else 1)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        state = self._get_state(state)
        self._update_route(state)
        current, visited = state

        artists = []
        for (start, end) in self._get_options(state):
            artists.extend(self._draw_option(start, end))
        for (i, (start, end)) in enumerate(self._route):
            artists.extend(self._draw_move(start, end, i == len(self._route) - 1))
        for n in self._nodes:
            artists.extend(self._draw_node(n, current, visited))

        # same rule as the termination condition
        done = len(visited) == len(self._nodes) and self._origins <= current
        route_cost = sum(self._cost(start, end) for (start, end) in self._route)
        title = (f'ROUTE COMPLETE: cost {route_cost:g}' if done else
                 f'{len(visited)} / {len(self._nodes)} visited, route cost {route_cost:g}')
        self.ax.set_title(title, fontsize=13, fontweight='bold',
                          color=DONE_COLOR if done else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
