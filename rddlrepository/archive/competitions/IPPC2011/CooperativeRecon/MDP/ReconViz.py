import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

AGENT_COLOR = '#1565c0'
EDGE_COLOR = '#37474f'
GRID_COLOR = '#b0bec5'
BASE_COLOR = '#cfd8dc'
HAZARD_COLOR = '#ffcdd2'
HAZARD_EDGE_COLOR = '#c62828'
NEAR_HAZARD_COLOR = to_rgba(HAZARD_EDGE_COLOR, 0.12)
UNTESTED_COLOR = 'white'
WATER_COLOR = '#bbdefb'
LIFE_COLOR = '#66bb6a'
RULED_OUT_COLOR = '#90a4ae'
ALERT_COLOR = '#ff8f00'

# where the objects and agents are drawn inside of a cell, as a fraction of its size
OBJECT_AREA = (0.06, 0.4, 0.88, 0.56)
MAX_TOKEN_RADIUS = 0.22
AGENT_AREA_BOTTOM = 0.05


class ReconVisualizer(BaseViz):
    '''Draws the grid with the base and hazards, the agents, the objects coloured by
    what the tools have found on them, and the condition of each tool.

    The MDP also damages tools next to a hazard and lets tests rule an object out,
    and these are only drawn when the domain has the corresponding fluents. The POMDP
    shows the true damage of each tool rather than the noisy observation of it.

    The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_cell_size=2.0,
                 max_size=5.5,
                 min_width=6.8,
                 tools_per_row=3,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._objects = model.type_to_objects['obj']
        self._agents = model.type_to_objects['agent']
        self._tools = model.type_to_objects['tool']
        self._tests_rule_out = 'waterChecked' in model.state_fluents

        # moving right and up follow the adjacency relations, whatever the objects are called
        self._xs = self._chain(model.type_to_objects['x_pos'], 'ADJACENT-RIGHT', 'ADJACENT-LEFT')
        self._ys = self._chain(model.type_to_objects['y_pos'], 'ADJACENT-UP', 'ADJACENT-DOWN')
        self._cells = [(col, row, x, y) for (col, x) in enumerate(self._xs)
                       for (row, y) in enumerate(self._ys)]

        # the objects never move, so each one has a fixed place in its cell
        self._cell_size = min(max_cell_size, max_size / max(len(self._xs), len(self._ys)))
        self._tokens = {}
        for (col, row, x, y) in self._cells:
            here = [o for o in self._objects
                    if self._nonfluent('objAt', o, x, y)]
            self._tokens.update(self._place_tokens(here, col, row))

        # title above, then the grid, its coordinates, the tools and the legend
        self._tool_cols = min(tools_per_row, max(len(self._tools), 1))
        tool_rows = math.ceil(len(self._tools) / self._tool_cols)
        grid_width = len(self._xs) * self._cell_size
        grid_height = len(self._ys) * self._cell_size
        tools_height = 0.36 * tool_rows
        legend_height = 0.6
        labels_height = 0.3
        width = max(grid_width + 0.8, min_width)
        height = legend_height + tools_height + labels_height + grid_height + 0.45
        self._figure_size = (width, height)
        self._tools_size = (width, tools_height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width),
             (legend_height + tools_height + labels_height) / height,
             grid_width / width, grid_height / height])
        self.ax.set_xlim(0, len(self._xs))
        self.ax.set_ylim(0, len(self._ys))
        self.ax.set_aspect('equal')
        self.ax.set_xticks(np.arange(len(self._xs)) + 0.5, self._xs)
        self.ax.set_yticks(np.arange(len(self._ys)) + 0.5, self._ys)
        self.ax.tick_params(length=0, labelsize=9, colors='#616161')
        self.ax.set_facecolor('#fafafa')
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        self._tools_ax = self.fig.add_axes(
            [0, legend_height / height, 1, tools_height / height])
        self._tools_ax.set_xlim(0, width)
        self._tools_ax.set_ylim(0, tools_height)
        self._tools_ax.axis('off')

        # the base, hazards and grid never change, so they are only drawn once
        for (col, row, x, y) in self._cells:
            self._draw_cell(col, row, x, y)

        self.fig.legend(handles=self._legend_handles(), loc='lower center',
                        ncol=5 if self._tests_rule_out else 4, frameon=False,
                        fontsize=8.5, columnspacing=1.0, handlelength=1.4)

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _chain(self, objects, forward, backward):

        # follow the relation from the objects nothing leads to, and any object left
        # over because of a cycle or missing relation keeps its declared position
        following = {}
        for a in objects:
            for b in objects:
                if a != b and (self._nonfluent(forward, a, b) or self._nonfluent(backward, b, a)):
                    following.setdefault(a, b)
        starts = [a for a in objects if a not in following.values()]
        order = []
        for start in starts + list(objects):
            current = start
            while current not in order:
                order.append(current)
                current = following.get(current, current)
        return order

    def _place_tokens(self, objects, col, row):
        if not objects:
            return {}

        # use the arrangement that leaves the most room for each object
        left, bottom, width, height = OBJECT_AREA
        cols = max(range(1, len(objects) + 1), key=lambda c: min(
            width / c, height / math.ceil(len(objects) / c)))
        rows = math.ceil(len(objects) / cols)
        radius = min(MAX_TOKEN_RADIUS, 0.45 * min(width / cols, height / rows))

        tokens = {}
        for (i, obj) in enumerate(objects):
            r, c = divmod(i, cols)
            per_row = min(cols, len(objects) - r * cols)
            x = col + left + width * (c + 0.5 + (cols - per_row) / 2) / cols
            y = row + bottom + height * (rows - r - 0.5) / rows
            tokens[obj] = (x, y, radius)
        return tokens

    def _is_near_hazard(self, x, y):

        # this mirrors the damage in the MDP, which applies along each axis separately
        return any(self._nonfluent('HAZARD', x2, y) and (
                   self._nonfluent('ADJACENT-LEFT', x, x2) or self._nonfluent('ADJACENT-RIGHT', x, x2))
                   for x2 in self._xs) or \
               any(self._nonfluent('HAZARD', x, y2) and (
                   self._nonfluent('ADJACENT-UP', y, y2) or self._nonfluent('ADJACENT-DOWN', y, y2))
                   for y2 in self._ys)

    def _draw_cell(self, col, row, x, y):
        base = self._nonfluent('BASE', x, y)
        hazard = self._nonfluent('HAZARD', x, y)

        # the base protects tools from damage in the MDP, but not in the POMDP
        if self._tests_rule_out and base:
            hazard = False
        near = self._tests_rule_out and not base and self._is_near_hazard(x, y)

        color = BASE_COLOR if base else (
            HAZARD_COLOR if hazard else (NEAR_HAZARD_COLOR if near else 'none'))
        self.ax.add_patch(patches.Rectangle(
            (col, row), 1, 1, facecolor=color, edgecolor=GRID_COLOR, zorder=1))
        if hazard:
            self.ax.add_patch(patches.Rectangle(
                (col, row), 1, 1, facecolor='none', edgecolor=HAZARD_EDGE_COLOR,
                hatch='///', linewidth=1.2, alpha=0.5, zorder=2))

    def _legend_handles(self):

        def token(color, label, **kwargs):
            style = {'markeredgecolor': EDGE_COLOR, **kwargs}
            return Line2D([], [], marker='o', color='none', markerfacecolor=color,
                          markersize=9, label=label, **style)

        handles = [
            patches.Patch(facecolor=BASE_COLOR, edgecolor=GRID_COLOR, label='Base'),
            patches.Patch(facecolor=HAZARD_COLOR, edgecolor=HAZARD_EDGE_COLOR, hatch='///', label='Hazard')
        ]
        if self._tests_rule_out:
            handles.append(patches.Patch(
                facecolor=NEAR_HAZARD_COLOR, edgecolor=GRID_COLOR, label='Near hazard'))
        handles += [
            Line2D([], [], marker='o', color='none', markerfacecolor=AGENT_COLOR,
                   markeredgecolor='white', markersize=9, label='Agent'),
            token(UNTESTED_COLOR, 'Not found yet' if not self._tests_rule_out else 'Not tested'),
            token(WATER_COLOR, 'Water found'),
            token(LIFE_COLOR, 'Life found')
        ]
        if self._tests_rule_out:
            handles += [
                token(RULED_OUT_COLOR, 'Ruled out'),
                Line2D([], [], marker='o', color='none', markerfacecolor=ALERT_COLOR,
                       markeredgecolor='white', markersize=7, label='Life test failed')
            ]
        handles.append(token('none', 'Photo taken', markeredgecolor=ALERT_COLOR, markeredgewidth=2))
        return handles

    def _get(self, state, var, *args):
        return bool(state[RDDLPlanningModel.ground_var(var, list(args))])

    def _token_color(self, state, obj):
        if self._get(state, 'lifeDetected', obj):
            return LIFE_COLOR
        water = self._get(state, 'waterDetected', obj)

        # a failed water test or a second failed life test can never succeed afterwards
        if self._tests_rule_out and (
                self._get(state, 'lifeChecked2', obj)
                or (self._get(state, 'waterChecked', obj) and not water)):
            return RULED_OUT_COLOR
        return WATER_COLOR if water else UNTESTED_COLOR

    def _draw_object(self, state, obj):
        x, y, radius = self._tokens[obj]
        color = self._token_color(state, obj)
        artists = [self.ax.add_patch(patches.Circle(
            (x, y), radius, facecolor=color, edgecolor=EDGE_COLOR, linewidth=1.2, zorder=4))]

        # the size of the name must fit inside of the token
        size = min(9, 2.8 * radius * self._cell_size * 72 / max(len(obj), 1))
        artists.append(self.ax.text(
            x, y, obj, ha='center', va='center', fontsize=size, zorder=6,
            color='white' if color == RULED_OUT_COLOR else 'black'))

        if self._get(state, 'pictureTaken', obj):
            artists.append(self.ax.add_patch(patches.Circle(
                (x, y), radius, facecolor='none', edgecolor=ALERT_COLOR, linewidth=3, zorder=5)))

        # the first failed life test leaves one more attempt
        if self._tests_rule_out and color != LIFE_COLOR and color != RULED_OUT_COLOR \
                and self._get(state, 'lifeChecked', obj):
            offset = radius * math.sqrt(0.5)
            artists.append(self.ax.add_patch(patches.Circle(
                (x + offset, y + offset), 0.28 * radius, facecolor=ALERT_COLOR,
                edgecolor='white', linewidth=1, zorder=7)))
        return artists

    def _draw_agents(self, col, row, agents):
        artists = []
        scale = min(1.0, 2.0 / len(agents))
        for (i, agent) in enumerate(agents):
            x = col + (i + 0.5) / len(agents)
            y = row + AGENT_AREA_BOTTOM
            artists.append(self.ax.add_patch(patches.Wedge(
                (x, y), 0.16 * scale, 0, 180, facecolor=AGENT_COLOR, zorder=4)))
            artists.append(self.ax.add_patch(patches.Circle(
                (x, y + 0.24 * scale), 0.08 * scale, facecolor=AGENT_COLOR, zorder=4)))

            # one agent needs no name, as there is nothing to tell it apart from
            if len(self._agents) > 1:
                artists.append(self.ax.text(
                    x, y + 0.06 * scale, agent, ha='center', va='center',
                    fontsize=min(7, 7 * scale * self._cell_size), color='white', zorder=5))
        return artists

    def _draw_tools(self, state):
        artists = []
        width, height = self._tools_size
        chip_width = min(1.9, (width - 0.4) / self._tool_cols)
        for (i, tool) in enumerate(self._tools):
            r, c = divmod(i, self._tool_cols)
            per_row = min(self._tool_cols, len(self._tools) - r * self._tool_cols)
            x = (width - per_row * chip_width) / 2 + c * chip_width
            y = height - 0.36 * (r + 1)
            damaged = self._get(state, 'damaged', tool)

            kinds = [name for (var, name) in (('WATER_TOOL', 'Water'), ('LIFE_TOOL', 'Life'),
                                              ('CAMERA_TOOL', 'Camera'))
                     if self._nonfluent(var, tool)]
            label = f'{" + ".join(kinds) or "Tool"} {tool}: {"damaged" if damaged else "ok"}'
            artists.append(self._tools_ax.add_patch(patches.Rectangle(
                (x + 0.05, y + 0.04), chip_width - 0.1, 0.28,
                facecolor=HAZARD_COLOR if damaged else '#fafafa',
                edgecolor=HAZARD_EDGE_COLOR if damaged else GRID_COLOR)))
            artists.append(self._tools_ax.text(
                x + chip_width / 2, y + 0.18, label, ha='center', va='center', fontsize=9,
                color=HAZARD_EDGE_COLOR if damaged else EDGE_COLOR))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        artists = []
        for obj in self._tokens:
            artists.extend(self._draw_object(state, obj))
        for (col, row, x, y) in self._cells:
            agents = [a for a in self._agents if self._get(state, 'agentAt', a, x, y)]
            if agents:
                artists.extend(self._draw_agents(col, row, agents))
        artists.extend(self._draw_tools(state))

        num_life = sum(self._get(state, 'lifeDetected', o) for o in self._objects)
        done = num_life == len(self._objects)
        self.ax.set_title(f'Life found on {num_life} / {len(self._objects)} objects',
                          fontsize=13, fontweight='bold',
                          color='#2e7d32' if done else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
