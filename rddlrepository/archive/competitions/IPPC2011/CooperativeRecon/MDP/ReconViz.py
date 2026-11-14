import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

GROUND_COLOR = '#f3e6d6'
GRID_COLOR = '#d7c4ad'
EDGE_COLOR = '#37474f'
TEXT_COLOR = '#37474f'
PAD_COLOR = '#90a4ae'
BASE_COLOR = '#e3e7ea'
HAZARD_COLOR = '#f8c9c4'
NEAR_HAZARD_COLOR = to_rgba('#e57373', 0.15)
TAPE_DARK, TAPE_LIGHT = '#212121', '#fdd835'
ROCK_COLOR = '#a1887f'
ROCK_EDGE_COLOR = '#5d4037'
RULED_OUT_COLOR = '#d7d7d7'
POND_COLOR = '#4fc3f7'
POND_EDGE_COLOR = '#0277bd'
LIFE_COLOR = '#43a047'
STEM_COLOR = '#2e7d32'
ALERT_COLOR = '#ff8f00'
ROVER_COLOR = '#1565c0'
WHEEL_COLOR = '#263238'
DAMAGED_COLOR = '#c62828'
DAMAGED_FILL = '#ffcdd2'

# where the objects are placed inside a cell, as a fraction of its size, and how large
# their rocks can be; the rovers drive along the bottom of the cell
OBJECT_AREA = (0.08, 0.36, 0.84, 0.5)
MAX_ROCK_RADIUS = 0.19
ROVER_BOTTOM = 0.06

# each entry of the legend takes this much room across, in inches
LEGEND_SPACING = 1.45


def _rock_outline(x, y, radius, seed):

    # an irregular boulder, the same for the same seed, flatter at its bottom
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + rng.uniform(0, 0.4)
    radii = radius * rng.uniform(0.78, 1.0, len(angles))
    points = np.column_stack([x + radii * np.cos(angles), y + 0.85 * radii * np.sin(angles)])
    points[:, 1] = np.maximum(points[:, 1], y - 0.6 * radius)
    return points


class ReconVisualizer(BaseViz):
    '''Draws the grid as ground with the base as a landing pad and hazards behind hazard
    tape, the agents as rovers, and the objects as rocks that show what the tools have
    found on them: a pond under a rock where water was found, a sprout on it where life
    was found, and a camera once it was photographed. Each tool is shown with whether it
    is damaged.

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
        self._rocks = {}
        for (col, row, x, y) in self._cells:
            here = [o for o in self._objects if self._nonfluent('objAt', o, x, y)]
            self._rocks.update(self._place_rocks(here, col, row))

        # title above, then the grid, its coordinates, the tools and the legend
        self._tool_cols = min(tools_per_row, max(len(self._tools), 1))
        tool_rows = math.ceil(len(self._tools) / self._tool_cols)
        grid_width = len(self._xs) * self._cell_size
        grid_height = len(self._ys) * self._cell_size
        tools_height = 0.4 * tool_rows
        labels_height = 0.3
        width = max(grid_width + 0.8, min_width)

        # the legend has as many entries in a row as fit, each this wide in inches
        self._legend_per_row = max(1, int((width - 0.3) / LEGEND_SPACING))
        legend_rows = math.ceil(len(self._legend_entries()) / self._legend_per_row)
        legend_height = 0.2 + 0.36 * legend_rows
        height = legend_height + tools_height + labels_height + grid_height + 0.45
        self._tools_size = (width, tools_height)

        self.fig = plt.figure(figsize=(width, height))
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
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        self._tools_ax = self.fig.add_axes(
            [0, legend_height / height, 1, tools_height / height])
        self._tools_ax.set_xlim(0, width)
        self._tools_ax.set_ylim(0, tools_height)
        self._tools_ax.set_aspect('equal')
        self._tools_ax.axis('off')

        # the ground, base, hazards and rocks never change, so they are only drawn once
        for (col, row, x, y) in self._cells:
            self._draw_cell(col, row, x, y)
        for (obj, (x, y, radius)) in self._rocks.items():
            self._add_name(x, y - 0.95 * radius, obj, radius)
        self._draw_legend(width, legend_height, height)

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

    def _place_rocks(self, objects, col, row):
        if not objects:
            return {}

        # use the arrangement that leaves the most room for each rock, with room above it
        # for a sprout and below it for its name
        left, bottom, width, height = OBJECT_AREA
        cols = max(range(1, len(objects) + 1), key=lambda c: min(
            width / c, height / math.ceil(len(objects) / c)))
        rows = math.ceil(len(objects) / cols)
        radius = min(MAX_ROCK_RADIUS, 0.36 * min(width / cols, height / rows))

        rocks = {}
        for (i, obj) in enumerate(objects):
            r, c = divmod(i, cols)
            per_row = min(cols, len(objects) - r * cols)
            x = col + left + width * (c + 0.5 + (cols - per_row) / 2) / cols
            y = row + bottom + height * (rows - r - 0.5) / rows
            rocks[obj] = (x, y, radius)
        return rocks

    def _is_near_hazard(self, x, y):

        # this mirrors the damage in the MDP, which applies along each axis separately
        return any(self._nonfluent('HAZARD', x2, y) and (
                   self._nonfluent('ADJACENT-LEFT', x, x2) or self._nonfluent('ADJACENT-RIGHT', x, x2))
                   for x2 in self._xs) or \
               any(self._nonfluent('HAZARD', x, y2) and (
                   self._nonfluent('ADJACENT-UP', y, y2) or self._nonfluent('ADJACENT-DOWN', y, y2))
                   for y2 in self._ys)

    def _add_name(self, x, y, name, radius):
        size = min(8, 2.6 * radius * self._cell_size * 72 / max(len(name), 1))
        if size >= 5:
            self.ax.text(x, y, name, ha='center', va='top', fontsize=size, color=TEXT_COLOR,
                         zorder=6)

    # ---------- pictures: each takes the axes, a place and a size, and returns its artists

    @staticmethod
    def _hazard_tape(ax, left, bottom, size, width):

        # black and yellow stripes along the inside of a square
        inset = 0.035 * size
        xs = [left + inset, left + size - inset, left + size - inset, left + inset, left + inset]
        ys = [bottom + inset, bottom + inset, bottom + size - inset, bottom + size - inset,
              bottom + inset]
        return [ax.plot(xs, ys, color=TAPE_DARK, linewidth=width, solid_joinstyle='miter',
                        zorder=1.5)[0],
                ax.plot(xs, ys, color=TAPE_LIGHT, linewidth=width, dashes=(1.2, 1.2),
                        zorder=1.6)[0]]

    @staticmethod
    def _landing_pad(ax, x, y, radius):
        return [ax.add_patch(patches.Circle(
                    (x, y), radius, facecolor='none', edgecolor=PAD_COLOR, linewidth=2,
                    zorder=1.5)),
                ax.add_patch(patches.Circle(
                    (x, y), 0.7 * radius, facecolor='none', edgecolor=PAD_COLOR, linewidth=1,
                    linestyle='--', zorder=1.5)),
                ax.plot([x - 0.35 * radius, x + 0.35 * radius], [y, y], color=PAD_COLOR,
                        linewidth=2, zorder=1.5)[0],
                ax.plot([x, x], [y - 0.35 * radius, y + 0.35 * radius], color=PAD_COLOR,
                        linewidth=2, zorder=1.5)[0]]

    @staticmethod
    def _rock(ax, x, y, radius, seed, ruled_out=False):
        artists = [ax.add_patch(patches.Polygon(
            _rock_outline(x, y, radius, seed),
            facecolor=RULED_OUT_COLOR if ruled_out else ROCK_COLOR,
            edgecolor='#9e9e9e' if ruled_out else ROCK_EDGE_COLOR, linewidth=1.2, zorder=4))]
        if ruled_out:
            for sign in (-1, 1):
                artists.extend(ax.plot([x - 0.45 * radius, x + 0.45 * radius],
                                       [y - sign * 0.4 * radius, y + sign * 0.4 * radius],
                                       color='#757575', linewidth=1.5, zorder=4.5))
        return artists

    @staticmethod
    def _pond(ax, x, y, radius):

        # water spreads around the bottom of a rock, with a glint on it
        return [ax.add_patch(patches.Ellipse(
                    (x, y - 0.45 * radius), 2.6 * radius, 0.9 * radius, facecolor=POND_COLOR,
                    edgecolor=POND_EDGE_COLOR, linewidth=1, zorder=3)),
                ax.add_patch(patches.Arc(
                    (x - 0.8 * radius, y - 0.45 * radius), 0.6 * radius, 0.3 * radius,
                    theta1=200, theta2=320, color='white', linewidth=1.5, zorder=3.1))]

    @staticmethod
    def _sprout(ax, x, y, radius):

        # a stem with two leaves growing out of the top of a rock
        top = y + 0.6 * radius
        artists = ax.plot([x, x], [top, top + 0.75 * radius], color=STEM_COLOR, linewidth=1.8,
                          zorder=5)
        for (side, angle) in ((-1, 30), (1, -30)):
            artists.append(ax.add_patch(patches.Ellipse(
                (x + side * 0.3 * radius, top + 0.65 * radius), 0.6 * radius, 0.28 * radius,
                angle=angle, facecolor=LIFE_COLOR, edgecolor=STEM_COLOR, linewidth=0.8,
                zorder=5)))
        return artists

    @staticmethod
    def _camera(ax, x, y, size):

        # a small camera: a body, a viewfinder on top and a lens
        return [ax.add_patch(patches.FancyBboxPatch(
                    (x - size, y - 0.6 * size), 2 * size, 1.2 * size,
                    boxstyle=f'round,pad=0,rounding_size={0.25 * size}',
                    facecolor=ALERT_COLOR, edgecolor='white', linewidth=0.8, zorder=6)),
                ax.add_patch(patches.Rectangle(
                    (x - 0.4 * size, y + 0.6 * size), 0.8 * size, 0.3 * size,
                    facecolor=ALERT_COLOR, edgecolor='none', zorder=6)),
                ax.add_patch(patches.Circle(
                    (x, y), 0.4 * size, facecolor='white', edgecolor=EDGE_COLOR,
                    linewidth=0.8, zorder=6.1))]

    @staticmethod
    def _droplet(ax, x, y, size, color):
        t = np.linspace(0, 2 * np.pi, 40)
        xs = x + size * np.sin(t) * np.sin(t / 2) ** 1.2
        ys = y + size * 1.1 * (np.cos(t) - 0.3)
        return [ax.add_patch(patches.Polygon(np.column_stack([xs, ys]), facecolor=color,
                                             edgecolor='none', zorder=6))]

    @staticmethod
    def _rover(ax, x, y, size, name=None, font_size=7):

        # a body on three wheels, with a mast and a camera head
        width, height = 1.8 * size, 0.7 * size
        artists = [ax.add_patch(patches.FancyBboxPatch(
            (x - width / 2, y + 0.35 * size), width, height,
            boxstyle=f'round,pad=0,rounding_size={0.2 * size}', facecolor=ROVER_COLOR,
            edgecolor='white', linewidth=0.8, zorder=5))]
        for k in (-1, 0, 1):
            artists.append(ax.add_patch(patches.Circle(
                (x + k * 0.65 * size, y + 0.3 * size), 0.27 * size, facecolor=WHEEL_COLOR,
                edgecolor='white', linewidth=0.6, zorder=5.1)))
        mast = x + 0.5 * size
        artists.extend(ax.plot([mast, mast], [y + 1.05 * size, y + 1.55 * size],
                               color=ROVER_COLOR, linewidth=1.5, zorder=5))
        artists.append(ax.add_patch(patches.Rectangle(
            (mast - 0.22 * size, y + 1.5 * size), 0.44 * size, 0.25 * size,
            facecolor=ROVER_COLOR, edgecolor='none', zorder=5)))
        if name is not None:
            artists.append(ax.text(x - 0.15 * size, y + 0.7 * size, name, ha='center',
                                   va='center', fontsize=font_size, color='white', zorder=6))
        return artists

    # ---------- the grid

    def _draw_cell(self, col, row, x, y):
        base = self._nonfluent('BASE', x, y)
        hazard = self._nonfluent('HAZARD', x, y)

        # the base protects tools from damage in the MDP, but not in the POMDP
        if self._tests_rule_out and base:
            hazard = False
        near = self._tests_rule_out and not base and self._is_near_hazard(x, y)

        color = BASE_COLOR if base else (
            HAZARD_COLOR if hazard else (NEAR_HAZARD_COLOR if near else GROUND_COLOR))
        if near:
            self.ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=GROUND_COLOR,
                                                edgecolor='none', zorder=0.9))
        self.ax.add_patch(patches.Rectangle(
            (col, row), 1, 1, facecolor=color, edgecolor=GRID_COLOR, zorder=1))
        if base:
            self._landing_pad(self.ax, col + 0.5, row + 0.5, 0.32)
        if hazard:
            self._hazard_tape(self.ax, col, row, 1, min(7.0, max(3.0, 5 * self._cell_size)))

    def _draw_object(self, state, obj):
        x, y, radius = self._rocks[obj]
        water = self._get(state, 'waterDetected', obj)
        life = self._get(state, 'lifeDetected', obj)

        # a failed water test or a second failed life test can never succeed afterwards
        ruled_out = self._tests_rule_out and not life and (
            self._get(state, 'lifeChecked2', obj)
            or (self._get(state, 'waterChecked', obj) and not water))

        artists = []
        if water:
            artists.extend(self._pond(self.ax, x, y, radius))
        artists.extend(self._rock(self.ax, x, y, radius, self._objects.index(obj), ruled_out))
        if life:
            artists.extend(self._sprout(self.ax, x, y, radius))
        if self._get(state, 'pictureTaken', obj):
            artists.extend(self._camera(self.ax, x + 0.85 * radius, y + 0.55 * radius,
                                        0.32 * radius))

        # the first failed life test leaves one more attempt
        if self._tests_rule_out and not life and not ruled_out \
                and self._get(state, 'lifeChecked', obj):
            artists.append(self.ax.add_patch(patches.Circle(
                (x - 0.75 * radius, y + 0.55 * radius), 0.22 * radius, facecolor=ALERT_COLOR,
                edgecolor='white', linewidth=1, zorder=6)))
        return artists

    def _draw_agents(self, col, row, agents):
        artists = []
        size = min(0.15, 0.4 / len(agents))
        for (i, agent) in enumerate(agents):
            x = col + (i + 0.5) / len(agents)

            # one agent needs no name, as there is nothing to tell it apart from
            name = agent if len(self._agents) > 1 else None
            artists.extend(self._rover(
                self.ax, x, row + ROVER_BOTTOM, size, name,
                min(7, 30 * size * self._cell_size)))
        return artists

    # ---------- the tools and the legend

    def _tool_icon(self, ax, tool, x, y, size, color):

        # the kind of each tool as its picture: a droplet, a sprout or a camera
        artists = []
        kinds = [var for var in ('WATER_TOOL', 'LIFE_TOOL', 'CAMERA_TOOL')
                 if self._nonfluent(var, tool)]
        for (k, kind) in enumerate(kinds):
            cx = x + (k - (len(kinds) - 1) / 2) * 2.4 * size
            if kind == 'WATER_TOOL':
                artists.extend(self._droplet(ax, cx, y, size, color or POND_COLOR))
            elif kind == 'LIFE_TOOL':
                artists.extend(self._sprout(ax, cx, y - 1.1 * size, 1.4 * size))
            else:
                artists.extend(self._camera(ax, cx, y, 0.9 * size))
        return artists, len(kinds)

    def _draw_tools(self, state):

        # each tool is a tile with its picture: a damaged one turns red, with its picture
        # faded and a crack across it; names only tell apart tools of the same kind
        artists = []
        width, height = self._tools_size
        tile = 0.62
        kinds = {t: tuple(self._nonfluent(v, t) for v in ('WATER_TOOL', 'LIFE_TOOL', 'CAMERA_TOOL'))
                 for t in self._tools}
        for (i, tool) in enumerate(self._tools):
            r, c = divmod(i, self._tool_cols)
            per_row = min(self._tool_cols, len(self._tools) - r * self._tool_cols)
            x = (width - per_row * (tile + 0.15)) / 2 + c * (tile + 0.15)
            y = height - 0.4 * (r + 1)
            damaged = self._get(state, 'damaged', tool)
            artists.append(self._tools_ax.add_patch(patches.FancyBboxPatch(
                (x, y + 0.03), tile, 0.34, boxstyle='round,pad=0,rounding_size=0.06',
                facecolor=DAMAGED_FILL if damaged else '#fafafa',
                edgecolor=DAMAGED_COLOR if damaged else GRID_COLOR, linewidth=1.5)))
            (icons, _) = self._tool_icon(self._tools_ax, tool, x + tile / 2, y + 0.2, 0.12, None)
            for icon in icons:
                icon.set_alpha(0.35 if damaged else 1.0)
            artists.extend(icons)
            if sum(k == kinds[tool] for k in kinds.values()) > 1:
                artists.append(self._tools_ax.text(
                    x + tile - 0.04, y + 0.06, tool, ha='right', va='bottom', fontsize=7,
                    color=TEXT_COLOR))
            if damaged:
                cx = x + tile / 2
                artists.extend(self._tools_ax.plot(
                    [cx - 0.18, cx - 0.04, cx - 0.1, cx + 0.06, cx + 0.18],
                    [y + 0.34, y + 0.24, y + 0.17, y + 0.1, y + 0.05],
                    color=DAMAGED_COLOR, linewidth=2))
        return artists

    def _legend_entries(self):
        entries = [('Base', 'base'), ('Hazard', 'hazard')]
        if self._tests_rule_out:
            entries.append(('Near hazard', 'near'))
        entries += [('Rover', 'rover'), ('Rock', 'rock'), ('Water found', 'water'),
                    ('Life found', 'life'), ('Photo taken', 'photo')]
        if self._tests_rule_out:
            entries += [('Ruled out', 'ruled'), ('Life test failed', 'failed')]
        entries.append(('Damaged tool', 'damaged'))
        return entries

    def _draw_legend(self, width, legend_height, height):

        # the legend shows the same pictures as the grid
        ax = self.fig.add_axes([0, 0, 1, legend_height / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, legend_height)
        ax.set_aspect('equal')
        ax.axis('off')
        entries = self._legend_entries()
        per_row = self._legend_per_row
        spacing = LEGEND_SPACING
        for (k, (label, kind)) in enumerate(entries):
            r, c = divmod(k, per_row)
            count = min(per_row, len(entries) - r * per_row)
            x = (width - count * spacing) / 2 + c * spacing + 0.15
            y = legend_height - 0.28 - 0.36 * r
            s = 0.13
            if kind in ('base', 'hazard', 'near'):
                color = {'base': BASE_COLOR, 'hazard': HAZARD_COLOR, 'near': NEAR_HAZARD_COLOR}[kind]
                ax.add_patch(patches.Rectangle((x - s, y - s), 2 * s, 2 * s, facecolor=color,
                                               edgecolor=GRID_COLOR, zorder=1))
                if kind == 'base':
                    self._landing_pad(ax, x, y, 0.8 * s)
                elif kind == 'hazard':
                    self._hazard_tape(ax, x - s, y - s, 2 * s, 2.5)
            elif kind == 'rover':
                self._rover(ax, x, y - s, 0.11)
            elif kind == 'damaged':
                ax.add_patch(patches.FancyBboxPatch(
                    (x - s, y - 0.75 * s), 2 * s, 1.5 * s, boxstyle='round,pad=0,rounding_size=0.03',
                    facecolor=DAMAGED_FILL, edgecolor=DAMAGED_COLOR, linewidth=1.2))
                ax.plot([x - 0.7 * s, x - 0.1 * s, x - 0.3 * s, x + 0.7 * s],
                        [y + 0.75 * s, y + 0.2 * s, y - 0.1 * s, y - 0.75 * s],
                        color=DAMAGED_COLOR, linewidth=1.5)
            else:
                radius = 0.09
                if kind == 'water':
                    self._pond(ax, x, y, radius)
                self._rock(ax, x, y, radius, 3, kind == 'ruled')
                if kind == 'life':
                    self._sprout(ax, x, y, radius)
                elif kind == 'photo':
                    self._camera(ax, x + 0.85 * radius, y + 0.55 * radius, 0.4 * radius)
                elif kind == 'failed':
                    ax.add_patch(patches.Circle((x - 0.75 * radius, y + 0.55 * radius),
                                                0.3 * radius, facecolor=ALERT_COLOR,
                                                edgecolor='white', linewidth=1, zorder=6))
            ax.text(x + 0.22, y, label, ha='left', va='center', fontsize=8.5, color=TEXT_COLOR)

    def _get(self, state, var, *args):
        return bool(state[RDDLPlanningModel.ground_var(var, list(args))])

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        artists = []
        for obj in self._rocks:
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
