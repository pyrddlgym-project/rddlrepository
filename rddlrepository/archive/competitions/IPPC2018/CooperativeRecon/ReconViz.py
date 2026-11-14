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
DONE_COLOR = '#2e7d32'
FAILED_COLOR = '#c62828'

# where the objects are placed inside a cell, as a fraction of its size, and how large
# their rocks can be; the rovers drive along the bottom of the cell
OBJECT_AREA = (0.08, 0.36, 0.84, 0.5)
MAX_ROCK_RADIUS = 0.19
ROVER_BOTTOM = 0.06

# cells where damaging a tool is more likely than not are hazards, and any other cell that
# can damage a tool is a risk
HAZARD_CHANCE = 0.5

# each rover has a row with its tools; sizes in inches
TOOL_TILE, TOOL_GAP = 0.62, 0.15
ROVER_SLOT = 0.45
TOOL_LINE = 0.4

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
    '''Draws the grid as ground with the base as a landing pad, the agents as rovers, and
    the objects as rocks that show what the tools have found on them: a pond under a rock
    where water was found, a sprout on it where life was found, and a camera once it was
    photographed. Cells where a tool is more likely than not to be damaged are taped off
    as hazards, and other cells that can damage a tool are tinted. Each rover has a row
    with the tools it carries, showing whether each one is damaged.

    The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_cell_size=2.0,
                 max_size=5.5,
                 min_width=6.8,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._objects = model.type_to_objects['object-of-interest']
        self._agents = model.type_to_objects['agent']
        self._tools = model.type_to_objects['tool']

        # moving right and up follow the adjacency relations, whatever the objects are called
        self._xs = self._chain(model.type_to_objects['xpos'], 'ADJACENT_RIGHT', 'ADJACENT_LEFT')
        self._ys = self._chain(model.type_to_objects['ypos'], 'ADJACENT_UP', 'ADJACENT_DOWN')
        self._cells = [(col, row, x, y) for (col, x) in enumerate(self._xs)
                       for (row, y) in enumerate(self._ys)]

        # the objects never move, so each one has a fixed place in its cell
        self._cell_size = min(max_cell_size, max_size / max(len(self._xs), len(self._ys)))
        self._rocks = {}
        for (col, row, x, y) in self._cells:
            here = [o for o in self._objects if self._nonfluent('OBJECT_AT', o, x, y)]
            self._rocks.update(self._place_rocks(here, col, row))

        # title above, then the grid, its coordinates, the tools and the legend
        grid_width = len(self._xs) * self._cell_size
        grid_height = len(self._ys) * self._cell_size
        width = max(grid_width + 0.8, min_width)
        self._tool_rows = self._layout_tools(width)
        tools_height = TOOL_LINE * len(self._tool_rows)
        labels_height = 0.3

        # the legend has as many entries in a row as fit, each this wide in inches
        self._legend_per_row = max(1, int((width - 0.3) / LEGEND_SPACING))
        legend_rows = math.ceil(len(self._legend_entries()) / self._legend_per_row)
        legend_height = 0.2 + 0.36 * legend_rows
        height = legend_height + tools_height + labels_height + grid_height + 0.45
        self._tools_height = tools_height

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

        # the ground, base, hazards, rocks and rovers next to their tools never change,
        # so they are only drawn once
        for (col, row, x, y) in self._cells:
            self._draw_cell(col, row, x, y)
        for (obj, (x, y, radius)) in self._rocks.items():
            self._add_name(x, y - 0.95 * radius, obj, radius)
        self._draw_tool_rovers()
        self._draw_legend(width, legend_height, height)

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _chain(self, objects, forward, backward):

        # the relations name the cell that is moved to first, so that b comes after a when
        # moving forward from a reaches b or moving backward from b reaches a, and any object
        # left over because of a cycle or missing relation keeps its declared position
        following = {}
        for a in objects:
            for b in objects:
                if a != b and (self._nonfluent(forward, b, a) or self._nonfluent(backward, a, b)):
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

    def _layout_tools(self, width):

        # each rover has a row with its tools, and tools no rover carries have a row of
        # their own; a rover with more tools than fit across takes several rows
        groups = [(agent, [t for t in self._tools if self._nonfluent('HAS_TOOL', agent, t)])
                  for agent in self._agents]
        carried = {tool for (_, tools) in groups for tool in tools}
        spare = [tool for tool in self._tools if tool not in carried]
        if spare:
            groups.append((None, spare))

        per_row = max(1, int((width - 0.3 - ROVER_SLOT) / (TOOL_TILE + TOOL_GAP)))
        rows = []
        for (agent, tools) in groups:
            chunks = [tools[i:i + per_row] for i in range(0, len(tools), per_row)] or [[]]
            for (k, chunk) in enumerate(chunks):
                has_slot = agent is not None
                row_width = (ROVER_SLOT if has_slot else 0) + len(chunk) * (TOOL_TILE + TOOL_GAP)
                rows.append((agent if k == 0 else None, has_slot,
                             chunk, (width - row_width + TOOL_GAP) / 2))
        return rows

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
        chance = self._nonfluent('DAMAGE_PROB', x, y)
        hazard = chance >= HAZARD_CHANCE
        risk = 0 < chance < HAZARD_CHANCE

        color = BASE_COLOR if base else (
            HAZARD_COLOR if hazard else (NEAR_HAZARD_COLOR if risk else GROUND_COLOR))
        if risk and not base:
            self.ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor=GROUND_COLOR,
                                                edgecolor='none', zorder=0.9))
        self.ax.add_patch(patches.Rectangle(
            (col, row), 1, 1, facecolor=color, edgecolor=GRID_COLOR, zorder=1))
        if base:
            self._landing_pad(self.ax, col + 0.5, row + 0.5, 0.32)
        if hazard:
            self._hazard_tape(self.ax, col, row, 1, min(7.0, max(3.0, 5 * self._cell_size)))

    def _ruled_out(self, state, obj):

        # a failed water test or a second failed life test can never succeed afterwards
        return not self._get(state, 'lifeDetected', obj) and (
            self._get(state, 'lifeChecked2', obj)
            or (self._get(state, 'waterChecked', obj) and not self._get(state, 'waterDetected', obj)))

    def _draw_object(self, state, obj):
        x, y, radius = self._rocks[obj]
        water = self._get(state, 'waterDetected', obj)
        life = self._get(state, 'lifeDetected', obj)
        ruled_out = self._ruled_out(state, obj)

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
        if not life and not ruled_out and self._get(state, 'lifeChecked', obj):
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

    def _tool_icon(self, ax, tool, x, y, size):

        # the kind of each tool as its picture: a droplet, a sprout or a camera
        artists = []
        kinds = [var for var in ('WATER_TOOL', 'LIFE_TOOL', 'CAMERA_TOOL')
                 if self._nonfluent(var, tool)]
        for (k, kind) in enumerate(kinds):
            cx = x + (k - (len(kinds) - 1) / 2) * 2.4 * size
            if kind == 'WATER_TOOL':
                artists.extend(self._droplet(ax, cx, y, size, POND_COLOR))
            elif kind == 'LIFE_TOOL':
                artists.extend(self._sprout(ax, cx, y - 1.1 * size, 1.4 * size))
            else:
                artists.extend(self._camera(ax, cx, y, 0.9 * size))
        return artists

    def _row_bottom(self, row):
        return self._tools_height - TOOL_LINE * (row + 1)

    def _draw_tool_rovers(self):

        # a rover in front of its tools, named when there are several
        for (r, (agent, has_slot, tools, x)) in enumerate(self._tool_rows):
            if agent is not None:
                name = self._shorten(agent, 4) if len(self._agents) > 1 else None
                self._rover(self._tools_ax, x + ROVER_SLOT / 2 - 0.05,
                            self._row_bottom(r) + 0.06, 0.14, name, 6.5)

    def _draw_tools(self, state):

        # each tool is a tile with its picture: a damaged one turns red, with its picture
        # faded and a crack across it; names only tell apart tools of the same kind in a row
        artists = []
        ax = self._tools_ax
        kinds = {t: tuple(self._nonfluent(v, t) for v in ('WATER_TOOL', 'LIFE_TOOL', 'CAMERA_TOOL'))
                 for t in self._tools}
        for (r, (agent, has_slot, tools, row_x)) in enumerate(self._tool_rows):
            y = self._row_bottom(r)
            for (i, tool) in enumerate(tools):
                x = row_x + (ROVER_SLOT if has_slot else 0) + i * (TOOL_TILE + TOOL_GAP)
                damaged = self._get(state, 'damaged', tool)
                artists.append(ax.add_patch(patches.FancyBboxPatch(
                    (x, y + 0.03), TOOL_TILE, 0.34, boxstyle='round,pad=0,rounding_size=0.06',
                    facecolor=DAMAGED_FILL if damaged else '#fafafa',
                    edgecolor=DAMAGED_COLOR if damaged else GRID_COLOR, linewidth=1.5)))
                icons = self._tool_icon(ax, tool, x + TOOL_TILE / 2, y + 0.2, 0.12)
                for icon in icons:
                    icon.set_alpha(0.35 if damaged else 1.0)
                artists.extend(icons)
                if sum(kinds[t] == kinds[tool] for t in tools) > 1:
                    artists.append(ax.text(
                        x + TOOL_TILE - 0.04, y + 0.06, tool, ha='right', va='bottom',
                        fontsize=7, color=TEXT_COLOR))
                if damaged:
                    cx = x + TOOL_TILE / 2
                    artists.extend(ax.plot(
                        [cx - 0.18, cx - 0.04, cx - 0.1, cx + 0.06, cx + 0.18],
                        [y + 0.34, y + 0.24, y + 0.17, y + 0.1, y + 0.05],
                        color=DAMAGED_COLOR, linewidth=2))
        return artists

    def _legend_entries(self):
        return [('Base', 'base'), ('Hazard', 'hazard'), ('Damage risk', 'near'),
                ('Rover', 'rover'), ('Rock', 'rock'), ('Water found', 'water'),
                ('Life found', 'life'), ('Photo taken', 'photo'), ('Ruled out', 'ruled'),
                ('Life test failed', 'failed'), ('Damaged tool', 'damaged')]

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
            agents = [a for a in self._agents if self._get(state, 'agent-at', a, x, y)]
            if agents:
                artists.extend(self._draw_agents(col, row, agents))
        artists.extend(self._draw_tools(state))

        total = len(self._objects)
        photos = sum(self._get(state, 'pictureTaken', o) for o in self._objects)
        missing = [o for o in self._objects if not self._get(state, 'pictureTaken', o)]
        if not missing:
            color = DONE_COLOR
        elif all(self._ruled_out(state, o) for o in missing):
            color = FAILED_COLOR
        else:
            color = 'black'
        self.ax.set_title(f'{photos} / {total} objects photographed', fontsize=13,
                          fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
