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
DAMAGE_COLOR = '#c62828'
UNTESTED_COLOR = 'white'
WATER_COLOR = '#bbdefb'
LIFE_COLOR = '#66bb6a'
RULED_OUT_COLOR = '#90a4ae'
ALERT_COLOR = '#ff8f00'
TOOL_COLOR = '#fafafa'
TOOL_DAMAGED_COLOR = '#ffcdd2'
PANEL_COLOR = '#f5f7f8'
DONE_COLOR = '#2e7d32'
FAILED_COLOR = '#c62828'

# where the objects and agents are drawn inside of a cell, as a fraction of its size, with
# the chance of damage along the top
OBJECT_AREA = (0.06, 0.38, 0.88, 0.48)
MAX_TOKEN_RADIUS = 0.22
AGENT_AREA_BOTTOM = 0.05
DAMAGE_TEXT_HEIGHT = 0.14

# the shading of a cell for the chance of damage there
MIN_DAMAGE_ALPHA = 0.08
DAMAGE_ALPHA_RANGE = 0.5

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5

# sizes in inches
BOTTOM_MARGIN = 0.62
LABELS_HEIGHT = 0.28
TOP_MARGIN = 0.45
LEFT_MARGIN = 0.55
RIGHT_MARGIN = 0.2
SIDE_GAP = 0.25
SIDE_WIDTH = 3.1
SIDE_PAD = 0.08
HEADER_HEIGHT = 0.28
SECTION_GAP = 0.1
AGENT_NAME_WIDTH = 0.5
TOOLS_PER_ROW = 3
TOOL_ROW_HEIGHT = 0.27
OBJECT_ROW_HEIGHT = 0.21
OBJECT_MARKER_RADIUS = 0.065
MAX_OBJECT_COLUMNS = 4
MIN_REWARD_COLUMN = 1.2

# the row of the side panel for any tools that no agent carries
NO_AGENT = 'None'


class ReconVisualizer(BaseViz):
    '''Draws the grid with the base, the chance of damaging a tool in each cell, the agents,
    and the objects coloured by what the tools have found on them. A side panel shows the
    tools that each agent carries with their condition, and the rewards for a photo of
    each object.

    The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_cell_size=2.0,
                 max_size=5.2,
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
        self._tokens = {}
        for (col, row, x, y) in self._cells:
            here = [o for o in self._objects if self._nonfluent('OBJECT_AT', o, x, y)]
            self._tokens.update(self._place_tokens(here, col, row))

        # the tools are listed by the agent that carries them, and the objects get the rest
        # of the room next to the grid
        grid_width = len(self._xs) * self._cell_size
        grid_height = len(self._ys) * self._cell_size
        self._tool_rows = self._assign_tools()
        agents_height = HEADER_HEIGHT + sum(
            max(math.ceil(len(tools) / TOOLS_PER_ROW), 1) * TOOL_ROW_HEIGHT
            for tools in self._tool_rows.values()) + SIDE_PAD
        room = grid_height - agents_height - SECTION_GAP - HEADER_HEIGHT - SIDE_PAD
        per_column = max(int(room / OBJECT_ROW_HEIGHT), 1)
        self._object_cols = min(max(math.ceil(len(self._objects) / per_column), 1), MAX_OBJECT_COLUMNS)
        object_rows = math.ceil(len(self._objects) / self._object_cols)
        objects_height = HEADER_HEIGHT + object_rows * OBJECT_ROW_HEIGHT + SIDE_PAD
        self._agents_height = agents_height
        self._objects_height = objects_height
        side_height = agents_height + SECTION_GAP + objects_height

        # title above, then the grid next to the side panel, the legend and the caption
        content = max(grid_height, side_height)
        width = LEFT_MARGIN + grid_width + SIDE_GAP + SIDE_WIDTH + RIGHT_MARGIN
        height = BOTTOM_MARGIN + LABELS_HEIGHT + content + TOP_MARGIN
        self._figure_size = (width, height)
        self._title_x = (width / 2 - LEFT_MARGIN) / grid_width

        self.fig = plt.figure(figsize=self._figure_size)
        grid_bottom = BOTTOM_MARGIN + LABELS_HEIGHT + (content - grid_height) / 2
        self.ax = self.fig.add_axes([LEFT_MARGIN / width, grid_bottom / height,
                                     grid_width / width, grid_height / height])
        self.ax.set_xlim(0, len(self._xs))
        self.ax.set_ylim(0, len(self._ys))
        self.ax.set_aspect('equal')
        x_limit = int(self._cell_size * 72 / (0.6 * 9))
        y_limit = int((LEFT_MARGIN - 0.08) * 72 / (0.6 * 9))
        self.ax.set_xticks(np.arange(len(self._xs)) + 0.5,
                           [self._shorten(x, x_limit) for x in self._xs])
        self.ax.set_yticks(np.arange(len(self._ys)) + 0.5,
                           [self._shorten(y, y_limit) for y in self._ys])
        self.ax.tick_params(length=0, labelsize=9, colors='#616161')
        self.ax.set_facecolor('#fafafa')
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        self._side_ax = self.fig.add_axes(
            [(LEFT_MARGIN + grid_width + SIDE_GAP) / width,
             (BOTTOM_MARGIN + LABELS_HEIGHT) / height, SIDE_WIDTH / width, content / height])
        self._side_ax.set_xlim(0, SIDE_WIDTH)
        self._side_ax.set_ylim(0, content)
        self._side_ax.axis('off')
        self._side_top = content

        # the base, the damage and the grid never change, so they are only drawn once
        for (col, row, x, y) in self._cells:
            self._draw_cell(col, row, x, y)
        self._draw_side_panels()

        handles = self._legend_handles()
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.25 / height),
                        ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                        handlelength=1.4)

        self.fig.text(0.5, 0.1 / height,
                      f'Tests work {self._nonfluent("DETECT_PROB"):.0%} of the time '
                      f'({self._nonfluent("DETECT_PROB_DAMAGED"):.0%} with a damaged tool), '
                      f'{self._nonfluent("DETECT_PROB_WITH_SUPPORT"):.0%} '
                      f'({self._nonfluent("DETECT_PROB_DAMAGED_WITH_SUPPORT"):.0%}) with support. '
                      f'Cells show the chance of damaging each tool there',
                      ha='center', va='bottom', fontsize=9, color=EDGE_COLOR)

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

    def _assign_tools(self):
        rows = {agent: [tool for tool in self._tools if self._nonfluent('HAS_TOOL', agent, tool)]
                for agent in self._agents}
        carried = {tool for tools in rows.values() for tool in tools}
        spare = [tool for tool in self._tools if tool not in carried]
        if spare:
            rows[NO_AGENT] = spare
        return rows

    def _draw_cell(self, col, row, x, y):
        self.ax.add_patch(patches.Rectangle(
            (col, row), 1, 1, facecolor=BASE_COLOR if self._nonfluent('BASE', x, y) else 'none',
            edgecolor=GRID_COLOR, zorder=1))

        chance = self._nonfluent('DAMAGE_PROB', x, y)
        if chance > 0:
            alpha = MIN_DAMAGE_ALPHA + DAMAGE_ALPHA_RANGE * min(chance, 1)
            self.ax.add_patch(patches.Rectangle(
                (col, row), 1, 1, facecolor=to_rgba(DAMAGE_COLOR, alpha), edgecolor='none', zorder=1))
            size = min(8, DAMAGE_TEXT_HEIGHT * self._cell_size * 72)
            if size >= MIN_FONT_SIZE:
                self.ax.text(col + 0.96, row + 1 - DAMAGE_TEXT_HEIGHT / 2 - 0.01, f'{chance:.0%}',
                             ha='right', va='center', fontsize=size, color=DAMAGE_COLOR, zorder=2)

    def _draw_side_panels(self):
        ax = self._side_ax
        for (top, height) in ((self._side_top, self._agents_height),
                              (self._side_top - self._agents_height - SECTION_GAP, self._objects_height)):
            ax.add_patch(patches.FancyBboxPatch(
                (0, top - height), SIDE_WIDTH, height, boxstyle='round,pad=0,rounding_size=0.05',
                facecolor=PANEL_COLOR, edgecolor='none', zorder=0))
        ax.text(SIDE_WIDTH / 2, self._side_top - HEADER_HEIGHT / 2, 'Tools', ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=1)

        # the names of the agents next to their tools never change
        y = self._side_top - HEADER_HEIGHT
        name_limit = int(AGENT_NAME_WIDTH * 72 / (0.6 * 8))
        for (agent, tools) in self._tool_rows.items():
            ax.text(SIDE_PAD, y - TOOL_ROW_HEIGHT / 2, self._shorten(agent, name_limit), ha='left',
                    va='center', fontsize=8, color=EDGE_COLOR, zorder=1)
            if not tools:
                ax.text(SIDE_PAD + AGENT_NAME_WIDTH + 0.04, y - TOOL_ROW_HEIGHT / 2, 'No tools', ha='left',
                        va='center', fontsize=7.5, color=RULED_OUT_COLOR, zorder=1)
            y -= max(math.ceil(len(tools) / TOOLS_PER_ROW), 1) * TOOL_ROW_HEIGHT

        top = self._side_top - self._agents_height - SECTION_GAP
        ax.text(SIDE_PAD, top - HEADER_HEIGHT / 2, 'Objects', ha='left', va='center',
                fontsize=9, fontweight='bold', zorder=1)
        ax.text(SIDE_WIDTH - SIDE_PAD, top - HEADER_HEIGHT / 2, 'Photo reward (damaged camera)',
                ha='right', va='center', fontsize=7, color=EDGE_COLOR, zorder=1)

    def _legend_handles(self):

        def token(color, label, **kwargs):
            style = {'markeredgecolor': EDGE_COLOR, **kwargs}
            return Line2D([], [], marker='o', color='none', markerfacecolor=color,
                          markersize=9, label=label, **style)

        return [
            patches.Patch(facecolor=BASE_COLOR, edgecolor=GRID_COLOR, label='Base'),
            patches.Patch(facecolor=to_rgba(DAMAGE_COLOR, 0.33), edgecolor=GRID_COLOR, label='Damage chance'),
            Line2D([], [], marker='o', color='none', markerfacecolor=AGENT_COLOR,
                   markeredgecolor='white', markersize=9, label='Agent'),
            token(UNTESTED_COLOR, 'Not tested'),
            token(WATER_COLOR, 'Water found'),
            token(LIFE_COLOR, 'Life found'),
            token(RULED_OUT_COLOR, 'Ruled out'),
            Line2D([], [], marker='o', color='none', markerfacecolor=ALERT_COLOR,
                   markeredgecolor='white', markersize=7, label='Life test failed'),
            token('none', 'Photo taken', markeredgecolor=ALERT_COLOR, markeredgewidth=2)
        ]

    def _get(self, state, var, *args):
        return bool(state[RDDLPlanningModel.ground_var(var, list(args))])

    def _token_color(self, state, obj):
        if self._get(state, 'lifeDetected', obj):
            return LIFE_COLOR
        water = self._get(state, 'waterDetected', obj)

        # a failed water test or a second failed life test can never succeed afterwards
        if self._get(state, 'lifeChecked2', obj) or (self._get(state, 'waterChecked', obj) and not water):
            return RULED_OUT_COLOR
        return WATER_COLOR if water else UNTESTED_COLOR

    def _draw_token(self, ax, state, obj, x, y, radius, color, ring):
        artists = [ax.add_patch(patches.Circle(
            (x, y), radius, facecolor=color, edgecolor=EDGE_COLOR, linewidth=1.2, zorder=4))]
        if self._get(state, 'pictureTaken', obj):
            artists.append(ax.add_patch(patches.Circle(
                (x, y), radius, facecolor='none', edgecolor=ALERT_COLOR, linewidth=ring, zorder=5)))

        # the first failed life test leaves one more attempt
        if color not in (LIFE_COLOR, RULED_OUT_COLOR) and self._get(state, 'lifeChecked', obj):
            offset = radius * math.sqrt(0.5)
            artists.append(ax.add_patch(patches.Circle(
                (x + offset, y + offset), 0.28 * radius, facecolor=ALERT_COLOR,
                edgecolor='white', linewidth=1, zorder=7)))
        return artists

    def _draw_object(self, state, obj):
        x, y, radius = self._tokens[obj]
        color = self._token_color(state, obj)
        artists = self._draw_token(self.ax, state, obj, x, y, radius, color, 3)

        # the name must fit inside of the token, and the side panel has the full name
        room = 3.4 * radius * self._cell_size * 72
        artists.append(self.ax.text(
            x, y, self._shorten(obj, int(room / MIN_FONT_SIZE)), ha='center', va='center',
            fontsize=max(min(9, room / max(len(obj), 1)), MIN_FONT_SIZE), zorder=6,
            color='white' if color == RULED_OUT_COLOR else 'black'))
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
            size = min(7, 0.32 * scale * self._cell_size * 72 / (0.6 * max(len(agent), 1)))
            if len(self._agents) > 1 and size >= MIN_FONT_SIZE:
                artists.append(self.ax.text(
                    x, y + 0.06 * scale, agent, ha='center', va='center', fontsize=size,
                    color='white', zorder=5))
        return artists

    def _draw_tools(self, state):
        artists = []
        ax = self._side_ax
        chip_width = (SIDE_WIDTH - 2 * SIDE_PAD - AGENT_NAME_WIDTH) / TOOLS_PER_ROW
        y = self._side_top - HEADER_HEIGHT
        for tools in self._tool_rows.values():
            for (i, tool) in enumerate(tools):
                r, c = divmod(i, TOOLS_PER_ROW)
                x = SIDE_PAD + AGENT_NAME_WIDTH + c * chip_width
                chip_y = y - (r + 1) * TOOL_ROW_HEIGHT
                damaged = self._get(state, 'damaged', tool)
                kinds = [name for (var, name) in (('WATER_TOOL', 'Water'), ('LIFE_TOOL', 'Life'),
                                                  ('CAMERA_TOOL', 'Camera'))
                         if self._nonfluent(var, tool)]
                label = f'{" + ".join(kinds) or tool}{" damaged" if damaged else ""}'
                artists.append(ax.add_patch(patches.FancyBboxPatch(
                    (x + 0.02, chip_y + 0.03), chip_width - 0.04, TOOL_ROW_HEIGHT - 0.06,
                    boxstyle='round,pad=0,rounding_size=0.03', linewidth=0.8, zorder=1,
                    facecolor=TOOL_DAMAGED_COLOR if damaged else TOOL_COLOR,
                    edgecolor=DAMAGE_COLOR if damaged else GRID_COLOR)))
                room = (chip_width - 0.08) * 72 / 0.62
                artists.append(ax.text(
                    x + chip_width / 2, chip_y + TOOL_ROW_HEIGHT / 2,
                    self._shorten(label, int(room / MIN_FONT_SIZE)),
                    ha='center', va='center', zorder=2,
                    fontsize=max(min(7, room / len(label)), MIN_FONT_SIZE),
                    color=DAMAGE_COLOR if damaged else EDGE_COLOR))
            y -= max(math.ceil(len(tools) / TOOLS_PER_ROW), 1) * TOOL_ROW_HEIGHT
        return artists

    def _draw_object_rows(self, state):
        artists = []
        ax = self._side_ax
        column_width = (SIDE_WIDTH - 2 * SIDE_PAD) / self._object_cols
        top = self._side_top - self._agents_height - SECTION_GAP - HEADER_HEIGHT
        rows = math.ceil(len(self._objects) / self._object_cols)
        has_reward = column_width >= MIN_REWARD_COLUMN
        reward_room = 0.8 if has_reward else 0
        name_left = 3.4 * OBJECT_MARKER_RADIUS
        name_limit = int((column_width - name_left - reward_room) * 72 / (0.6 * 8))
        for (i, obj) in enumerate(self._objects):
            c, r = divmod(i, rows)
            x = SIDE_PAD + c * column_width
            y = top - (r + 0.5) * OBJECT_ROW_HEIGHT
            color = self._token_color(state, obj)
            artists.extend(self._draw_token(ax, state, obj, x + OBJECT_MARKER_RADIUS + 0.01, y,
                                            OBJECT_MARKER_RADIUS, color, 1.6))

            # a photo of an object that was photographed or ruled out is no longer possible
            done = color == RULED_OUT_COLOR or self._get(state, 'pictureTaken', obj)
            text_color = RULED_OUT_COLOR if done else 'black'
            artists.append(ax.text(
                x + name_left, y, self._shorten(obj, name_limit), ha='left',
                va='center', fontsize=8, color=text_color, zorder=2))
            if has_reward:
                good = self._nonfluent('GOOD_PIC_REWARD', obj)
                bad = self._nonfluent('BAD_PIC_REWARD', obj)
                artists.append(ax.text(
                    x + column_width - 0.1, y, f'{good:.3g} ({bad:.3g})', ha='right', va='center',
                    fontsize=7.5, color=RULED_OUT_COLOR if done else EDGE_COLOR, zorder=2))
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
            agents = [a for a in self._agents if self._get(state, 'agent-at', a, x, y)]
            if agents:
                artists.extend(self._draw_agents(col, row, agents))
        artists.extend(self._draw_tools(state))
        artists.extend(self._draw_object_rows(state))

        total = len(self._objects)
        photos = sum(self._get(state, 'pictureTaken', o) for o in self._objects)
        missing = [o for o in self._objects if not self._get(state, 'pictureTaken', o)]
        plural = '' if total == 1 else 's'
        if not missing:
            title, color = f'{total} / {total} object{plural} photographed', DONE_COLOR
        elif all(self._token_color(state, o) == RULED_OUT_COLOR for o in missing):
            title, color = f'{photos} / {total} object{plural} photographed, the rest ruled out', FAILED_COLOR
        else:
            title, color = f'{photos} / {total} object{plural} photographed', 'black'
        self.ax.set_title(title, x=self._title_x, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
