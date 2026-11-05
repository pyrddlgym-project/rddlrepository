import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

EMPTY_COLOR = '#ffffff'
EMPTY_EDGE_COLOR = '#b0bec5'
NATIVE_COLOR = '#c8e6c9'
NATIVE_EDGE_COLOR = '#2e7d32'
TAMARISK_COLOR = '#ffcdd2'
TAMARISK_EDGE_COLOR = '#c62828'
COMPETING_COLOR = '#ffe0b2'
COMPETING_EDGE_COLOR = '#ff8f00'
REACH_COLOR = '#eceff1'
REACH_EDGE_COLOR = '#90a4ae'
INVADED_COLOR = '#ffebee'
RIVER_COLOR = '#64b5f6'
TRUNK_COLOR = '#6d4c41'
DETAIL_COLOR = '#37474f'

# sizes of the river in inches before it is scaled to fit the figure
CELL_SIZE = 0.55
CELL_GAP = 0.04
NAME_HEIGHT = 0.26
PANEL_PAD = 0.06
RIVER_GAP = 0.34
ROW_GAP = 0.3

# more reaches than this at one distance from the source continue on another column
MAX_COLUMN_REACHES = 3

# slots that are not on any reach are shown together, and a reach name cannot have a space
NO_REACH = 'No reach'

# text smaller than this is not readable and plants smaller than this are not recognizable,
# so they are left out
MIN_FONT_SIZE = 5
MIN_PLANT_SIZE = 0.2


class TamariskVisualizer(BaseViz):
    '''Draws the river flowing from left to right, with each reach as a panel of its slots
    and each slot coloured by the plants growing there.

    The river comes from the relation between reaches rather than their names, so it can
    fork and merge. Eradicating and restoring are not part of the state, so they only show
    as the plants changing afterwards. The POMDP shows the true plants rather than only the
    noisy observations of each reach. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_scale=1.5,
                 max_size=(9.0, 5.5),
                 min_width=8.2,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._reaches = list(model.type_to_objects['reach'])
        self._slots = list(model.type_to_objects['slot'])
        self._flows = [(up, down) for up in self._reaches for down in self._reaches
                       if self._nonfluent('DOWNSTREAM-REACH', down, up)]

        self._panel_slots = {reach: [slot for slot in self._slots
                                     if self._nonfluent('SLOT-AT-REACH', slot, reach)]
                             for reach in self._reaches}
        placed = {slot for slots in self._panel_slots.values() for slot in slots}
        unplaced = [slot for slot in self._slots if slot not in placed]
        if unplaced:
            self._panel_slots[NO_REACH] = unplaced

        # a panel taller than it is wide, as the panels of a long river are side by side
        most = max(max(map(len, self._panel_slots.values()), default=1), 1)
        self._slot_cols = math.ceil(most / math.ceil(math.sqrt(2 * most)))
        self._slot_rows = math.ceil(most / self._slot_cols)
        self._panel_width = self._slot_cols * CELL_SIZE + 2 * PANEL_PAD
        self._panel_height = NAME_HEIGHT + self._slot_rows * CELL_SIZE + PANEL_PAD

        # lay out the river, then scale it to fit the figure, where a river that needs to
        # go around other panels also has a row gap above the top row and a gap at the right
        self._places = self._make_places()
        num_cols = max((col for (col, _) in self._places.values()), default=0) + 1
        num_rows = max((row for (_, row) in self._places.values()), default=0) + 1
        self._routed = [(up, down) for (up, down) in self._flows
                        if self._places[down][0] != self._places[up][0] + 1]
        river_width = num_cols * self._panel_width + (num_cols - 1) * RIVER_GAP
        river_height = num_rows * self._panel_height + (num_rows - 1) * ROW_GAP
        extra_right, extra_top = (RIVER_GAP, ROW_GAP) if self._routed else (0, 0)
        total_width, total_height = river_width + 2 * extra_right, river_height + extra_top
        scale = min(max_scale, max_size[0] / total_width, max_size[1] / total_height)

        # title above, then the river, the legend and the caption
        bottom, top = 0.62, 0.45
        width = max(total_width * scale + 0.4, min_width)
        height = total_height * scale + 0.1 + bottom + top
        self._figure_size = (width, height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, bottom / height, 1, (height - bottom - top) / height])
        half_width, half_height = width / (2 * scale), (height - bottom - top) / (2 * scale)
        center_y = (extra_top - river_height) / 2
        self.ax.set_xlim(river_width / 2 - half_width, river_width / 2 + half_width)
        self.ax.set_ylim(center_y - half_height, center_y + half_height)
        self.ax.axis('off')

        for (up, down) in self._flows:
            self.ax.add_patch(patches.FancyArrowPatch(
                path=self._flow_path(up, down), arrowstyle='-|>', mutation_scale=13 * scale,
                linewidth=2.5 * min(scale, 1), color=RIVER_COLOR, shrinkA=0, shrinkB=0, zorder=1))

        # the panels and slots never move, so they are only drawn once and then recolored
        self._has_plants = (CELL_SIZE - CELL_GAP) * scale >= MIN_PLANT_SIZE
        self._panels, self._names, self._cells = {}, {}, []
        name_fit = 0.9 * self._panel_width * scale * 72 / 0.62
        most_size = min(9 * min(scale, 1), 0.65 * NAME_HEIGHT * scale * 72)
        for (panel, slots) in self._panel_slots.items():
            (left, panel_top) = self._panel_corner(panel)
            self._panels[panel] = self.ax.add_patch(patches.FancyBboxPatch(
                (left, panel_top - self._panel_height), self._panel_width, self._panel_height,
                boxstyle='round,pad=0,rounding_size=0.06', zorder=2, linewidth=1.3))
            if most_size >= MIN_FONT_SIZE:
                name = self._shorten(panel, int(name_fit / MIN_FONT_SIZE))
                self._names[panel] = self.ax.text(
                    left + self._panel_width / 2, panel_top - NAME_HEIGHT / 2, name, ha='center',
                    va='center', fontsize=max(min(most_size, name_fit / len(name)), MIN_FONT_SIZE),
                    fontweight='bold', zorder=3)
            for (i, slot) in enumerate(slots):
                (row, col) = divmod(i, self._slot_cols)
                x = left + PANEL_PAD + col * CELL_SIZE + CELL_GAP / 2
                y = panel_top - NAME_HEIGHT - (row + 1) * CELL_SIZE + CELL_GAP / 2
                cell = self.ax.add_patch(patches.Rectangle(
                    (x, y), CELL_SIZE - CELL_GAP, CELL_SIZE - CELL_GAP, linewidth=1, zorder=3))
                self._cells.append((slot, x, y, cell))

        self.fig.legend(handles=[
            patches.Patch(facecolor=EMPTY_COLOR, edgecolor=EMPTY_EDGE_COLOR, label='Empty'),
            patches.Patch(facecolor=NATIVE_COLOR, edgecolor=NATIVE_EDGE_COLOR, label='Native'),
            patches.Patch(facecolor=TAMARISK_COLOR, edgecolor=TAMARISK_EDGE_COLOR, label='Tamarisk'),
            patches.Patch(facecolor=COMPETING_COLOR, edgecolor=COMPETING_EDGE_COLOR, label='Competing'),
            patches.Patch(facecolor=INVADED_COLOR, edgecolor=TAMARISK_EDGE_COLOR, linewidth=2, label='Invaded reach'),
            Line2D([], [], color=RIVER_COLOR, linewidth=2.5, marker='>', markersize=5, label='River flow')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / height), ncol=6, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)

        down_rate = self._nonfluent('DOWNSTREAM-SPREAD-RATE')
        up_rate = self._nonfluent('UPSTREAM-SPREAD-RATE')
        reach_cost = self._nonfluent('COST-PER-INVADED-REACH')
        self.fig.text(0.5, 0.1 / height,
                      f'Tamarisk spreads downstream with chance {down_rate:g} and upstream with '
                      f'chance {up_rate:g}, and each invaded reach costs {reach_cost:g} per step',
                      ha='center', va='bottom', fontsize=9, color=DETAIL_COLOR)

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _make_places(self):

        # each reach goes to the right of all reaches flowing into it, and a cycle has no
        # such place so the one reach missing the fewest of them goes first to break it
        upstream = {reach: [up for (up, down) in self._flows if down == reach and up != reach]
                    for reach in self._reaches}
        levels, placed = [], set()
        remaining = list(self._reaches)
        while remaining:
            missing = {reach: sum(up not in placed for up in upstream[reach]) for reach in remaining}
            level = [reach for reach in remaining if missing[reach] == 0]
            if not level:
                level = [min(remaining, key=missing.get)]
            remaining = [reach for reach in remaining if reach not in level]
            placed.update(level)
            levels.append(level)
        if NO_REACH in self._panel_slots:
            levels.append([NO_REACH])

        # each reach stays as level as it can with the reaches flowing into it
        places, col = {}, 0
        for level in levels:
            for start in range(0, len(level), MAX_COLUMN_REACHES):
                part = level[start:start + MAX_COLUMN_REACHES]
                wanted = {}
                for (i, panel) in enumerate(part):
                    rows = [places[up][1] for up in upstream.get(panel, []) if up in places]
                    wanted[panel] = np.mean(rows) if rows else i
                row = -1
                for panel in sorted(part, key=wanted.get):
                    row = max(row + 1, int(round(wanted[panel])))
                    places[panel] = (col, row)
                col += 1
        return places

    def _panel_corner(self, panel):
        (col, row) = self._places[panel]
        return (col * (self._panel_width + RIVER_GAP), -row * (self._panel_height + ROW_GAP))

    def _flow_path(self, up, down):
        (left, top) = self._panel_corner(up)
        (down_left, down_top) = self._panel_corner(down)
        start = (left + self._panel_width, top - self._panel_height / 2)

        # the next column over is reached directly through the gap between them
        if self._places[down][0] == self._places[up][0] + 1:
            end = (down_left, down_top - self._panel_height / 2)
            middle = (start[0] + end[0]) / 2
            return Path([start, (middle, start[1]), (middle, end[1]), end],
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])

        # any other reach is around the panels, along the gaps between columns and rows, with
        # each flow on its own lane so that the flows sharing a gap can still be told apart
        lane = (self._routed.index((up, down)) + 1) / (len(self._routed) + 1)
        gap_x = start[0] + RIVER_GAP * lane
        gap_y = down_top + ROW_GAP * lane
        end_x = down_left + self._panel_width * lane
        return Path([start, (gap_x, start[1]), (gap_x, gap_y), (end_x, gap_y), (end_x, down_top)],
                    [Path.MOVETO] + [Path.LINETO] * 4)

    def _draw_plant(self, native, x, y, size):

        # a round native tree and a feathery tamarisk plant, standing at the given place
        trunk_width = 0.09 * size
        artists = [self.ax.add_patch(patches.Rectangle(
            (x - trunk_width / 2, y), trunk_width, 0.42 * size, facecolor=TRUNK_COLOR,
            edgecolor='none', zorder=4))]
        if native:
            artists.append(self.ax.add_patch(patches.Circle(
                (x, y + 0.62 * size), 0.3 * size, facecolor=NATIVE_EDGE_COLOR,
                edgecolor='none', zorder=5)))
        else:
            for angle in (-52, -26, 0, 26, 52):
                radians = math.radians(angle)
                center = (x + 0.46 * size * math.sin(radians), y + 0.25 * size + 0.46 * size * math.cos(radians))
                artists.append(self.ax.add_patch(patches.Ellipse(
                    center, 0.15 * size, 0.62 * size, angle=-angle, facecolor=TAMARISK_EDGE_COLOR,
                    edgecolor='none', zorder=5)))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        tamarisk = {slot for slot in self._slots
                    if state[RDDLPlanningModel.ground_var('tamarisk-at', [slot])]}
        native = {slot for slot in self._slots
                  if state[RDDLPlanningModel.ground_var('native-at', [slot])]}

        artists = []
        side = CELL_SIZE - CELL_GAP
        for (slot, x, y, cell) in self._cells:
            if slot in tamarisk and slot in native:
                fill, edge = COMPETING_COLOR, COMPETING_EDGE_COLOR
            elif slot in tamarisk:
                fill, edge = TAMARISK_COLOR, TAMARISK_EDGE_COLOR
            elif slot in native:
                fill, edge = NATIVE_COLOR, NATIVE_EDGE_COLOR
            else:
                fill, edge = EMPTY_COLOR, EMPTY_EDGE_COLOR
            cell.set_facecolor(fill)
            cell.set_edgecolor(edge)

            if self._has_plants:
                if slot in tamarisk and slot in native:
                    artists.extend(self._draw_plant(True, x + 0.28 * side, y + 0.15 * side, 0.55 * side))
                    artists.extend(self._draw_plant(False, x + 0.72 * side, y + 0.15 * side, 0.55 * side))
                elif slot in tamarisk or slot in native:
                    artists.extend(self._draw_plant(slot in native, x + 0.5 * side, y + 0.08 * side, 0.85 * side))

        invaded = [reach for reach in self._reaches
                   if any(slot in tamarisk for slot in self._panel_slots[reach])]
        for (panel, patch) in self._panels.items():
            is_invaded = any(slot in tamarisk for slot in self._panel_slots[panel])
            patch.set_facecolor(INVADED_COLOR if is_invaded else REACH_COLOR)
            patch.set_edgecolor(TAMARISK_EDGE_COLOR if is_invaded else REACH_EDGE_COLOR)
            patch.set_linewidth(2 if is_invaded else 1.3)
            if panel in self._names:
                self._names[panel].set_color(TAMARISK_EDGE_COLOR if is_invaded else 'black')

        if not tamarisk:
            title, color = 'No tamarisk', NATIVE_EDGE_COLOR
        elif self._reaches and len(invaded) == len(self._reaches):
            title, color = 'All reaches invaded', TAMARISK_EDGE_COLOR
        else:
            title, color = f'{len(invaded)} / {len(self._reaches)} reaches invaded', 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
