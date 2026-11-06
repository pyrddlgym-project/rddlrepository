import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

PLATE_COLOR = '#f5f7f8'
PLATE_EDGE_COLOR = '#cfd8dc'
ROOM_COLOR = '#ffffff'
LINE_COLOR = '#90a4ae'
GOOD_COLOR = '#2e7d32'
ACTIVE_COLOR = '#ff8f00'
BAD_COLOR = '#c62828'
DETAIL_COLOR = '#37474f'
HOSPITAL_COLOR = '#1565c0'
UNCOUNTED_COLOR = '#eceff1'

# recovered sits next to infected when nobody is in hospital, and the darker green is hard to
# tell from that red with red-green colour blindness
RECOVERED_COLOR = '#66bb6a'

# the state fluents of a group in the order that people pass through them
COMPARTMENTS = ('S', 'E', 'I', 'H', 'R', 'D')
COMPARTMENT_COLORS = ('#b0bec5', ACTIVE_COLOR, BAD_COLOR, HOSPITAL_COLOR, RECOVERED_COLOR, DETAIL_COLOR)
COMPARTMENT_NAMES = ('Susceptible', 'Exposed', 'Infected', 'Hospital', 'Recovered', 'Dead')

# sizes in inches, before the scene is scaled to fit the figure
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
MIN_WIDTH = 8.6
SCENE_WIDTH = 8.8
ROW_WIDTH = 6.1
ROW_HEIGHT = 1.0
MAX_ROW_HEIGHT = 1.6
ROW_GAP = 0.08
NAME_WIDTH = 1.05
CROWD_X = 1.15
CROWD_WIDTH = 2.25
VALUES_X = 3.47
VALUE_WIDTH = 0.62
CURVE_X = 4.74
CURVE_WIDTH = 1.26
HOSPITAL_X = 6.2
HOSPITAL_WIDTH = 2.6
HOSPITAL_HEIGHT = 3.2
BED_SIZE = (0.36, 0.19)
BED_PITCH = (0.44, 0.24)
PATIENT_PITCH = 0.16
TILE_SIZE = (1.05, 0.7)
TILE_GAP = 0.08

# the crowd of each group, where one person stands for 1% of it
NUM_CROWD_COLS = 20
NUM_CROWD_ROWS = 5
CROWD_SIZE = NUM_CROWD_COLS * NUM_CROWD_ROWS
MAX_CROWD_PITCH = 0.17

# the ward, where a full ward is exactly the capacity of the hospital
NUM_BED_COLS = 5
NUM_BED_ROWS = 4
NUM_BEDS = NUM_BED_COLS * NUM_BED_ROWS
MAX_OUTSIDE_PATIENTS = 12

# a smaller share of a group is reported as text, but is not worth drawing a person for
VISIBLE_SHARE = 0.001

# less than this much of a bed is the tail of a decay, and not worth drawing a patient for
VISIBLE_BED = 0.01

# rows scaled more than this are too small to read, so groups are drawn as tiles
MIN_ROW_SCALE = 0.6

# the thinnest band in a stacked bar that still shows on every tile
MIN_BAND_PIXELS = 1.5

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5
NAME_FONT_SIZE = 10
TEXT_FONT_SIZE = 7.5
CHAR_WIDTH = 0.56
NAME_CHAR_WIDTH = 0.72
BOLD_CHAR_WIDTH = 0.62
LINE_HEIGHT = 1.3


class EpidemicVisualizer(BaseViz):
    '''Draws a crowd for each age group, where every person is 1% of the group coloured by the
    compartment they are in, in the order that people pass through the compartments. Next to it
    are the exact shares and how the infected share of the group has changed so far, on one
    scale for all groups.

    The hospital is shared by the groups. A full ward is its capacity, patients beyond it wait
    outside, and its load is the sum over the groups that the overload cost uses. The domain has
    no group sizes, so this is not a share of the population.

    The compartments of a group need not add up to 1, so what is left is drawn as pale people
    instead of being rescaled away. Too many groups to draw like this show as tiles with one
    stacked bar each.

    The actions are not part of the state, so vaccination, distancing and school closures are
    not shown. The history is built from the states seen so far, so it requires render to be
    called on every step. The state does not hold the time, so it is the number of different
    states seen.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_size=(9.0, 5.5),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._groups = list(model.type_to_objects['group'])
        self._capacity = float(self._value('hosp_capacity'))
        self._overload_cost = float(self._value('overload_cost'))
        self._bed_share = (self._capacity if self._capacity > 0 else 1.0) / NUM_BEDS
        self._horizon = model.horizon
        self._artists = []

        # the state is only compared in this order, to notice repeats and new episodes
        self._state_keys = [RDDLPlanningModel.ground_var(var, [group])
                            for group in self._groups for var in COMPARTMENTS]
        init_state = model.ground_vars_with_values(model.state_fluents)
        self._init_state = tuple(float(init_state[key]) for key in self._state_keys)
        self._last_state = None
        self._load_history = []
        self._infected_history = {group: [] for group in self._groups}

        # rows if these stay readable, otherwise tiles beside the hospital
        count = len(self._groups)
        height = max(count * ROW_HEIGHT, HOSPITAL_HEIGHT)
        self._scale = min(1.0, max_size[0] / SCENE_WIDTH, max_size[1] / height)
        self._has_rows = self._scale >= MIN_ROW_SCALE
        if not self._has_rows:
            self._scale = min(1.0, max_size[0] / SCENE_WIDTH, max_size[1] / HOSPITAL_HEIGHT)
            height = max_size[1] / self._scale

        # title above, then the scene, the legend and the caption
        grid_width = SCENE_WIDTH * self._scale
        grid_height = height * self._scale
        width = max(grid_width + 0.4, MIN_WIDTH)
        figure_height = grid_height + BOTTOM_MARGIN + TOP_MARGIN
        self._figure_size = (width, figure_height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width), BOTTOM_MARGIN / figure_height,
             grid_width / width, grid_height / figure_height])
        self.ax.set_xlim(0, SCENE_WIDTH)
        self.ax.set_ylim(0, height)
        self.ax.axis('off')

        # the rooms never move, so they are only drawn once and then filled in
        if self._has_rows:
            self._rows = {}
            self._row_height = min(height / count, MAX_ROW_HEIGHT)
            for (i, group) in enumerate(self._groups):
                self._rows[group] = self._draw_row(group, height - (i + 1) * self._row_height)
        else:
            self._draw_tiles(height)
        self._hospital = self._draw_hospital(height)

        handles = [patches.Patch(facecolor=color, edgecolor='none', label=name)
                   for (color, name) in zip(COMPARTMENT_COLORS, COMPARTMENT_NAMES)]
        self.fig.legend(handles=handles, loc='lower center',
                        bbox_to_anchor=(0.5, 0.25 / figure_height), ncol=len(handles),
                        frameon=False, fontsize=8.5, columnspacing=1.0, handlelength=1.4)
        first = 'Each person is 1% of their group' if self._has_rows else self._tile_caption
        self.fig.text(0.5, 0.1 / figure_height,
                      f'{first}. Hospital load adds up the share of each group in hospital',
                      ha='center', va='bottom', fontsize=9, color=DETAIL_COLOR)

    def _value(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _font(self, size):
        return max(size * self._scale, MIN_FONT_SIZE)

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    # ==========================================================================
    # WHAT NEVER MOVES
    # ==========================================================================

    def _draw_row(self, group, y):
        left, bottom = 0.0, y + ROW_GAP / 2
        height = self._row_height - ROW_GAP
        top = bottom + height
        frame = self.ax.add_patch(patches.FancyBboxPatch(
            (left, bottom), ROW_WIDTH, height, boxstyle='round,pad=0,rounding_size=0.07',
            facecolor=PLATE_COLOR, edgecolor=PLATE_EDGE_COLOR, linewidth=1.4, zorder=1))

        # the name, how the group is doing and whether school closures reach it on the left
        name_x = left + 0.1
        limit = int((NAME_WIDTH - 0.1) * 72 / (NAME_CHAR_WIDTH * NAME_FONT_SIZE))
        self.ax.text(name_x, top - 0.18, self._shorten(group, limit), ha='left', va='center',
                     fontsize=self._font(NAME_FONT_SIZE), fontweight='bold', zorder=3)
        status = self.ax.text(name_x, top - 0.38, '', ha='left', va='center', fontweight='bold',
                              fontsize=self._font(TEXT_FONT_SIZE), zorder=3)
        uncounted = self.ax.text(name_x, top - 0.56, '', ha='left', va='center', color=DETAIL_COLOR,
                                 fontsize=self._font(TEXT_FONT_SIZE), zorder=3)
        if self._value('affect_by_school_shutdown', group):
            self._draw_school(name_x, bottom + 0.12)

        room_bottom, room_height = bottom + 0.06, height - 0.12
        self.ax.add_patch(patches.FancyBboxPatch(
            (CROWD_X, room_bottom), CROWD_WIDTH, room_height, boxstyle='round,pad=0,rounding_size=0.05',
            facecolor=ROOM_COLOR, edgecolor=LINE_COLOR, linewidth=1.0, zorder=2))
        crowd = self._draw_crowd(room_bottom, room_height)

        # the exact shares, with a swatch that ties each one to the crowd and the legend
        values = []
        for (i, var) in enumerate(COMPARTMENTS):
            row, col = i % 3, i // 3
            value_x = VALUES_X + col * VALUE_WIDTH
            value_y = bottom + height / 2 + (1 - row) * 0.24
            self.ax.add_patch(patches.Rectangle(
                (value_x, value_y - 0.04), 0.08, 0.08, facecolor=COMPARTMENT_COLORS[i],
                edgecolor='none', zorder=3))
            values.append(self.ax.text(value_x + 0.1, value_y, '', ha='left', va='center', zorder=3,
                                       fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR))

        # how the infected share has changed so far, on one scale for all groups
        curve_top = top - 0.24
        box = (CURVE_X + 0.04, bottom + 0.1, CURVE_WIDTH - 0.08, curve_top - bottom - 0.14)
        self.ax.text(CURVE_X, top - 0.13, 'Infected', ha='left', va='center', zorder=3,
                     fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR)
        peak = self.ax.text(CURVE_X + CURVE_WIDTH, top - 0.13, '', ha='right', va='center', zorder=3,
                            fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR)
        line = self._draw_curve_box(box, BAD_COLOR)
        return {'frame': frame, 'status': status, 'uncounted': uncounted, 'crowd': crowd,
                'values': values, 'peak': peak, 'line': line, 'box': box}

    def _draw_school(self, x, y):
        self.ax.add_patch(patches.Rectangle(
            (x, y - 0.05), 0.14, 0.08, facecolor=LINE_COLOR, edgecolor='none', zorder=3))
        self.ax.add_patch(patches.Polygon(
            [(x - 0.015, y + 0.03), (x + 0.07, y + 0.09), (x + 0.155, y + 0.03)],
            closed=True, facecolor=DETAIL_COLOR, edgecolor='none', zorder=3))
        self.ax.text(x + 0.2, y, 'In school', ha='left', va='center', color=DETAIL_COLOR,
                     fontsize=self._font(TEXT_FONT_SIZE), zorder=3)

    def _draw_crowd(self, bottom, height):

        # filled a column at a time, so that the compartments read as bands from left to right
        pitch_x = CROWD_WIDTH / NUM_CROWD_COLS
        pitch_y = min(height / NUM_CROWD_ROWS, MAX_CROWD_PITCH)
        size = min(pitch_x, pitch_y) * 0.5
        lowest = bottom + (height - NUM_CROWD_ROWS * pitch_y) / 2
        bodies, heads = [], []
        for col in range(NUM_CROWD_COLS):
            for row in range(NUM_CROWD_ROWS):
                x = CROWD_X + (col + 0.5) * pitch_x
                foot = lowest + (NUM_CROWD_ROWS - row - 1) * pitch_y + 0.2 * pitch_y
                bodies.append(patches.Wedge((x, foot), size * 0.85, 0, 180))
                heads.append(patches.Circle((x, foot + size * 1.32), size * 0.46))
        crowd = PatchCollection(bodies + heads, edgecolor='none', zorder=4)
        return self.ax.add_collection(crowd)

    def _draw_curve_box(self, box, color):
        (x, y, width, height) = box
        self.ax.add_patch(patches.Rectangle(
            (x, y), width, height, facecolor=ROOM_COLOR, edgecolor=LINE_COLOR, linewidth=0.8, zorder=2))
        line = Line2D([], [], color=color, linewidth=1.3, marker='o', markersize=3, markevery=[-1],
                      zorder=4)
        self.ax.add_line(line)
        return line

    def _draw_tiles(self, height):

        # as many columns as give the largest tiles
        count = len(self._groups)
        best = None
        for cols in range(1, count + 1):
            rows = math.ceil(count / cols)
            size = min(ROW_WIDTH / (cols * TILE_SIZE[0]), height / (rows * TILE_SIZE[1]))
            if best is None or size >= best[0]:
                best = (size, cols, rows)
        (size, num_cols, num_rows) = best
        tile_width, tile_height = TILE_SIZE[0] * size, TILE_SIZE[1] * size
        left = (ROW_WIDTH - num_cols * tile_width) / 2
        top = (height + num_rows * tile_height) / 2

        # text only when it can be read, and otherwise only the bar
        gap = min(TILE_GAP, 0.2 * tile_width)
        inner_width, inner_height = (tile_width - gap) * self._scale, (tile_height - gap) * self._scale
        name_fit = 0.9 * inner_width * 72 / BOLD_CHAR_WIDTH
        line = LINE_HEIGHT * MIN_FONT_SIZE
        has_name = name_fit / MIN_FONT_SIZE >= 3 and inner_height * 72 >= 2.6 * line
        has_value = has_name and inner_height * 72 >= 3.6 * line and \
            inner_width * 72 >= len('I 100.0%') * CHAR_WIDTH * MIN_FONT_SIZE
        bar_share = 0.3 if has_name else 0.7

        frames, self._bars, self._tile_values = [], [], {}
        content = min(size, 1.0) * self._scale
        font = max(TEXT_FONT_SIZE * content, MIN_FONT_SIZE)
        self._tile_has_dead = has_value and \
            inner_width * 72 >= len('I 100.0%  D 100.0%') * CHAR_WIDTH * font
        self._tile_caption = 'Each bar is a whole group'
        if has_value:
            self._tile_caption += ', with its infected (I) and dead (D) shares' if self._tile_has_dead \
                else ', with its infected (I) share'
        for (i, group) in enumerate(self._groups):
            row, col = divmod(i, num_cols)
            x = left + col * tile_width + gap / 2
            y = top - (row + 1) * tile_height + gap / 2
            box_width, box_height = tile_width - gap, tile_height - gap
            frames.append(patches.FancyBboxPatch(
                (x, y), box_width, box_height, facecolor=PLATE_COLOR, edgecolor=PLATE_EDGE_COLOR,
                boxstyle=f'round,pad=0,rounding_size={min(0.07, 0.2 * box_width)}'))

            inset = 0.08 * box_width
            bar_height = bar_share * box_height
            bar_y = y + 0.1 * box_height if has_name else y + (box_height - bar_height) / 2
            self._bars.append((x + inset, bar_y, box_width - 2 * inset, bar_height))
            if has_name:
                name = self._shorten(group, int(name_fit / MIN_FONT_SIZE))
                center = x + box_width / 2
                self.ax.text(center, y + box_height * (0.82 if has_value else 0.72), name, ha='center',
                             va='center', fontweight='bold', zorder=3,
                             fontsize=max(min(NAME_FONT_SIZE * content, name_fit / len(name)),
                                          MIN_FONT_SIZE))
            if has_value:
                self._tile_values[group] = self.ax.text(
                    center, y + box_height * 0.56, '', ha='center', va='center', fontsize=font,
                    color=DETAIL_COLOR, zorder=3)

        self._tile_frames = self.ax.add_collection(PatchCollection(
            frames, match_original=True, linewidth=min(1.2, tile_width * 72 / 30), zorder=1))

        # one collection for every bar, so that many groups stay quick to draw
        colors = list(COMPARTMENT_COLORS) + [UNCOUNTED_COLOR]
        self._bar_polys = self.ax.add_collection(PolyCollection(
            np.zeros((count * len(colors), 4, 2)), facecolors=colors * count, edgecolors='none', zorder=3))

    def _draw_hospital(self, height):
        left = HOSPITAL_X
        panel_bottom = ROW_GAP / 2
        panel_top = height - ROW_GAP / 2
        frame = self.ax.add_patch(patches.FancyBboxPatch(
            (left, panel_bottom), HOSPITAL_WIDTH, panel_top - panel_bottom,
            boxstyle='round,pad=0,rounding_size=0.07', facecolor=PLATE_COLOR, edgecolor=PLATE_EDGE_COLOR,
            linewidth=1.4, zorder=1))

        # the building with its sign and ward
        wall_x, wall_width = left + 0.12, HOSPITAL_WIDTH - 0.24
        wall_top = panel_top - 0.08
        sign_height = 0.26
        ward_height = NUM_BED_ROWS * BED_PITCH[1] + 0.1
        wall_bottom = wall_top - sign_height - ward_height
        self.ax.add_patch(patches.Rectangle(
            (wall_x, wall_bottom), wall_width, sign_height + ward_height, facecolor=ROOM_COLOR,
            edgecolor=LINE_COLOR, linewidth=1.0, zorder=2))
        self.ax.add_patch(patches.Rectangle(
            (wall_x, wall_top - sign_height), wall_width, sign_height, facecolor=HOSPITAL_COLOR,
            edgecolor='none', zorder=3))
        cross_x, cross_y = wall_x + 0.16, wall_top - sign_height / 2
        for (w, h) in ((0.15, 0.05), (0.05, 0.15)):
            self.ax.add_patch(patches.Rectangle(
                (cross_x - w / 2, cross_y - h / 2), w, h, facecolor=ROOM_COLOR, edgecolor='none', zorder=4))
        self.ax.text(cross_x + 0.14, cross_y, 'Hospital', ha='left', va='center', color=ROOM_COLOR,
                     fontweight='bold', fontsize=self._font(NAME_FONT_SIZE), zorder=4)

        beds = []
        if self._capacity > 0:
            ward_x = wall_x + (wall_width - NUM_BED_COLS * BED_PITCH[0]) / 2
            for i in range(NUM_BEDS):
                row, col = divmod(i, NUM_BED_COLS)
                x = ward_x + col * BED_PITCH[0] + (BED_PITCH[0] - BED_SIZE[0]) / 2
                y = wall_top - sign_height - 0.07 - (row + 1) * BED_PITCH[1] + (BED_PITCH[1] - BED_SIZE[1])
                beds.append(self._draw_bed(x, y))
        else:
            self.ax.text(wall_x + wall_width / 2, wall_bottom + ward_height / 2, 'No beds', ha='center',
                         va='center', color=DETAIL_COLOR, fontsize=self._font(NAME_FONT_SIZE), zorder=3)

        # how full it is and who is waiting outside
        text_x = wall_x
        load = self.ax.text(text_x, wall_bottom - 0.13, '', ha='left', va='center', color=DETAIL_COLOR,
                            fontsize=self._font(TEXT_FONT_SIZE), zorder=3)
        status = self.ax.text(text_x, wall_bottom - 0.31, '', ha='left', va='center', fontweight='bold',
                              fontsize=self._font(TEXT_FONT_SIZE), zorder=3)
        outside_y = wall_bottom - 0.52

        # how the load has changed so far, against the capacity
        self.ax.text(text_x, wall_bottom - 0.72, 'Load', ha='left', va='center', color=DETAIL_COLOR,
                     fontsize=self._font(TEXT_FONT_SIZE), zorder=3)
        step = self.ax.text(wall_x + wall_width, wall_bottom - 0.72, '', ha='right', va='center',
                            color=DETAIL_COLOR, fontsize=self._font(TEXT_FONT_SIZE), zorder=3)
        box_top = wall_bottom - 0.82
        box_bottom = panel_bottom + 0.1
        box = (wall_x, box_bottom, wall_width, box_top - box_bottom)
        line = self._draw_curve_box(box, HOSPITAL_COLOR)
        capacity_line, capacity_label = None, None
        if self._capacity > 0:
            capacity_line = Line2D([], [], color=BAD_COLOR, linestyle='--', linewidth=1.0, zorder=3)
            self.ax.add_line(capacity_line)
            capacity_label = self.ax.text(
                box[0] + box[2] - 0.04, 0.0, 'Capacity', ha='right', va='bottom', color=DETAIL_COLOR,
                fontsize=self._font(TEXT_FONT_SIZE), zorder=5)
        return {'frame': frame, 'beds': beds, 'load': load, 'status': status, 'outside': (text_x, outside_y),
                'step': step, 'box': box, 'line': line, 'capacity_line': capacity_line,
                'capacity_label': capacity_label}

    def _draw_bed(self, x, y):
        (width, height) = BED_SIZE
        self.ax.add_patch(patches.Rectangle(
            (x, y), width, height, facecolor=PLATE_COLOR, edgecolor=LINE_COLOR, linewidth=0.7, zorder=3))
        self.ax.add_patch(patches.Rectangle(
            (x + 0.02, y + 0.03), 0.09, height - 0.06, facecolor=ROOM_COLOR, edgecolor=LINE_COLOR,
            linewidth=0.5, zorder=4))
        patient = [
            self.ax.add_patch(patches.Circle(
                (x + 0.065, y + height / 2), 0.04, facecolor=HOSPITAL_COLOR, edgecolor='none', zorder=5)),
            self.ax.add_patch(patches.Rectangle(
                (x + 0.12, y + 0.02), width - 0.14, height - 0.04, facecolor=HOSPITAL_COLOR,
                edgecolor='none', zorder=5))
        ]
        return patient

    # ==========================================================================
    # WHAT MOVES
    # ==========================================================================

    def _update_history(self, values, load, infected):

        # rendering the same state again must not change the history
        if values == self._last_state:
            return
        if self._last_state is None or values == self._init_state:
            self._load_history = []
            self._infected_history = {group: [] for group in self._groups}
        self._last_state = values
        self._load_history.append(load)
        if self._has_rows:
            for group in self._groups:
                self._infected_history[group].append(infected[group])

    def _people(self, shares):
        '''The number of people for each compartment and then those in no compartment, which add
        up to the crowd. Anything worth seeing keeps at least one person, however small.'''

        # compartments adding up to more than 1 can only be drawn by their proportions
        total = sum(shares)
        if total > 1:
            shares = [share / total for share in shares]
        shares = list(shares) + [max(1 - sum(shares), 0.0)]

        counts = [max(int(share * CROWD_SIZE), 1 if share >= VISIBLE_SHARE else 0) for share in shares]
        order = sorted((i for i in range(len(shares)) if shares[i] >= VISIBLE_SHARE),
                       key=lambda i: counts[i] - shares[i] * CROWD_SIZE)
        spare = CROWD_SIZE - sum(counts)
        for k in range(max(spare, 0)):
            counts[order[k % len(order)]] += 1
        for _ in range(max(-spare, 0)):
            counts[counts.index(max(counts))] -= 1
        return counts

    def _percent(self, share):
        if share <= 0:
            return '0%'
        return '<0.1%' if share < 0.001 else f'{100 * share:.1f}%'

    def _x_end(self, count):

        # the horizon when there is one, so that the curve also shows how much time is left
        end = max(count - 1, 1)
        if self._horizon is not None and math.isfinite(self._horizon):
            end = max(end, self._horizon)
        return end

    def _plot(self, line, box, history, top):
        (x, y, width, height) = box
        times = np.arange(len(history)) / self._x_end(len(history))
        line.set_data(x + width * times, y + height * np.asarray(history, dtype=float) / top)

    def _fill_row(self, row, group, shares, is_active, top):
        counts = self._people(shares)
        colors = []
        for (count, color) in zip(counts, list(COMPARTMENT_COLORS) + [UNCOUNTED_COLOR]):
            colors.extend([color] * count)
        row['crowd'].set_facecolor(colors + colors)

        for (text, var, share) in zip(row['values'], COMPARTMENTS, shares):
            text.set_text(f'{var} {self._percent(share)}')

        if is_active:
            row['status'].set(text='Active', color=ACTIVE_COLOR)
            row['frame'].set_edgecolor(ACTIVE_COLOR)
        else:
            row['status'].set(text='Clear', color=GOOD_COLOR)
            row['frame'].set_edgecolor(GOOD_COLOR)

        missing = 1 - sum(shares)
        row['uncounted'].set_text(
            f'{self._percent(missing)} uncounted' if missing >= VISIBLE_SHARE else '')

        row['peak'].set_text(f'top {100 * top:.3g}%')
        self._plot(row['line'], row['box'], self._infected_history[group], top)

    def _bands(self, shares, width):
        '''The share of a bar for each compartment and then those in no compartment, where anything
        worth seeing stays wide enough to show.'''
        total = sum(shares)
        bands = [share / total for share in shares] if total > 1 else list(shares)
        bands.append(max(1 - sum(bands), 0.0))
        thinnest = MIN_BAND_PIXELS / (width * self._scale * self.fig.dpi)
        bands = [max(band, thinnest) if band >= VISIBLE_SHARE else band for band in bands]
        total = sum(bands)
        return [band / total for band in bands]

    def _fill_tiles(self, all_shares, edges):
        polys = []
        for ((x, y, width, height), shares) in zip(self._bars, all_shares):
            left = x
            for segment in self._bands(shares, width):
                right = left + segment * width
                polys.append([(left, y), (right, y), (right, y + height), (left, y + height)])
                left = right
        self._bar_polys.set_verts(polys)
        self._tile_frames.set_edgecolor(edges)
        for (group, text) in self._tile_values.items():
            shares = all_shares[self._groups.index(group)]
            label = f'I {self._percent(shares[COMPARTMENTS.index("I")])}'
            if self._tile_has_dead:
                label += f'  D {self._percent(shares[COMPARTMENTS.index("D")])}'
            text.set_text(label)

    def _draw_outside_patient(self, x, y):
        size = PATIENT_PITCH * 0.5
        foot = y - size * 0.9
        self._artists.append(self.ax.add_patch(patches.Wedge(
            (x, foot), size * 0.72, 0, 180, facecolor=BAD_COLOR, edgecolor='none', zorder=4)))
        self._artists.append(self.ax.add_patch(patches.Circle(
            (x, foot + size * 1.14), size * 0.38, facecolor=BAD_COLOR, edgecolor='none', zorder=4)))

    def _fill_hospital(self, load):
        hospital = self._hospital
        over = max(load - self._capacity, 0.0)

        # a patient for each bed that is at least partly used, so that a full ward means capacity
        used = load / self._bed_share
        num_used = math.ceil(used) if used >= VISIBLE_BED else 0
        for (i, bed) in enumerate(hospital['beds']):
            for artist in bed:
                artist.set_visible(i < num_used)

        hospital['load'].set_text(f'Load {load:.3g} of {self._capacity:g}')
        if over > 0:
            cost = self._overload_cost * over
            hospital['status'].set(text=f'Over by {over:.3g}, costs {cost:.3g} a step', color=BAD_COLOR)
            hospital['frame'].set_edgecolor(BAD_COLOR)
        else:
            hospital['status'].set(text='Within capacity', color=GOOD_COLOR)
            hospital['frame'].set_edgecolor(GOOD_COLOR)

        # those beyond the capacity wait outside, at the same value per person as a bed
        num_outside = math.ceil(over / self._bed_share)
        (x, y) = hospital['outside']
        shown = min(num_outside, MAX_OUTSIDE_PATIENTS)
        for i in range(shown):
            self._draw_outside_patient(x + 0.08 + i * PATIENT_PITCH, y)
        if num_outside > shown:
            self._artists.append(self.ax.text(
                x + 0.04 + shown * PATIENT_PITCH, y, f'+{num_outside - shown:.3g}', ha='left', va='center',
                color=BAD_COLOR, fontsize=self._font(TEXT_FONT_SIZE), zorder=4))

        # the load so far on a scale that always holds both it and the capacity
        history = self._load_history
        top = 1.15 * max(max(history), self._capacity)
        top = top if top > 0 else 1.0
        self._plot(hospital['line'], hospital['box'], history, top)
        (box_x, box_y, box_width, box_height) = hospital['box']
        if hospital['capacity_line'] is not None:
            level = box_y + box_height * self._capacity / top
            hospital['capacity_line'].set_data([box_x, box_x + box_width], [level, level])

            # on the side of the line with room for it, inside the box
            hospital['capacity_label'].set_y(level)
            hospital['capacity_label'].set_va('top' if level > box_y + 0.6 * box_height else 'bottom')
        step = max(len(history) - 1, 0)
        end = self._x_end(len(history))
        hospital['step'].set_text(f'Step {step} / {end:g}')

    def _title(self, num_active, load):
        total = len(self._groups)
        over = load > self._capacity
        if over:
            hospital = 'hospital over capacity'
        elif self._capacity > 0:
            hospital = f'hospital {100 * load / self._capacity:.0f}% full'
        else:
            hospital = 'hospital has no beds'
        title = f'Infection active in {num_active} / {total} group{"" if total == 1 else "s"}, {hospital}'
        color = BAD_COLOR if over else (GOOD_COLOR if num_active == 0 else 'black')
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        all_shares = [[max(float(state[fluent(var, [group])]), 0.0) for var in COMPARTMENTS]
                      for group in self._groups]
        hospital_index = COMPARTMENTS.index('H')
        infected_index = COMPARTMENTS.index('I')
        exposed_index = COMPARTMENTS.index('E')
        load = sum(shares[hospital_index] for shares in all_shares)
        infected = {group: shares[infected_index] for (group, shares) in zip(self._groups, all_shares)}
        self._update_history(tuple(float(state[key]) for key in self._state_keys), load, infected)

        # active means someone can still pass the disease on or end up in hospital
        active = [shares[exposed_index] + shares[infected_index] + shares[hospital_index] >= VISIBLE_SHARE
                  for shares in all_shares]

        if self._has_rows:

            # one scale for all groups so that their curves compare
            peak = max((max(history, default=0.0) for history in self._infected_history.values()), default=0.0)
            top = max(1.15 * peak, VISIBLE_SHARE)
            for (group, shares, is_active) in zip(self._groups, all_shares, active):
                self._fill_row(self._rows[group], group, shares, is_active, top)
        else:
            self._fill_tiles(all_shares, [ACTIVE_COLOR if is_active else GOOD_COLOR for is_active in active])
        self._fill_hospital(load)
        self._title(sum(active), load)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in self._artists:
            artist.remove()
        self._artists = []
        return img
