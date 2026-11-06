import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

PLATE_COLOR = '#f5f7f8'
PLATE_EDGE_COLOR = '#cfd8dc'
ROOM_COLOR = '#ffffff'
LINE_COLOR = '#90a4ae'
WORKER_COLOR = '#1565c0'
SPARE_COLOR = '#90caf9'
TRAINEE_COLOR = '#ffb74d'
COVERED_COLOR = '#2e7d32'
SHORT_COLOR = '#c62828'
COVERED_FILL_COLOR = '#c8e6c9'
SHORT_FILL_COLOR = '#ffcdd2'
DOT_COLOR = '#cfd8dc'
CURRENT_COLOR = '#37474f'
BURST_COLOR = '#ff8f00'
DETAIL_COLOR = '#37474f'

# sizes in inches, before the rows are scaled to fit the figure
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
MIN_WIDTH = 8.6
ROW_WIDTH = 8.8
ROW_HEIGHT = 1.02
ROW_GAP = 0.08
NAME_WIDTH = 1.25
ROOM_WIDTH = 1.3
ARROW_WIDTH = 0.28
FLOOR_LABEL_WIDTH = 0.92
MONEY_WIDTH = 1.3
ICON_PITCH = 0.18
TILE_SIZE = (1.05, 0.8)
TILE_GAP = 0.08

# the number of icons a row can hold before each one has to stand for more than one
MAX_FLOOR_ICONS = 20
MAX_ROOM_ICONS = 5

# more months than this are not told apart as dots
MAX_CALENDAR_DOTS = 12

# rows scaled more than this are too small to read, so they are drawn as tiles
MIN_ROW_SCALE = 0.6

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5
NAME_FONT_SIZE = 10
TEXT_FONT_SIZE = 7.5
PHONE_FONT_SIZE = 11
CHAR_WIDTH = 0.56
NAME_CHAR_WIDTH = 0.72
BOLD_CHAR_WIDTH = 0.62
LINE_HEIGHT = 1.3


class WorkforceVisualizer(BaseViz):
    '''Draws a row for each skill level in their declared order: the trainees in the training
    room, then the call floor with a row of workers above a row of ringing phones with the same
    spacing, so that it shows at a glance whether there are enough workers. Phones are green
    when a worker can take them and red otherwise, and workers without a call are pale. When
    there are too many to draw one by one, an icon stands for several, which the row says.

    A small calendar shows the month the level is in. Too many levels to draw like this show
    as tiles with counts. The actions are not part of the state, so hiring only shows as
    trainees afterwards. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_size=(9.0, 5.5),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._levels = list(model.type_to_objects['lvl'])
        self._frequency = int(self._value('SEASONFREQ'))
        self._artists = []

        # rows if these stay readable, otherwise small tiles
        count = len(self._levels)
        self._scale = min(1.0, max_size[0] / ROW_WIDTH, max_size[1] / (count * ROW_HEIGHT))
        self._has_rows = self._scale >= MIN_ROW_SCALE
        if self._has_rows:
            num_cols, num_rows = 1, count
            size = (ROW_WIDTH, ROW_HEIGHT)
        else:
            best = None
            for cols in range(1, count + 1):
                rows = math.ceil(count / cols)
                scale = min(1.0, max_size[0] / (cols * TILE_SIZE[0]), max_size[1] / (rows * TILE_SIZE[1]))
                if best is None or scale >= best[0]:
                    best = (scale, cols, rows)
            (self._scale, num_cols, num_rows) = best
            size = TILE_SIZE

        # title above, then the rows, the legend and the caption
        grid_width = num_cols * size[0] * self._scale
        grid_height = num_rows * size[1] * self._scale
        width = max(grid_width + 0.4, MIN_WIDTH)
        height = grid_height + BOTTOM_MARGIN + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width), BOTTOM_MARGIN / height,
             grid_width / width, grid_height / height])
        self.ax.set_xlim(0, num_cols * size[0])
        self.ax.set_ylim(0, num_rows * size[1])
        self.ax.axis('off')

        # the rooms never move, so they are only drawn once and then filled in
        self._rows = {}
        for (i, level) in enumerate(self._levels):
            row, col = divmod(i, num_cols)
            x, y = col * size[0], (num_rows - row - 1) * size[1]
            self._rows[level] = (self._draw_row if self._has_rows else self._draw_tile)(level, x, y)

        if self._has_rows:
            handles = [
                Line2D([], [], linestyle='', marker='o', markersize=8, markerfacecolor=WORKER_COLOR,
                       markeredgecolor='none', label='Worker on a call'),
                Line2D([], [], linestyle='', marker='o', markersize=8, markerfacecolor=SPARE_COLOR,
                       markeredgecolor='none', label='Spare worker'),
                Line2D([], [], linestyle='', marker='o', markersize=8, markerfacecolor=TRAINEE_COLOR,
                       markeredgecolor='none', label='Trainee'),
                Line2D([], [], linestyle='', marker='$☎$', markersize=9, markerfacecolor=COVERED_COLOR,
                       markeredgecolor=COVERED_COLOR, label='Call covered'),
                Line2D([], [], linestyle='', marker='$☎$', markersize=9, markerfacecolor=SHORT_COLOR,
                       markeredgecolor=SHORT_COLOR, label='Call not covered')
            ]
        else:
            handles = [
                patches.Patch(facecolor=COVERED_FILL_COLOR, edgecolor=COVERED_COLOR, label='All calls covered'),
                patches.Patch(facecolor=SHORT_FILL_COLOR, edgecolor=SHORT_COLOR, label='Short of workers')
            ]
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.25 / height),
                        ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                        handlelength=1.4)
        self.fig.text(0.5, 0.1 / height, self._caption(), ha='center', va='bottom', fontsize=9,
                      color=DETAIL_COLOR)

    def _caption(self):
        earns = 'A level only earns when all of its calls are covered'
        if not self._has_rows:
            return f'Staff shows workers + trainees, who start work the next month. {earns}'

        caption = f'Trainees start work the next month. {earns}'
        if any(self._value('SEASONALPEAK', level) > 0 for level in self._levels):
            if 1 <= self._frequency <= MAX_CALENDAR_DOTS:
                caption += '. Calls burst after the orange month'
            elif self._frequency > MAX_CALENDAR_DOTS:
                caption += f'. Calls burst every {self._frequency} months'
        return caption

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

    def _draw_row(self, level, x, y):
        left, bottom = x, y + ROW_GAP / 2
        height = ROW_HEIGHT - ROW_GAP
        top = bottom + height
        frame = self.ax.add_patch(patches.FancyBboxPatch(
            (left, bottom), ROW_WIDTH, height, boxstyle='round,pad=0,rounding_size=0.07',
            facecolor=PLATE_COLOR, edgecolor=PLATE_EDGE_COLOR, linewidth=1.4, zorder=1))

        # the name, how the level is doing and its calendar on the left
        name_x = left + 0.1
        limit = int((NAME_WIDTH - 0.15) * 72 / (NAME_CHAR_WIDTH * NAME_FONT_SIZE))
        self.ax.text(name_x, top - 0.2, self._shorten(level, limit), ha='left', va='center',
                     fontsize=self._font(NAME_FONT_SIZE), fontweight='bold', zorder=3)
        status = self.ax.text(name_x, top - 0.43, '', ha='left', va='center', fontweight='bold',
                              fontsize=self._font(TEXT_FONT_SIZE), zorder=3)
        dots = self._draw_calendar(level, name_x, top - 0.68)

        # the training room, and the way out of it onto the call floor
        room_x = left + NAME_WIDTH
        room_bottom, room_height = bottom + 0.08, height - 0.16
        self.ax.add_patch(patches.FancyBboxPatch(
            (room_x, room_bottom), ROOM_WIDTH, room_height, boxstyle='round,pad=0,rounding_size=0.05',
            facecolor=ROOM_COLOR, edgecolor=LINE_COLOR, linewidth=1.0, zorder=2))
        room_label = self.ax.text(room_x + 0.07, room_bottom + room_height - 0.13, '', ha='left', va='center',
                                  fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR, zorder=3)
        room_value = self.ax.text(room_x + 0.07, room_bottom + room_height - 0.27, '', ha='left', va='center',
                                  fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR, zorder=3)
        floor_x = room_x + ROOM_WIDTH + ARROW_WIDTH
        floor_width = ROW_WIDTH - NAME_WIDTH - ROOM_WIDTH - ARROW_WIDTH - MONEY_WIDTH
        middle = room_bottom + room_height / 2
        self.ax.add_patch(patches.FancyArrowPatch(
            (room_x + ROOM_WIDTH, middle), (floor_x, middle), arrowstyle='-|>', mutation_scale=10,
            linewidth=1.5, color=LINE_COLOR, shrinkA=1, shrinkB=1, zorder=2))

        # the call floor: its workers above its phones, and what one icon stands for
        self.ax.add_patch(patches.FancyBboxPatch(
            (floor_x, room_bottom), floor_width, room_height, boxstyle='round,pad=0,rounding_size=0.05',
            facecolor=ROOM_COLOR, edgecolor=LINE_COLOR, linewidth=1.0, zorder=2))
        staff_y, calls_y = middle + room_height / 4, middle - room_height / 4
        labels = [self.ax.text(floor_x + 0.07, row_y, '', ha='left', va='center', zorder=3,
                               fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR)
                  for row_y in (staff_y, calls_y)]

        # what it earns and pays on the right, with how many leave
        money_x = floor_x + floor_width + 0.1
        money_top = top - 0.2
        money = [self.ax.text(money_x, money_top - i * 0.21, '', ha='left', va='center', zorder=3,
                              fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR)
                 for i in range(3)]
        money[1].set_text(f'Pay {self._value("LABORCOST", level):g} per worker')
        money[2].set_text(f'{self._value("ATTRITIONAMOUNT", level):g} leave, chance {self._value("ATTRITION", level):g}')
        return {'frame': frame, 'status': status, 'dots': dots, 'room_label': room_label,
                'room_value': room_value, 'labels': labels,
                'room': (room_x + 0.07, room_bottom + 0.27), 'staff': (floor_x + FLOOR_LABEL_WIDTH, staff_y),
                'calls': (floor_x + FLOOR_LABEL_WIDTH, calls_y), 'earning': money[0]}

    def _draw_calendar(self, level, x, y):

        # a dot per month of the season, with the month before the burst in orange
        if self._value('SEASONALPEAK', level) <= 0 or not 1 <= self._frequency <= MAX_CALENDAR_DOTS:
            return {}
        dots = {}
        for i in range(self._frequency):
            month = self._frequency - 1 - i
            dots[month] = self.ax.add_patch(patches.Circle(
                (x + 0.04 + i * 0.09, y), 0.035, facecolor=DOT_COLOR,
                edgecolor=BURST_COLOR if month == 0 else 'none', linewidth=1.0, zorder=3))
        return dots

    def _draw_tile(self, level, x, y):
        (tile_width, tile_height) = TILE_SIZE
        frame = self.ax.add_patch(patches.FancyBboxPatch(
            (x + TILE_GAP / 2, y + TILE_GAP / 2), tile_width - TILE_GAP, tile_height - TILE_GAP,
            boxstyle='round,pad=0,rounding_size=0.07', facecolor=PLATE_COLOR, edgecolor=PLATE_EDGE_COLOR,
            linewidth=1.2, zorder=1))
        scale = self._scale
        center = x + tile_width / 2

        # counts stay at the smallest readable text for as long as they fit on the tile
        font = max(TEXT_FONT_SIZE * scale, MIN_FONT_SIZE)
        limit = int((tile_width - TILE_GAP) * scale * 72 / (CHAR_WIDTH * font))
        has_details = (limit >= len('Calls 9.9')
                       and tile_height * scale * 72 >= 3 * LINE_HEIGHT * font)

        name_fit = 0.9 * (tile_width - TILE_GAP) * scale * 72 / BOLD_CHAR_WIDTH
        name = self._shorten(level, int(name_fit / MIN_FONT_SIZE))
        self.ax.text(center, y + tile_height * (0.75 if has_details else 0.5), name, ha='center',
                     va='center', fontweight='bold', zorder=3,
                     fontsize=max(min(NAME_FONT_SIZE * scale, name_fit / len(name)), MIN_FONT_SIZE))
        counts = None
        if has_details:
            counts = [self.ax.text(center, y + tile_height * share, '', ha='center', va='center',
                                   fontsize=font, color=DETAIL_COLOR, zorder=3)
                      for share in (0.48, 0.24)]
        return {'frame': frame, 'counts': counts}

    # ==========================================================================
    # WHAT MOVES
    # ==========================================================================

    def _icon_value(self, largest, max_icons):

        # the smallest round number of people for each icon that lets them all fit
        value = 1
        while largest / value > max_icons:
            for step in (2, 5):
                if largest / (value * step) <= max_icons:
                    return value * step
            value *= 10
        return value

    def _count(self, value):

        # very large counts would run into the icons, so they lose their last digits
        return f'{value:.3g}' if abs(value) >= 100000 else f'{value:g}'

    def _draw_person(self, x, y, color, has_headset):
        size = ICON_PITCH * 0.5
        foot = y - size * 0.95
        body = size * 0.72
        head_y = foot + body + size * 0.42
        self._artists.append(self.ax.add_patch(patches.Wedge(
            (x, foot), body, 0, 180, facecolor=color, edgecolor='none', zorder=4)))
        self._artists.append(self.ax.add_patch(patches.Circle(
            (x, head_y), size * 0.38, facecolor=color, edgecolor='none', zorder=4)))
        if has_headset:
            self._artists.append(self.ax.add_patch(patches.Arc(
                (x, head_y), size * 1.15, size * 1.15, theta1=0, theta2=180, color=DETAIL_COLOR,
                linewidth=0.8, zorder=5)))

    def _draw_phone(self, x, y, color):
        self._artists.append(self.ax.text(x, y, '☎', ha='center', va='center', color=color,
                                          fontsize=self._font(PHONE_FONT_SIZE), zorder=4))

    def _fill_row(self, row, labor, demand, training, month, level):
        staff = max(labor, 0)
        calls = max(demand, 0)
        waiting = max(training, 0)
        covered = labor >= demand

        # the same value per icon on both rows of the floor, so that their lengths compare
        value = self._icon_value(max(staff, calls), MAX_FLOOR_ICONS)
        num_workers = math.ceil(staff / value)
        num_calls = math.ceil(calls / value)
        (staff_x, staff_y) = row['staff']
        (calls_x, calls_y) = row['calls']

        # icons round up, so one phone stays red whenever there are not enough workers
        taken = num_calls if covered else min(num_workers, max(num_calls - 1, 0))
        for i in range(num_workers):
            self._draw_person(staff_x + i * ICON_PITCH, staff_y, WORKER_COLOR if i < taken else SPARE_COLOR, True)
        for i in range(num_calls):
            self._draw_phone(calls_x + i * ICON_PITCH, calls_y, COVERED_COLOR if i < taken else SHORT_COLOR)

        (staff_label, calls_label) = row['labels']
        staff_label.set_text(f'Staff {self._count(labor)}')
        calls_label.set_text(f'Calls {self._count(round(demand, 1))}')
        if value > 1:
            self._artists.append(self.ax.text(
                calls_x + MAX_FLOOR_ICONS * ICON_PITCH - 0.02, (staff_y + calls_y) / 2, f'One icon is {value:g}',
                ha='right', va='center', fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR, zorder=5))

        # trainees are in their room, which is smaller and so has its own value per icon
        room_value = self._icon_value(waiting, MAX_ROOM_ICONS)
        row['room_label'].set_text(f'Training {self._count(training)}')
        row['room_value'].set_text(f'One icon is {room_value:g}' if room_value > 1 else '')
        (room_x, room_y) = row['room']
        for i in range(math.ceil(waiting / room_value)):
            self._draw_person(room_x + 0.1 + i * ICON_PITCH * 1.2, room_y, TRAINEE_COLOR, False)

        # how the level is doing, and how much it brings in
        if covered:
            row['status'].set(text='Covered', color=COVERED_COLOR)
            row['frame'].set_edgecolor(COVERED_COLOR)
        else:
            row['status'].set(text=f'Short by {demand - labor:.1f}', color=SHORT_COLOR)
            row['frame'].set_edgecolor(SHORT_COLOR)
        earns = self._value('SERVICEREWARD', level) if covered else 0
        row['earning'].set(text=f'Earning {earns:g}', color=COVERED_COLOR if earns > 0 else DETAIL_COLOR)

        for (dot_month, dot) in row['dots'].items():
            is_now = dot_month == month
            is_burst = dot_month == 0
            dot.set_facecolor((BURST_COLOR if is_burst else CURRENT_COLOR) if is_now else DOT_COLOR)

    def _fill_tile(self, tile, labor, demand, training):
        covered = labor >= demand
        tile['frame'].set_facecolor(COVERED_FILL_COLOR if covered else SHORT_FILL_COLOR)
        tile['frame'].set_edgecolor(COVERED_COLOR if covered else SHORT_COLOR)
        if tile['counts'] is not None:
            (first, second) = tile['counts']
            first.set_text(f'Staff {self._count(labor)}' + (f'+{self._count(training)}' if training else ''))
            second.set_text(f'Calls {self._count(round(demand, 1))}')

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        num_covered = 0
        net = 0.0
        for level in self._levels:
            labor = int(state[fluent('labor', [level])])
            demand = float(state[fluent('demand', [level])])
            training = int(state[fluent('training', [level])])
            month = int(state[fluent('month', [level])])

            covered = labor >= demand
            num_covered += covered
            net += (self._value('SERVICEREWARD', level) if covered else 0)
            net -= self._value('LABORCOST', level) * (labor + training)
            if self._has_rows:
                self._fill_row(self._rows[level], labor, demand, training, month, level)
            else:
                self._fill_tile(self._rows[level], labor, demand, training)

        total = len(self._levels)
        title = f'{num_covered} / {total} skill level{"" if total == 1 else "s"} covered, net {net:g} this month'
        color = COVERED_COLOR if num_covered == total else 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in self._artists:
            artist.remove()
        self._artists = []
        return img
