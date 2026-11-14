import math
import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

OUTSIDE_COLOR = '#eceff1'
WALL_COLOR = '#37474f'
WINDOW_COLOR = '#81d4fa'
HALL_COLOR = '#fafafa'
OTHER_COLOR = '#cfd8dc'
# floors shade from deep to light blue below the comfort range, and light to deep red above
COLD_COLORS = ('#64b5f6', '#e3f2fd')
COMFORT_COLOR = '#c5e1a5'
HOT_COLORS = ('#ffebee', '#ef5350')
TEXT_COLOR = '#263238'
MUTED_COLOR = '#78909c'
OK_COLOR = '#2e7d32'
BAD_COLOR = '#c62828'

# a doorway or a window is this fraction of the wall it is in
DOORWAY = 0.36
WINDOW = 0.5

# the hall is a corridor this deep, in rooms, along the bottom of the building
HALL_DEPTH = 0.32

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


def _mix(colors, fraction):
    (a, b) = (np.array(plt.matplotlib.colors.to_rgb(c)) for c in colors)
    return tuple(a + (b - a) * min(max(fraction, 0.0), 1.0))


class HVACVisualizer(BaseViz):
    '''Draws the spaces as rooms of one floor plan, laid out on a grid, with the hall as a
    corridor along the bottom of the building. The layout is chosen once, so that as many
    adjacent spaces as possible are next to each other with a doorway between them, rooms
    next to the outside have windows, and rooms next to the hall have a door to it. Rooms
    next to each other without a link share a solid wall, and each room lists the
    adjacent spaces, hall or outside that it could not be placed next to. Each floor is
    shaded by its temperature against its comfort range.

    The outside and the hall take random temperatures that are not part of the state,
    so they are not shown. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_room_size=1.6,
                 max_size=(12.0, 7.5),
                 min_width=6.0,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._spaces = model.type_to_objects['space']
        self._rooms = [s for s in self._spaces if self._get_nonfluent('IS_ROOM', s)]
        self._low = {s: self._get_nonfluent('TEMP_LOW', s) for s in self._spaces}
        self._up = {s: self._get_nonfluent('TEMP_UP', s) for s in self._spaces}
        self._outside = {s for s in self._spaces if self._get_nonfluent('ADJ_OUTSIDE', s)}
        self._hall = {s for s in self._spaces if self._get_nonfluent('ADJ_HALL', s)}
        self._links = [
            (s1, s2) for (i, s1) in enumerate(self._spaces) for s2 in self._spaces[i + 1:]
            if self._nonfluents[RDDLPlanningModel.ground_var('ADJ', [s1, s2])]
            or self._nonfluents[RDDLPlanningModel.ground_var('ADJ', [s2, s1])]]
        self._scale = self._get_scale(model)
        self._cell, (rows, columns) = self._get_layout()

        # the building with its hall and a margin of outside around it, then the legend
        room = min(max_room_size, max_size[0] / columns, max_size[1] / (rows + HALL_DEPTH))
        margin, bottom, top = 0.35, 0.95, 0.5
        hall = HALL_DEPTH if self._hall else 0.0
        width = max(columns * room + 2 * margin, min_width)
        height = (rows + hall) * room + 2 * margin + bottom + top
        self._room = room
        self.fig = plt.figure(figsize=(width, height))
        self.fig.patch.set_facecolor(OUTSIDE_COLOR)
        self.ax = self.fig.add_axes([0, bottom / height, 1, (height - bottom - top) / height])
        side = (width - columns * room) / 2 / room
        self.ax.set_xlim(-side, columns + side)
        self.ax.set_ylim(-hall - margin / room, rows + margin / room)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # everything is drawn once: render only changes colors and texts
        self._floors, self._readouts = {}, {}
        if self._hall:
            self._draw_hall(columns)
        for s in self._spaces:
            self._draw_room(s)
        self._draw_walls()

        self.fig.legend(handles=[
            patches.Patch(facecolor=COLD_COLORS[0], label='Too cold'),
            patches.Patch(facecolor=COMFORT_COLOR, label='In range'),
            patches.Patch(facecolor=HOT_COLORS[1], label='Too hot'),
            patches.Patch(facecolor=OTHER_COLOR, label='Not a room'),
            Line2D([], [], color=WINDOW_COLOR, linewidth=3, label='Window')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.45 / height), ncol=5, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)
        self.fig.text(0.5, 0.08 / height,
                      'Doorways join adjacent spaces, ↔ lists the ones elsewhere\n'
                      f'Floors are deepest at {self._scale[0]:.1f}° and as far above the range',
                      ha='center', va='bottom', fontsize=8, color=TEXT_COLOR)

    def _get_nonfluent(self, var, space):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, [space])]

    def _get_scale(self, model):

        # temperature moves towards the outside, the hallway, the air and other spaces,
        # so these values and the initial temperatures bound where it normally stays
        init_state = model.ground_vars_with_values(model.state_fluents)
        temps = [self._nonfluents['TEMP_AIR']] if self._rooms else []
        temps.extend(init_state[RDDLPlanningModel.ground_var('temp', [s])] for s in self._spaces)
        temps.extend(self._get_nonfluent('TEMP_OUTSIDE_MEAN', s) for s in self._outside)
        temps.extend(self._get_nonfluent('TEMP_HALL_MEAN', s) for s in self._hall)
        for s in self._rooms:
            temps.extend((self._low[s], self._up[s]))
        low, high = min(temps), max(temps)
        return (low, high) if high > low else (low - 1.0, low + 1.0)

    def _get_layout(self):

        # spaces are placed on a grid to put adjacent spaces next to each other, spaces
        # next to the hall in the bottom row, and spaces next to the outside on the other
        # edges: starting from the declared order and from a breadth-first order, on a grid
        # about as wide as it is tall and on one a column wider, the layout with the fewest
        # of these apart wins, then the smaller one
        n = len(self._spaces)
        neighbors = {s: [] for s in self._spaces}
        for (s1, s2) in self._links:
            neighbors[s1].append(s2)
            neighbors[s2].append(s1)
        breadth_first = []
        for root in sorted(self._spaces, key=lambda s: -len(neighbors[s])):
            queue = [root] if root not in breadth_first else []
            while queue:
                s = queue.pop(0)
                if s not in breadth_first:
                    breadth_first.append(s)
                    queue.extend(sorted(neighbors[s], key=lambda s2: -len(neighbors[s2])))
        best = None
        for columns in {max(math.ceil(math.sqrt(n)), 1), math.ceil(math.sqrt(n)) + 1}:
            rows = max(math.ceil(n / columns), 1)
            for order in (list(self._spaces), breadth_first):
                (apart, cells, where) = self._search(order, rows, columns, neighbors)
                if best is None or (apart, rows * columns) < best[0]:
                    best = ((apart, rows * columns), rows, columns, cells, where)
        (_, rows, columns, cells, where) = best

        # the first row of the grid is drawn at the top
        layout = {s: (cells[where[s]][1], rows - 1 - cells[where[s]][0]) for s in self._spaces}
        return layout, (rows, columns)

    def _search(self, order, rows, columns, neighbors):

        # swapping two cells, empty or not, is kept when it brings more adjacent spaces
        # together, or else their total distance down
        cells = [(r, c) for r in range(rows) for c in range(columns)]
        at = list(order) + [None] * (len(cells) - len(order))
        where = {s: i for (i, s) in enumerate(order)}
        miss = len(cells) ** 2

        def edges(s, r, c):
            hall = s in self._hall and r != rows - 1
            outside = s in self._outside and not (
                r == 0 or c == 0 or c == columns - 1 or (r == rows - 1 and not self._hall))
            return hall + outside

        def cost(s):
            if s is None:
                return 0
            (r1, c1) = cells[where[s]]
            total = miss * edges(s, r1, c1)
            for s2 in neighbors[s]:
                (r2, c2) = cells[where[s2]]
                distance = abs(r1 - r2) + abs(c1 - c2)
                total += distance + (0 if distance == 1 else miss)
            return total

        def place(a, i, b, j):
            (at[i], at[j]) = (a, b)
            for (s, k) in ((a, i), (b, j)):
                if s is not None:
                    where[s] = k

        # ponytail: local search, so some layouts could put more adjacent spaces together
        for _ in range(100):
            improved = False
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    (a, b) = (at[i], at[j])
                    if a is None and b is None:
                        continue
                    before = cost(a) + cost(b)
                    place(b, i, a, j)
                    if cost(a) + cost(b) < before:
                        improved = True
                    else:
                        place(a, i, b, j)
            if not improved:
                break
        apart = sum(cost(s) // miss for s in order)
        return apart, cells, where

    def _add_text(self, x, y, text, font_size, width, ha='center', va='center', **kwargs):

        # text shrinks to the given width in rooms, and is left out if unreadable
        size = min(font_size, 72 * width * self._room / (0.62 * max(len(text), 1)))
        if size < MIN_FONT_SIZE:
            return None
        return self.ax.text(x, y, text, ha=ha, va=va, fontsize=size, zorder=6, **kwargs)

    def _sides(self, space):

        # each side of a room, with what is beyond it: another space, the hall or nothing
        (x, y) = self._cell[space]
        occupant = self._occupant
        for ((dx, dy), start, end) in (((0, -1), (x, y), (x + 1, y)),
                                       ((0, 1), (x, y + 1), (x + 1, y + 1)),
                                       ((-1, 0), (x, y), (x, y + 1)),
                                       ((1, 0), (x + 1, y), (x + 1, y + 1))):
            other = occupant.get((x + dx, y + dy))
            if other is None and dy == -1 and y == 0 and self._hall:
                other = 'hall'
            yield (other, start, end)

    @property
    def _occupant(self):
        return {cell: s for (s, cell) in self._cell.items()}

    def _remote_text(self, space):

        # adjacent spaces, the hall and the outside that are not next to this room
        sides = list(self._sides(space))
        beyond = {other for (other, _, _) in sides}
        names = [s2 for (s1, s2) in self._links if s1 == space and s2 not in beyond]
        names += [s1 for (s1, s2) in self._links if s2 == space and s1 not in beyond]
        names.sort(key=self._spaces.index)
        if space in self._hall and 'hall' not in beyond:
            names.append('hall')
        if space in self._outside and None not in beyond:
            names.append('outside')

        # as many names as fit on two lines, and how many more there are
        size = min(9, 72 * 0.1 * self._room)
        if not names or size < MIN_FONT_SIZE:
            return None
        budget = max(int(0.9 * self._room * 72 / (0.62 * size)), 1)
        for k in range(len(names), -1, -1):
            words = ['\u2194'] + names[:k] + ([f'+{len(names) - k}'] if k < len(names) else [])
            lines = textwrap.wrap(' '.join(words), budget, break_long_words=False)
            if len(lines) <= 2:
                break
        return '\n'.join(lines), size

    def _draw_hall(self, columns):
        self.ax.add_patch(patches.Rectangle(
            (0, -HALL_DEPTH), columns, HALL_DEPTH, facecolor=HALL_COLOR, edgecolor='none',
            zorder=1))
        self._add_text(columns / 2, -HALL_DEPTH / 2, 'hall', 9, columns, color=MUTED_COLOR)
        for (start, end) in (((0, -HALL_DEPTH), (columns, -HALL_DEPTH)),
                             ((0, -HALL_DEPTH), (0, 0)), ((columns, -HALL_DEPTH), (columns, 0))):
            self.ax.plot(*zip(start, end), color=WALL_COLOR, linewidth=4.0,
                         solid_capstyle='projecting', zorder=3)

        # the hall is outside the building where no room is above it
        bottom_row = {x for (x, y) in self._cell.values() if y == 0}
        for x in range(columns):
            if x not in bottom_row:
                self.ax.plot([x, x + 1], [0, 0], color=WALL_COLOR, linewidth=4.0,
                             solid_capstyle='projecting', zorder=3)

    def _draw_room(self, space):
        (x, y) = self._cell[space]
        self._floors[space] = self.ax.add_patch(patches.Rectangle(
            (x, y), 1, 1, facecolor=COMFORT_COLOR if space in self._rooms else OTHER_COLOR,
            edgecolor='none', zorder=1))
        self._add_text(x + 0.07, y + 0.93, space, 11, 0.8, ha='left', va='top',
                       color=TEXT_COLOR, fontweight='bold')
        self._readouts[space] = self._add_text(x + 0.5, y + 0.6, '00.0°', 12, 0.6,
                                               color=TEXT_COLOR, fontweight='bold')
        tag = self._remote_text(space)
        if tag is not None:
            self.ax.text(x + 0.5, y + 0.3, tag[0], ha='center', va='center', fontsize=tag[1],
                         linespacing=1.15, color=TEXT_COLOR, zorder=6)

    def _draw_walls(self):

        # each side of a room is an outer wall, with a window if the room is next to the
        # outside, a solid wall to the space next to it, or a wall with a doorway to an
        # adjacent space or the hall
        thin, thick, windows = [], [], []
        for s in self._spaces:
            for (other, start, end) in self._sides(s):
                if other is None:
                    thick.append((start, end))
                    if s in self._outside:
                        windows.append(self._middle(start, end, WINDOW))
                elif other == 'hall' or self._spaces.index(s) < self._spaces.index(other):
                    linked = s in self._hall if other == 'hall' else (
                        (s, other) in self._links or (other, s) in self._links)
                    if linked:
                        (left, right) = self._middle(start, end, DOORWAY)
                        thin.extend(((start, left), (right, end)))
                    else:
                        thin.append((start, end))
        for (lines, width, color) in ((thin, 2.0, WALL_COLOR), (thick, 4.0, WALL_COLOR),
                                      (windows, 2.5, WINDOW_COLOR)):
            for (start, end) in lines:
                self.ax.plot(*zip(start, end), color=color, linewidth=width,
                             solid_capstyle='butt' if color == WINDOW_COLOR else 'projecting',
                             zorder=4 if color == WINDOW_COLOR else 3)

    @staticmethod
    def _middle(start, end, fraction):
        (sx, sy), (ex, ey) = start, end
        gap = (1 - fraction) / 2
        return ((sx + (ex - sx) * gap, sy + (ey - sy) * gap),
                (ex - (ex - sx) * gap, ey - (ey - sy) * gap))

    def _floor_color(self, space, temp):

        # floors take their deepest shades at the bottom of the scale, and as far above the
        # comfort range as the bottom of the scale is below it, or the top of the scale
        if space not in self._rooms:
            return OTHER_COLOR
        (low, high) = self._scale
        (lower, upper) = (self._low[space], self._up[space])
        if temp < lower:
            return _mix(COLD_COLORS, (temp - low) / max(lower - low, 1e-6))
        if temp > upper:
            return _mix(HOT_COLORS, (temp - upper) / max(high - upper, lower - low, 1e-6))
        return COMFORT_COLOR

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        temps = {s: float(state[RDDLPlanningModel.ground_var('temp', [s])]) for s in self._spaces}
        for s in self._spaces:
            self._floors[s].set_facecolor(self._floor_color(s, temps[s]))
            if self._readouts[s] is not None:
                self._readouts[s].set_text(f'{temps[s]:.1f}°')

        # same rule as the penalty in the reward
        comfortable = sum(self._low[s] <= temps[s] <= self._up[s] for s in self._rooms)
        self.ax.set_title(f'{comfortable} / {len(self._rooms)} rooms in range',
                          fontsize=13, fontweight='bold',
                          color=OK_COLOR if comfortable == len(self._rooms) else BAD_COLOR)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
