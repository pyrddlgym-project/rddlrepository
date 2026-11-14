import math
import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

OUTSIDE_COLOR = '#eceff1'
WALL_COLOR = '#37474f'
# floors shade from deep to light blue below the comfort range, and light to deep red above
COLD_COLORS = ('#64b5f6', '#e3f2fd')
COMFORT_COLOR = '#c5e1a5'
HOT_COLORS = ('#ffebee', '#ef5350')
# a heater that serves one zone is orange, and each that serves several has a color of its
# own, away from the colors of temperatures, in all of its radiators
HEATER_COLOR = '#ff6d00'
SHARED_COLORS = ('#8e24aa', '#6d4c41', '#d81b60', '#3949ab', '#9e9d24', '#00897b', '#546e7a',
                 '#fbc02d')
PERSON_COLOR = '#455a64'
TEXT_COLOR = '#263238'
MUTED_COLOR = '#78909c'
OK_COLOR = '#2e7d32'
BAD_COLOR = '#c62828'

# a doorway is this fraction of the wall it is in
DOORWAY = 0.36

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


def _person_marker():
    '''A head above rounded shoulders, as one path.'''
    t = np.linspace(0, np.pi, 13)
    body = np.column_stack([np.r_[0.34 * np.cos(t), -0.34, 0.34, 0.34],
                            np.r_[0.34 * np.sin(t) - 0.16, -0.75, -0.75, 0]])
    return Path.make_compound_path(Path.circle((0, 0.5), 0.24), Path(body, closed=True))


PERSON = _person_marker()


def _mix(colors, fraction):
    (a, b) = (np.array(plt.matplotlib.colors.to_rgb(c)) for c in colors)
    return tuple(a + (b - a) * min(max(fraction, 0.0), 1.0))


class HVACVisualizer(BaseViz):
    '''Draws the zones as rooms of one floor plan, laid out on a grid. The layout is
    chosen once, so that as many adjacent zones as possible are next to each other with a
    doorway between them, and zones that share a heater are close. Rooms next to each
    other without a doorway share a solid wall, and each room lists the adjacent zones
    that are not next to it. Each heater has a radiator in every zone it serves, which
    fills up as the heater heats up, and the radiators of a heater that serves several
    zones share a color of their own. Each floor is shaded by its temperature against the
    comfort range, and a person stands in each occupied zone.

    Discomfort is only penalized in occupied zones. The drawing only depends on the
    current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_room_size=1.6,
                 max_size=(12.0, 7.5),
                 min_width=6.0,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._zones = model.type_to_objects['zone']
        self._heaters = model.type_to_objects['heater']
        self._low = self._nonfluents['TEMP-ZONE-MIN']
        self._high = self._nonfluents['TEMP-ZONE-MAX']
        self._zone_heaters = {z: [h for h in self._heaters if self._nonfluents[
            RDDLPlanningModel.ground_var('ADJ-HEATER', [h, z])]] for z in self._zones}
        self._served = {h: [z for z in self._zones if h in self._zone_heaters[z]]
                        for h in self._heaters}
        self._shared = [h for h in self._heaters if len(self._served[h]) > 1]
        self._links = [
            (z1, z2) for (i, z1) in enumerate(self._zones) for z2 in self._zones[i + 1:]
            if self._nonfluents[RDDLPlanningModel.ground_var('ADJ-ZONES', [z1, z2])]
            or self._nonfluents[RDDLPlanningModel.ground_var('ADJ-ZONES', [z2, z1])]]
        self._scale = self._get_scale(model)

        # floors take their deepest shades at the bottom of the scale, and as far above the
        # comfort range as the bottom of the scale is below it, or the top of the scale
        (low, high) = self._scale
        self._shades = (low, self._high + max(high - self._high, self._low - low, 1e-6))
        self._cell, (rows, columns) = self._get_layout()

        # the building and a margin of outside around it, then the legend below
        room = min(max_room_size, max_size[0] / columns, max_size[1] / rows)
        margin, bottom, top = 0.35, 0.95, 0.5
        width = max(columns * room + 2 * margin, min_width)
        height = rows * room + 2 * margin + bottom + top
        self._room = room
        self.fig = plt.figure(figsize=(width, height))
        self.fig.patch.set_facecolor(OUTSIDE_COLOR)
        self.ax = self.fig.add_axes([0, bottom / height, 1, (height - bottom - top) / height])
        self.ax.set_xlim(-(width - columns * room) / 2 / room, columns + (width - columns * room) / 2 / room)
        self.ax.set_ylim(-margin / room, rows + margin / room)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self._rows = rows

        # everything is drawn once: render only changes colors, texts and who is there
        self._floors, self._readouts, self._people, self._radiators = {}, {}, {}, []
        for z in self._zones:
            self._draw_room(z)
        self._draw_walls()
        self._add_text(columns / 2, rows + 0.5 * margin / room,
                       f'outside {self._nonfluents["TEMP-OUT"]:.1f}°', 9, columns,
                       color=MUTED_COLOR)

        person = dict(marker=PERSON, color='none', markeredgecolor='none', markersize=11)
        self.fig.legend(handles=[
            patches.Patch(facecolor=COLD_COLORS[0], label='Too cold'),
            patches.Patch(facecolor=COMFORT_COLOR, edgecolor=MUTED_COLOR, label='In range'),
            patches.Patch(facecolor=HOT_COLORS[1], label='Too hot'),
            patches.Patch(facecolor=HEATER_COLOR, label='Heater'),
            Line2D([], [], markerfacecolor=PERSON_COLOR, label='Occupied', **person)
        ] + ([patches.Patch(facecolor=SHARED_COLORS[0], label='Shared heater')]
             if self._shared else []),
           loc='lower center', bbox_to_anchor=(0.5, 0.45 / height), ncol=6, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)
        self.fig.text(0.5, 0.08 / height,
                      'Doorways join adjacent zones, ↔ lists the ones elsewhere\n'
                      f'Floors are deepest at {self._shades[0]:.1f}° and {self._shades[1]:.1f}°, '
                      f'radiators are half full at {self._scale[1]:.1f}°',
                      ha='center', va='bottom', fontsize=8, color=TEXT_COLOR)

    def _get_scale(self, model):

        # the outside, the comfort range and the initial temperatures bound where zones
        # normally stay: anything beyond takes the deepest shade
        init_state = model.ground_vars_with_values(model.state_fluents)
        temps = [self._nonfluents['TEMP-OUT'], self._low, self._high]
        temps.extend(init_state[RDDLPlanningModel.ground_var('temp-zone', [z])] for z in self._zones)
        temps.extend(init_state[RDDLPlanningModel.ground_var('temp-heater', [h])] for h in self._heaters)
        low, high = min(temps), max(temps)
        return (low, high) if high > low else (low - 1.0, low + 1.0)

    def _get_layout(self):

        # zones are placed on a grid so that adjacent zones are next to each other, and zones
        # that share a heater are close: starting from the declared order and from a
        # breadth-first order, on a grid about as wide as it is tall and on one a column
        # wider, the layout with the fewest adjacent zones apart wins, then the smaller one
        n = len(self._zones)
        neighbors = {z: [] for z in self._zones}
        for (z1, z2) in self._links:
            neighbors[z1].append(z2)
            neighbors[z2].append(z1)
        groups = [self._served[h] for h in self._shared]
        related = {z: neighbors[z] + [z2 for g in groups if z in g for z2 in g if z2 != z]
                   for z in self._zones}
        breadth_first = []
        for root in sorted(self._zones, key=lambda z: -len(related[z])):
            queue = [root] if root not in breadth_first else []
            while queue:
                z = queue.pop(0)
                if z not in breadth_first:
                    breadth_first.append(z)
                    queue.extend(sorted(related[z], key=lambda z2: -len(related[z2])))
        best = None
        for columns in {max(math.ceil(math.sqrt(n)), 1), math.ceil(math.sqrt(n)) + 1}:
            rows = max(math.ceil(n / columns), 1)
            for order in (list(self._zones), breadth_first):
                (apart, cells, where) = self._search(order, rows, columns, neighbors, groups)
                if best is None or (apart, rows * columns) < best[0]:
                    best = ((apart, rows * columns), rows, columns, cells, where)
        (_, rows, columns, cells, where) = best

        # the first row of the grid is drawn at the top
        layout = {z: (cells[where[z]][1], rows - 1 - cells[where[z]][0]) for z in self._zones}
        return layout, (rows, columns)

    @staticmethod
    def _search(order, rows, columns, neighbors, groups):

        # swapping two cells, empty or not, is kept when it brings more adjacent zones
        # together, or else the total distance of them and of zones sharing heaters down
        cells = [(r, c) for r in range(rows) for c in range(columns)]
        at = list(order) + [None] * (len(cells) - len(order))
        where = {z: i for (i, z) in enumerate(order)}
        member = {z: [g for g in groups if z in g] for z in order}
        miss = len(cells) ** 2

        def cost(z):
            if z is None:
                return 0
            (r1, c1) = cells[where[z]]
            total = 0
            for z2 in neighbors[z]:
                (r2, c2) = cells[where[z2]]
                distance = abs(r1 - r2) + abs(c1 - c2)
                total += distance + (0 if distance == 1 else miss)
            for g in member[z]:
                for z2 in g:
                    (r2, c2) = cells[where[z2]]
                    total += abs(r1 - r2) + abs(c1 - c2)
            return total

        def place(a, i, b, j):
            (at[i], at[j]) = (a, b)
            for (z, k) in ((a, i), (b, j)):
                if z is not None:
                    where[z] = k

        # ponytail: local search, so some layouts could put more adjacent zones together
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
        apart = sum(abs(cells[where[z1]][0] - cells[where[z2]][0])
                    + abs(cells[where[z1]][1] - cells[where[z2]][1]) != 1
                    for z1 in order for z2 in neighbors[z1]) // 2
        return apart, cells, where

    def _add_text(self, x, y, text, font_size, width, ha='center', va='center', **kwargs):

        # text shrinks to the given width in rooms, and is left out if unreadable
        size = min(font_size, 72 * width * self._room / (0.62 * max(len(text), 1)))
        if size < MIN_FONT_SIZE:
            return None
        return self.ax.text(x, y, text, ha=ha, va=va, fontsize=size, zorder=6, **kwargs)

    def _remote_text(self, others):
        names = sorted(others, key=self._zones.index)

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

    def _draw_room(self, zone):
        (x, y) = self._cell[zone]
        self._floors[zone] = self.ax.add_patch(patches.Rectangle(
            (x, y), 1, 1, facecolor=COMFORT_COLOR, edgecolor='none', zorder=1))
        self._add_text(x + 0.07, y + 0.93, zone, 11, 0.6, ha='left', va='top',
                       color=TEXT_COLOR, fontweight='bold')
        self._readouts[zone] = self._add_text(x + 0.5, y + 0.67, '00.0°', 12, 0.6,
                                              color=TEXT_COLOR, fontweight='bold')
        self._people[zone] = self.ax.plot(
            [x + 0.85], [y + 0.85], marker=PERSON, markersize=72 * 0.16 * self._room,
            color=PERSON_COLOR, markeredgecolor='none', zorder=5)[0]

        # adjacent zones that are not next to this one
        remote = [z2 for (z1, z2) in self._links if z1 == zone and not self._next_to(z1, z2)]
        remote += [z1 for (z1, z2) in self._links if z2 == zone and not self._next_to(z1, z2)]
        tag = self._remote_text(remote)
        if tag is not None:
            self.ax.text(x + 0.5, y + 0.45, tag[0], ha='center', va='center', fontsize=tag[1],
                         linespacing=1.15, color=TEXT_COLOR, zorder=6)

        # radiators stand along the bottom wall, as many as the zone has heaters: each is a
        # pale shade of the color of its heater, and fills up from the bottom as it heats up
        heaters = self._zone_heaters[zone]
        width = min(0.3, 0.84 / max(len(heaters), 1) - 0.04)
        for (i, h) in enumerate(heaters):
            left = x + 0.5 + (i - len(heaters) / 2) * (width + 0.04) + 0.02
            color = self._heater_color(h)
            self.ax.add_patch(patches.Rectangle(
                (left, y + 0.08), width, 0.12, facecolor=_mix(('white', color), 0.25),
                edgecolor='none', zorder=4))
            fill = self.ax.add_patch(patches.Rectangle(
                (left, y + 0.08), width, 0.0, facecolor=color, edgecolor='none', zorder=4.1))
            self.ax.add_patch(patches.Rectangle(
                (left, y + 0.08), width, 0.12, facecolor='none', edgecolor=WALL_COLOR,
                linewidth=0.6, zorder=4.2))
            for k in range(1, 4):
                self.ax.plot([left + k * width / 4] * 2, [y + 0.1, y + 0.18], color='white',
                             linewidth=0.6, alpha=0.7, zorder=5)
            label = self._add_text(left + width / 2, y + 0.24, f'{h} 00°', 8, width + 0.04,
                                   va='bottom', color=TEXT_COLOR)
            self._radiators.append((h, fill, label))

    def _heater_color(self, heater):
        if heater not in self._shared:
            return HEATER_COLOR

        # ponytail: colors repeat beyond the eighth shared heater; add patterns if needed
        return SHARED_COLORS[self._shared.index(heater) % len(SHARED_COLORS)]

    def _next_to(self, z1, z2):
        (x1, y1), (x2, y2) = self._cell[z1], self._cell[z2]
        return abs(x1 - x2) + abs(y1 - y2) == 1

    def _draw_walls(self):

        # each side of a room is an outer wall, a solid wall to the room next to it, or a
        # wall with a doorway to an adjacent zone next to it
        occupant = {cell: z for (z, cell) in self._cell.items()}
        segments, thick = [], []
        for (z, (x, y)) in self._cell.items():
            for ((dx, dy), start, end) in (((0, -1), (x, y), (x + 1, y)),
                                           ((0, 1), (x, y + 1), (x + 1, y + 1)),
                                           ((-1, 0), (x, y), (x, y + 1)),
                                           ((1, 0), (x + 1, y), (x + 1, y + 1))):
                other = occupant.get((x + dx, y + dy))
                if other is None:
                    thick.append((start, end))
                elif self._zones.index(z) < self._zones.index(other):
                    if (z, other) in self._links or (other, z) in self._links:
                        (sx, sy), (ex, ey) = start, end
                        gap = (1 - DOORWAY) / 2
                        segments.append((start, (sx + (ex - sx) * gap, sy + (ey - sy) * gap)))
                        segments.append(((ex - (ex - sx) * gap, ey - (ey - sy) * gap), end))
                    else:
                        segments.append((start, end))
        for (lines, width) in ((segments, 2.0), (thick, 4.0)):
            for (start, end) in lines:
                self.ax.plot(*zip(start, end), color=WALL_COLOR, linewidth=width,
                             solid_capstyle='projecting', zorder=3)

    def _floor_color(self, temp):
        (low, high) = self._shades
        if temp < self._low:
            return _mix(COLD_COLORS, (temp - low) / max(self._low - low, 1e-6))
        if temp > self._high:
            return _mix(HOT_COLORS, (temp - self._high) / (high - self._high))
        return COMFORT_COLOR

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        occupied_zones = 0
        comfortable = 0
        for z in self._zones:
            temp = float(state[RDDLPlanningModel.ground_var('temp-zone', [z])])
            occupied = bool(state[RDDLPlanningModel.ground_var('occupied', [z])])
            self._floors[z].set_facecolor(self._floor_color(temp))
            if self._readouts[z] is not None:
                self._readouts[z].set_text(f'{temp:.1f}°')
            self._people[z].set_visible(occupied)

            # same rule as the penalty in the reward
            if occupied:
                occupied_zones += 1
                comfortable += self._low <= temp <= self._high
        (low, high) = self._scale
        for (h, fill, label) in self._radiators:
            temp = float(state[RDDLPlanningModel.ground_var('temp-heater', [h])])

            # heaters have no highest temperature, so radiators fill up ever more slowly:
            # empty at the bottom of the scale, half full at its top
            heat = max(temp - low, 0.0)
            fill.set_height(0.12 * heat / (heat + high - low))
            if label is not None:
                label.set_text(f'{h} {temp:.0f}°')

        self.ax.set_title(f'{comfortable} / {occupied_zones} occupied zones in range',
                          fontsize=13, fontweight='bold',
                          color=OK_COLOR if comfortable == occupied_zones else BAD_COLOR)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
