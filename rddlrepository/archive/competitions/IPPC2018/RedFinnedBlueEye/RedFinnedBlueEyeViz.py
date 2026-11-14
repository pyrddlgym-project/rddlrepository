import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

# the water of a spring with each population, which is murky with Gambusia
WATER_COLORS = {
    'red-finned-blue-eye': '#b3e5fc',
    'gambusia': '#d4dcae',
    'none': '#e3f4fc'
}
BANK_COLOR = '#c5e1a5'
BANK_EDGE_COLOR = '#8bc34a'
RIPPLE_COLOR = '#81d4fa'
BLUE_EYE_COLOR = '#cfe3f5'
BLUE_EYE_EDGE_COLOR = '#5c7f9e'
FIN_COLOR = '#e53935'
EYE_COLOR = '#1e88e5'
GAMBUSIA_COLOR = '#a19f63'
GAMBUSIA_EDGE_COLOR = '#5d5b36'
GAMBUSIA_FIN_COLOR = '#7f7d4f'
SPREAD_COLOR = '#c62828'
QUIET_COLOR = '#b0bec5'
GOOD_COLOR = '#2e7d32'
BAD_COLOR = '#c62828'
DETAIL_COLOR = '#37474f'

# sizes of the graph in inches before it is scaled to fit the figure
NODE_WIDTH = 0.66
NODE_HEIGHT = 0.36
COLUMN_PITCH = 1.08
ROW_PITCH = 0.42
BLOCK_GAP = 0.5
ARC_BULGE = 0.34
SHORT_ARC_BULGE = 0.2
ARC_ROOM = 0.42
LABEL_INSET = 0.15
LABEL_WIDTH = 0.3
LABEL_HEIGHT = 0.13
LABEL_POSITIONS = (0.5, 0.35, 0.65, 0.25, 0.75)
GRAPH_PADDING = 0.1
MIN_LINE_WIDTH = 0.7
MAX_LINE_WIDTH = 2.6

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5
NAME_FONT_SIZE = 7.5
DETAIL_FONT_SIZE = 6.5
LABEL_FONT_SIZE = 6.5
CHAR_WIDTH = 0.62
LINE_HEIGHT = 1.3

# room around the figure
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45


class RedFinnedBlueEyeVisualizer(BaseViz):
    '''Draws the springs as ponds with the fish that live in them, as trees that branch out
    from the middle of each group of connected springs: slim red-finned blue-eyes, stout
    olive Gambusia in murky water, or only ripples in an empty spring. Each line shows the
    chance that the water connects two springs in the next step, and is red when Gambusia
    could spread along it.

    The high-water mark is not part of the state, so only its chances show. The drawing only
    depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_scale=1.2,
                 max_size=(9.0, 5.5),
                 min_size=(9.0, 2.8),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._springs = list(model.type_to_objects['spring'])
        self._levels = list(model.type_to_objects['high-water-level'])
        self._links = self._find_links()
        self._neighbors = {spring: [] for spring in self._springs}
        for (a, b) in self._links:
            self._neighbors[a].append(b)
            self._neighbors[b].append(a)

        # lay out the trees and scale them to fit the figure, but so many springs that this
        # would make them unreadable have too many lines to follow, so they are listed by
        # group instead
        (places, graph_width, graph_height, scale) = self._layout(max_size, max_scale)
        self._has_lines = DETAIL_FONT_SIZE * min(scale, 1) >= MIN_FONT_SIZE

        self._ponds, self._lines = {}, {}
        if self._has_lines:
            self._places = places
            self._draw_graph(graph_width, graph_height, scale, min_size)
        else:
            self._draw_list(max_size, min_size)
        height = self._figure_size[1]

        self._draw_legend()

        points = self._nonfluent('ACTION_POINTS')
        lines = 'Percentages are chances that springs connect. ' if self._has_lines else ''
        self.fig.text(0.5, 0.1 / height,
                      f'{lines}Each step has {points:g} '
                      f'point{"" if points == 1 else "s"}: poison 1 '
                      f'({self._nonfluent("POISON_SUCCESS_PROB"):.0%}, '
                      f'penalty {self._nonfluent("PENALTY_POISON_USAGE"):g}), by hand 2 '
                      f'({self._nonfluent("REMOVE_MANUALLY_SUCCESS_PROB"):.0%}), translocate 3 '
                      f'({self._nonfluent("TRANSLOCATION_SUCCESS_PROB"):.0%})',
                      ha='center', va='bottom', fontsize=9, color=DETAIL_COLOR)

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _connected(self, a, b, level):
        return (self._nonfluent('SPRINGS_CONNECTED', a, b, level)
                or self._nonfluent('SPRINGS_CONNECTED', b, a, level))

    def _find_links(self):

        # either way round connects two springs, and only at high-water levels that can occur
        links = {}
        for (i, a) in enumerate(self._springs):
            for b in self._springs[i + 1:]:
                chance = sum(self._nonfluent('HIGH_WATER_PROB', level) for level in self._levels
                             if self._connected(a, b, level))
                if chance > 0:
                    links[(a, b)] = chance
        return links

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _reach(self, start, without=None):

        # all springs that can be reached from one, with how many steps it takes
        steps = {start: 0}
        queue = [start]
        for spring in queue:
            for other in self._neighbors[spring]:
                if other not in steps and other != without:
                    steps[other] = steps[spring] + 1
                    queue.append(other)
        return steps

    def _groups(self):
        groups, seen = [], set()
        for spring in self._springs:
            if spring not in seen:
                group = list(self._reach(spring))
                seen.update(group)
                groups.append(group)
        return groups

    def _center(self, group):

        # the spring that is the fewest steps from the rest makes for the shallowest tree
        return min(group, key=lambda spring: (max(self._reach(spring).values()),
                                              -len(self._neighbors[spring])))

    def _tree(self, root):

        # each spring hangs from the first one that reaches it, so that any other connection
        # is between springs that are in the same column or next to each other
        children = {root: []}
        queue = [root]
        for spring in queue:
            for other in self._neighbors[spring]:
                if other not in children:
                    children[other] = []
                    children[spring].append(other)
                    queue.append(other)
        return children

    def _side(self, roots, children, place, direction):

        # leaves go one per row, and every other spring goes in the middle of its children,
        # which keeps the lines of a tree from crossing
        next_row = [0]

        def visit(spring, column):
            if not children[spring]:
                row = next_row[0]
                next_row[0] += 1
            else:
                rows = [visit(child, column + direction) for child in children[spring]]
                row = (min(rows) + max(rows)) / 2
            place[spring] = (column, row)
            return row

        rows = [visit(root, direction) for root in roots]
        return next_row[0], ((min(rows) + max(rows)) / 2 if rows else 0)

    def _block(self, root, two_sided):
        children = self._tree(root)
        leaves = {}

        def count(spring):
            leaves[spring] = sum(count(child) for child in children[spring]) or 1
            return leaves[spring]

        count(root)

        # the children of the root that connect to each other in some other way stay on the
        # same side of it, so that no line has to cross the column of the root
        sides = ([], [])
        units = {}
        for child in children[root]:
            reached = self._reach(child, without=root)
            unit = next((key for key in units if key in reached), child)
            units.setdefault(unit, []).append(child)
        for unit in sorted(units.values(), key=lambda u: -sum(leaves[c] for c in u)):
            load = [sum(leaves[c] for c in side) for side in sides]
            sides[1 if two_sided and load[1] < load[0] else 0].extend(unit)

        place = {}
        (rows_right, middle_right) = self._side(sides[0], children, place, 1)
        before = dict(place)
        (rows_left, middle_left) = self._side(sides[1], children, place, -1)

        # the two sides are centered around the root
        offset = middle_right - middle_left
        for spring in place:
            if spring not in before:
                (column, row) = place[spring]
                place[spring] = (column, row + offset)
        place[root] = (0, middle_right)

        left = min(column for (column, _) in place.values())
        top = min(row for (_, row) in place.values())
        place = {spring: (column - left, row - top) for (spring, (column, row)) in place.items()}
        last = max(column for (column, _) in place.values())
        width = last * COLUMN_PITCH + NODE_WIDTH
        height = max(row for (_, row) in place.values()) * ROW_PITCH + NODE_HEIGHT

        # an arc in the last column needs room of its own, or it would reach what comes next
        if any(a in place and b in place and place[a][0] == last and place[b][0] == last
               for (a, b) in self._links):
            width += ARC_ROOM
        return (width, height, place)

    def _pack(self, blocks, limit, stacked):

        # blocks go next to each other until they reach the limit, and then start a new row,
        # or the same with rows and columns swapped
        along, across, room = 0, 0, 0
        places = {}
        total_x, total_y = 0, 0
        for (width, height, place) in blocks:
            size, depth = (height, width) if stacked else (width, height)
            if along > 0 and along + size > limit:
                across += room + BLOCK_GAP
                along, room = 0, 0
            (dx, dy) = (across, along) if stacked else (along, across)
            for (spring, (column, row)) in place.items():
                places[spring] = (dx + column * COLUMN_PITCH + NODE_WIDTH / 2,
                                  dy + row * ROW_PITCH + NODE_HEIGHT / 2)
            total_x, total_y = max(total_x, dx + width), max(total_y, dy + height)
            along += size + BLOCK_GAP
            room = max(room, depth)
        return places, total_x, total_y

    def _layout(self, max_size, max_scale):

        # try trees with one and two sides, and all ways of breaking the groups across rows
        # or columns, for what comes out the largest after scaling to fit the figure
        roots = [self._center(group) for group in self._groups()]
        best = None
        for two_sided in (False, True):
            blocks = sorted((self._block(root, two_sided) for root in roots),
                            key=lambda block: (-block[1], -block[0]))
            for stacked in (False, True):
                sizes = [block[1] if stacked else block[0] for block in blocks]
                limit = max(sizes)
                while True:
                    (places, width, height) = self._pack(blocks, limit, stacked)
                    scale = min(max_scale, max_size[0] / (width + GRAPH_PADDING),
                                max_size[1] / (height + GRAPH_PADDING))
                    if best is None or scale > best[3] + 1e-9:
                        best = (places, width, height, scale)
                    if limit >= sum(sizes) + BLOCK_GAP * len(sizes):
                        break
                    limit += NODE_WIDTH if not stacked else NODE_HEIGHT
        return best

    @staticmethod
    def _fish(ax, x, y, length, kind, facing, up):

        # a fish seen from the side, facing right or left, where up is the sign of up on
        # the screen: blue-eyes are slim, with red fins and a blue eye, and Gambusia are
        # stout and olive, with a round tail
        L, d = length, facing

        def at(dx, dy):
            return (x + d * dx * L, y + up * dy * L)

        if kind == 'red-finned-blue-eye':
            fins = [[at(-0.33, 0), at(-0.55, 0.17), at(-0.46, 0), at(-0.55, -0.17)],
                    [at(-0.04, 0.1), at(-0.22, 0.26), at(-0.24, 0.08)],
                    [at(-0.04, -0.1), at(-0.2, -0.22), at(-0.22, -0.08)]]
            artists = [ax.add_patch(patches.Polygon(points, closed=True, facecolor=FIN_COLOR,
                                                    edgecolor='none', zorder=4))
                       for points in fins]
            artists += [
                ax.add_patch(patches.Ellipse((x, y), 0.76 * L, 0.26 * L, facecolor=BLUE_EYE_COLOR,
                                             edgecolor=BLUE_EYE_EDGE_COLOR, linewidth=0.6, zorder=4.1)),
                ax.add_patch(patches.Circle(at(0.24, 0.02), 0.055 * L, facecolor=EYE_COLOR,
                                            edgecolor='white', linewidth=0.4, zorder=4.2))]
        else:
            angles = (140, 220) if d > 0 else (-40, 40)
            artists = [
                ax.add_patch(patches.Wedge(at(-0.3, 0), 0.24 * L, *angles,
                                           facecolor=GAMBUSIA_FIN_COLOR, edgecolor='none', zorder=4)),
                ax.add_patch(patches.Polygon([at(-0.08, 0.13), at(-0.2, 0.24), at(-0.24, 0.1)],
                                             closed=True, facecolor=GAMBUSIA_FIN_COLOR,
                                             edgecolor='none', zorder=4)),
                ax.add_patch(patches.Ellipse((x, y), 0.68 * L, 0.36 * L, facecolor=GAMBUSIA_COLOR,
                                             edgecolor=GAMBUSIA_EDGE_COLOR, linewidth=0.6, zorder=4.1)),
                ax.add_patch(patches.Circle(at(0.2, 0.04), 0.045 * L, facecolor='black',
                                            edgecolor='none', zorder=4.2))]
        return artists

    @classmethod
    def _pond(cls, ax, x, y, width, height, up):

        # a pond on its bank, with two fish of each kind and ripples, all hidden: the
        # pictures sit below the middle so that a name fits above them
        bank = ax.add_patch(patches.Ellipse((x, y), width, height, facecolor=BANK_COLOR,
                                            edgecolor=BANK_EDGE_COLOR, linewidth=1, zorder=2))
        water = ax.add_patch(patches.Ellipse((x, y), 0.88 * width, 0.78 * height,
                                             facecolor=WATER_COLORS['none'], edgecolor='none',
                                             zorder=2.1))
        length = min(0.36 * width, 0.9 * height)
        low = y - up * 0.12 * height
        pictures = {kind: cls._fish(ax, x - 0.16 * width, low + up * 0.04 * height, length, kind, 1, up)
                    + cls._fish(ax, x + 0.17 * width, low - up * 0.08 * height, length, kind, -1, up)
                    for kind in ('red-finned-blue-eye', 'gambusia')}
        pictures['none'] = [ax.add_patch(patches.Ellipse(
            (x, low), size * width, size * 0.45 * height, facecolor='none',
            edgecolor=RIPPLE_COLOR, linewidth=0.8, zorder=2.5)) for size in (0.18, 0.36)]
        for artists in pictures.values():
            for artist in artists:
                artist.set_visible(False)
        return bank, water, pictures

    @staticmethod
    def _show_pond(pond, kind):
        (_, water, pictures) = pond
        water.set_facecolor(WATER_COLORS[kind])
        for (other, artists) in pictures.items():
            for artist in artists:
                artist.set_visible(other == kind)

    def _add_spring(self, spring, x, y, width, height, scale):

        # the first row goes on top, so that up is a smaller y here
        self._ponds[spring] = self._pond(self.ax, x, y, width, height, -1)
        fonts = min(scale, 1)
        room = 0.6 * width * scale * 72 / CHAR_WIDTH
        if room / MIN_FONT_SIZE < 2:
            return
        name_size = max(min(NAME_FONT_SIZE * fonts, room / len(spring)), MIN_FONT_SIZE)
        if name_size * LINE_HEIGHT / 72 <= 0.45 * height * scale:
            self.ax.text(x, y - 0.2 * height, self._shorten(spring, int(room / MIN_FONT_SIZE)),
                         ha='center', va='center', fontsize=name_size, fontweight='bold',
                         color='#37474f', zorder=3)

    def _draw_legend(self):

        # the legend shows the same ponds and lines as the springs, in a row of pictures
        (width, height) = self._figure_size
        reward = self._nonfluent('REWARD_RED_FINNED_BLUE_EYE_POPULATION')
        entries = [('red-finned-blue-eye', f'Blue-eyes, +{reward:g} per step'),
                   ('gambusia', 'Gambusia'), ('none', 'Empty')]
        if self._has_lines:
            entries += [('spread', 'Gambusia could spread'), ('quiet', 'No threat')]
        widths = [0.6 + 0.066 * len(label) + 0.3 for (_, label) in entries]
        shrink = min(1.0, (width - 0.2) / sum(widths))
        ax = self.fig.add_axes([0, 0.24 / height, 1, 0.34 / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, 0.34)
        ax.set_aspect('equal')
        ax.axis('off')
        x = (width - shrink * sum(widths)) / 2
        for ((kind, label), entry_width) in zip(entries, widths):
            if kind in WATER_COLORS:
                self._show_pond(self._pond(ax, x + 0.26, 0.15, 0.52, 0.3, 1), kind)
            else:
                ax.plot([x + 0.04, x + 0.48], [0.17, 0.17], linewidth=1.6,
                        color=SPREAD_COLOR if kind == 'spread' else QUIET_COLOR)
            ax.text(x + 0.58, 0.17, label, ha='left', va='center', fontsize=8.5, color=DETAIL_COLOR)
            x += shrink * entry_width

    def _open_figure(self, width, height):

        # title above, then the springs, the legend and the caption
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, BOTTOM_MARGIN / height, 1,
                                     (height - BOTTOM_MARGIN - TOP_MARGIN) / height])
        self.ax.axis('off')

    def _draw_graph(self, graph_width, graph_height, scale, min_size):
        width = max(graph_width * scale + 0.4, min_size[0])
        height = max((graph_height + GRAPH_PADDING) * scale + BOTTOM_MARGIN + TOP_MARGIN, min_size[1])
        self._open_figure(width, height)
        half_width = width / (2 * scale)
        half_height = (height - BOTTOM_MARGIN - TOP_MARGIN) / (2 * scale)
        self.ax.set_xlim(graph_width / 2 - half_width, graph_width / 2 + half_width)

        # the first row goes on top
        self.ax.set_ylim(graph_height / 2 + half_height, graph_height / 2 - half_height)

        for spring in self._springs:
            (x, y) = self._places[spring]
            self._add_spring(spring, x, y, NODE_WIDTH, NODE_HEIGHT, scale)

        # arcs have only one place for a label, so they go first
        self._labels = []
        arcs_first = sorted(self._links, key=lambda link: self._places[link[0]][0] != self._places[link[1]][0])
        for link in arcs_first:
            self._add_line(link, self._links[link], scale)

    def _curve_point(self, vertices, t):
        weights = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3)
        return (sum(w * x for (w, (x, _)) in zip(weights, vertices)),
                sum(w * y for (w, (_, y)) in zip(weights, vertices)))

    def _label_is_free(self, point):
        return all(abs(point[0] - x) > LABEL_WIDTH or abs(point[1] - y) > LABEL_HEIGHT
                   for (x, y) in self._labels)

    def _add_line(self, link, chance, scale):
        (a, b) = link
        (ax, ay), (bx, by) = self._places[a], self._places[b]
        half = NODE_WIDTH / 2

        # lines only run in the gaps between columns, so they never pass behind a spring
        if abs(ax - bx) < 1e-9:

            # a short arc stays inside of a longer one around it
            right = ax + half
            bulge = ARC_BULGE if abs(ay - by) > ROW_PITCH * 1.5 else SHORT_ARC_BULGE
            vertices = [(right, ay), (right + bulge, ay), (right + bulge, by), (right, by)]
            places = []
        else:
            if ax > bx:
                (a, b), (ax, ay), (bx, by) = (b, a), (bx, by), (ax, ay)
            (start, end) = (ax + half, bx - half)
            middle = (start + end) / 2
            vertices = [(start, ay), (middle, ay), (middle, by), (end, by)]

            # lines fan out from a spring with many of them, so the label goes at the other end
            # where it has a row to itself, unless another line already took that place
            ends = {a: (start + LABEL_INSET, ay), b: (end - LABEL_INSET, by)}
            order = sorted((b, a), key=lambda spring: len(self._neighbors[spring]))
            places = [ends[spring] for spring in order]

        # otherwise anywhere along the line that is free, or else the middle
        places += [self._curve_point(vertices, t) for t in LABEL_POSITIONS]
        label = next((point for point in places if self._label_is_free(point)), self._curve_point(vertices, 0.5))
        self._labels.append(label)

        width = MIN_LINE_WIDTH + (MAX_LINE_WIDTH - MIN_LINE_WIDTH) * chance
        line = self.ax.add_patch(patches.PathPatch(
            Path(vertices, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]), facecolor='none',
            linewidth=width * min(scale, 1), zorder=1))
        text = self.ax.text(label[0], label[1], f'{chance:.0%}', ha='center', va='center',
                            fontsize=max(LABEL_FONT_SIZE * min(scale, 1), MIN_FONT_SIZE), zorder=1.8,
                            bbox={'boxstyle': 'round,pad=0.12', 'facecolor': 'white',
                                  'edgecolor': 'none'})
        self._lines[link] = (line, text)

    def _draw_list(self, max_size, min_size):

        # the springs of a group stay next to each other, in rows that fill the figure
        count = len(self._springs)
        columns = max(1, math.ceil(math.sqrt(count * max_size[0] / max_size[1] * NODE_HEIGHT / NODE_WIDTH)))
        rows = math.ceil(count / columns)
        cell_x = min(NODE_WIDTH + 0.04, max_size[0] / columns)
        cell_y = min(NODE_HEIGHT + 0.04, max_size[1] / rows)
        width = max(columns * cell_x + 0.4, min_size[0])
        height = max(rows * cell_y + BOTTOM_MARGIN + TOP_MARGIN, min_size[1])
        self._open_figure(width, height)
        self.ax.set_xlim(columns * cell_x / 2 - width / 2, columns * cell_x / 2 + width / 2)
        room = height - BOTTOM_MARGIN - TOP_MARGIN
        self.ax.set_ylim(rows * cell_y / 2 + room / 2, rows * cell_y / 2 - room / 2)

        ordered = [spring for group in self._groups() for spring in group]
        for (i, spring) in enumerate(ordered):
            self._add_spring(spring, (i % columns + 0.5) * cell_x, (i // columns + 0.5) * cell_y,
                             0.9 * cell_x, 0.9 * cell_y, 1)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        population = {spring: str(state[fluent('population', [spring])]) for spring in self._springs}
        gambusia = [spring for spring in self._springs if population[spring] == 'gambusia']
        natives = sum(kind == 'red-finned-blue-eye' for kind in population.values())

        for spring in self._springs:
            self._show_pond(self._ponds[spring], population[spring])

        # lines that Gambusia could spread along stay on top of the others
        for ((a, b), (line, text)) in self._lines.items():
            spreading = (population[a] == 'gambusia') != (population[b] == 'gambusia')
            line.set_edgecolor(SPREAD_COLOR if spreading else QUIET_COLOR)
            line.set_zorder(1.5 if spreading else 1)
            text.set_color(SPREAD_COLOR if spreading else DETAIL_COLOR)

        total = len(self._springs)
        springs = f'{natives} / {total} spring{"" if total == 1 else "s"} with blue-eyes'
        if natives == 0:
            penalty = self._nonfluent('PENALTY_RED_FINNED_BLUE_EYE_EXTINCT')
            title, color = f'Blue-eyes extinct, penalty {penalty:g} per step', BAD_COLOR
        elif not gambusia:
            title, color = f'No Gambusia left, {springs}', GOOD_COLOR
        else:
            wait = int(str(state['translocate-possible-in']))
            if wait > 0:
                move = f'translocation in {wait} step{"" if wait == 1 else "s"}'
            elif 'none' in population.values():
                move = 'translocation ready'
            else:
                move = 'no empty spring'
            title, color = f'{springs}, {move}', 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
