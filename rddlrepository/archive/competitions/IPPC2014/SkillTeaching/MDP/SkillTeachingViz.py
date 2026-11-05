import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

LOW_COLOR = '#eceff1'
LOW_EDGE_COLOR = '#90a4ae'
MEDIUM_COLOR = '#bbdefb'
MEDIUM_EDGE_COLOR = '#1565c0'
HIGH_COLOR = '#c8e6c9'
HIGH_EDGE_COLOR = '#2e7d32'
ALERT_COLOR = '#ff8f00'
WRONG_COLOR = '#c62828'
DETAIL_COLOR = '#37474f'

# sizes of the graph in inches before it is scaled to fit the figure
NODE_WIDTH = 1.05
NODE_HEIGHT = 0.56
ROW_PITCH = 1.0
NODE_GAP = 0.22
PASSING_WIDTH = 0.1
ARC_HEIGHT = 0.36
PORT_SPACING = 0.2
PORT_SPAN = 0.8
ORDERING_SWEEPS = 4

# more skills than this at one level continue on another row, as they would not be readable
MAX_ROW_SKILLS = 8

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class SkillTeachingVisualizer(BaseViz):
    '''Draws the prerequisite graph of the skills with the foundations on top, where each
    skill is coloured by the proficiency of the student and shows its weight in the reward.

    Skills being taught are ringed and show how the student's turn went. The POMDP shows
    the true proficiency rather than only what is observed. The drawing only depends on
    the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_scale=1.25,
                 max_size=(9.0, 5.5),
                 min_size=(6.6, 2.6),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._has_false_positives = 'fpos' in model.state_fluents

        self._skills = list(model.type_to_objects['skill'])
        self._prereqs = {skill: [other for other in self._skills
                                 if self._nonfluent('PRE_REQ', other, skill)]
                         for skill in self._skills}
        self._edges = [(prereq, skill) for skill in self._skills
                       for prereq in self._prereqs[skill]]

        # lay out the graph, then scale it to fit the figure
        self._layout()
        xs = [x for x in self._x.values()]
        graph_width = max(xs) - min(xs) + NODE_WIDTH + 0.1
        arc_height = ARC_HEIGHT if any(self._row[a] == self._row[b] for (a, b) in self._edges) else 0
        graph_height = (len(self._rows) - 1) * ROW_PITCH + NODE_HEIGHT + 2 * arc_height + 0.1
        scale = min(max_scale, max_size[0] / graph_width, max_size[1] / graph_height)

        # title above, then the graph and the legend
        bottom, top = 0.4, 0.45
        width = max(graph_width * scale + 0.4, min_size[0])
        height = max(graph_height * scale + bottom + top, min_size[1])
        self._figure_size = (width, height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, bottom / height, 1, (height - bottom - top) / height])
        center_x = (max(xs) + min(xs)) / 2
        center_y = -(len(self._rows) - 1) * ROW_PITCH / 2
        half_width = width / (2 * scale)
        half_height = (height - bottom - top) / (2 * scale)
        self.ax.set_xlim(center_x - half_width, center_x + half_width)
        self.ax.set_ylim(center_y - half_height, center_y + half_height)
        self.ax.axis('off')

        # the edges and skills never move, so they are only drawn once and then recolored
        self._arrows = {edge: self.ax.add_patch(patches.FancyArrowPatch(
            path=self._edge_path(edge), arrowstyle='-|>', mutation_scale=10 * scale,
            linewidth=1.3, shrinkA=0, shrinkB=0, zorder=1)) for edge in self._edges}

        self._nodes, self._details = {}, {}
        name_limit = 0.9 * NODE_WIDTH * scale * 72 / 0.62
        detail_size = 7.5 * min(scale, 1)
        for skill in self._skills:
            x, y = self._x[skill], self._y(skill)
            self._nodes[skill] = self.ax.add_patch(patches.FancyBboxPatch(
                (x - NODE_WIDTH / 2, y - NODE_HEIGHT / 2), NODE_WIDTH, NODE_HEIGHT,
                boxstyle='round,pad=0,rounding_size=0.09', zorder=2))

            name_size = min(10 * min(scale, 1), name_limit / len(skill))
            has_detail = detail_size >= MIN_FONT_SIZE
            self.ax.text(
                x, y + (0.1 if has_detail else 0),
                self._shorten(skill, int(name_limit / MIN_FONT_SIZE)), ha='center', va='center',
                fontsize=max(name_size, MIN_FONT_SIZE), fontweight='bold', zorder=3)
            if has_detail:
                self._details[skill] = self.ax.text(
                    x, y - 0.13, '', ha='center', va='center', fontsize=detail_size, zorder=3)

        self.fig.legend(handles=[
            patches.Patch(facecolor=LOW_COLOR, edgecolor=LOW_EDGE_COLOR, label='Low'),
            patches.Patch(facecolor=MEDIUM_COLOR, edgecolor=MEDIUM_EDGE_COLOR, label='Medium'),
            patches.Patch(facecolor=HIGH_COLOR, edgecolor=HIGH_EDGE_COLOR, label='High'),
            patches.Patch(facecolor='none', edgecolor=ALERT_COLOR, linewidth=2, label='Being taught'),
            Line2D([], [], color=HIGH_EDGE_COLOR, marker='>', markersize=4, label='Prerequisite met'),
            Line2D([], [], color=LOW_EDGE_COLOR, marker='>', markersize=4, label='Prerequisite not met')
        ], loc='lower center', ncol=6, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _fluent(self, state, var, skill):
        return state[RDDLPlanningModel.ground_var(var, [skill])]

    def _y(self, item):
        return -self._row[item] * ROW_PITCH

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _layout(self):

        # each skill goes below all of its prerequisites, and a cycle of prerequisites has
        # no such place so the one skill missing the fewest of them goes first to break it
        self._rows, self._row = [], {}
        remaining = list(self._skills)
        while remaining:
            missing = {skill: sum(prereq not in self._row and prereq != skill
                                  for prereq in self._prereqs[skill]) for skill in remaining}
            row = [skill for skill in remaining if missing[skill] == 0]
            if not row:
                row = [min(remaining, key=missing.get)]
            remaining = [skill for skill in remaining if skill not in row]
            for start in range(0, len(row), MAX_ROW_SKILLS):
                part = row[start:start + MAX_ROW_SKILLS]
                self._row.update((skill, len(self._rows)) for skill in part)
                self._rows.append(part)

        # an edge that crosses rows passes through a slot in every row in between, so that
        # it goes around the skills there rather than behind them
        self._passing = {}
        neighbors = {skill: [] for skill in self._skills}
        for edge in self._edges:
            (start, end) = edge
            step = 1 if self._row[end] > self._row[start] else -1
            passing = [(edge, row) for row in range(self._row[start] + step, self._row[end], step)]
            self._passing[edge] = passing if self._row[start] != self._row[end] else []
            chain = [start] + self._passing[edge] + [end]
            for item in self._passing[edge]:
                self._row[item] = item[1]
                self._rows[item[1]].append(item)
                neighbors[item] = []
            if self._row[start] != self._row[end]:
                for (a, b) in zip(chain, chain[1:]):
                    neighbors[a].append(b)
                    neighbors[b].append(a)

        # sweep down and up the rows to pull each item towards the ones it connects to
        self._x = {}
        for row in self._rows:
            self._x.update(zip(row, self._spread([0.0] * len(row), row)))
        for sweep in range(ORDERING_SWEEPS):
            rows = self._rows if sweep % 2 == 0 else self._rows[::-1]
            for row in rows:
                wanted = {item: np.mean([self._x[n] for n in neighbors[item]])
                          if neighbors[item] else self._x[item] for item in row}
                row.sort(key=wanted.get)
                self._x.update(zip(row, self._spread([wanted[item] for item in row], row)))

        # edges meeting a skill on the same side connect in the order they arrive from,
        # which keeps them from crossing and their arrowheads from piling up
        sides = {}
        for edge in self._edges:
            (start, end) = edge
            if self._row[start] != self._row[end]:
                chain = [start] + self._passing[edge] + [end]
                down = self._row[end] > self._row[start]
                sides.setdefault((start, down), []).append((self._x[chain[1]], id(edge), edge, 0))
                sides.setdefault((end, not down), []).append((self._x[chain[-2]], id(edge), edge, 1))
        self._ports = {}
        for ((skill, _), entries) in sides.items():
            spacing = min(PORT_SPACING, PORT_SPAN / len(entries))
            for (i, (_, _, edge, which)) in enumerate(sorted(entries)):
                self._ports[(edge, which)] = self._x[skill] + spacing * (i - (len(entries) - 1) / 2)

    def _item_width(self, item):
        return NODE_WIDTH if isinstance(item, str) else PASSING_WIDTH

    def _spread(self, wanted, row):

        # the closest positions to those wanted that keep the items in a row apart, as the
        # average of pushing them apart to the right and to the left
        gaps = [(self._item_width(a) + self._item_width(b)) / 2 + NODE_GAP
                for (a, b) in zip(row, row[1:])]
        right, left = list(wanted), list(wanted)
        for i, gap in enumerate(gaps):
            right[i + 1] = max(right[i + 1], right[i] + gap)
        for i, gap in reversed(list(enumerate(gaps))):
            left[i] = min(left[i], left[i + 1] - gap)
        return [(a + b) / 2 for (a, b) in zip(right, left)]

    def _edge_path(self, edge):
        (start, end) = edge
        (start_x, start_y), (end_x, end_y) = (self._x[start], self._y(start)), (self._x[end], self._y(end))

        # edges within a row arc around the skills, over them to the right and under
        # them to the left so that two opposite edges stay apart
        if start_y == end_y:
            sign = 1 if end_x >= start_x else -1
            edge_y = start_y + sign * NODE_HEIGHT / 2
            offset = sign * NODE_WIDTH / 4
            return Path([(start_x + offset, edge_y),
                         ((start_x + end_x) / 2, edge_y + sign * 2 * ARC_HEIGHT),
                         (end_x - offset, edge_y)],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])

        # edges across rows go straight past the skills in a row, and only curve in the
        # gaps between rows where nothing can be in the way
        sign = 1 if end_y < start_y else -1
        half = sign * NODE_HEIGHT / 2
        vertices = [(self._ports[(edge, 0)], start_y - half)]
        codes = [Path.MOVETO]
        for item in self._passing[edge] + [None]:
            x, y = (self._ports[(edge, 1)], end_y) if item is None else (self._x[item], self._y(item))
            (last_x, last_y) = vertices[-1]
            middle = (last_y + y + half) / 2
            vertices.extend([(last_x, middle), (x, middle), (x, y + half)])
            codes.extend([Path.CURVE4] * 3)
            if item is not None:
                vertices.append((x, y - half))
                codes.append(Path.LINETO)
        return Path(vertices, codes)

    def _outcome(self, state, skill):
        if self._fluent(state, 'answeredRight', skill):
            if self._has_false_positives and self._fluent(state, 'fpos', skill):
                return 'False positive', WRONG_COLOR
            return 'Right', HIGH_EDGE_COLOR
        if self._fluent(state, 'hintDelayVar', skill):
            if self._fluent(state, 'hintedRight', skill):
                return 'Hint worked', HIGH_EDGE_COLOR
            return 'Hint failed', WRONG_COLOR
        return 'Wrong', WRONG_COLOR

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        high = {skill for skill in self._skills if self._fluent(state, 'proficiencyHigh', skill)}

        for skill in self._skills:
            if skill in high:
                fill, edge = HIGH_COLOR, HIGH_EDGE_COLOR
            elif self._fluent(state, 'proficiencyMed', skill):
                fill, edge = MEDIUM_COLOR, MEDIUM_EDGE_COLOR
            else:
                fill, edge = LOW_COLOR, LOW_EDGE_COLOR

            taught = self._fluent(state, 'updateTurn', skill)
            node = self._nodes[skill]
            node.set_facecolor(fill)
            node.set_edgecolor(ALERT_COLOR if taught else edge)
            node.set_linewidth(2.5 if taught else 1.2)

            if skill in self._details:
                detail, color = (f'weight {self._nonfluent("SKILL_WEIGHT", skill):.2f}', DETAIL_COLOR)
                if taught:
                    detail, color = self._outcome(state, skill)
                self._details[skill].set_text(detail)
                self._details[skill].set_color(color)
                self._details[skill].set_fontweight('bold' if taught else 'normal')

        for ((prereq, skill), arrow) in self._arrows.items():
            arrow.set_color(HIGH_EDGE_COLOR if prereq in high else LOW_EDGE_COLOR)

        if len(high) == len(self._skills):
            self.ax.set_title('All skills at high proficiency', fontsize=13, fontweight='bold',
                              color=HIGH_EDGE_COLOR)
        else:
            self.ax.set_title(f'{len(high)} / {len(self._skills)} skills at high proficiency',
                              fontsize=13, fontweight='bold', color='black')

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
