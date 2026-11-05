import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

WAITING_COLOR = '#eceff1'
WAITING_EDGE_COLOR = '#90a4ae'
READY_COLOR = '#bbdefb'
READY_EDGE_COLOR = '#1565c0'
PASSED_COLOR = '#c8e6c9'
PASSED_EDGE_COLOR = '#2e7d32'
FAILED_COLOR = '#ffcdd2'
FAILED_EDGE_COLOR = '#c62828'
REQUIRED_COLOR = '#263238'
UNMET_EDGE_COLOR = '#b0bec5'
DETAIL_COLOR = '#37474f'

# sizes of the graph in inches before it is scaled to fit the figure, which are tight
# since the arrows between the rows of a dense curriculum take up most of the room
NODE_WIDTH = 0.95
NODE_HEIGHT = 0.42
ROW_PITCH = 0.72
NODE_GAP = 0.14
PASSING_WIDTH = 0.02
PASSING_GAP = 0.05
ARC_HEIGHT = 0.2
PORT_SPACING = 0.14
PORT_SPAN = 0.8
ORDERING_SWEEPS = 8

# more courses than this at one level continue on another row, as they would not be readable
MAX_ROW_COURSES = 8

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class AdvisingVisualizer(BaseViz):
    '''Draws the prerequisite graph of the courses with the foundations on top, where each
    course is coloured by how the student stands with it and shows the chance of passing
    and the cost of taking it now.

    Courses being taken are not part of the state, so they only show as passed or failed
    afterwards. The POMDP shows which courses are really passed rather than the delayed
    grades that are observed. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_scale=1.25,
                 max_size=(9.0, 5.5),
                 min_size=(8.2, 2.8),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)

        self._courses = list(model.type_to_objects['course'])
        self._prereqs = {course: [other for other in self._courses
                                  if self._nonfluent('PREREQ', other, course)]
                         for course in self._courses}
        self._edges = [(prereq, course) for course in self._courses
                       for prereq in self._prereqs[course]]
        self._required = {course for course in self._courses
                          if self._nonfluent('PROGRAM_REQUIREMENT', course)}

        # lay out the graph, then scale it to fit the figure
        self._layout()
        xs = [x for x in self._x.values()]
        graph_width = max(xs) - min(xs) + NODE_WIDTH + 0.1
        arc_height = ARC_HEIGHT if any(self._row[a] == self._row[b] for (a, b) in self._edges) else 0
        graph_height = (len(self._rows) - 1) * ROW_PITCH + NODE_HEIGHT + 2 * arc_height + 0.1
        scale = min(max_scale, max_size[0] / graph_width, max_size[1] / graph_height)

        # title above, then the graph, the legend and the caption
        bottom, top = 0.62, 0.45
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

        # the edges and courses never move, so they are only drawn once and then recolored
        self._arrows = {edge: self.ax.add_patch(patches.FancyArrowPatch(
            path=self._edge_path(edge), arrowstyle='-|>', mutation_scale=7 * scale,
            linewidth=0.9, shrinkA=0, shrinkB=0, zorder=1)) for edge in self._edges}

        self._nodes, self._details = {}, {}
        name_limit = 0.9 * NODE_WIDTH * scale * 72 / 0.62
        detail_size = 7.5 * min(scale, 1)
        for course in self._courses:
            x, y = self._x[course], self._y(course)
            required = course in self._required
            self._nodes[course] = self.ax.add_patch(patches.FancyBboxPatch(
                (x - NODE_WIDTH / 2, y - NODE_HEIGHT / 2), NODE_WIDTH, NODE_HEIGHT,
                boxstyle='round,pad=0,rounding_size=0.07', zorder=2,
                edgecolor=REQUIRED_COLOR if required else 'none',
                linewidth=2.2 if required else 1.2))

            name_size = min(9 * min(scale, 1), name_limit / len(course))
            has_detail = detail_size >= MIN_FONT_SIZE
            self.ax.text(
                x, y + (0.08 if has_detail else 0),
                self._shorten(course, int(name_limit / MIN_FONT_SIZE)), ha='center', va='center',
                fontsize=max(name_size, MIN_FONT_SIZE), fontweight='bold', zorder=3)
            if has_detail:
                self._details[course] = self.ax.text(
                    x, y - 0.1, '', ha='center', va='center', fontsize=detail_size,
                    color=DETAIL_COLOR, zorder=3)

        self.fig.legend(handles=[
            patches.Patch(facecolor=WAITING_COLOR, edgecolor=WAITING_EDGE_COLOR, label='Waiting'),
            patches.Patch(facecolor=READY_COLOR, edgecolor=READY_EDGE_COLOR, label='Ready'),
            patches.Patch(facecolor=FAILED_COLOR, edgecolor=FAILED_EDGE_COLOR, label='Failed'),
            patches.Patch(facecolor=PASSED_COLOR, edgecolor=PASSED_EDGE_COLOR, label='Passed'),
            patches.Patch(facecolor='none', edgecolor=REQUIRED_COLOR, linewidth=2, label='Required'),
            Line2D([], [], color=PASSED_EDGE_COLOR, marker='>', markersize=4, label='Prerequisite passed'),
            Line2D([], [], color=UNMET_EDGE_COLOR, marker='>', markersize=4, label='Prerequisite not passed')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / height), ncol=7, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)

        penalty = self._nonfluent('PROGRAM_INCOMPLETE_PENALTY')
        self.fig.text(0.5, 0.1 / height,
                      f'Percentages are the chances of passing if taken now, and an incomplete '
                      f'program costs {-penalty:g} per step',
                      ha='center', va='bottom', fontsize=9, color=DETAIL_COLOR)

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

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

        # each course goes below all of its prerequisites, and a cycle of prerequisites has
        # no such place so the one course missing the fewest of them goes first to break it
        self._rows, self._row = [], {}
        remaining = list(self._courses)
        while remaining:
            missing = {course: sum(prereq not in self._row and prereq != course
                                   for prereq in self._prereqs[course]) for course in remaining}
            row = [course for course in remaining if missing[course] == 0]
            if not row:
                row = [min(remaining, key=missing.get)]
            remaining = [course for course in remaining if course not in row]
            for start in range(0, len(row), MAX_ROW_COURSES):
                part = row[start:start + MAX_ROW_COURSES]
                self._row.update((course, len(self._rows)) for course in part)
                self._rows.append(part)

        # an edge that crosses rows passes through a slot in every row in between, so that
        # it goes around the courses there rather than behind them
        self._passing = {}
        neighbors = {course: [] for course in self._courses}
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

        # edges meeting a course on the same side connect in the order they arrive from,
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
        for ((course, _), entries) in sides.items():
            spacing = min(PORT_SPACING, PORT_SPAN / len(entries))
            for (i, (_, _, edge, which)) in enumerate(sorted(entries)):
                self._ports[(edge, which)] = self._x[course] + spacing * (i - (len(entries) - 1) / 2)

    def _gap(self, a, b):
        if isinstance(a, str) and isinstance(b, str):
            return NODE_WIDTH + NODE_GAP
        if isinstance(a, str) or isinstance(b, str):
            return (NODE_WIDTH + PASSING_WIDTH) / 2 + PASSING_GAP * 2
        return PASSING_WIDTH + PASSING_GAP

    def _spread(self, wanted, row):

        # the closest positions to those wanted that keep the items in a row apart, as the
        # average of pushing them apart to the right and to the left
        gaps = [self._gap(a, b) for (a, b) in zip(row, row[1:])]
        right, left = list(wanted), list(wanted)
        for i, gap in enumerate(gaps):
            right[i + 1] = max(right[i + 1], right[i] + gap)
        for i, gap in reversed(list(enumerate(gaps))):
            left[i] = min(left[i], left[i + 1] - gap)
        return [(a + b) / 2 for (a, b) in zip(right, left)]

    def _edge_path(self, edge):
        (start, end) = edge
        (start_x, start_y), (end_x, end_y) = (self._x[start], self._y(start)), (self._x[end], self._y(end))

        # edges within a row arc around the courses, over them to the right and under
        # them to the left so that two opposite edges stay apart
        if start_y == end_y:
            sign = 1 if end_x >= start_x else -1
            edge_y = start_y + sign * NODE_HEIGHT / 2
            offset = sign * NODE_WIDTH / 4
            return Path([(start_x + offset, edge_y),
                         ((start_x + end_x) / 2, edge_y + sign * 2 * ARC_HEIGHT),
                         (end_x - offset, edge_y)],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])

        # edges across rows go straight past the courses in a row, and only curve in the
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

    def _pass_chance(self, course, passed):

        # as in the domain, a course with prerequisites gets closer to certain with each one
        # passed but never reaches it, and a course without them has its own chance
        prereqs = self._prereqs[course]
        if not prereqs:
            return self._nonfluent('PRIOR_PROB_PASS_NO_PREREQ', course)
        prior = self._nonfluent('PRIOR_PROB_PASS', course)
        return prior + (1 - prior) * sum(p in passed for p in prereqs) / (1 + len(prereqs))

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        passed = {course for course in self._courses
                  if state[RDDLPlanningModel.ground_var('passed', [course])]}

        for course in self._courses:
            taken = state[RDDLPlanningModel.ground_var('taken', [course])]
            if course in passed:
                fill, edge = PASSED_COLOR, PASSED_EDGE_COLOR
            elif taken:
                fill, edge = FAILED_COLOR, FAILED_EDGE_COLOR
            elif all(prereq in passed for prereq in self._prereqs[course]):
                fill, edge = READY_COLOR, READY_EDGE_COLOR
            else:
                fill, edge = WAITING_COLOR, WAITING_EDGE_COLOR

            self._nodes[course].set_facecolor(fill)
            if course not in self._required:
                self._nodes[course].set_edgecolor(edge)

            if course in self._details:
                if course in passed:
                    detail = 'Passed'
                else:
                    cost = -self._nonfluent('COURSE_RETAKE_COST' if taken else 'COURSE_COST', course)
                    detail = f'{self._pass_chance(course, passed):.0%} · cost {cost:.3g}'
                self._details[course].set_text(detail)

        # passed prerequisites stay on top of the others so that they can be followed
        for ((prereq, _), arrow) in self._arrows.items():
            met = prereq in passed
            arrow.set_color(PASSED_EDGE_COLOR if met else UNMET_EDGE_COLOR)
            arrow.set_zorder(1.5 if met else 1)

        done = sum(course in passed for course in self._required)
        if done == len(self._required):
            self.ax.set_title('Program complete', fontsize=13, fontweight='bold',
                              color=PASSED_EDGE_COLOR)
        else:
            self.ax.set_title(f'{done} / {len(self._required)} required courses passed',
                              fontsize=13, fontweight='bold', color='black')

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
