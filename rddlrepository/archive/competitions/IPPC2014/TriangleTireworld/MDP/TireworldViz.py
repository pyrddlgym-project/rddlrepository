import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

LOCATION_COLOR = '#eceff1'
LOCATION_EDGE_COLOR = '#90a4ae'
SPARE_COLOR = '#bbdefb'
SPARE_EDGE_COLOR = '#263238'
GOAL_COLOR = '#c8e6c9'
GOAL_EDGE_COLOR = '#2e7d32'
ROAD_COLOR = '#b0bec5'
ACTIVE_ROAD_COLOR = '#1565c0'
CAR_COLOR = '#ffe0b2'
CAR_EDGE_COLOR = '#ef6c00'
FLAT_COLOR = '#ffcdd2'
FLAT_EDGE_COLOR = '#c62828'
WINDOW_COLOR = '#37474f'
DETAIL_COLOR = '#37474f'

# more locations than this at one distance from the start continue in the next column
MAX_COLUMN_LOCATIONS = 6

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class TireworldVisualizer(BaseViz):
    '''Draws the roads flowing from left to right, with the locations that have a spare,
    the goal, and the car, which turns red when it has a flat tire.

    The map comes from the roads rather than the names of the locations. Roads forming a
    triangular lattice are drawn as that triangle, and any other roads in columns by their
    distance from the start. The POMDP shows the true car, spares and tire rather than only
    the noisy observations. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_step=(0.65, 1.1),
                 max_size=(9.0, 5.5),
                 min_width=8.2,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._locations = list(model.type_to_objects['location'])
        self._roads = [(a, b) for a in self._locations for b in self._locations
                       if nonfluents[RDDLPlanningModel.ground_var('road', [a, b])]]
        self._goals = {location for location in self._locations
                       if nonfluents[RDDLPlanningModel.ground_var('goal-location', [location])]}
        self._moves_to_goal = self._find_moves_to_goal()

        # lay out the map, then fit it to the figure
        places = self._lattice_places() or self._column_places()
        columns = [column for (column, _) in places.values()]
        rows = [row for (_, row) in places.values()]
        num_columns, num_rows = max(columns) - min(columns), max(rows) - min(rows)
        step_x = min(max_step[0], max_size[0] / (num_columns + 1))
        step_y = min(max_step[1], max_size[1] / (num_rows + 1.4))
        self._places = {location: (column * step_x, row * step_y)
                        for (location, (column, row)) in places.items()}
        points = list(self._places.values())
        closest = min((math.dist(a, b) for (i, a) in enumerate(points) for b in points[i + 1:]),
                      default=1.0)
        self._radius = min(0.3, 0.4 * closest)

        # title above, then the map, the legend and the caption
        bottom, top = 0.62, 0.45
        map_width, map_height = num_columns * step_x, num_rows * step_y
        margin = 2 * self._radius + 0.15
        loop_room = 1.2 * self._radius if any(a == b for (a, b) in self._roads) else 0
        width = max(map_width + margin + 0.25, min_width)
        height = map_height + margin + loop_room + bottom + top
        self._figure_size = (width, height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, bottom / height, 1, (height - bottom - top) / height])
        center_x = (min(columns) + max(columns)) * step_x / 2
        center_y = (min(rows) + max(rows)) * step_y / 2 + loop_room / 2
        self.ax.set_xlim(center_x - width / 2, center_x + width / 2)
        self.ax.set_ylim(center_y - (height - bottom - top) / 2, center_y + (height - bottom - top) / 2)
        self.ax.axis('off')

        # the roads and locations never move, so they are only drawn once and then recolored
        self._arrows = {road: self._draw_road(*road) for road in self._roads}
        self._nodes = {}
        name_fit = 0.9 * 2 * self._radius * 72 / 0.62
        for location in self._locations:
            (x, y) = self._places[location]
            self._nodes[location] = self.ax.add_patch(patches.Circle(
                (x, y), self._radius, zorder=2))
            name = self._shorten(location, int(name_fit / MIN_FONT_SIZE))
            size = min(8, name_fit / len(name))
            if size >= MIN_FONT_SIZE:
                self.ax.text(x, y, name, ha='center', va='center', fontsize=size,
                             color=DETAIL_COLOR, zorder=3)

        self.fig.legend(handles=[
            patches.Patch(facecolor=CAR_COLOR, edgecolor=CAR_EDGE_COLOR, label='Car'),
            patches.Patch(facecolor=FLAT_COLOR, edgecolor=FLAT_EDGE_COLOR, label='Flat tire'),
            patches.Patch(facecolor=SPARE_COLOR, edgecolor=SPARE_EDGE_COLOR, linewidth=2, label='Spare here'),
            patches.Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_EDGE_COLOR, label='Goal'),
            Line2D([], [], color=ROAD_COLOR, marker='>', markersize=4, label='Road'),
            Line2D([], [], color=ACTIVE_ROAD_COLOR, marker='>', markersize=4, label='Road car can take')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / height), ncol=6, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)

        # the domain draws having no flat with this chance, despite its name
        keep_tire = nonfluents[RDDLPlanningModel.ground_var('FLAT-PROB', [])]
        self.fig.text(0.5, 0.1 / height,
                      f'Each move gets a flat tire with chance {1 - keep_tire:.3g}, and a spare '
                      f'has to be loaded at its location to change one',
                      ha='center', va='bottom', fontsize=9, color=DETAIL_COLOR)

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _find_moves_to_goal(self):

        # the fewest moves to a goal, going backwards along the roads from all of the goals
        moves = {goal: 0 for goal in self._goals}
        frontier = list(self._goals)
        while frontier:
            next_frontier = []
            for (a, b) in self._roads:
                if b in frontier and a not in moves:
                    moves[a] = moves[b] + 1
                    next_frontier.append(a)
            frontier = next_frontier
        return moves

    def _lattice_places(self):

        # in a triangular lattice each road goes right across two columns and stays in its
        # row, or across one column and goes one row up or down, which leaves one place for
        # each location if the columns are the longest paths from the start
        previous = {location: [a for (a, b) in self._roads if b == location]
                    for location in self._locations}
        columns, remaining = {}, list(self._locations)
        while remaining:
            ready = [location for location in remaining
                     if all(a in columns for a in previous[location])]
            if not ready:
                return None
            for location in ready:
                columns[location] = max((columns[a] + 1 for a in previous[location]), default=0)
            remaining = [location for location in remaining if location not in ready]

        rows = {}
        for column in sorted(set(columns.values())):
            options = {}
            for location in (other for other in self._locations if columns[other] == column):
                choices = {0}
                for (i, a) in enumerate(previous[location]):
                    span = column - columns[a]
                    if span > 2:
                        return None
                    choice = {rows[a]} if span == 2 else {rows[a] - 1, rows[a] + 1}
                    choices = choice if i == 0 else choices & choice
                options[location] = choices

            # the locations with the fewest choices settle which are left for the others,
            # and the triangle has its top above the road from the start to the goal
            taken = set()
            for location in sorted(options, key=lambda other: len(options[other])):
                if not options[location] - taken:
                    return None
                rows[location] = max(options[location] - taken)
                taken.add(rows[location])

        return {location: (columns[location], rows[location]) for location in self._locations}

    def _column_places(self):

        # each location goes to the right of all roads into it, and a cycle has no such
        # place so the one location missing the fewest of them goes first to break it
        previous = {location: [a for (a, b) in self._roads if b == location and a != location]
                    for location in self._locations}
        levels, placed = [], set()
        remaining = list(self._locations)
        while remaining:
            missing = {location: sum(a not in placed for a in previous[location]) for location in remaining}
            level = [location for location in remaining if missing[location] == 0]
            if not level:
                level = [min(remaining, key=missing.get)]
            remaining = [location for location in remaining if location not in level]
            placed.update(level)
            levels.append(level)

        # each location goes towards the roads into it, and the columns are centered
        places, column = {}, 0
        for level in levels:
            for start in range(0, len(level), MAX_COLUMN_LOCATIONS):
                part = level[start:start + MAX_COLUMN_LOCATIONS]
                wanted = {}
                for (i, location) in enumerate(part):
                    rows = [places[a][1] for a in previous[location] if a in places]
                    wanted[location] = np.mean(rows) if rows else i
                ordered = sorted(part, key=wanted.get)
                for (i, location) in enumerate(ordered):
                    places[location] = (column, 2 * i - (len(ordered) - 1))
                column += 1
        return places

    def _is_clear(self, a, b):

        # whether a straight road between the two would stay clear of the other locations
        (ax, ay), (bx, by) = self._places[a], self._places[b]
        length = math.hypot(bx - ax, by - ay)
        for location in self._locations:
            if location not in (a, b):
                (x, y) = self._places[location]
                along = ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / (length * length)
                if 0 < along < 1 and math.hypot(ax + along * (bx - ax) - x, ay + along * (by - ay) - y) < 1.2 * self._radius:
                    return False
        return True

    def _draw_road(self, a, b):
        (x, y) = self._places[a]
        shrink = self._radius * 72 + 1
        style = dict(arrowstyle='-|>', mutation_scale=9, linewidth=1.2, zorder=1, color=ROAD_COLOR)

        # a road back to the same location loops over it
        if a == b:
            r = self._radius
            return self.ax.add_patch(patches.FancyArrowPatch(path=Path(
                [(x - 0.5 * r, y + 0.85 * r), (x - r, y + 2 * r), (x + r, y + 2 * r), (x + 0.5 * r, y + 0.85 * r)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]), shrinkA=0, shrinkB=0, **style))

        # a road curves when it would hide behind a location or a road the opposite way
        curved = (b, a) in self._roads or not self._is_clear(a, b)
        return self.ax.add_patch(patches.FancyArrowPatch(
            (x, y), self._places[b], connectionstyle=f'arc3,rad={0.25 if curved else 0}',
            shrinkA=shrink, shrinkB=shrink, **style))

    def _draw_car(self, x, y, flat):

        # the window is at the front of the car, as in the other domains with cars
        r = self._radius
        return [
            self.ax.add_patch(patches.FancyBboxPatch(
                (x - 0.85 * r, y - 0.55 * r), 1.7 * r, 1.1 * r, boxstyle=f'round,pad=0,rounding_size={0.25 * r}',
                facecolor=FLAT_COLOR if flat else CAR_COLOR, edgecolor=FLAT_EDGE_COLOR if flat else CAR_EDGE_COLOR,
                linewidth=1.8, zorder=4)),
            self.ax.add_patch(patches.FancyBboxPatch(
                (x + 0.25 * r, y - 0.35 * r), 0.35 * r, 0.7 * r, boxstyle=f'round,pad=0,rounding_size={0.08 * r}',
                facecolor=WINDOW_COLOR, edgecolor='none', zorder=5))
        ]

    def _title(self, state, cars, spares, flat, has_spare):
        goal_reached = (state[RDDLPlanningModel.ground_var('goal-reward-received', [])]
                        or any(car in self._goals for car in cars))
        if goal_reached:
            return 'Goal reached', GOAL_EDGE_COLOR
        if not cars:
            return 'No car on the map', 'black'
        if flat:
            if has_spare:
                return 'Flat tire, carrying a spare', 'black'
            if any(car in spares for car in cars):
                return 'Flat tire, spare here', 'black'
            return 'Stuck with a flat tire', FLAT_EDGE_COLOR
        moves = [self._moves_to_goal[car] for car in cars if car in self._moves_to_goal]
        if not moves:
            return 'No road to the goal', FLAT_EDGE_COLOR
        return (f'{min(moves)} move{"" if min(moves) == 1 else "s"} to goal, '
                f'{"carrying a spare" if has_spare else "no spare"}'), 'black'

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        cars = [location for location in self._locations
                if state[RDDLPlanningModel.ground_var('vehicle-at', [location])]]
        spares = {location for location in self._locations
                  if state[RDDLPlanningModel.ground_var('spare-in', [location])]}
        flat = not state[RDDLPlanningModel.ground_var('not-flattire', [])]
        has_spare = state[RDDLPlanningModel.ground_var('hasspare', [])]

        for (location, node) in self._nodes.items():
            if location in self._goals:
                fill, edge = GOAL_COLOR, GOAL_EDGE_COLOR
            elif location in spares:
                fill, edge = SPARE_COLOR, SPARE_EDGE_COLOR
            else:
                fill, edge = LOCATION_COLOR, LOCATION_EDGE_COLOR
            node.set_facecolor(fill)
            node.set_edgecolor(SPARE_EDGE_COLOR if location in spares else edge)
            node.set_linewidth(2.5 if location in spares else 1.2)

        for ((a, _), arrow) in self._arrows.items():
            active = a in cars and not flat
            arrow.set_color(ACTIVE_ROAD_COLOR if active else ROAD_COLOR)
            arrow.set_zorder(1.5 if active else 1)

        artists = []
        for car in cars:
            artists.extend(self._draw_car(*self._places[car], flat))

        title, color = self._title(state, cars, spares, flat, has_spare)
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
