import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

ROAD_COLOR = '#cfd8dc'
GRID_COLOR = '#b0bec5'
INTERSECTION_COLOR = '#78909c'
ENTRY_COLOR = '#bbdefb'
ENTRY_EDGE_COLOR = '#1565c0'
EXIT_COLOR = '#c8e6c9'
EXIT_EDGE_COLOR = '#2e7d32'
CAR_COLOR = '#ffe0b2'
CAR_EDGE_COLOR = '#ef6c00'
CONGESTED_COLOR = '#ffcdd2'
CONGESTED_EDGE_COLOR = '#c62828'
WINDOW_COLOR = '#37474f'
CONNECTOR_COLOR = '#546e7a'
GREEN_LIGHT_COLOR = '#43a047'
RED_LIGHT_COLOR = '#c62828'


class TrafficVisualizer(BaseViz):
    '''Draws the roads with traffic flowing right and down, the cars on them, and a bar at
    the stopline of each intersection for the color of its light.

    The map comes from how the cells flow into each other and into intersections. Roads
    that do not line up as a grid are instead drawn in rows, one for each road. A car is
    congested if it is directly behind another, which is what the reward penalizes. The
    POMDP shows every car rather than only those observed at the stoplines. The drawing
    only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_cell_size=0.7,
                 max_size=(9.0, 5.5),
                 min_width=6.8,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._cells = list(model.type_to_objects['cell'])
        self._intersections = list(model.type_to_objects['intersection'])
        self._previous = {cell: [other for other in self._cells
                                 if self._nonfluent('FLOWS-INTO-CELL', other, cell)]
                          for cell in self._cells}

        self._roads = self._make_roads()
        self._positions, self._horizontal = self._grid_layout() or self._row_layout()

        points = np.array(list(self._positions.values()))
        num_x, num_y = points.max(axis=0) + 1

        # one empty cell around the map leaves room for the chances at the entries
        cell_size = min(max_cell_size, max_size[0] / (num_x + 2), max_size[1] / (num_y + 2))
        self._cell_size = cell_size

        # title above, then the map, the legend and the caption
        bottom, top = 0.62, 0.45
        grid_width, grid_height = (num_x + 2) * cell_size, (num_y + 2) * cell_size
        width = max(grid_width + 0.4, min_width)
        height = grid_height + bottom + top
        self._figure_size = (width, height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width), bottom / height,
             grid_width / width, grid_height / height])
        self.ax.set_xlim(-1, num_x + 1)
        self.ax.set_ylim(num_y + 1, -1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        self._draw_map()

        self.fig.legend(handles=[
            patches.Patch(facecolor=CAR_COLOR, edgecolor=CAR_EDGE_COLOR, label='Car'),
            patches.Patch(facecolor=CONGESTED_COLOR, edgecolor=CONGESTED_EDGE_COLOR, label='Congested car'),
            patches.Patch(facecolor=ENTRY_COLOR, edgecolor=ENTRY_EDGE_COLOR, label='Entry'),
            patches.Patch(facecolor=EXIT_COLOR, edgecolor=EXIT_EDGE_COLOR, label='Exit'),
            patches.Patch(facecolor=GREEN_LIGHT_COLOR, edgecolor='none', label='Green light'),
            patches.Patch(facecolor=RED_LIGHT_COLOR, edgecolor='none', label='Red light')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / height), ncol=6, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)
        self.fig.text(0.5, 0.1 / height,
                      'Traffic flows right and down, and numbers are the chance of a car arriving',
                      ha='center', va='bottom', fontsize=9, color='#37474f')

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _phases(self, cell, intersection):
        return [phase for phase in ('NS', 'EW')
                if self._nonfluent(f'FLOWS-INTO-INTERSECTION-{phase}', cell, intersection)]

    def _make_roads(self):

        # follow the flow from the cells nothing leads to, and any cell left over because
        # of a loop or a branch in the road starts another road
        roads, placed = [], set()
        starts = [cell for cell in self._cells if not self._previous[cell]]
        for start in starts + self._cells:
            if start in placed:
                continue
            road, current = [], start
            while current is not None:
                placed.add(current)
                road.append(('cell', current))
                road.extend(('intersection', intersection) for intersection in self._intersections
                            if self._phases(current, intersection))
                current = next((cell for cell in self._cells if cell not in placed
                                and current in self._previous[cell]), None)
            roads.append(road)
        return roads

    def _road_phases(self, road):
        cells = [item[1] for item in road if item[0] == 'cell']
        return {phase for cell in cells for intersection in self._intersections
                for phase in self._phases(cell, intersection)}

    def _grid_layout(self):

        # along a road each item is after the last, and across it all items are in line, so
        # intersections shared by roads line those roads up
        group = {}

        def find(key):
            group.setdefault(key, key)
            while group[key] != key:
                key = group[key]
            return key

        horizontal = []
        orders = []
        for road in self._roads:
            phases = self._road_phases(road)
            if phases == {'NS', 'EW'}:
                return None
            is_horizontal = phases != {'NS'}
            horizontal.append(is_horizontal)
            along, across = (0, 1) if is_horizontal else (1, 0)
            for (a, b) in zip(road, road[1:]):
                group[find((across, a))] = find((across, b))
                orders.append(((along, a), (along, b)))
            find((along, road[0]))
            find((across, road[0]))

        orders = [(find(a), find(b)) for (a, b) in orders]
        if any(a == b for (a, b) in orders):
            return None
        values = {find(key): 0 for key in list(group)}
        for _ in range(len(values) + 1):
            changed = False
            for (a, b) in orders:
                if values[b] < values[a] + 1:
                    values[b] = values[a] + 1
                    changed = True
            if not changed:
                break
        else:
            return None

        positions = {item: (values[find((0, item))], values[find((1, item))])
                     for road in self._roads for item in road}

        # cells before the first intersection of a road stay right behind it
        for (road, is_horizontal) in zip(self._roads, horizontal):
            along = 0 if is_horizontal else 1
            first = next((i for (i, item) in enumerate(road) if item[0] == 'intersection'), None)
            for i in range(first or 0):
                position = list(positions[road[i]])
                position[along] = positions[road[first]][along] - (first - i)
                positions[road[i]] = tuple(position)

        if len(set(positions.values())) != len(positions):
            return None
        return ({(index, item): positions[item] for (index, road) in enumerate(self._roads)
                 for item in road}, horizontal)

    def _row_layout(self):
        positions = {(index, item): (col, 2 * index) for (index, road) in enumerate(self._roads)
                     for (col, item) in enumerate(road)}
        return positions, [True] * len(self._roads)

    def _draw_map(self):

        # the roads and chances never change, so they are only drawn once
        self._cell_places = {}
        self._bars = []
        size = min(8, 72 * self._cell_size / (0.62 * 4))
        for (index, road) in enumerate(self._roads):
            is_horizontal = self._horizontal[index]
            places = [self._positions[(index, item)] for item in road]

            # the road runs on through any gap left for a road crossing further along
            for ((x1, y1), (x2, y2)) in zip(places, places[1:]):
                self.ax.add_patch(patches.Rectangle(
                    (x1, y1), x2 - x1 + 1, y2 - y1 + 1, facecolor=ROAD_COLOR,
                    edgecolor='none', zorder=0))

            for ((kind, name), (x, y)) in zip(road, places):
                if kind == 'intersection':
                    self.ax.add_patch(patches.Rectangle(
                        (x, y), 1, 1, facecolor=INTERSECTION_COLOR, edgecolor=GRID_COLOR, zorder=2))
                    self._add_bar(last_cell, name, x, y, is_horizontal)
                    continue

                last_cell = name
                self._cell_places[name] = (x, y, is_horizontal)
                face, edge = ROAD_COLOR, GRID_COLOR
                if self._nonfluent('PERIMETER-INPUT-CELL', name):
                    face, edge = ENTRY_COLOR, ENTRY_EDGE_COLOR
                    self.ax.text(
                        x - 0.5 if is_horizontal else x + 0.5, y + 0.5 if is_horizontal else y - 0.5,
                        f'{self._nonfluent("PERIMETER-INPUT-RATE", name):.2f}',
                        ha='center', va='center', fontsize=size, color='#37474f', zorder=3)
                elif self._nonfluent('PERIMETER-EXIT-CELL', name):
                    face, edge = EXIT_COLOR, EXIT_EDGE_COLOR
                self.ax.add_patch(patches.Rectangle(
                    (x, y), 1, 1, facecolor=face, edgecolor=edge, zorder=1))

        # flows that branch or merge between roads are not along a road, so have an arrow
        along_roads = set()
        for road in self._roads:
            cells = [name for (kind, name) in road if kind == 'cell']
            along_roads.update(zip(cells, cells[1:]))
        for (cell, others) in self._previous.items():
            for other in others:
                if other != cell and (other, cell) not in along_roads:
                    self.ax.add_patch(patches.FancyArrowPatch(
                        np.add(self._cell_places[other][:2], 0.5), np.add(self._cell_places[cell][:2], 0.5),
                        arrowstyle='-|>', mutation_scale=9, linewidth=1.2, color=CONNECTOR_COLOR,
                        connectionstyle='arc3,rad=0.15', zorder=3))

    def _add_bar(self, cell, intersection, x, y, is_horizontal):

        # the bar is on the side of the intersection the road comes in from
        width, height = (0.16, 1) if is_horizontal else (1, 0.16)
        bar = self.ax.add_patch(patches.Rectangle(
            (x, y), width, height, edgecolor='none', zorder=3))
        self._bars.append((bar, intersection, self._phases(cell, intersection)))

    def _draw_car(self, x, y, is_horizontal, congested):
        face = CONGESTED_COLOR if congested else CAR_COLOR
        edge = CONGESTED_EDGE_COLOR if congested else CAR_EDGE_COLOR
        linewidth = min(2, 3 * self._cell_size)

        # the window is at the front of the car, which shows the direction of travel
        if is_horizontal:
            body, window = (x + 0.12, y + 0.26, 0.76, 0.48), (x + 0.64, y + 0.34, 0.16, 0.32)
        else:
            body, window = (x + 0.26, y + 0.12, 0.48, 0.76), (x + 0.34, y + 0.64, 0.32, 0.16)
        return [
            self.ax.add_patch(patches.FancyBboxPatch(
                body[:2], body[2], body[3], boxstyle='round,pad=0,rounding_size=0.1',
                facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=4)),
            self.ax.add_patch(patches.FancyBboxPatch(
                window[:2], window[2], window[3], boxstyle='round,pad=0,rounding_size=0.04',
                facecolor=WINDOW_COLOR, edgecolor='none', zorder=5))
        ]

    def _is_green(self, state, intersection, phases):
        signal1 = state[RDDLPlanningModel.ground_var('light-signal1', [intersection])]
        signal2 = state[RDDLPlanningModel.ground_var('light-signal2', [intersection])]
        return ('NS' in phases and signal2 and not signal1) or ('EW' in phases and signal1 and not signal2)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        occupied = {cell for cell in self._cells
                    if state[RDDLPlanningModel.ground_var('occupied', [cell])]}
        congested = {cell for cell in occupied
                     if any(other in occupied for other in self._previous[cell])}

        artists = []
        for cell in occupied:
            (x, y, is_horizontal) = self._cell_places[cell]
            artists.extend(self._draw_car(x, y, is_horizontal, cell in congested))

        for (bar, intersection, phases) in self._bars:
            green = self._is_green(state, intersection, phases)
            bar.set_facecolor(GREEN_LIGHT_COLOR if green else RED_LIGHT_COLOR)

        if congested:
            title = f'{len(congested)} / {len(occupied)} cars congested'
            color = 'black'
        else:
            title, color = 'No congestion', EXIT_EDGE_COLOR
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
