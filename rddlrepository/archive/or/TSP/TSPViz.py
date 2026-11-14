import itertools
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

GROUND_COLOR = '#e6edd5'
TREE_COLOR = '#a5c48a'
PLOT_COLOR = '#f3ead8'
PLOT_EDGE_COLOR = '#d6c7a8'
UNVISITED_COLOR = '#b0bec5'
VISITED_COLOR = '#e07b53'
BUILDING_EDGE_COLOR = '#546e7a'
WINDOW_COLOR = '#fff3c4'
WALL_COLOR = '#fafafa'
ROOF_COLOR = '#c62828'
DOOR_COLOR = '#795548'
ROAD_COLOR = '#78909c'
ROAD_LINE_COLOR = '#f5f5f5'
CAR_COLOR = '#1e88e5'
CAR_DARK_COLOR = '#263238'
GLASS_COLOR = '#bbdefb'
TEXT_COLOR = '#37474f'
DONE_COLOR = '#2e7d32'

# sizes in inches: the largest map, the room around it, and the size of a town, which
# shrinks so that towns close together do not overlap
MAX_MAP = (6.4, 5.6)
TOP_MARGIN = 0.45
LEGEND_HEIGHT = 0.55
SIDE_MARGIN = 0.2
MAX_TOWN = 0.42
MIN_TOWN = 0.16


# ---------- pictures: each takes the axes, a centre and a size in map units

def _town(ax, x, y, s):

    # three buildings of different heights on a round plot, which are recoloured once visited
    ax.add_patch(patches.Ellipse((x, y - 0.32 * s), 1.05 * s, 0.32 * s, facecolor=PLOT_COLOR,
                                 edgecolor=PLOT_EDGE_COLOR, linewidth=0.6, zorder=2))
    buildings = []
    for (dx, w, h) in ((-0.3, 0.26, 0.55), (-0.02, 0.3, 0.85), (0.27, 0.26, 0.42)):
        (left, bottom) = (x + (dx - w / 2) * s, y - 0.35 * s)
        buildings.append(ax.add_patch(patches.Rectangle(
            (left, bottom), w * s, h * s, facecolor=UNVISITED_COLOR, edgecolor=BUILDING_EDGE_COLOR,
            linewidth=0.5, zorder=2.1)))
        for row in np.arange(0.12, h - 0.1, 0.17):
            for col in (0.25, 0.6):
                ax.add_patch(patches.Rectangle((left + col * w * s, bottom + row * s), 0.15 * w * s, 0.07 * s,
                                               facecolor=WINDOW_COLOR, edgecolor='none', zorder=2.2))
    return buildings


def _home(ax, x, y, s):

    # a house with a red roof and a door, on the same plot as a town
    ax.add_patch(patches.Ellipse((x, y - 0.32 * s), 1.05 * s, 0.32 * s, facecolor=PLOT_COLOR,
                                 edgecolor=PLOT_EDGE_COLOR, linewidth=0.6, zorder=2))
    ax.add_patch(patches.Rectangle((x - 0.32 * s, y - 0.35 * s), 0.64 * s, 0.5 * s, facecolor=WALL_COLOR,
                                   edgecolor=BUILDING_EDGE_COLOR, linewidth=0.6, zorder=2.1))
    ax.add_patch(patches.Polygon([(x - 0.42 * s, y + 0.13 * s), (x, y + 0.5 * s), (x + 0.42 * s, y + 0.13 * s)],
                                 closed=True, facecolor=ROOF_COLOR, edgecolor='none', zorder=2.2))
    ax.add_patch(patches.Rectangle((x - 0.08 * s, y - 0.35 * s), 0.16 * s, 0.28 * s, facecolor=DOOR_COLOR,
                                   edgecolor='none', zorder=2.2))
    ax.add_patch(patches.Rectangle((x + 0.12 * s, y - 0.12 * s), 0.13 * s, 0.13 * s, facecolor=WINDOW_COLOR,
                                   edgecolor=BUILDING_EDGE_COLOR, linewidth=0.4, zorder=2.2))


def _car(ax, x, y, s, facing):

    # a car seen from the side, with its nose towards the facing side
    def at(dx, dy):
        return (x + facing * dx * s, y + dy * s)
    artists = [
        ax.add_patch(patches.Polygon([at(-0.5, 0), at(0.5, 0), at(0.5, 0.2), at(0.3, 0.26), at(-0.5, 0.26)],
                                     closed=True, facecolor=CAR_COLOR, edgecolor=CAR_DARK_COLOR,
                                     linewidth=0.6, zorder=5)),
        ax.add_patch(patches.Polygon([at(-0.36, 0.26), at(0.18, 0.26), at(0.06, 0.46), at(-0.3, 0.46)],
                                     closed=True, facecolor=GLASS_COLOR, edgecolor=CAR_DARK_COLOR,
                                     linewidth=0.6, zorder=5))]
    for dx in (-0.28, 0.28):
        artists.append(ax.add_patch(plt.Circle(at(dx, 0), 0.11 * s, facecolor=CAR_DARK_COLOR,
                                               edgecolor='none', zorder=5.1)))
    return artists


def _road(ax, xs, ys, width):

    # asphalt with a dashed centre line, in points
    return (ax.plot(xs, ys, color=ROAD_COLOR, linewidth=width, solid_capstyle='round', zorder=1)
            + ax.plot(xs, ys, color=ROAD_LINE_COLOR, linewidth=max(0.4, 0.12 * width),
                      dashes=(3, 3), zorder=1.1))


class TSPVisualizer(BaseViz):
    '''Draws the cities as towns on a map at their coordinates, with the origin as the
    salesman's home. Towns turn from grey to brick red once visited, the salesman is a car
    at the current city, and the roads travelled so far join the cities in the order visited.

    The route is built from the states seen so far, so it requires render to be called on
    every step. Cities visited between two renders are not on the route.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._cities = model.type_to_objects['city']
        self._origins = {c for c in self._cities if self._value('ORIGIN', c)}
        self._positions = {c: (self._value('X', c), self._value('Y', c)) for c in self._cities}
        self._init_state = self._get_state(model.ground_vars_with_values(model.state_fluents))

        self._route = []
        self._last_state = None

        # the map keeps the shape of the coordinates, with room for a town around each city
        xs = [x for (x, _) in self._positions.values()] or [0.0]
        ys = [y for (_, y) in self._positions.values()] or [0.0]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        closest = min((math.dist(self._positions[a], self._positions[b])
                       for (a, b) in itertools.combinations(self._cities, 2)), default=span)
        (map_w, map_h) = MAX_MAP
        units = span / (min(map_w, map_h) - 2 * MAX_TOWN)
        town = max(MIN_TOWN, min(MAX_TOWN, 0.9 * closest / units))
        pad = 0.7 * town * units
        (low_x, high_x) = (min(xs) - pad, max(xs) + pad + 1e-6)
        (low_y, high_y) = (min(ys) - pad, max(ys) + pad + 1e-6)
        inches = min(map_w / (high_x - low_x), map_h / (high_y - low_y))
        (map_w, map_h) = ((high_x - low_x) * inches, (high_y - low_y) * inches)
        self._size = town / inches
        self._road_width = max(2.0, min(6.0, 72 * 0.18 * town))

        width = max(map_w, 5.2) + 2 * SIDE_MARGIN
        height = LEGEND_HEIGHT + map_h + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.fig.patch.set_facecolor('white')
        self.ax = self.fig.add_axes([(width - map_w) / 2 / width, LEGEND_HEIGHT / height,
                                     map_w / width, map_h / height])
        self.ax.set_xlim(low_x, high_x)
        self.ax.set_ylim(low_y, high_y)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor(GROUND_COLOR)
        for spine in self.ax.spines.values():
            spine.set_color(PLOT_EDGE_COLOR)

        # the map never changes, so it is only drawn once
        self._draw_trees(low_x, high_x, low_y, high_y)
        self._buildings = {}
        for c in self._cities:
            (x, y) = self._positions[c]
            if c in self._origins:
                _home(self.ax, x, y, self._size)
            else:
                self._buildings[c] = _town(self.ax, x, y, self._size)
        self._draw_legend()

    def _value(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _draw_trees(self, low_x, high_x, low_y, high_y):

        # clumps of trees in the countryside, always in the same places, away from the towns
        rng = np.random.default_rng(3)
        s = self._size
        for _ in range(60):
            (x, y) = (rng.uniform(low_x, high_x), rng.uniform(low_y, high_y))
            if all(math.dist((x, y), p) > 0.9 * s for p in self._positions.values()):
                for (dx, dy) in ((0, 0), (0.12, 0.05), (0.05, 0.13)):
                    self.ax.add_patch(plt.Circle((x + dx * s, y + dy * s), 0.1 * s, facecolor=TREE_COLOR,
                                                 edgecolor='none', zorder=0.5))

    def _draw_legend(self):
        (width, height) = self._figure_size
        ax = self.fig.add_axes([0, 0.04 / height, 1, 0.46 / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, 0.46)
        ax.set_aspect('equal')
        ax.axis('off')
        entries = [('home', 'Home'), ('town', 'To visit'), ('visited', 'Visited'),
                   ('car', 'Salesman'), ('road', 'Route')]
        widths = [0.42 + 0.066 * len(label) + 0.3 for (_, label) in entries]
        x = (width - sum(widths)) / 2
        for ((kind, label), entry_width) in zip(entries, widths):
            (cx, cy, s) = (x + 0.17, 0.25, 0.32)
            if kind == 'home':
                _home(ax, cx, cy, s)
            elif kind in ('town', 'visited'):
                for building in _town(ax, cx, cy, s):
                    building.set_facecolor(VISITED_COLOR if kind == 'visited' else UNVISITED_COLOR)
            elif kind == 'car':
                _car(ax, cx, cy - 0.1, s, 1)
            else:
                _road(ax, [cx - 0.17, cx + 0.17], [cy - 0.04, cy - 0.04], 4)
            ax.text(x + 0.42, 0.21, label, ha='left', va='center', fontsize=8.5, color=TEXT_COLOR)
            x += entry_width

    # ---------- the state

    def _get_state(self, state):
        return (frozenset(c for c in self._cities
                          if state[RDDLPlanningModel.ground_var('current', [c])]),
                frozenset(c for c in self._cities
                          if state[RDDLPlanningModel.ground_var('visited', [c])]))

    def _update_route(self, state):

        # rendering the same state again must not change the route
        if state == self._last_state:
            return
        last_state, self._last_state = self._last_state, state

        # visited cities are never forgotten during an episode
        if last_state is None or state == self._init_state or not state[1] >= last_state[1]:
            self._route = []
            return
        self._route.extend((start, end) for start in last_state[0] for end in state[0]
                           if start != end)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        state = self._get_state(state)
        self._update_route(state)
        current, visited = state

        for (c, buildings) in self._buildings.items():
            for building in buildings:
                building.set_facecolor(VISITED_COLOR if c in visited else UNVISITED_COLOR)

        artists = []
        for (start, end) in self._route:
            ((x1, y1), (x2, y2)) = (self._positions[start], self._positions[end])
            artists += _road(self.ax, [x1, x2], [y1, y2], self._road_width)

        # the car parks in front of its town, facing the way it came
        s = self._size
        for c in current:
            (x, y) = self._positions[c]
            facing = 1
            if self._route and self._route[-1][1] == c:
                facing = 1 if x >= self._positions[self._route[-1][0]][0] else -1
            artists += _car(self.ax, x + 0.35 * s, y - 0.5 * s, 0.75 * s, facing)

        # same rule as the termination condition
        done = len(visited) == len(self._cities) and self._origins <= current
        total = len(self._cities)
        self.ax.set_title('Back home' if done else f'{len(visited)} / {total} cities visited',
                          fontsize=13, fontweight='bold', color=DONE_COLOR if done else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
