import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

PLATE_COLOR = '#f5f7f8'
PLATE_EDGE_COLOR = '#cfd8dc'
SHOP_PLATE_COLOR = '#e3f2fd'
ROAD_COLOR = '#b0bec5'
ACTIVE_ROAD_COLOR = '#1565c0'
WALL_COLOR = '#ffffff'
WALL_EDGE_COLOR = '#78909c'
ROOF_COLOR = '#cfd8dc'
SHOP_COLOR = '#90caf9'
SHOP_EDGE_COLOR = '#1565c0'
OPEN_COLOR = '#ffb74d'
OPEN_EDGE_COLOR = '#ff8f00'
PAID_COLOR = '#c8e6c9'
PAID_EDGE_COLOR = '#2e7d32'
UNPAID_COLOR = '#ffcdd2'
UNPAID_EDGE_COLOR = '#c62828'
TRUCK_COLOR = '#eceff1'
TRUCK_EDGE_COLOR = '#546e7a'
WINDOW_COLOR = '#37474f'
WHEEL_COLOR = '#263238'
HOT_COLOR = '#ffcc80'
HOT_EDGE_COLOR = '#d84315'
COLD_COLOR = '#bbdefb'
COLD_EDGE_COLOR = '#1565c0'
SLOT_COLOR = '#90a4ae'
DETAIL_COLOR = '#37474f'

# sizes in inches, before the map is scaled to fit the figure
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
MIN_WIDTH = 8.6
STOP_WIDTH = 1.8
BUILDING_ROW = 0.58
ROADS_ROW = 0.16
TRUCK_ROW = 0.42
STOP_PADDING = 0.05
ROAD_GAP = 0.55
ROW_GAP = 0.25
LANE_GAP = 0.13
MAP_MARGIN = 0.04
GRID_GAP = 0.12
STRIP_HEIGHT = 0.42
PIZZA_RADIUS = 0.115
PIZZA_PITCH = 0.26
TRUCK_NAME_WIDTH = 0.36
CAB_WIDTH = 0.22
TILE_SIZE = (1.05, 0.8)
TILE_GAP = 0.08

# more than this shows how many more there are
MAX_TRUCK_ROWS = 3
MAX_ORDER_SLOTS = 4
MAX_TRUCK_SLOTS = 4
MAX_COLUMN_LOCATIONS = 6

# more roads than this for each location cannot be followed as lines, so they are listed
MAX_ROADS_PER_LOCATION = 2

# a map scaled more than this is too small to read, so it is drawn as tiles
MIN_MAP_SCALE = 0.6

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5
NAME_FONT_SIZE = 8.5
TEXT_FONT_SIZE = 7
PIZZA_FONT_SIZE = 5.5
CHAR_WIDTH = 0.56
BOLD_CHAR_WIDTH = 0.62
NAME_CHAR_WIDTH = 0.72
PIZZA_CHAR_WIDTH = 0.7
LINE_HEIGHT = 1.3

AT_SHOP = 'At the shop'


class PizzaVisualizer(BaseViz):
    '''Draws a map of the roads flowing from left to right, as in the other domains with
    roads. Each location is a house with a place for each pizza it ordered, or a shop, and
    trucks stand at their location with the pizzas they carry in their cargo box. The roads a
    truck can take next are in blue. The pizzas that are not loaded yet are not at any one
    location, so they are in a strip above the map.

    The map comes from the roads rather than the names of the locations. Roads going a long
    way around the map take their own lane along its edge. Roads too many to follow as lines
    are listed on the locations instead, and too many locations to draw like this show as
    tiles with counts. The actions are not part of the state, so a
    failed drive only shows as the truck staying where it is. The drawing only depends on the
    current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_size=(9.0, 5.5),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._trucks = list(model.type_to_objects['truck'])
        self._locations = list(model.type_to_objects['location'])
        self._pizzas = list(model.type_to_objects['pizza'])
        self._roads = [(a, b) for a in self._locations for b in self._locations
                       if self._value('CONNECTED', a, b)]
        self._shops = {location for location in self._locations if self._value('SHOP', location)}
        self._orders = {location: max(int(self._value('ORDERS', location)), 0)
                        for location in self._locations}

        self._truck_rows = min(len(self._trucks), MAX_TRUCK_ROWS)
        self._choose_layout(max_size)
        self._artists = []
        self._places_drawn = {}
        if self._has_map:
            self._create_map_figure(max_size)
        else:
            self._create_tile_figure(max_size)
        height = self._figure_size[1]

        orders = [
            patches.Patch(facecolor=OPEN_COLOR, edgecolor=OPEN_EDGE_COLOR, label='Order open'),
            patches.Patch(facecolor=PAID_COLOR, edgecolor=PAID_EDGE_COLOR, label='Order filled')
        ]
        if self._has_map:
            handles = [
                Line2D([], [], linestyle='', marker='o', markersize=8, markerfacecolor=HOT_COLOR,
                       markeredgecolor=HOT_EDGE_COLOR, label='Hot pizza'),
                Line2D([], [], linestyle='', marker='o', markersize=8, markerfacecolor=COLD_COLOR,
                       markeredgecolor=COLD_EDGE_COLOR, label='Cold pizza'),
                Line2D([], [], linestyle='', marker='o', markersize=8, markerfacecolor=PAID_COLOR,
                       markeredgecolor=PAID_EDGE_COLOR, markeredgewidth=2, label='Delivered, paid'),
                Line2D([], [], linestyle='', marker='o', markersize=8, markerfacecolor=UNPAID_COLOR,
                       markeredgecolor=UNPAID_EDGE_COLOR, markeredgewidth=2, label='Delivered, not paid')
            ] + orders
            if self._has_lines:
                handles.append(Line2D([], [], color=ACTIVE_ROAD_COLOR, marker='>', markersize=4,
                                      label='Road a truck can take'))
        else:
            handles = orders + [
                patches.Patch(facecolor=UNPAID_COLOR, edgecolor=UNPAID_EDGE_COLOR, label='Too many delivered'),
                patches.Patch(facecolor=SHOP_PLATE_COLOR, edgecolor=PLATE_EDGE_COLOR, label='Shop')
            ]
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.25 / height),
                        ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                        handlelength=1.4)

        caption = 'A pizza in a truck with n hot pizzas stays hot with chance n / (n + 1)'
        chances = [self._value('DRIVEPROB', truck) for truck in self._trucks]
        if chances:
            low, high = min(chances), max(chances)
            chance = f'{low:g}' if low == high else f'{low:g} to {high:g}'
            caption += f'. Trucks drive with chance {chance}'
        self.fig.text(0.5, 0.1 / height, caption, ha='center', va='bottom', fontsize=9,
                      color=DETAIL_COLOR)

    def _value(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    # ==========================================================================
    # LAYOUT
    # ==========================================================================

    def _choose_layout(self, max_size):

        # a map with its roads as lines, otherwise a grid listing them, otherwise tiles
        room = (max_size[0], max_size[1] - STRIP_HEIGHT)
        self._has_lines = (sum(a != b for (a, b) in self._roads)
                           <= MAX_ROADS_PER_LOCATION * len(self._locations))
        if self._has_lines:
            self._stop_height = self._height_of_stop(False)
            self._places = self._road_places()
            self._route_roads()
            self._scale = self._map_scale(room)
            if self._scale < MIN_MAP_SCALE:
                self._has_lines = False

        if not self._has_lines:
            self._stop_height = self._height_of_stop(True)
            self._places = self._grid_places(room)
            self._routes = {}
            self._lanes_above, self._lanes_below = 0, 0
            self._scale = self._map_scale(room)
        self._has_map = self._scale >= MIN_MAP_SCALE

    def _height_of_stop(self, lists_roads):
        return (BUILDING_ROW + (ROADS_ROW if lists_roads else 0)
                + self._truck_rows * TRUCK_ROW + STOP_PADDING)

    def _map_extent(self):
        xs = [x for (x, _) in self._places.values()]
        ys = [y for (_, y) in self._places.values()]
        return (min(xs) - STOP_WIDTH / 2, max(xs) + STOP_WIDTH / 2,
                min(ys) - self._stop_height / 2 - self._lanes_below * LANE_GAP - MAP_MARGIN,
                max(ys) + self._stop_height / 2 + self._lanes_above * LANE_GAP + MAP_MARGIN)

    def _map_scale(self, room):
        (left, right, bottom, top) = self._map_extent()
        return min(1.0, room[0] / (right - left), room[1] / (top - bottom))

    def _road_places(self):

        # each location goes to the right of all roads into it, and a cycle has no such
        # place so the one location missing the fewest of them goes first to break it
        previous = {location: [a for (a, b) in self._roads if b == location and a != location]
                    for location in self._locations}
        levels, placed = [], set()
        remaining = list(self._locations)
        while remaining:
            missing = {location: sum(a not in placed for a in previous[location])
                       for location in remaining}

            # shops start the map, since the pizzas leave from there
            shops = [location for location in remaining if location in self._shops]
            level = [location for location in remaining if missing[location] == 0]
            if not level:
                level = shops[:1] or [min(remaining, key=missing.get)]
            remaining = [location for location in remaining if location not in level]
            placed.update(level)
            levels.append(level)

        # each location goes towards the roads into it, and the columns are centered
        rows, places, column = {}, {}, 0
        step_x = STOP_WIDTH + ROAD_GAP
        step_y = (self._stop_height + ROW_GAP) / 2
        for level in levels:
            for start in range(0, len(level), MAX_COLUMN_LOCATIONS):
                part = level[start:start + MAX_COLUMN_LOCATIONS]
                wanted = {}
                for (i, location) in enumerate(part):
                    from_rows = [rows[a] for a in previous[location] if a in rows]
                    wanted[location] = np.mean(from_rows) if from_rows else i
                for (i, location) in enumerate(sorted(part, key=wanted.get)):
                    rows[location] = 2 * i - (len(part) - 1)
                    places[location] = (column * step_x, -rows[location] * step_y)
                column += 1
        return places

    def _grid_places(self, room):

        # the layout with the largest locations, and of those the one with the fewest rows
        cell_width, cell_height = STOP_WIDTH + GRID_GAP, self._stop_height + GRID_GAP
        count = len(self._locations)
        best = None
        for num_cols in range(1, count + 1):
            num_rows = math.ceil(count / num_cols)
            scale = min(1.0, room[0] / (num_cols * cell_width), room[1] / (num_rows * cell_height))
            if best is None or scale >= best[0]:
                best = (scale, num_cols)
        return {location: (divmod(i, best[1])[1] * cell_width, -divmod(i, best[1])[0] * cell_height)
                for (i, location) in enumerate(self._locations)}

    def _create_map_figure(self, max_size):
        (left, right, bottom, top) = self._map_extent()
        scale = self._scale
        map_width = (right - left) * scale
        map_height = (top - bottom + STRIP_HEIGHT) * scale

        # title above, then the strip, the map, the legend and the caption
        width = max(map_width + 0.4, MIN_WIDTH)
        height = map_height + BOTTOM_MARGIN + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, BOTTOM_MARGIN / height, 1, map_height / height])
        center = (left + right) / 2
        half_view = width / (2 * scale)
        self.ax.set_xlim(center - half_view, center + half_view)
        self.ax.set_ylim(bottom, top + STRIP_HEIGHT)
        self.ax.axis('off')

        self._strip_y = top + STRIP_HEIGHT / 2
        self._strip_left = center - min(half_view - 0.15, max((right - left) / 2, 3.2))
        self._strip_right = 2 * center - self._strip_left

        # the locations and roads never move, so they are only drawn once and then recolored
        self._plates = {location: self._draw_plate(location) for location in self._locations}
        self._road_arrows = {}
        if self._has_lines:
            self._road_arrows = {road: self._draw_road(*road) for road in self._roads}
        self._roofs = {location: self._draw_building(location) for location in self._locations}
        for location in self._locations:
            self._draw_labels(location)

    def _create_tile_figure(self, max_size):
        places = [AT_SHOP] + self._locations
        count = len(places)
        best = None
        for num_cols in range(1, count + 1):
            num_rows = math.ceil(count / num_cols)
            scale = min(1.0, max_size[0] / (num_cols * TILE_SIZE[0]), max_size[1] / (num_rows * TILE_SIZE[1]))
            if best is None or scale >= best[0]:
                best = (scale, num_cols, num_rows)
        (self._scale, num_cols, num_rows) = best

        # title above, then the tiles, the legend and the caption
        grid_width = num_cols * TILE_SIZE[0] * self._scale
        grid_height = num_rows * TILE_SIZE[1] * self._scale
        width = max(grid_width + 0.4, MIN_WIDTH)
        height = grid_height + BOTTOM_MARGIN + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width), BOTTOM_MARGIN / height,
             grid_width / width, grid_height / height])
        self.ax.set_xlim(0, num_cols * TILE_SIZE[0])
        self.ax.set_ylim(num_rows * TILE_SIZE[1], 0)
        self.ax.axis('off')
        self._tiles = {}
        for (i, place) in enumerate(places):
            row, col = divmod(i, num_cols)
            self._tiles[place] = self._draw_tile(place, col * TILE_SIZE[0], row * TILE_SIZE[1])

    # ==========================================================================
    # WHAT NEVER MOVES
    # ==========================================================================

    def _font(self, size):
        return max(size * self._scale, MIN_FONT_SIZE)

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _list_names(self, names, limit):

        # as many names as fit on the line, then how many more there are
        text = ''
        for (i, name) in enumerate(names):
            longer = f'{text} {name}' if text else name
            rest = len(names) - i - 1
            if len(longer) + (len(f' +{rest}') if rest else 0) > limit:
                return f'{text} +{len(names) - i}' if text else f'+{len(names)}'
            text = longer
        return text

    def _stop_corner(self, location):
        (x, y) = self._places[location]
        return x - STOP_WIDTH / 2, y + self._stop_height / 2

    def _draw_plate(self, location):
        (left, top) = self._stop_corner(location)
        return self.ax.add_patch(patches.FancyBboxPatch(
            (left, top - self._stop_height), STOP_WIDTH, self._stop_height,
            boxstyle='round,pad=0,rounding_size=0.06', linewidth=1.0, zorder=1,
            facecolor=SHOP_PLATE_COLOR if location in self._shops else PLATE_COLOR,
            edgecolor=PLATE_EDGE_COLOR))

    def _blocks_road(self, points, a, b):
        for location in self._locations:
            if location in (a, b):
                continue
            (x, y) = self._places[location]
            for (px, py) in points:
                if abs(px - x) < STOP_WIDTH / 2 + 0.03 and abs(py - y) < self._stop_height / 2 + 0.03:
                    return True
        return False

    def _road_bend(self, a, b):

        # a road curves when it would hide behind a location or a road the opposite way, and
        # takes the smallest bend that stays on the map and clear of the other locations
        (ax, ay), (bx, by) = self._places[a], self._places[b]
        ys = [y for (_, y) in self._places.values()]
        top, bottom = max(ys) + self._stop_height / 2, min(ys) - self._stop_height / 2
        bends = ([0.0] if (b, a) not in self._roads else []) + [0.2, -0.2, 0.35, -0.35]
        for bend in bends:
            control = ((ax + bx) / 2 + bend * (by - ay), (ay + by) / 2 - bend * (bx - ax))
            points = [((1 - t) ** 2 * ax + 2 * t * (1 - t) * control[0] + t * t * bx,
                       (1 - t) ** 2 * ay + 2 * t * (1 - t) * control[1] + t * t * by)
                      for t in np.linspace(0, 1, 40)]
            inside = all(bottom <= y <= top for (_, y) in points)
            if inside and not self._blocks_road(points, a, b):
                return bend
        return None

    def _lane_path(self, a, b, lane_y, shift):

        # out of the source, along the lane and back into the destination
        (ax, ay), (bx, by) = self._places[a], self._places[b]
        side = 1 if lane_y > ay else -1
        toward = 1 if bx >= ax else -1
        return [(ax + toward * shift, ay + side * self._stop_height / 2), (ax + toward * shift, lane_y),
                (bx - toward * shift, lane_y), (bx - toward * shift, by + side * self._stop_height / 2)]

    def _sample_path(self, path):
        return [(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
                for (p, q) in zip(path, path[1:]) for t in np.linspace(0, 1, 40)]

    def _route_roads(self):
        self._routes = {}
        long_roads = []
        for (a, b) in self._roads:
            if a != b:
                bend = self._road_bend(a, b)
                if bend is None:
                    long_roads.append((a, b))
                else:
                    self._routes[(a, b)] = ('bend', bend)

        # the longest roads take the outermost lanes, so that lanes do not cross each other
        ys = [y for (_, y) in self._places.values()]
        top, bottom = max(ys) + self._stop_height / 2, min(ys) - self._stop_height / 2
        distance = lambda road: abs(self._places[road[0]][0] - self._places[road[1]][0])
        self._lanes_above, self._lanes_below = 0, 0
        for road in sorted(long_roads, key=distance):
            shift = 0.15 + 0.1 * (self._lanes_above + self._lanes_below)
            path = self._lane_path(*road, top + (self._lanes_above + 1) * LANE_GAP, shift)
            if self._blocks_road(self._sample_path(path), *road):
                path = self._lane_path(*road, bottom - (self._lanes_below + 1) * LANE_GAP, shift)
                self._lanes_below += 1
            else:
                self._lanes_above += 1
            self._routes[road] = ('lane', path)

    def _draw_road(self, a, b):
        style = dict(arrowstyle='-|>', mutation_scale=11, linewidth=1.8, zorder=0.5, color=ROAD_COLOR)

        # a road back to the same location loops over its top corner
        if a == b:
            (left, top) = self._stop_corner(a)
            start = (left + STOP_WIDTH - 0.3, top)
            end = (left + STOP_WIDTH, top - 0.3)
            return self.ax.add_patch(patches.FancyArrowPatch(
                start, end, connectionstyle='arc3,rad=-1.2', shrinkA=0, shrinkB=0, **style))

        (kind, route) = self._routes[(a, b)]
        if kind == 'lane':
            return self.ax.add_patch(patches.FancyArrowPatch(
                path=Path(route, [Path.MOVETO] + [Path.LINETO] * (len(route) - 1)),
                shrinkA=0, shrinkB=0, **style))
        return self.ax.add_patch(patches.FancyArrowPatch(
            self._places[a], self._places[b], patchA=self._plates[a], patchB=self._plates[b],
            connectionstyle=f'arc3,rad={route}', shrinkA=1, shrinkB=2, **style))

    def _draw_building(self, location):

        # a house, or a shop with a striped awning, and its roof shows the state of its order
        (left, top) = self._stop_corner(location)
        x, ground = left + 0.07, top - BUILDING_ROW + 0.05
        width, height = 0.42, 0.26
        self.ax.add_patch(patches.Rectangle(
            (x, ground), width, height, facecolor=WALL_COLOR, edgecolor=WALL_EDGE_COLOR,
            linewidth=1.0, zorder=2))
        self.ax.add_patch(patches.Rectangle(
            (x + width / 2 - 0.05, ground), 0.1, 0.15, facecolor=WALL_EDGE_COLOR, edgecolor='none', zorder=3))

        if location in self._shops:
            roof = self.ax.add_patch(patches.Rectangle(
                (x - 0.03, ground + height), width + 0.06, 0.12, facecolor=SHOP_COLOR,
                edgecolor=SHOP_EDGE_COLOR, linewidth=1.0, zorder=3))
            for i in range(1, 4, 2):
                self.ax.add_patch(patches.Rectangle(
                    (x - 0.03 + i * (width + 0.06) / 4, ground + height), (width + 0.06) / 4, 0.12,
                    facecolor=WALL_COLOR, edgecolor='none', zorder=4))
            return roof
        return self.ax.add_patch(plt.Polygon(
            [(x - 0.05, ground + height), (x + width / 2, ground + height + 0.17),
             (x + width + 0.05, ground + height)], closed=True, linewidth=1.0, zorder=3))

    def _draw_labels(self, location):
        (left, top) = self._stop_corner(location)
        text_x = left + 0.56

        # the name leaves room for the size of the order next to it
        orders = self._orders[location]
        order_text = f'Order {orders}/{orders}' if orders > 0 else ''
        room = STOP_WIDTH - 0.68 - len(order_text) * CHAR_WIDTH * TEXT_FONT_SIZE / 72
        limit = max(int(room * 72 / (NAME_CHAR_WIDTH * NAME_FONT_SIZE)), 3)
        self.ax.text(text_x, top - 0.13, self._shorten(location, limit), ha='left', va='center',
                     fontsize=self._font(NAME_FONT_SIZE), fontweight='bold', zorder=3)
        if not self._has_lines:
            limit = int((STOP_WIDTH - 0.14) * 72 / (CHAR_WIDTH * TEXT_FONT_SIZE))
            names = [self._shorten(b, 7) for (a, b) in self._roads if a == location]
            text = 'To ' + self._list_names(names, limit - 3) if names else 'No roads out'
            self.ax.text(left + 0.07, top - BUILDING_ROW - ROADS_ROW / 2, text, ha='left', va='center',
                         fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR, zorder=3)

    def _draw_tile(self, place, x, y):
        (tile_width, tile_height) = TILE_SIZE
        frame = self.ax.add_patch(patches.FancyBboxPatch(
            (x + TILE_GAP / 2, y + TILE_GAP / 2), tile_width - TILE_GAP, tile_height - TILE_GAP,
            boxstyle='round,pad=0,rounding_size=0.07', edgecolor=PLATE_EDGE_COLOR, linewidth=1.2, zorder=1,
            facecolor=SHOP_PLATE_COLOR if place in self._shops else PLATE_COLOR))
        scale = self._scale
        center = x + tile_width / 2

        # counts stay at the smallest readable text for as long as they fit on the tile
        size = max(TEXT_FONT_SIZE * scale, MIN_FONT_SIZE)
        limit = int((tile_width - TILE_GAP) * scale * 72 / (CHAR_WIDTH * size))
        has_details = (limit >= len('9 trucks')
                       and tile_height * scale * 72 >= 3 * LINE_HEIGHT * size)

        name_fit = 0.9 * (tile_width - TILE_GAP) * scale * 72 / BOLD_CHAR_WIDTH
        name = self._shorten(place, int(name_fit / MIN_FONT_SIZE))
        self.ax.text(center, y + tile_height * (0.25 if has_details else 0.5), name, ha='center',
                     va='center', fontweight='bold', zorder=3,
                     fontsize=max(min(NAME_FONT_SIZE * scale, name_fit / len(name)), MIN_FONT_SIZE))
        counts = None
        if has_details:
            counts = [self.ax.text(center, y + tile_height * share, '', ha='center', va='center',
                                   fontsize=size, color=DETAIL_COLOR, zorder=3)
                      for share in (0.52, 0.76)]
        return {'frame': frame, 'counts': counts, 'limit': limit}

    # ==========================================================================
    # WHAT MOVES
    # ==========================================================================

    def _add(self, artist):
        self._artists.append(artist)
        return artist

    def _add_patch(self, patch):
        return self._add(self.ax.add_patch(patch))

    def _add_text(self, x, y, text, **kwargs):
        return self._add(self.ax.text(x, y, text, **kwargs))

    def _draw_slot(self, x, y):
        self._add_patch(patches.Circle((x, y), PIZZA_RADIUS, facecolor='none', edgecolor=SLOT_COLOR,
                                       linewidth=0.9, linestyle=(0, (2, 2)), zorder=5))

    def _draw_pizza(self, x, y, pizza, is_hot):

        # a crust and toppings, orange and red when hot and blue when cold
        fill, edge = (HOT_COLOR, HOT_EDGE_COLOR) if is_hot else (COLD_COLOR, COLD_EDGE_COLOR)
        self._add_patch(patches.Circle((x, y), PIZZA_RADIUS, facecolor=fill, edgecolor=edge,
                                       linewidth=1.3, zorder=6))
        for angle in (40, 160, 280):
            self._add_patch(patches.Circle(
                (x + 0.078 * math.cos(math.radians(angle)), y + 0.078 * math.sin(math.radians(angle))),
                0.016, facecolor=edge, edgecolor='none', zorder=7))

        # the name only stays when it fits inside the pizza
        size = self._font(PIZZA_FONT_SIZE)
        limit = int(2 * PIZZA_RADIUS * 0.9 * self._scale * 72 / (PIZZA_CHAR_WIDTH * size))
        if limit >= 1:
            self._add_text(x, y, self._shorten(pizza, limit), ha='center', va='center',
                           fontsize=size, zorder=8)

    def _draw_more(self, x, y, count):
        self._add_patch(patches.Circle((x, y), PIZZA_RADIUS, facecolor=WALL_COLOR, edgecolor=SLOT_COLOR,
                                       linewidth=1.0, zorder=6))
        self._add_text(x, y, f'+{count}', ha='center', va='center',
                       fontsize=self._font(PIZZA_FONT_SIZE), zorder=8)

    def _draw_truck(self, x, ground, truck, pizzas, hot):

        # the truck faces right, and its cargo box is open to show one place per capacity
        capacity = int(self._value('CAPACITY', truck))
        slots = max(min(capacity, MAX_TRUCK_SLOTS), 1)
        box_width = slots * PIZZA_PITCH + 0.04
        bottom = ground + 0.05
        height = 0.3
        self._add_patch(patches.FancyBboxPatch(
            (x, bottom), box_width, height, boxstyle='round,pad=0,rounding_size=0.03',
            facecolor=TRUCK_COLOR, edgecolor=TRUCK_EDGE_COLOR, linewidth=1.2, zorder=4))
        self._add_patch(patches.FancyBboxPatch(
            (x + box_width + 0.01, bottom), CAB_WIDTH, 0.22, boxstyle='round,pad=0,rounding_size=0.04',
            facecolor=TRUCK_COLOR, edgecolor=TRUCK_EDGE_COLOR, linewidth=1.2, zorder=4))
        self._add_patch(patches.Rectangle(
            (x + box_width + 0.09, bottom + 0.1), 0.1, 0.08, facecolor=WINDOW_COLOR, edgecolor='none', zorder=5))
        for wheel in (x + 0.1, x + box_width + 0.1):
            self._add_patch(patches.Circle((wheel, ground + 0.05), 0.05, facecolor=WHEEL_COLOR,
                                           edgecolor='none', zorder=6))

        # cold pizzas go first so that they stay in view when not all of them fit
        pizzas = sorted(pizzas, key=lambda pizza: hot[pizza])
        middle = bottom + height / 2
        first = x + 0.02 + PIZZA_PITCH / 2
        for i in range(slots):
            place = first + i * PIZZA_PITCH
            if i == slots - 1 and len(pizzas) > slots:
                self._draw_more(place, middle, len(pizzas) - i)
            elif i < len(pizzas):
                self._draw_pizza(place, middle, pizzas[i], hot[pizzas[i]])
            else:
                self._draw_slot(place, middle)
        return x + box_width + CAB_WIDTH + 0.01

    def _draw_trucks(self, location, trucks, carried, hot):
        (left, top) = self._stop_corner(location)
        first_top = top - BUILDING_ROW - (0 if self._has_lines else ROADS_ROW)
        limit = int(TRUCK_NAME_WIDTH * 72 / (BOLD_CHAR_WIDTH * TEXT_FONT_SIZE))
        for row in range(self._truck_rows):
            if row >= len(trucks):
                break
            ground = first_top - (row + 1) * TRUCK_ROW + 0.03

            # the last row says how many trucks did not fit
            if row == self._truck_rows - 1 and len(trucks) > self._truck_rows:
                self._add_text(left + 0.07, ground + 0.17, f'+{len(trucks) - row} more trucks', ha='left',
                               va='center', fontsize=self._font(TEXT_FONT_SIZE), fontweight='bold',
                               color=DETAIL_COLOR, zorder=3)
                break
            truck = trucks[row]
            self._add_text(left + 0.07, ground + 0.2, self._shorten(truck, limit), ha='left', va='center',
                           fontsize=self._font(TEXT_FONT_SIZE), fontweight='bold', zorder=3)
            end = self._draw_truck(left + 0.07 + TRUCK_NAME_WIDTH, ground, truck, carried[truck], hot)

            # how full it is when not all of the places are drawn
            capacity = int(self._value('CAPACITY', truck))
            if capacity > MAX_TRUCK_SLOTS:
                self._add_text(end + 0.03, ground + 0.2, f'{len(carried[truck])}/{capacity}', ha='left',
                               va='center', fontsize=self._font(PIZZA_FONT_SIZE), color=DETAIL_COLOR, zorder=3)

    def _draw_order(self, location, pizzas, pays, hot):

        # a place per pizza ordered, filling up with those delivered, and any extra ones after
        (left, top) = self._stop_corner(location)
        orders = self._orders[location]
        y = top - 0.39
        x = left + 0.56 + PIZZA_RADIUS
        if orders > 0:
            self._add_text(left + STOP_WIDTH - 0.07, top - 0.13, f'Order {len(pizzas)}/{orders}', ha='right',
                           va='center', fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR, zorder=3)
        count = max(orders, len(pizzas))
        for i in range(min(count, MAX_ORDER_SLOTS)):
            place = x + i * PIZZA_PITCH
            if i == MAX_ORDER_SLOTS - 1 and count > MAX_ORDER_SLOTS:
                self._draw_more(place, y, count - i)
            elif i < len(pizzas):
                is_paid = pays and hot[pizzas[i]]
                self._add_patch(patches.Circle(
                    (place, y), PIZZA_RADIUS + 0.018, zorder=5,
                    facecolor=PAID_COLOR if is_paid else UNPAID_COLOR,
                    edgecolor=PAID_EDGE_COLOR if is_paid else UNPAID_EDGE_COLOR, linewidth=1.8))
                self._draw_pizza(place, y, pizzas[i], hot[pizzas[i]])
            else:
                self._draw_slot(place, y)

    def _draw_strip(self, waiting, disposed, hot):

        # the pizzas not loaded yet on the left and those disposed on the right
        y = self._strip_y
        size = self._font(TEXT_FONT_SIZE)
        label = 'Ready at the shop'
        self._add_text(self._strip_left, y, label, ha='left', va='center', fontsize=size,
                       fontweight='bold', color=DETAIL_COLOR, zorder=3)
        x = self._strip_left + (len(label) + 1) * BOLD_CHAR_WIDTH * TEXT_FONT_SIZE / 72 + PIZZA_RADIUS

        disposed_room = 0
        if disposed:
            limit = int(1.7 * 72 / (CHAR_WIDTH * TEXT_FONT_SIZE))
            names = [self._shorten(pizza, 6) for pizza in disposed]
            text = 'Disposed ' + self._list_names(names, limit - 9)
            self._add_text(self._strip_right, y, text, ha='right', va='center', fontsize=size,
                           color=DETAIL_COLOR, zorder=3)
            disposed_room = len(text) * CHAR_WIDTH * TEXT_FONT_SIZE / 72 + 0.15

        room = self._strip_right - disposed_room - x
        fits = max(int(room / PIZZA_PITCH), 1)
        for i in range(min(len(waiting), fits)):
            if i == fits - 1 and len(waiting) > fits:
                self._draw_more(x + i * PIZZA_PITCH, y, len(waiting) - i)
            else:
                self._draw_pizza(x + i * PIZZA_PITCH, y, waiting[i], hot[waiting[i]])

    def _count_text(self, count, one, many, extra, limit):

        # nothing here says nothing, and the extra is left out when it does not fit
        if not count:
            return ''
        text = f'{count} {one if count == 1 else many}'
        return f'{text}, {extra}' if extra and len(text) + 2 + len(extra) <= limit else text

    def _fill_tile(self, location, trucks, pizzas, color, carried, hot):
        tile = self._tiles[location]

        # the color still tells about the order when the tile is too small for its counts
        tile['frame'].set_edgecolor(color[1])
        tile['frame'].set_facecolor(color[0])
        if tile['counts'] is not None:
            cold = sum(not hot[pizza] for truck in trucks for pizza in carried[truck])
            orders = self._orders[location]
            (first, second) = tile['counts']
            first.set_text(self._count_text(len(trucks), 'truck', 'trucks',
                                            f'{cold} cold' if cold else '', tile['limit']))
            second.set_text(f'Order {len(pizzas)} / {orders}' if orders > 0 else
                            self._count_text(len(pizzas), 'pizza', 'pizzas', '', tile['limit']))

    def _order_color(self, location, num_delivered, paid):
        orders = self._orders[location]
        if orders <= 0:
            return None
        if num_delivered > orders:
            return UNPAID_COLOR, UNPAID_EDGE_COLOR
        return (PAID_COLOR, PAID_EDGE_COLOR) if paid == orders else (OPEN_COLOR, OPEN_EDGE_COLOR)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        hot = {pizza: bool(state[fluent('hot', [pizza])]) for pizza in self._pizzas}
        carried = {truck: [pizza for pizza in self._pizzas
                           if state[fluent('pizzaInTruck', [pizza, truck])]]
                   for truck in self._trucks}
        delivered = {location: [pizza for pizza in self._pizzas
                                if state[fluent('delivered', [pizza, location])]]
                     for location in self._locations}
        disposed = [pizza for pizza in self._pizzas if state[fluent('disposed', [pizza])]]
        taken = set(disposed).union(*carried.values(), *delivered.values())
        waiting = [pizza for pizza in self._pizzas if pizza not in taken]

        total_paid = 0
        truck_locations = set()
        for location in self._locations:
            trucks = [truck for truck in self._trucks if state[fluent('truckAt', [truck, location])]]
            if trucks:
                truck_locations.add(location)
            pizzas = delivered[location]

            # the domain only pays for the hot pizzas where no more were delivered than ordered
            pays = self._orders[location] > 0 and len(pizzas) <= self._orders[location]
            paid = sum(hot[pizza] for pizza in pizzas) if pays else 0
            total_paid += paid
            color = self._order_color(location, len(pizzas), paid)

            if self._has_map:
                if location not in self._shops:
                    roof = self._roofs[location]
                    roof.set_facecolor(color[0] if color else ROOF_COLOR)
                    roof.set_edgecolor(color[1] if color else WALL_EDGE_COLOR)
                self._draw_order(location, pizzas, pays, hot)
                self._draw_trucks(location, trucks, carried, hot)
            else:
                plate = SHOP_PLATE_COLOR if location in self._shops else PLATE_COLOR
                self._fill_tile(location, trucks, pizzas, color or (plate, PLATE_EDGE_COLOR), carried, hot)

        if self._has_map:
            self._draw_strip(waiting, disposed, hot)
            for ((a, _), arrow) in self._road_arrows.items():
                active = a in truck_locations
                arrow.set_color(ACTIVE_ROAD_COLOR if active else ROAD_COLOR)
                arrow.set_zorder(0.6 if active else 0.5)
        else:
            tile = self._tiles[AT_SHOP]
            if tile['counts'] is not None:
                (first, second) = tile['counts']
                first.set_text(self._count_text(len(waiting), 'pizza', 'pizzas', '', tile['limit']))
                second.set_text(f'{len(disposed)} disposed' if disposed else '')

        total = sum(self._orders.values())
        if total == 0:
            title, color = 'No orders to fill', 'black'
        elif total_paid == total:
            title, color = 'All orders filled', PAID_EDGE_COLOR
        else:
            title = f'{total_paid} / {total} ordered pizza{"" if total == 1 else "s"} paid'
            color = 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in self._artists:
            artist.remove()
        self._artists = []
        return img
