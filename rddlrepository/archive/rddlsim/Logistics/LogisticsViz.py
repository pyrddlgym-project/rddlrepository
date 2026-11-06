import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

PLATE_COLOR = '#f5f7f8'
PLATE_EDGE_COLOR = '#cfd8dc'
BUSY_COLOR = '#ff8f00'
CROWDED_COLOR = '#c62828'
DEPOT_COLOR = '#ffffff'
DEPOT_EDGE_COLOR = '#78909c'
ROOF_COLOR = '#b0bec5'
DOOR_COLOR = '#546e7a'
TRUCK_COLOR = '#eceff1'
TRUCK_EDGE_COLOR = '#546e7a'
DOCKED_COLOR = '#90caf9'
DOCKED_EDGE_COLOR = '#1565c0'
WINDOW_COLOR = '#37474f'
WHEEL_COLOR = '#263238'
WAITING_COLOR = '#ffecb3'
WAITING_EDGE_COLOR = '#ff8f00'
DELIVERED_COLOR = '#c8e6c9'
DELIVERED_EDGE_COLOR = '#2e7d32'
SLOT_COLOR = '#90a4ae'
DETAIL_COLOR = '#37474f'

# sizes in inches, before the cities are scaled to fit the figure
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
MIN_WIDTH = 8.6
CITY_WIDTH = 2.15
CITY_GAP = 0.08
HEADER_ROW = 0.22
DEPOT_ROW = 0.25
TRUCK_ROW = 0.47
GROUND_ROW = 0.36
PADDING = 0.08
CRATE_SIZE = (0.33, 0.28)
CRATE_PITCH = 0.36
TRUCK_NAME_LIMIT = 6
CAB_WIDTH = 0.22
TILE_SIZE = (1.05, 0.8)
TILE_GAP = 0.08

# more than this shows how many more there are
MAX_TRUCK_ROWS = 3
MAX_TRUCK_SLOTS = 3
MAX_DOORS = 4

# the number of trucks at which a city gets busier, from the domain
BUSY_TRUCKS = 2
CROWDED_TRUCKS = 4

# cities scaled more than this are too small to read, so they are drawn as tiles
MIN_CITY_SCALE = 0.6

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5
NAME_FONT_SIZE = 10
TEXT_FONT_SIZE = 7
CRATE_FONT_SIZE = 5.5
CHAR_WIDTH = 0.56
BOLD_CHAR_WIDTH = 0.62
NAME_CHAR_WIDTH = 0.72
CRATE_CHAR_WIDTH = 0.72
LINE_HEIGHT = 1.3

NOT_IN_CITY = 'Not in a city'


class LogisticsVisualizer(BaseViz):
    '''Draws each city as a scene in their declared order, since a truck can drive from any
    city to any other. A city has a depot with its loading docks, the trucks there with the
    boxes in their cargo box, docked ones first and in blue, and the boxes lying on the ground.
    Each box says where it has to go, and turns green when delivered. Too many cities to draw
    like this show as tiles with counts.

    Trucks and boxes that start outside every city, which the domain allows, get a scene of
    their own. They can only get there from the start, so it is left out otherwise. The
    actions are not part of the state, so a failed drive or load only shows as nothing
    changing. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_size=(9.0, 5.5),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._trucks = list(model.type_to_objects['truck'])
        self._cities = list(model.type_to_objects['city'])
        self._boxes = list(model.type_to_objects['box'])
        self._destinations = {box: [city for city in self._cities
                                    if self._value('DESTINATION', box, city)]
                              for box in self._boxes}

        initial = model.ground_vars_with_values(model.state_fluents)
        has_lost = self._trucks_in(NOT_IN_CITY, initial) or self._free_in(NOT_IN_CITY, initial)
        self._places = self._cities + ([NOT_IN_CITY] if has_lost else [])
        self._artists = []

        # room for the longest truck name, as far as it can be read next to the truck
        longest = min(max((len(truck) for truck in self._trucks), default=0), TRUCK_NAME_LIMIT)
        self._truck_name_chars = longest
        self._truck_name_width = longest * NAME_CHAR_WIDTH * TEXT_FONT_SIZE / 72 + 0.04

        # as many trucks on a row as fit next to each other, so the largest decides
        capacities = [self._slots_of(truck) for truck in self._trucks]
        self._truck_width = self._width_of_truck(max(capacities, default=1))
        self._trucks_per_row = max(int((CITY_WIDTH - 2 * PADDING) / self._truck_width), 1)
        self._truck_rows = min(math.ceil(len(self._trucks) / self._trucks_per_row), MAX_TRUCK_ROWS)
        self._crates_per_row = int((CITY_WIDTH - 2 * PADDING) / CRATE_PITCH)

        # scenes if these stay readable, otherwise small tiles
        scene = (CITY_WIDTH + CITY_GAP, HEADER_ROW + DEPOT_ROW + self._truck_rows * TRUCK_ROW
                 + GROUND_ROW + PADDING + CITY_GAP)
        count = max(len(self._places), 1)
        scale, num_cols, num_rows = self._fit(count, scene, max_size)
        self._has_scenes = scale >= MIN_CITY_SCALE
        size = scene
        if not self._has_scenes:
            size = TILE_SIZE
            scale, num_cols, num_rows = self._fit(count, size, max_size)
        self._scale = scale

        # title above, then the cities, the legend and the caption
        grid_width = num_cols * size[0] * scale
        grid_height = num_rows * size[1] * scale
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

        # the cities never move, so they are only drawn once and then filled in
        self._scenes = {}
        draw = self._draw_scene if self._has_scenes else self._draw_tile
        for (i, place) in enumerate(self._places):
            row, col = divmod(i, num_cols)
            self._scenes[place] = draw(place, col * size[0], (num_rows - row - 1) * size[1], size)

        handles = [
            patches.Patch(facecolor=WAITING_COLOR, edgecolor=WAITING_EDGE_COLOR, label='Box waiting'),
            patches.Patch(facecolor=DELIVERED_COLOR, edgecolor=DELIVERED_EDGE_COLOR, label='Box delivered'),
            patches.Patch(facecolor=PLATE_COLOR, edgecolor=BUSY_COLOR, label='Busy city'),
            patches.Patch(facecolor=PLATE_COLOR, edgecolor=CROWDED_COLOR, label='Crowded city')
        ]
        if self._has_scenes:
            handles = [
                patches.Patch(facecolor=TRUCK_COLOR, edgecolor=TRUCK_EDGE_COLOR, label='Truck'),
                patches.Patch(facecolor=DOCKED_COLOR, edgecolor=DOCKED_EDGE_COLOR, label='Truck docked')
            ] + handles
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.25 / height),
                        ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                        handlelength=1.4)
        caption = (f'Driving into a city works with chance {self._value("LOW_CROWD_PROB"):g} below '
                   f'{BUSY_TRUCKS} trucks, {self._value("MED_CROWD_PROB"):g} below {CROWDED_TRUCKS} '
                   f'and {self._value("BIG_CROWD_PROB"):g} otherwise. A delivered box earns 1 per step')
        self.fig.text(0.5, 0.1 / height, caption, ha='center', va='bottom', fontsize=9,
                      color=DETAIL_COLOR)

    def _value(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _fit(self, count, size, max_size):

        # the layout with the largest cities, and of those the one with the fewest rows
        best = None
        for num_cols in range(1, count + 1):
            num_rows = math.ceil(count / num_cols)
            scale = min(1.0, max_size[0] / (num_cols * size[0]), max_size[1] / (num_rows * size[1]))
            if best is None or scale >= best[0]:
                best = (scale, num_cols, num_rows)
        return best

    def _font(self, size):
        return max(size * self._scale, MIN_FONT_SIZE)

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _slots_of(self, truck):
        return max(min(int(self._value('CAPACITY', truck)), MAX_TRUCK_SLOTS), 1)

    def _width_of_truck(self, slots):
        return self._truck_name_width + slots * CRATE_PITCH + 0.04 + CAB_WIDTH + 0.06

    # ==========================================================================
    # WHAT NEVER MOVES
    # ==========================================================================

    def _draw_scene(self, place, x, y, size):
        left, right = x + CITY_GAP / 2, x + size[0] - CITY_GAP / 2
        bottom, top = y + CITY_GAP / 2, y + size[1] - CITY_GAP / 2
        frame = self.ax.add_patch(patches.FancyBboxPatch(
            (left, bottom), right - left, top - bottom, boxstyle='round,pad=0,rounding_size=0.07',
            facecolor=PLATE_COLOR, edgecolor=PLATE_EDGE_COLOR, linewidth=1.4, zorder=1))

        # the name on the left and the docks on the right
        is_city = place != NOT_IN_CITY
        name_y = top - HEADER_ROW / 2 - 0.01
        room = (right - left - 2 * PADDING) * (0.62 if is_city else 1)
        limit = int(room * 72 / (NAME_CHAR_WIDTH * NAME_FONT_SIZE))
        self.ax.text(left + PADDING, name_y, self._shorten(place, limit), ha='left', va='center',
                     fontsize=self._font(NAME_FONT_SIZE), fontweight='bold', zorder=3)
        docks = None
        if is_city:
            docks = self.ax.text(right - PADDING, name_y, '', ha='right', va='center',
                                 fontsize=self._font(TEXT_FONT_SIZE), color=DETAIL_COLOR, zorder=3)
            self._draw_depot(place, left + PADDING, right - PADDING, top - HEADER_ROW)
        return {'frame': frame, 'docks': docks, 'left': left + PADDING, 'right': right - PADDING,
                'top': top - HEADER_ROW - DEPOT_ROW, 'bottom': bottom + PADDING / 2}

    def _draw_depot(self, city, left, right, top):

        # a long low building with a door per dock, as many as can be told apart
        height = DEPOT_ROW - 0.05
        self.ax.add_patch(patches.Rectangle(
            (left, top - height), right - left, height, facecolor=DEPOT_COLOR, edgecolor=DEPOT_EDGE_COLOR,
            linewidth=1.0, zorder=2))
        self.ax.add_patch(patches.Rectangle(
            (left - 0.02, top - 0.05), right - left + 0.04, 0.05, facecolor=ROOF_COLOR,
            edgecolor=DEPOT_EDGE_COLOR, linewidth=1.0, zorder=3))
        doors = min(max(int(self._value('MAX-DOCKED', city)), 0), MAX_DOORS)
        for i in range(doors):
            door_x = left + (i + 0.5) * (right - left) / doors
            self.ax.add_patch(patches.Rectangle(
                (door_x - 0.11, top - height), 0.22, height - 0.07, facecolor=DOOR_COLOR, edgecolor='none', zorder=3))

    def _draw_tile(self, place, x, y, size):
        (tile_width, tile_height) = size
        frame = self.ax.add_patch(patches.FancyBboxPatch(
            (x + TILE_GAP / 2, y + TILE_GAP / 2), tile_width - TILE_GAP, tile_height - TILE_GAP,
            boxstyle='round,pad=0,rounding_size=0.07', facecolor=PLATE_COLOR, edgecolor=PLATE_EDGE_COLOR,
            linewidth=1.2, zorder=1))
        scale = self._scale
        center = x + tile_width / 2

        # counts stay at the smallest readable text for as long as they fit on the tile
        font = max(TEXT_FONT_SIZE * scale, MIN_FONT_SIZE)
        limit = int((tile_width - TILE_GAP) * scale * 72 / (CHAR_WIDTH * font))
        has_details = (limit >= len('9 trucks')
                       and tile_height * scale * 72 >= 3 * LINE_HEIGHT * font)

        name_fit = 0.9 * (tile_width - TILE_GAP) * scale * 72 / BOLD_CHAR_WIDTH
        name = self._shorten(place, int(name_fit / MIN_FONT_SIZE))
        self.ax.text(center, y + tile_height * (0.75 if has_details else 0.5), name, ha='center',
                     va='center', fontweight='bold', zorder=3,
                     fontsize=max(min(NAME_FONT_SIZE * scale, name_fit / len(name)), MIN_FONT_SIZE))
        counts = None
        if has_details:
            counts = [self.ax.text(center, y + tile_height * share, '', ha='center', va='center',
                                   fontsize=font, color=DETAIL_COLOR, zorder=3)
                      for share in (0.48, 0.24)]
        return {'frame': frame, 'counts': counts, 'limit': limit}

    # ==========================================================================
    # WHAT MOVES
    # ==========================================================================

    def _add_patch(self, patch):
        self._artists.append(self.ax.add_patch(patch))
        return patch

    def _add_text(self, x, y, text, **kwargs):
        artist = self.ax.text(x, y, text, **kwargs)
        self._artists.append(artist)
        return artist

    def _draw_crate(self, x, y, box, fill, edge):

        # a wooden crate with slats, saying which box it is and where it has to go
        (width, height) = CRATE_SIZE
        self._add_patch(patches.Rectangle((x, y), width, height, facecolor=fill, edgecolor=edge,
                                          linewidth=1.1, zorder=6))
        for slat_x in (x + 0.03, x + width - 0.03):
            self._add_patch(patches.Rectangle((slat_x - 0.006, y), 0.012, height, facecolor=edge,
                                              edgecolor='none', alpha=0.45, zorder=6))

        font = self._font(CRATE_FONT_SIZE)
        limit = int(width * 0.92 * self._scale * 72 / (CRATE_CHAR_WIDTH * font))
        if limit < 1 or 2 * LINE_HEIGHT * font > height * self._scale * 72:
            return
        destinations = self._destinations[box]
        if destinations:
            more = f'+{len(destinations) - 1}' if len(destinations) > 1 else ''
            target = '→' + self._shorten(destinations[0], max(limit - 1 - len(more), 1)) + more
        else:
            target = ''
        self._add_text(x + width / 2, y + height * (0.71 if target else 0.5), self._shorten(box, limit),
                       ha='center', va='center', fontsize=font, fontweight='bold', zorder=7)
        if target:
            self._add_text(x + width / 2, y + height * 0.27, target, ha='center', va='center',
                           fontsize=font, color=DETAIL_COLOR, zorder=7)

    def _draw_more(self, x, y, text):
        (width, height) = CRATE_SIZE
        self._add_patch(patches.Rectangle((x, y), width, height, facecolor=DEPOT_COLOR, edgecolor=SLOT_COLOR,
                                          linewidth=1.0, zorder=6))
        self._add_text(x + width / 2, y + height / 2, text, ha='center', va='center',
                       fontsize=self._font(CRATE_FONT_SIZE), zorder=7)

    def _draw_truck(self, x, ground, truck, boxes, is_docked):

        # the truck faces right, and its cargo box is open to show one place per capacity
        fill, edge = (DOCKED_COLOR, DOCKED_EDGE_COLOR) if is_docked else (TRUCK_COLOR, TRUCK_EDGE_COLOR)
        self._add_text(x, ground + 0.2, self._shorten(truck, self._truck_name_chars), ha='left', va='center',
                       fontsize=self._font(TEXT_FONT_SIZE), fontweight='bold', zorder=3)

        slots = self._slots_of(truck)
        box_x = x + self._truck_name_width
        box_width = slots * CRATE_PITCH + 0.04
        bottom = ground + 0.05
        height = CRATE_SIZE[1] + 0.06
        self._add_patch(patches.FancyBboxPatch(
            (box_x, bottom), box_width, height, boxstyle='round,pad=0,rounding_size=0.03',
            facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=4))
        cab_x = box_x + box_width + 0.01
        self._add_patch(patches.FancyBboxPatch(
            (cab_x, bottom), CAB_WIDTH, 0.24, boxstyle='round,pad=0,rounding_size=0.04',
            facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=4))
        self._add_patch(patches.Rectangle(
            (cab_x + 0.08, bottom + 0.12), 0.1, 0.08, facecolor=WINDOW_COLOR, edgecolor='none', zorder=5))
        for wheel in (box_x + 0.1, cab_x + 0.1):
            self._add_patch(patches.Circle((wheel, ground + 0.05), 0.05, facecolor=WHEEL_COLOR,
                                           edgecolor='none', zorder=6))

        first = box_x + 0.02 + (CRATE_PITCH - CRATE_SIZE[0]) / 2
        crate_y = bottom + 0.03
        for i in range(slots):
            place = first + i * CRATE_PITCH
            if i == slots - 1 and len(boxes) > slots:
                self._draw_more(place, crate_y, f'+{len(boxes) - i}')
            elif i < len(boxes):
                self._draw_crate(place, crate_y, boxes[i], WAITING_COLOR, WAITING_EDGE_COLOR)
            else:
                self._add_patch(patches.Rectangle(
                    (place, crate_y), CRATE_SIZE[0], CRATE_SIZE[1], facecolor='none', edgecolor=SLOT_COLOR,
                    linewidth=0.9, linestyle=(0, (2, 2)), zorder=5))

        # how full it is when not all of the places are drawn
        capacity = int(self._value('CAPACITY', truck))
        if capacity > MAX_TRUCK_SLOTS:
            self._add_text(cab_x + CAB_WIDTH / 2, bottom + 0.31, f'{len(boxes)}/{capacity}', ha='center',
                           va='center', fontsize=self._font(CRATE_FONT_SIZE), color=DETAIL_COLOR, zorder=7)

    def _draw_trucks(self, scene, trucks, carried, docked):

        # docked trucks first, so that they stand right below the depot
        ordered = sorted(trucks, key=lambda truck: truck not in docked)
        room = self._truck_rows * self._trucks_per_row
        for (i, truck) in enumerate(ordered[:room]):
            row, col = divmod(i, self._trucks_per_row)
            x = scene['left'] + col * self._truck_width
            ground = scene['top'] - (row + 1) * TRUCK_ROW + 0.03

            # the last place says how many trucks did not fit
            if i == room - 1 and len(ordered) > room:
                self._add_text(x, ground + 0.2, f'+{len(ordered) - i} more trucks', ha='left', va='center',
                               fontsize=self._font(TEXT_FONT_SIZE), fontweight='bold', color=DETAIL_COLOR,
                               zorder=3)
                break
            self._draw_truck(x, ground, truck, carried[truck], truck in docked)

    def _draw_ground(self, scene, place, boxes):
        y = scene['bottom']
        fits = self._crates_per_row
        for (i, box) in enumerate(boxes[:fits]):
            x = scene['left'] + i * CRATE_PITCH
            if i == fits - 1 and len(boxes) > fits:
                self._draw_more(x, y, f'+{len(boxes) - i}')
            elif place in self._destinations[box]:
                self._draw_crate(x, y, box, DELIVERED_COLOR, DELIVERED_EDGE_COLOR)
            else:
                self._draw_crate(x, y, box, WAITING_COLOR, WAITING_EDGE_COLOR)

    def _trucks_in(self, place, state):
        fluent = RDDLPlanningModel.ground_var
        if place == NOT_IN_CITY:
            return [truck for truck in self._trucks
                    if not any(state[fluent('truckAt', [truck, city])] for city in self._cities)]
        return [truck for truck in self._trucks if state[fluent('truckAt', [truck, place])]]

    def _free_in(self, place, state):

        # a box in a truck is also in the city of that truck, so it does not count as lying there
        fluent = RDDLPlanningModel.ground_var
        free = [box for box in self._boxes
                if not any(state[fluent('boxInTruck', [box, truck])] for truck in self._trucks)]
        if place == NOT_IN_CITY:
            return [box for box in free
                    if not any(state[fluent('boxInCity', [box, city])] for city in self._cities)]
        return [box for box in free if state[fluent('boxInCity', [box, place])]]

    def _count_text(self, count, one, many, extra, limit):

        # an empty city says nothing, and the extra is left out when it does not fit
        if not count:
            return ''
        text = f'{count} {one if count == 1 else many}'
        return f'{text}, {extra}' if extra and len(text) + 2 + len(extra) <= limit else text

    def _fill_tile(self, tile, place, trucks, boxes, docked):

        # the color still tells about the boxes when the tile is too small for its counts
        delivered = sum(place in self._destinations[box] for box in boxes)
        if delivered < len(boxes):
            tile['frame'].set_facecolor(WAITING_COLOR)
        elif delivered:
            tile['frame'].set_facecolor(DELIVERED_COLOR)
        else:
            tile['frame'].set_facecolor(PLATE_COLOR)

        if tile['counts'] is not None:
            at_dock = sum(truck in docked for truck in trucks)
            (first, second) = tile['counts']
            first.set_text(self._count_text(len(trucks), 'truck', 'trucks',
                                            f'{at_dock} dock' if at_dock else '', tile['limit']))
            second.set_text(self._count_text(len(boxes), 'box', 'boxes',
                                             f'{delivered} done' if delivered else '', tile['limit']))

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        carried = {truck: [box for box in self._boxes if state[fluent('boxInTruck', [box, truck])]]
                   for truck in self._trucks}
        docked = {truck for truck in self._trucks
                  if any(state[fluent('atDepot', [truck, city])] for city in self._cities)}

        delivered = set()
        for place in self._places:
            trucks = self._trucks_in(place, state)
            boxes = self._free_in(place, state)
            delivered.update(box for box in boxes if place in self._destinations[box])

            scene = self._scenes[place]
            if place == NOT_IN_CITY or len(trucks) < BUSY_TRUCKS:
                scene['frame'].set_edgecolor(PLATE_EDGE_COLOR)
            elif len(trucks) < CROWDED_TRUCKS:
                scene['frame'].set_edgecolor(BUSY_COLOR)
            else:
                scene['frame'].set_edgecolor(CROWDED_COLOR)

            if self._has_scenes:
                if scene['docks'] is not None:
                    count = sum(state[fluent('atDepot', [truck, place])] for truck in self._trucks)
                    scene['docks'].set_text(f'Dock {count} / {self._value("MAX-DOCKED", place)}')
                self._draw_trucks(scene, trucks, carried, docked)
                self._draw_ground(scene, place, boxes)
            else:
                self._fill_tile(scene, place, trucks, boxes, docked)

        total = sum(bool(self._destinations[box]) for box in self._boxes)
        if total == 0:
            title, color = 'No boxes to deliver', 'black'
        elif len(delivered) == total:
            title, color = 'All boxes delivered', DELIVERED_EDGE_COLOR
        else:
            title = f'{len(delivered)} / {total} box{"" if total == 1 else "es"} delivered'
            color = 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in self._artists:
            artist.remove()
        self._artists = []
        return img
