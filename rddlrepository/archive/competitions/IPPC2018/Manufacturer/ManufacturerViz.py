import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

RAW_COLOR = '#ffffff'
RAW_EDGE_COLOR = '#90a4ae'
NONE_COLOR = '#eceff1'
NONE_EDGE_COLOR = '#90a4ae'
BUILDING_COLOR = '#ffecb3'
BUILDING_EDGE_COLOR = '#ff8f00'
FACTORY_COLOR = '#e3f2fd'
FACTORY_EDGE_COLOR = '#64b5f6'
READY_COLOR = '#90caf9'
READY_EDGE_COLOR = '#1565c0'
HIRED_COLOR = '#c8e6c9'
GREEN_COLOR = '#2e7d32'
STOCK_COLOR = '#2e7d32'
NEEDED_COLOR = '#b0bec5'
BAR_ON_COLOR = '#37474f'
BAR_OFF_COLOR = '#cfd8dc'
DETAIL_COLOR = '#37474f'
CARD_COLOR = '#fafafa'
CARD_EDGE_COLOR = '#b0bec5'
GROUND_COLOR = '#bcaaa4'
LOG_COLOR = '#d7b48a'
LOG_EDGE_COLOR = '#8d6e63'
LOT_COLOR = '#90a4ae'
WALL_COLOR = '#ffcc80'
CRANE_COLOR = '#f9a825'
FACTORY_BODY_COLOR = '#90a4ae'
FACTORY_DARK_COLOR = '#546e7a'
WINDOW_COLOR = '#eceff1'
LIT_COLOR = '#ffd54f'
SMOKE_COLOR = '#b0bec5'
CRATE_COLOR = '#c8a165'
CRATE_EDGE_COLOR = '#8d6e63'
STAFF_COLOR = '#3949ab'
APPLYING_COLOR = '#9fa8da'
VACANT_COLOR = '#cfd8dc'
HARD_HAT_COLOR = '#fbc02d'
MEGAPHONE_COLOR = '#ef6c00'
BRIEFCASE_COLOR = '#6d4c41'
TREND_MARKS = {'up': ('▲', '#2e7d32'), 'stable': ('▬', '#78909c'), 'down': ('▼', '#c62828')}

# sizes of the graph in inches before it is scaled to fit the figure
NODE_WIDTH = 1.22
NODE_HEIGHT = 0.62
ROW_PITCH = 0.9
NODE_GAP = 0.12
PASSING_WIDTH = 0.02
PASSING_GAP = 0.05
ARC_HEIGHT = 0.2
PORT_SPACING = 0.14
PORT_SPAN = 1.0
ORDERING_SWEEPS = 8

# where things go on a card, in inches before scaling and from its middle or its sides:
# the name and price on top, and below them the site of the factory, a crate when the
# good is for sale, its production manager and the price level
PADDING = 0.06
PRICE_ROW = 0.19
TREND_WIDTH = 0.13
PRICE_ROOM = 0.48
SITE_BOTTOM = -0.25
SITE_WIDTH, SITE_HEIGHT = 0.46, 0.3
CRATE_LEFT, CRATE_SIZE = 0.51, 0.12
MANAGER_X, MANAGER_HEIGHT = 0.71, 0.26
GAUGE_HEIGHT = 0.17
GAUGE_WIDTH = 0.32

# more goods than this at one level continue on another row, as they would not be readable
MAX_ROW_GOODS = 8

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5
NAME_FONT_SIZE = 8.5
PRICE_FONT_SIZE = 8.5
DETAIL_FONT_SIZE = 6.5
CHAR_WIDTH = 0.62
LINE_HEIGHT = 1.3

# room around the figure, and sizes in inches of the tiles of a very large company
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
STAFF_HEIGHT = 0.5
STAFF_PITCH = 2.7
LEGEND_SPACING = 1.4
LEGEND_ROW = 0.36
TILE_GAP = 0.04
LEVEL_COLOR = '#f5f7f8'
LEVEL_LABEL_WIDTH = 0.72
LEVEL_GAP = 0.05
LEVEL_PADDING = 0.03
MAX_LEVEL_BANDS = 12
MAX_TILE_NAME = 12


class ManufacturerVisualizer(BaseViz):
    '''Draws the production chain with the raw goods on top. Each good is a card with its
    price now, the price level and long-term trend, and a picture of where it comes from:
    a pile of logs for a raw good, an empty lot where a factory could be built, walls
    rising while one is built, and a factory, which smokes when all its inputs are in
    stock. A crate shows a good for sale and a worker in a hard hat its production
    manager. Each arrow turns green when the good it starts from is in stock to make the
    one it points to. A row of people on top shows the staff, in color when hired, with a
    raised hand when applying and greyed out otherwise. A chain too large to follow shows
    the goods as colored tiles grouped by level instead.

    The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_scale=1.15,
                 max_size=(9.0, 5.5),
                 min_size=(8.8, 2.8),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        initial = model.ground_vars_with_values(model.state_fluents)

        self._goods = list(model.type_to_objects['good'])
        self._price_levels = list(model.type_to_objects['price-level'])
        self._stages = [stage for stage in model.type_to_objects['construction-progress']
                        if stage != 'not-building']
        self._inputs = {good: [other for other in self._goods
                               if self._nonfluent('PROCESSED_TO', other, good)]
                        for good in self._goods}
        self._edges = [(source, good) for good in self._goods for source in self._inputs[good]]

        # staff is left out when the instance has no use for it
        fluent = RDDLPlanningModel.ground_var
        self._has_marketing = (self._nonfluent('PROB_MARKETING_MANAGER_APPLIES') > 0
                               or self._nonfluent('MARKETING_MANAGER_BONUS') != 0
                               or initial[fluent('have-marketing-manager', [])])
        self._has_lobbyist = (self._nonfluent('PROB_LOBBYIST_APPLIES') > 0
                              or self._nonfluent('LOBBYIST_BONUS') != 0
                              or initial[fluent('have-lobbyist', [])])
        self._has_production = (self._nonfluent('PROB_PRODUCTION_MANAGER_APPLIES') > 0
                                or self._nonfluent('SALARY_PRODUCTION_MANAGER') != 0
                                or any(initial[fluent('have-production-manager', [good])]
                                       for good in self._goods))
        self._staff_room = (STAFF_HEIGHT if self._has_marketing or self._has_lobbyist
                            or self._has_production else 0)

        # lay out the graph and scale it to fit the figure, but a chain so large that this
        # would make the cards unreadable has too many arrows to follow, so it is drawn as
        # tiles grouped by level instead
        levels = self._levels()
        self._layout(levels)
        xs = list(self._x.values())
        graph_width = max(xs) - min(xs) + NODE_WIDTH + 0.1
        arc_height = ARC_HEIGHT if any(self._row[a] == self._row[b] for (a, b) in self._edges) else 0
        graph_height = (len(self._rows) - 1) * ROW_PITCH + NODE_HEIGHT + 2 * arc_height + 0.1
        scale = min(max_scale, max_size[0] / graph_width,
                    (max_size[1] - self._staff_room) / graph_height)

        self._boxes, self._details, self._arrows = {}, {}, {}
        self._prices, self._trends, self._bars, self._sale_marks = {}, {}, {}, {}
        self._sites, self._managers = {}, {}
        self._has_arrows = DETAIL_FONT_SIZE * min(scale, 1) >= MIN_FONT_SIZE
        if self._has_arrows:
            self._draw_graph(graph_width, graph_height, scale, xs, min_size)
        else:
            self._draw_tiles(levels, max_size, min_size)
        height = self._figure_size[1]

        if self._has_arrows:
            self._draw_legend()
            caption = 'Numbers and bars are prices now, before staff bonuses'
        else:
            handles = self._legend_handles()
            self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.25 / height),
                            ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                            handlelength=1.4)
            caption = 'Tiles show the price now before staff bonuses, its trend and the inputs in stock.'
        self.fig.text(0.5, 0.1 / height, caption, ha='center', va='bottom', fontsize=9,
                      color=DETAIL_COLOR)

        self._add_staff()

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

    def _fit_name(self, name, room, size):

        # a long name first gets smaller and then shorter to fit in this many inches
        fit = room * 72 / CHAR_WIDTH
        return (self._shorten(name, int(fit / MIN_FONT_SIZE)),
                max(min(size, fit / len(name)), MIN_FONT_SIZE))

    def _open_figure(self, width, height):

        # title above, then the staff, the goods, the legend and the caption
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, self._bottom / height, 1,
                                     (height - self._bottom - TOP_MARGIN) / height])
        self.ax.axis('off')

    def _inch_axes(self, bottom, height):

        # axes across the figure, in inches from their bottom left, for pictures
        (width, fig_height) = self._figure_size
        ax = self.fig.add_axes([0, bottom / fig_height, 1, height / fig_height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.axis('off')
        return ax

    def _add_box(self, good, x, y, width, height, ring):
        self._boxes[good] = self.ax.add_patch(patches.FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle=f'round,pad=0,rounding_size={height / 8:g}', linewidth=ring, zorder=2))

    def _add_card(self, good, scale):
        x, y = self._x[good], self._y(good)
        fonts = min(scale, 1)
        left, right = x - NODE_WIDTH / 2 + PADDING, x + NODE_WIDTH / 2 - PADDING
        self._add_box(good, x, y, NODE_WIDTH, NODE_HEIGHT, 1.4)
        self._boxes[good].set(facecolor=CARD_COLOR, edgecolor=CARD_EDGE_COLOR)

        room = (NODE_WIDTH - 2 * PADDING - PRICE_ROOM - TREND_WIDTH) * scale
        (name, size) = self._fit_name(good, room, NAME_FONT_SIZE * fonts)
        self.ax.text(left, y + PRICE_ROW, name, ha='left', va='center', fontsize=size,
                     fontweight='bold', zorder=3)
        self._prices[good] = self.ax.text(
            right - TREND_WIDTH, y + PRICE_ROW, '', ha='right', va='center',
            fontsize=PRICE_FONT_SIZE * fonts, fontweight='bold', zorder=3)
        self._trends[good] = self.ax.text(
            right, y + PRICE_ROW, '', ha='right', va='center',
            fontsize=PRICE_FONT_SIZE * fonts, zorder=3)

        # the bars rise with the price level, as far as the price has
        count = len(self._price_levels)
        pitch = GAUGE_WIDTH / count
        self._bars[good] = [self.ax.add_patch(patches.Rectangle(
            (right - GAUGE_WIDTH + i * pitch, y + SITE_BOTTOM), 0.75 * pitch,
            GAUGE_HEIGHT * (i + 2) / (count + 1), edgecolor='none', zorder=3))
            for i in range(count)]

        # all the pictures are drawn once, and render shows the ones that apply
        bottom = y + SITE_BOTTOM
        self._sites[good] = self._site(self.ax, left, bottom, SITE_WIDTH, SITE_HEIGHT)
        self._sale_marks[good] = self._crate(self.ax, left + CRATE_LEFT, bottom, CRATE_SIZE)
        self._managers[good] = self._person(self.ax, left + MANAGER_X, bottom, MANAGER_HEIGHT,
                                            STAFF_COLOR, 'production')
        for artist in self._sale_marks[good] + self._managers[good]:
            artist.set_visible(False)

    # ---------- pictures: each takes the axes, a place and a size, and returns its artists

    @staticmethod
    def _silhouette(left, bottom, width, height):

        # a factory hall with a sawtooth roof of three teeth
        points = [(left, bottom), (left, bottom + 0.5 * height)]
        for k in range(3):
            points += [(left + (k + 1) * 0.24 * width, bottom + 0.78 * height),
                       (left + (k + 1) * 0.24 * width, bottom + 0.5 * height)]
        return points[:-1] + [(left + 0.72 * width, bottom)]

    @classmethod
    def _site(cls, ax, left, bottom, width, height):

        # the ground, then the pictures for a raw good, an empty lot, a factory being
        # built and one that is built, the last with smoke and lit windows when working;
        # all but the ground are hidden, and the walls being built are set by render
        def add(patch, zorder=3):
            patch.set_zorder(zorder)
            return ax.add_patch(patch)

        ax.add_patch(patches.Rectangle((left - 0.02 * width, bottom - 0.03 * height),
                                       1.04 * width, 0.04 * height, facecolor=GROUND_COLOR,
                                       edgecolor='none', zorder=3))
        site = {'wall_height': 0.5 * height}

        # a pile of logs, seen from their ends
        radius = 0.1 * width
        site['raw'] = []
        for (row, count) in enumerate((3, 2, 1)):
            for k in range(count):
                cx = left + 0.4 * width + (k - (count - 1) / 2) * 2 * radius
                cy = bottom + radius + row * 1.72 * radius
                site['raw'] += [
                    add(patches.Circle((cx, cy), radius, facecolor=LOG_COLOR,
                                       edgecolor=LOG_EDGE_COLOR, linewidth=0.8)),
                    add(patches.Circle((cx, cy), 0.45 * radius, facecolor='none',
                                       edgecolor=LOG_EDGE_COLOR, linewidth=0.5))]

        outline = cls._silhouette(left, bottom, width, height)
        site['lot'] = [add(patches.Polygon(outline, closed=True, facecolor='none',
                                           edgecolor=LOT_COLOR, linewidth=0.9, linestyle='--'))]

        # walls rising inside the outline, next to a crane
        site['walls'] = add(patches.Rectangle((left, bottom), 0.72 * width, 0, facecolor=WALL_COLOR,
                                              edgecolor='none'), 2.9)
        site['building'] = [
            site['walls'], site['lot'][0],
            add(patches.Rectangle((left + 0.84 * width, bottom), 0.05 * width, height,
                                  facecolor=CRANE_COLOR, edgecolor='none')),
            add(patches.Rectangle((left + 0.45 * width, bottom + 0.95 * height), 0.55 * width,
                                  0.06 * height, facecolor=CRANE_COLOR, edgecolor='none')),
            add(patches.Rectangle((left + 0.52 * width, bottom + 0.62 * height), 0.015 * width,
                                  0.33 * height, facecolor=FACTORY_DARK_COLOR, edgecolor='none'))]

        windows = [(left + (0.06 + 0.23 * k) * width, bottom + 0.14 * height)
                   for k in range(3)]
        site['factory'] = [
            add(patches.Rectangle((left + 0.76 * width, bottom), 0.1 * width, 0.92 * height,
                                  facecolor=FACTORY_DARK_COLOR, edgecolor='none')),
            add(patches.Polygon(outline, closed=True, facecolor=FACTORY_BODY_COLOR,
                                edgecolor=FACTORY_DARK_COLOR, linewidth=0.8))]
        site['factory'] += [add(patches.Rectangle(xy, 0.12 * width, 0.17 * height,
                                                  facecolor=WINDOW_COLOR, edgecolor='none'), 3.1)
                            for xy in windows]
        site['working'] = [add(patches.Rectangle(xy, 0.12 * width, 0.17 * height,
                                                 facecolor=LIT_COLOR, edgecolor='none'), 3.2)
                           for xy in windows]
        site['working'] += [
            add(patches.Circle((left + 0.84 * width, bottom + 1.04 * height), 0.1 * height,
                               facecolor=SMOKE_COLOR, edgecolor='none')),
            add(patches.Circle((left + 0.96 * width, bottom + 1.2 * height), 0.13 * height,
                               facecolor=SMOKE_COLOR, edgecolor='none'))]
        for kind in ('raw', 'lot', 'building', 'factory', 'working'):
            for artist in site[kind]:
                artist.set_visible(False)
        return site

    @staticmethod
    def _show_site(site, kind, progress=0.0):

        # a factory that is working is one that is built with smoke and lit windows
        shown = {'working': ('factory', 'working')}.get(kind, (kind,))
        for other in ('raw', 'lot', 'building', 'factory', 'working'):
            for artist in site[other]:
                artist.set_visible(False)
        for other in shown:
            for artist in site[other]:
                artist.set_visible(True)
        site['walls'].set_height(progress * site['wall_height'])

    @staticmethod
    def _crate(ax, left, bottom, size):

        # a wooden crate with a green price tag on its corner
        return [ax.add_patch(patches.Rectangle((left, bottom), size, size, facecolor=CRATE_COLOR,
                                               edgecolor=CRATE_EDGE_COLOR, linewidth=0.8, zorder=3)),
                ax.add_patch(patches.Rectangle((left, bottom + 0.45 * size), size, 0.1 * size,
                                               facecolor=CRATE_EDGE_COLOR, edgecolor='none', zorder=3.1)),
                ax.add_patch(patches.Circle((left + size, bottom + size), 0.25 * size,
                                            facecolor=GREEN_COLOR, edgecolor='white',
                                            linewidth=0.6, zorder=3.2))]

    @staticmethod
    def _person(ax, x, bottom, height, color, kind, raised=False):

        # a person standing on the given point, with what tells their job: a hard hat,
        # a megaphone or a briefcase, and an arm raised when applying
        h = height
        artists = [
            ax.add_patch(patches.FancyBboxPatch(
                (x - 0.2 * h, bottom), 0.4 * h, 0.58 * h,
                boxstyle=f'round,pad=0,rounding_size={0.15 * h}', facecolor=color,
                edgecolor='none', zorder=4)),
            ax.add_patch(patches.Circle((x, bottom + 0.76 * h), 0.15 * h, facecolor=color,
                                        edgecolor='none', zorder=4))]
        accessory = VACANT_COLOR if color == VACANT_COLOR else None
        if kind == 'production':
            artists += [
                ax.add_patch(patches.Wedge((x, bottom + 0.8 * h), 0.17 * h, 0, 180,
                                           facecolor=accessory or HARD_HAT_COLOR,
                                           edgecolor='none', zorder=4.1)),
                ax.add_patch(patches.Rectangle((x - 0.22 * h, bottom + 0.78 * h), 0.44 * h,
                                               0.04 * h, facecolor=accessory or HARD_HAT_COLOR,
                                               edgecolor='none', zorder=4.1))]
        elif kind == 'marketing':
            artists.append(ax.add_patch(patches.Polygon(
                [(x + 0.12 * h, bottom + 0.74 * h), (x + 0.4 * h, bottom + 0.6 * h),
                 (x + 0.4 * h, bottom + 0.92 * h)], closed=True,
                facecolor=accessory or MEGAPHONE_COLOR, edgecolor='none', zorder=4.1)))
        elif kind == 'lobbyist':
            artists += [
                ax.add_patch(patches.Rectangle((x + 0.14 * h, bottom + 0.04 * h), 0.3 * h,
                                               0.22 * h, facecolor=accessory or BRIEFCASE_COLOR,
                                               edgecolor='none', zorder=4.1)),
                ax.add_patch(patches.Rectangle((x + 0.24 * h, bottom + 0.26 * h), 0.1 * h,
                                               0.05 * h, facecolor=accessory or BRIEFCASE_COLOR,
                                               edgecolor='none', zorder=4.1))]
        if raised:
            (sx, sy), (hx, hy) = (x - 0.14 * h, bottom + 0.48 * h), (x - 0.3 * h, bottom + 0.98 * h)
            (dx, dy) = (0.045 * h, 0.015 * h)
            artists += [
                ax.add_patch(patches.Polygon([(sx - dx, sy - dy), (sx + dx, sy + dy),
                                              (hx + dx, hy + dy), (hx - dx, hy - dy)],
                                             closed=True, facecolor=color, edgecolor='none',
                                             zorder=4)),
                ax.add_patch(patches.Circle((hx, hy), 0.06 * h, facecolor=color,
                                            edgecolor='none', zorder=4))]
        return artists

    def _add_tile(self, good, x, y, width, height, name_size, has_detail):
        self._add_box(good, x, y, width, height, 1.2)

        # the mark goes in the corner to stay clear of the name, which is in the middle
        radius = min(width, height) / 11
        self._sale_marks[good] = [self.ax.add_patch(patches.Circle(
            (x + width / 2 - 1.6 * radius, y + height / 2 - 1.6 * radius), radius,
            facecolor=GREEN_COLOR, edgecolor='none', zorder=4))]
        if name_size is None:
            return
        (name, size) = self._fit_name(good, 0.9 * width, name_size)
        self.ax.text(x, y + (0.19 * height if has_detail else 0), name, ha='center',
                     va='center', fontsize=size, fontweight='bold', zorder=3)
        if has_detail:
            self._details[good] = self.ax.text(
                x, y - 0.24 * height, '', ha='center', va='center', color=DETAIL_COLOR, zorder=3,
                fontsize=max(name_size * DETAIL_FONT_SIZE / NAME_FONT_SIZE, MIN_FONT_SIZE))

    def _draw_graph(self, graph_width, graph_height, scale, xs, min_size):
        width = max(graph_width * scale + 0.4, min_size[0])
        self._bottom = 0.42 + LEGEND_ROW * self._legend_rows(width)[1]
        height = max(graph_height * scale + self._bottom + TOP_MARGIN + self._staff_room, min_size[1])
        self._open_figure(width, height)
        center_x = (max(xs) + min(xs)) / 2
        center_y = -(len(self._rows) - 1) * ROW_PITCH / 2
        half_width = width / (2 * scale)
        half_height = (height - self._bottom - TOP_MARGIN - self._staff_room) / (2 * scale)
        self.ax.set_xlim(center_x - half_width, center_x + half_width)
        self.ax.set_ylim(center_y - half_height, center_y + half_height + self._staff_room / scale)

        # the edges and cards never move, so they are only drawn once and then recolored
        self._arrows = {edge: self.ax.add_patch(patches.FancyArrowPatch(
            path=self._edge_path(edge), arrowstyle='-|>', mutation_scale=7 * scale,
            linewidth=0.9, shrinkA=0, shrinkB=0, zorder=1)) for edge in self._edges}
        for good in self._goods:
            self._add_card(good, scale)

    def _draw_tiles(self, levels, max_size, min_size):

        self._bottom = BOTTOM_MARGIN

        # a long chain of levels would leave no room for anything but the bands, so past a
        # limit neighbouring levels share a band, which keeps the goods in the same order
        per_band = math.ceil(len(levels) / MAX_LEVEL_BANDS)
        bands = []
        for start in range(0, len(levels), per_band):
            last = min(start + per_band, len(levels))
            label = f'Level {last}' if last == start + 1 else f'Levels {start + 1}–{last}'
            bands.append((label, [good for level in levels[start:last] for good in level]))

        # the number of columns that leaves room for the largest text, since more of them
        # make the tiles narrower and fewer make long levels wrap onto more rows, where
        # longer names than can be read at any size are shortened rather than made room for
        longest = min(max(len(good) for good in self._goods), MAX_TILE_NAME)
        room_x = max_size[0] - LEVEL_LABEL_WIDTH
        room_y = max_size[1] - self._staff_room - len(bands) * (LEVEL_GAP + 2 * LEVEL_PADDING)
        best = None
        for columns in range(1, max(len(goods) for (_, goods) in bands) + 1):
            rows = sum(math.ceil(len(goods) / columns) for (_, goods) in bands)
            cell_x = min(room_x / columns, NODE_WIDTH + TILE_GAP)
            cell_y = min(room_y / rows, NODE_HEIGHT + TILE_GAP)
            tile_x, tile_y = cell_x - self._tile_gap(cell_x), cell_y - self._tile_gap(cell_y)
            font = min(NAME_FONT_SIZE, 0.9 * tile_x * 72 / (CHAR_WIDTH * longest),
                       tile_y * 72 / (2 * LINE_HEIGHT))
            if best is None or (font, -rows) > best[0]:
                best = ((font, -rows), columns, cell_x, cell_y, tile_x, tile_y, font)
        (_, columns, cell_x, cell_y, tile_x, tile_y, font) = best

        # text goes first from the details and then from the names when there is no room
        line_room = tile_y * 72 / LINE_HEIGHT
        has_detail = line_room >= 2 * MIN_FONT_SIZE
        name_size = max(font, MIN_FONT_SIZE) if line_room >= MIN_FONT_SIZE else None

        rows = [math.ceil(len(goods) / columns) for (_, goods) in bands]
        band_width = LEVEL_LABEL_WIDTH + columns * cell_x + TILE_GAP
        grid_height = sum(rows) * cell_y + len(bands) * (2 * LEVEL_PADDING + LEVEL_GAP) - LEVEL_GAP
        width = max(band_width + 0.4, min_size[0])
        height = max(grid_height + BOTTOM_MARGIN + TOP_MARGIN + self._staff_room, min_size[1])
        self._open_figure(width, height)
        self.ax.set_xlim(-(width - band_width) / 2, (width + band_width) / 2)
        extra = height - BOTTOM_MARGIN - TOP_MARGIN - self._staff_room - grid_height
        self.ax.set_ylim(-grid_height - extra / 2, extra / 2 + self._staff_room)

        y = 0
        for ((label, goods), band_rows) in zip(bands, rows):
            band_height = band_rows * cell_y + 2 * LEVEL_PADDING
            self.ax.add_patch(patches.FancyBboxPatch(
                (0, y - band_height), band_width, band_height,
                boxstyle='round,pad=0,rounding_size=0.04', facecolor=LEVEL_COLOR,
                edgecolor='none', zorder=0))
            self.ax.text(LEVEL_LABEL_WIDTH / 2, y - band_height / 2, label,
                         ha='center', va='center', fontsize=7.5, color=DETAIL_COLOR, zorder=1)
            for (i, good) in enumerate(goods):
                x = LEVEL_LABEL_WIDTH + (i % columns + 0.5) * cell_x
                center_y = y - LEVEL_PADDING - (i // columns + 0.5) * cell_y
                self._add_tile(good, x, center_y, tile_x, tile_y, name_size, has_detail)
            y -= band_height + LEVEL_GAP

    def _tile_gap(self, cell):

        # however many tiles there are, the gaps between them never take up the tiles
        return min(TILE_GAP, cell / 4)

    def _legend_handles(self):

        # the tiles of a very large company are colored rather than drawn
        handles = [
            patches.Patch(facecolor=RAW_COLOR, edgecolor=RAW_EDGE_COLOR, label='Raw good'),
            patches.Patch(facecolor=NONE_COLOR, edgecolor=NONE_EDGE_COLOR, label='No factory'),
            patches.Patch(facecolor=BUILDING_COLOR, edgecolor=BUILDING_EDGE_COLOR, label='Building'),
            patches.Patch(facecolor=FACTORY_COLOR, edgecolor=FACTORY_EDGE_COLOR, label='Factory'),
            patches.Patch(facecolor=READY_COLOR, edgecolor=READY_EDGE_COLOR, label='Ready'),
            Line2D([], [], marker='o', color='none', markerfacecolor=GREEN_COLOR,
                   markeredgecolor='none', markersize=7, label='For sale')
        ]
        for (label, marker, trend) in (('Rising', '^', 'up'), ('Stable', 's', 'stable'),
                                       ('Falling', 'v', 'down')):
            handles.append(Line2D([], [], marker=marker, color='none', markersize=6, label=label,
                                  markerfacecolor=TREND_MARKS[trend][1], markeredgecolor='none'))
        return handles

    def _legend_entries(self):
        entries = [('raw', 'Raw good'), ('lot', 'No factory'), ('building', 'Building'),
                   ('factory', 'Factory'), ('working', 'Inputs ready'), ('sale', 'For sale'),
                   ('stock', 'In stock'), ('needed', 'Needed')]
        if self._staff_room:
            entries += [('hired', 'Hired'), ('applying', 'Applying'), ('vacant', 'Vacant')]
        return entries

    def _legend_rows(self, width):
        per_row = max(1, int((width - 0.3) / LEGEND_SPACING))
        return per_row, math.ceil(len(self._legend_entries()) / per_row)

    def _draw_legend(self):

        # the legend shows the same pictures as the cards, between the graph and the caption
        (width, _) = self._figure_size
        (per_row, rows) = self._legend_rows(width)
        ax = self._inch_axes(0.3, LEGEND_ROW * rows)
        entries = self._legend_entries()
        for (k, (kind, label)) in enumerate(entries):
            r, c = divmod(k, per_row)
            count = min(per_row, len(entries) - r * per_row)
            x = (width - count * LEGEND_SPACING) / 2 + c * LEGEND_SPACING + 0.1
            y = LEGEND_ROW * (rows - r) - 0.3
            if kind in ('raw', 'lot', 'building', 'factory', 'working'):
                site = self._site(ax, x, y, 0.34, 0.22)
                self._show_site(site, kind, 0.5)
            elif kind == 'sale':
                self._crate(ax, x + 0.1, y, 0.16)
            elif kind in ('stock', 'needed'):
                ax.add_patch(patches.FancyArrowPatch(
                    (x, y + 0.1), (x + 0.34, y + 0.1), arrowstyle='-|>', mutation_scale=7,
                    linewidth=1.6 if kind == 'stock' else 0.9,
                    color=STOCK_COLOR if kind == 'stock' else NEEDED_COLOR))
            else:
                color = {'hired': STAFF_COLOR, 'applying': APPLYING_COLOR, 'vacant': VACANT_COLOR}[kind]
                self._person(ax, x + 0.17, y - 0.02, 0.26, color, None, kind == 'applying')
            ax.text(x + 0.42, y + 0.1, label, ha='left', va='center', fontsize=8.5,
                    color=DETAIL_COLOR)

    def _add_staff(self):

        # each kind of staff is a person, drawn hired, applying and vacant and shown as it
        # stands, next to its name and what it does and costs
        terms = {}
        if self._has_production:
            salary = self._nonfluent('SALARY_PRODUCTION_MANAGER')
            terms['production'] = ('Production managers', f'One-step production, pay {salary:g} each')
        if self._has_marketing:
            bonus = self._nonfluent('MARKETING_MANAGER_BONUS')
            terms['marketing'] = ('Marketing manager',
                                  f'Sales +{bonus:.0%}, pay {self._nonfluent("SALARY_MARKETING_MANAGER"):g}')
        if self._has_lobbyist:
            bonus = self._nonfluent('LOBBYIST_BONUS')
            terms['lobbyist'] = ('Lobbyist',
                                 f'Purchases −{bonus:.0%}, pay {self._nonfluent("SALARY_LOBBYIST"):g}')
        self._staff = {}
        if not terms:
            return
        (width, height) = self._figure_size
        ax = self._inch_axes(height - TOP_MARGIN - STAFF_HEIGHT, STAFF_HEIGHT)
        for (i, (kind, (name, term))) in enumerate(terms.items()):
            x = width / 2 + (i - (len(terms) - 1) / 2) * STAFF_PITCH - 0.75
            self._staff[kind] = {
                state: self._person(ax, x, 0.06, 0.38, color, kind, state == 'applying')
                for (state, color) in (('hired', STAFF_COLOR), ('applying', APPLYING_COLOR),
                                       ('vacant', VACANT_COLOR))}
            ax.text(x + 0.25, 0.25, f'{name}\n{term}', ha='left', va='center', fontsize=7.5,
                    linespacing=1.4, color=DETAIL_COLOR)

    def _layout(self, levels):
        self._rows, self._row = [], {}
        for level in levels:
            for start in range(0, len(level), MAX_ROW_GOODS):
                part = level[start:start + MAX_ROW_GOODS]
                self._row.update((good, len(self._rows)) for good in part)
                self._rows.append(part)

        # an edge that crosses rows passes through a slot in every row in between, so that
        # it goes around the cards there rather than behind them
        self._passing = {}
        neighbors = {good: [] for good in self._goods}
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

        # edges meeting a card on the same side connect in the order they arrive from,
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
        for ((good, _), entries) in sides.items():
            spacing = min(PORT_SPACING, PORT_SPAN / len(entries))
            for (i, (_, _, edge, which)) in enumerate(sorted(entries)):
                self._ports[(edge, which)] = self._x[good] + spacing * (i - (len(entries) - 1) / 2)

    def _levels(self):

        # each good goes below all it is made from, and a cycle has no such place so the one
        # good missing the fewest of its inputs goes first to break it
        levels, placed = [], set()
        remaining = list(self._goods)
        while remaining:
            missing = {good: sum(source not in placed and source != good
                                 for source in self._inputs[good]) for good in remaining}
            level = [good for good in remaining if missing[good] == 0]
            if not level:
                level = [min(remaining, key=missing.get)]
            remaining = [good for good in remaining if good not in level]
            placed.update(level)
            levels.append(level)
        return levels

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

        # edges within a row arc around the cards, over them to the right and under them to
        # the left so that two opposite edges stay apart
        if start_y == end_y:
            sign = 1 if end_x >= start_x else -1
            edge_y = start_y + sign * NODE_HEIGHT / 2
            offset = sign * NODE_WIDTH / 4
            return Path([(start_x + offset, edge_y),
                         ((start_x + end_x) / 2, edge_y + sign * 2 * ARC_HEIGHT),
                         (end_x - offset, edge_y)],
                        [Path.MOVETO, Path.CURVE3, Path.CURVE3])

        # edges across rows go straight past the cards in a row, and only curve in the gaps
        # between rows where nothing can be in the way
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

    def _market_price(self, good, state):
        level = str(state[RDDLPlanningModel.ground_var('current-price-level', [good])])
        return self._nonfluent('PRICE', good) * self._nonfluent('PRICE_LEVEL_FACTOR', level)

    def _card_colors(self, good, has_factory, is_building, in_stock):
        if has_factory:
            if in_stock == len(self._inputs[good]):
                return READY_COLOR, READY_EDGE_COLOR
            return FACTORY_COLOR, FACTORY_EDGE_COLOR
        if is_building:
            return BUILDING_COLOR, BUILDING_EDGE_COLOR
        if self._inputs[good]:
            return NONE_COLOR, NONE_EDGE_COLOR
        return RAW_COLOR, RAW_EDGE_COLOR

    def _render_staff(self, state):
        fluent = RDDLPlanningModel.ground_var
        hired = {
            'production': any(bool(state[fluent('have-production-manager', [good])])
                              for good in self._goods),
            'marketing': bool(state[fluent('have-marketing-manager', [])]),
            'lobbyist': bool(state[fluent('have-lobbyist', [])])}
        applies = {'production': 'production-manager-applies',
                   'marketing': 'marketing-manager-applies', 'lobbyist': 'lobbyist-applies'}

        # production managers are hired for each good and can apply while others work, but
        # the others can only apply while not hired
        for (kind, people) in self._staff.items():
            applying = bool(state[fluent(applies[kind], [])])
            if applying and (kind == 'production' or not hired[kind]):
                shown = 'applying'
            else:
                shown = 'hired' if hired[kind] else 'vacant'
            for (other, artists) in people.items():
                for artist in artists:
                    artist.set_visible(other == shown)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        bonus = (self._nonfluent('MARKETING_MANAGER_BONUS')
                 if bool(state[fluent('have-marketing-manager', [])]) else 0)

        factories, for_sale, worth = 0, 0, 0.0
        for good in self._goods:
            price = self._market_price(good, state)
            level = str(state[fluent('current-price-level', [good])])
            (mark, mark_color) = TREND_MARKS[str(state[fluent('price-trend', [good])])]
            has_factory = bool(state[fluent('have-factory', [good])])
            stage = str(state[fluent('construction-progress-factory', [good])])
            has_manager = bool(state[fluent('have-production-manager', [good])])
            in_stock = sum(bool(state[fluent('good-in-stock', [source, good])])
                           for source in self._inputs[good])
            selling = bool(state[fluent('good-in-stock-for-sale', [good])])

            factories += has_factory
            if selling:
                for_sale += 1
                worth += price * (1 + bonus)

            for mark_artist in self._sale_marks[good]:
                mark_artist.set_visible(selling)

            if self._has_arrows:
                self._prices[good].set_text(f'{price:.0f}')
                self._trends[good].set_text(mark)
                self._trends[good].set_color(mark_color)
                reached = self._price_levels.index(level)
                for (i, bar) in enumerate(self._bars[good]):
                    bar.set_facecolor(BAR_ON_COLOR if i <= reached else BAR_OFF_COLOR)
                for artist in self._managers[good]:
                    artist.set_visible(has_manager)

                # walls rise with each stage of building, short of a whole factory
                if has_factory:
                    kind = 'working' if in_stock == len(self._inputs[good]) else 'factory'
                elif stage != 'not-building':
                    kind = 'building'
                else:
                    kind = 'lot' if self._inputs[good] else 'raw'
                progress = ((self._stages.index(stage) + 1) / (len(self._stages) + 1)
                            if stage in self._stages else 0.0)
                self._show_site(self._sites[good], kind, progress)
            else:
                (fill, edge) = self._card_colors(good, has_factory, stage != 'not-building', in_stock)
                self._boxes[good].set_facecolor(fill)
                self._boxes[good].set_edgecolor(edge)
                if good in self._details:
                    self._details[good].set_text(f'{price:.0f} {mark} {in_stock}/{len(self._inputs[good])}')

        # inputs in stock stay on top of the others so that they can be followed
        for ((source, good), arrow) in self._arrows.items():
            stocked = bool(state[fluent('good-in-stock', [source, good])])
            arrow.set_color(STOCK_COLOR if stocked else NEEDED_COLOR)
            arrow.set_linewidth(1.6 if stocked else 0.9)
            arrow.set_zorder(1.5 if stocked else 1)

        self._render_staff(state)

        limit = self._nonfluent('MAX_NUM_FACTORIES')
        title = f'{factories} / {limit} factor{"y" if limit == 1 else "ies"}, '
        if for_sale:
            title += f'{for_sale} good{"" if for_sale == 1 else "s"} for sale worth {worth:.0f}'
        else:
            title += 'nothing for sale'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color='black')

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
