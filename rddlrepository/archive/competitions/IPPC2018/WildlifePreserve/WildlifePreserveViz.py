import math
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

SAVANNA_COLOR = '#efe3c2'
SAVANNA_EDGE_COLOR = '#d7c9a3'
GRASS_COLOR = '#9ccc65'
THREAT_COLOR = '#e53935'
ELEPHANT_COLOR = '#90a4ae'
ELEPHANT_DARK_COLOR = '#78909c'
SKIN_COLOR = '#d7b49e'
RANGER_SHIRT_COLOR = '#6b8e23'
RANGER_TROUSER_COLOR = '#a1887f'
HAT_COLOR = '#8d6e63'
BINOCULAR_COLOR = '#263238'
POACHER_COLOR = '#37474f'
POACHER_TROUSER_COLOR = '#212121'
MASK_COLOR = '#263238'
RIFLE_COLOR = '#4e342e'
BARS_COLOR = '#607d8b'
TRACK_COLOR = '#8d6e63'
TRUNK_COLOR = '#6d4c41'
CANOPY_COLOR = '#7cb342'
WINNING_COLOR = '#2e7d32'
DETAIL_COLOR = '#37474f'

# the tint of an area for the chance that a free poacher attacks it
MIN_THREAT_ALPHA, THREAT_ALPHA_RANGE = 0.08, 0.55

# the tracks of rangers in earlier steps fade, from the step before the last on
TRACK_ALPHAS = (0.8, 0.55, 0.35, 0.2)

# sizes in inches
TOP_MARGIN = 0.45
MAP_SIZE = (8.4, 3.9)
MAX_TILE = 1.5
TILE_GAP = 0.06
POACHER_ROW = 0.6
LEGEND_HEIGHT = 0.42
CAPTION_HEIGHT = 0.1
MIN_WIDTH = 8.6
LEGEND_GAP = 0.3


class WildlifePreserveVisualizer(BaseViz):
    '''Draws the preserve as a map of savanna areas with their elephants, where more
    elephants mean more is at stake: the reward when the area is safe less the penalty when
    it is attacked. A ranger stands in each area defended in the last step, and fading
    tracks show the areas defended in the steps before, which poachers remember. Each area is
    tinted red by the chance that a free poacher attacks it next. The poachers stand below
    the map, behind bars when caught.

    The attacks are not part of the state, so they only show as chances. The drawing only
    depends on the current state. All versions of the domain share this class, and the
    number of areas, remembered steps and attack weights all come from the model.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._areas = list(model.type_to_objects['area'])
        self._poachers = list(model.type_to_objects['poacher'])
        self._steps = list(model.type_to_objects['number'])
        self._rangers = list(model.type_to_objects['ranger'])

        # the fluent ATTACK-WEIGHT_k applies to an area remembered as defended k times
        self._weight_names = {}
        for name in model.non_fluents:
            match = re.fullmatch(r'ATTACK-WEIGHT_(\d+)', name)
            if match:
                self._weight_names[int(match.group(1))] = name
        self._memory = {poacher: [step for step in self._steps
                                  if self._nonfluent('POACHER-REMEMBERS', poacher, step)]
                        for poacher in self._poachers}

        # the map has the shape with the largest areas, and the fewest rows and gaps among those
        count = max(len(self._areas), 1)
        (self._tile, _, _, self._cols) = max(
            (min(MAX_TILE, MAP_SIZE[0] / cols, MAP_SIZE[1] / math.ceil(count / cols)),
             -math.ceil(count / cols), -(math.ceil(count / cols) * cols - count), cols)
            for cols in range(1, count + 1))
        self._rows = math.ceil(count / self._cols)
        map_width, map_height = self._cols * self._tile, self._rows * self._tile

        # title above, then the map, the poachers, the legend and the caption
        bottom = CAPTION_HEIGHT + LEGEND_HEIGHT + POACHER_ROW
        width = max(map_width + 0.4, MIN_WIDTH)
        height = bottom + map_height + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self._inch_axes((width - map_width) / 2, bottom, map_width, map_height)

        # the areas, their elephants and the rangers never move, so they are drawn once and
        # then shown or tinted
        self._threats, self._rangers_at, self._tracks = {}, {}, {}
        values = {area: self._nonfluent('DEFENDER-REWARD', area) - self._nonfluent('DEFENDER-PENALTY', area)
                  for area in self._areas}
        (low, high) = (min(values.values()), max(values.values()))
        for (i, area) in enumerate(self._areas):
            row, col = divmod(i, self._cols)
            herd = 2 if high <= low else 1 + round(2 * (values[area] - low) / (high - low))
            self._draw_area(area, col * self._tile, (self._rows - 1 - row) * self._tile, herd)

        self._poacher_ax = self._inch_axes(0, CAPTION_HEIGHT + LEGEND_HEIGHT, width, POACHER_ROW)
        self._cages = {}
        size = min(0.48, (width - 0.6) / max(len(self._poachers), 1) / 0.8)
        for (i, poacher) in enumerate(self._poachers):
            x = width / 2 + (i - (len(self._poachers) - 1) / 2) * 0.8 * size
            self._person(self._poacher_ax, x, 0.06, size, 'poacher')
            self._cages[poacher] = self._bars(self._poacher_ax, x, 0.03, size)

        self._draw_legend()

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _inch_axes(self, left, bottom, width, height):

        # axes in inches from their bottom left, for pictures
        (fig_width, fig_height) = self._figure_size
        ax = self.fig.add_axes([left / fig_width, bottom / fig_height,
                                width / fig_width, height / fig_height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.axis('off')
        return ax

    # ---------- pictures: each takes the axes, a place and a size, and returns its artists

    @staticmethod
    def _elephant(ax, x, y, size):

        # an elephant facing right, standing on the given point
        s = size
        artists = [ax.add_patch(patches.Rectangle((x + dx * s, y), 0.13 * s, 0.32 * s,
                                                  facecolor=ELEPHANT_DARK_COLOR, edgecolor='none',
                                                  zorder=3))
                   for dx in (-0.4, -0.2, 0.08, 0.26)]
        artists += [
            ax.add_patch(patches.Ellipse((x, y + 0.5 * s), 1.0 * s, 0.6 * s, facecolor=ELEPHANT_COLOR,
                                         edgecolor='none', zorder=3.1)),
            ax.add_patch(patches.Polygon([(x + 0.55 * s, y + 0.6 * s), (x + 0.75 * s, y + 0.55 * s),
                                          (x + 0.78 * s, y + 0.12 * s), (x + 0.68 * s, y + 0.12 * s)],
                                         closed=True, facecolor=ELEPHANT_COLOR, edgecolor='none',
                                         zorder=3.2)),
            ax.add_patch(patches.Circle((x + 0.5 * s, y + 0.64 * s), 0.24 * s,
                                        facecolor=ELEPHANT_COLOR, edgecolor='none', zorder=3.2)),
            ax.add_patch(patches.Ellipse((x + 0.38 * s, y + 0.64 * s), 0.24 * s, 0.38 * s,
                                         facecolor=ELEPHANT_DARK_COLOR, edgecolor='none', zorder=3.3)),
            ax.add_patch(patches.Circle((x + 0.6 * s, y + 0.7 * s), 0.035 * s, facecolor='#263238',
                                        edgecolor='none', zorder=3.4))]
        return artists

    @staticmethod
    def _person(ax, x, y, size, kind):

        # a person standing on the given point: a ranger in uniform with a bush hat and
        # binoculars, or a poacher in dark clothes and a balaclava with a rifle on the back
        h = size
        ranger = kind == 'ranger'
        (shirt, trousers) = ((RANGER_SHIRT_COLOR, RANGER_TROUSER_COLOR) if ranger
                             else (POACHER_COLOR, POACHER_TROUSER_COLOR))

        def box(x0, y0, w, hh, color, zorder=5, radius=0.0):
            return ax.add_patch(patches.FancyBboxPatch(
                (x + x0 * h, y + y0 * h), w * h, hh * h,
                boxstyle=f'round,pad=0,rounding_size={radius * h}', facecolor=color,
                edgecolor='none', zorder=zorder))

        artists = []
        if not ranger:
            # a rifle slung across the back, its stock below one side and barrel above the other
            (bx, by, tx, ty, d) = (0.36, 0.22, -0.3, 1.08, 0.022)
            artists += [
                ax.add_patch(patches.Polygon(
                    [(x + (bx - d) * h, y + by * h), (x + (bx + d) * h, y + by * h),
                     (x + (tx + d) * h, y + ty * h), (x + (tx - d) * h, y + ty * h)],
                    closed=True, facecolor=RIFLE_COLOR, edgecolor='none', zorder=4.9)),
                ax.add_patch(patches.Polygon(
                    [(x + (bx - 0.07) * h, y + (by - 0.02) * h), (x + (bx + 0.05) * h, y + (by - 0.06) * h),
                     (x + (bx - 0.02) * h, y + (by + 0.14) * h)],
                    closed=True, facecolor=RIFLE_COLOR, edgecolor='none', zorder=4.9))]
        artists += [
            box(-0.12, 0, 0.1, 0.42, trousers), box(0.02, 0, 0.1, 0.42, trousers),
            box(-0.18, 0.38, 0.36, 0.34, shirt, 5.1, 0.06),
            box(-0.26, 0.4, 0.08, 0.3, shirt, 5.05, 0.04), box(0.18, 0.4, 0.08, 0.3, shirt, 5.05, 0.04),
            ax.add_patch(patches.Circle((x, y + 0.83 * h), 0.11 * h,
                                        facecolor=SKIN_COLOR if ranger else MASK_COLOR,
                                        edgecolor='none', zorder=5.2))]
        if ranger:
            artists += WildlifePreserveVisualizer._hat(ax, x, y + 0.9 * h, h)
            artists += [ax.add_patch(patches.Circle((x + dx * h, y + 0.6 * h), 0.045 * h,
                                                    facecolor=BINOCULAR_COLOR, edgecolor='none',
                                                    zorder=5.3)) for dx in (-0.05, 0.05)]
        else:
            artists.append(box(-0.08, 0.81, 0.16, 0.045, SKIN_COLOR, 5.3))
        return artists

    @staticmethod
    def _hat(ax, x, y, size):

        # a bush hat: a wide brim under a round crown
        return [ax.add_patch(patches.Ellipse((x, y), 0.42 * size, 0.06 * size, facecolor=HAT_COLOR,
                                             edgecolor='none', zorder=5.3)),
                ax.add_patch(patches.Wedge((x, y), 0.12 * size, 0, 180, facecolor=HAT_COLOR,
                                           edgecolor='none', zorder=5.3))]

    @staticmethod
    def _track(ax, x, y, size, alpha):

        # a pair of boot prints, one a little ahead of the other
        return [ax.add_patch(patches.Ellipse((x + dx * size, y + dy * size), 0.32 * size, 0.6 * size,
                                             angle=-15, facecolor=TRACK_COLOR, edgecolor='none',
                                             alpha=alpha, zorder=1.4))
                for (dx, dy) in ((-0.25, 0), (0.25, 0.45))]

    @staticmethod
    def _acacia(ax, x, y, size):

        # a flat-topped acacia tree standing on the given point
        return [ax.add_patch(patches.Polygon(
                    [(x - 0.04 * size, y), (x + 0.04 * size, y), (x + 0.03 * size, y + 0.6 * size),
                     (x + 0.25 * size, y + 0.8 * size), (x + 0.2 * size, y + 0.82 * size),
                     (x, y + 0.66 * size), (x - 0.2 * size, y + 0.82 * size),
                     (x - 0.25 * size, y + 0.8 * size), (x - 0.03 * size, y + 0.6 * size)],
                    closed=True, facecolor=TRUNK_COLOR, edgecolor='none', zorder=1.3)),
                ax.add_patch(patches.Ellipse((x, y + 0.88 * size), 0.9 * size, 0.22 * size,
                                             facecolor=CANOPY_COLOR, edgecolor='none', zorder=1.35))]

    @staticmethod
    def _bars(ax, x, y, size):

        # the bars of a cage around a person, hidden
        artists = [ax.add_patch(patches.Rectangle((x + dx * size, y), 0.035 * size, 1.05 * size,
                                                  facecolor=BARS_COLOR, edgecolor='none', zorder=6))
                   for dx in (-0.33, -0.12, 0.09, 0.3)]
        artists.append(ax.add_patch(patches.Rectangle((x - 0.36 * size, y + 1.02 * size), 0.72 * size,
                                                      0.05 * size, facecolor=BARS_COLOR,
                                                      edgecolor='none', zorder=6)))
        artists.append(ax.add_patch(patches.Rectangle((x - 0.36 * size, y), 0.72 * size, 0.05 * size,
                                                      facecolor=BARS_COLOR, edgecolor='none', zorder=6)))
        for artist in artists:
            artist.set_visible(False)
        return artists

    @staticmethod
    def _savanna(ax, left, bottom, size):

        # a tile of dry grassland with a few tufts of grass
        artists = [ax.add_patch(patches.FancyBboxPatch(
            (left, bottom), size, size, boxstyle=f'round,pad=0,rounding_size={0.06 * size}',
            facecolor=SAVANNA_COLOR, edgecolor=SAVANNA_EDGE_COLOR, linewidth=1, zorder=1))]
        for (tx, ty) in ((0.15, 0.62), (0.55, 0.85), (0.4, 0.08), (0.88, 0.45)):
            for angle in (-0.35, 0, 0.35):
                artists.append(ax.add_patch(patches.Polygon(
                    [(left + (tx - 0.012) * size, bottom + ty * size),
                     (left + (tx + 0.012) * size, bottom + ty * size),
                     (left + (tx + angle * 0.08) * size, bottom + (ty + 0.07) * size)],
                    closed=True, facecolor=GRASS_COLOR, edgecolor='none', zorder=1.2)))
        return artists

    # ---------- the map, the legend and the state

    def _draw_area(self, area, left, bottom, herd):
        tile = self._tile - TILE_GAP
        (left, bottom) = (left + TILE_GAP / 2, bottom + TILE_GAP / 2)
        self._savanna(self.ax, left, bottom, tile)
        self._threats[area] = self.ax.add_patch(patches.FancyBboxPatch(
            (left, bottom), tile, tile, boxstyle=f'round,pad=0,rounding_size={0.06 * tile}',
            facecolor=THREAT_COLOR, edgecolor='none', alpha=0, zorder=1.5))

        # the herd stays on the left, clear of the rangers on the right, under a tree
        self._acacia(self.ax, left + 0.5 * tile, bottom + 0.56 * tile, 0.32 * tile)
        size = 0.2 * tile
        spots = {1: [(0.3, 0.25)], 2: [(0.2, 0.42), (0.42, 0.18)],
                 3: [(0.18, 0.46), (0.45, 0.4), (0.28, 0.14)]}[herd]
        for (sx, sy) in spots:
            self._elephant(self.ax, left + sx * tile, bottom + sy * tile, size)

        self._rangers_at[area] = self._person(self.ax, left + 0.8 * tile, bottom + 0.06 * tile,
                                              0.5 * tile, 'ranger')
        self._tracks[area] = [self._track(self.ax, left + (0.9 - 0.14 * k) * tile,
                                          bottom + (0.78 - 0.06 * k) * tile, 0.09 * tile, alpha)
                              for (k, alpha) in zip(range(len(self._steps) - 1), TRACK_ALPHAS)]
        for artist in self._rangers_at[area] + [a for track in self._tracks[area] for a in track]:
            artist.set_visible(False)

        size = min(8.5, 0.16 * tile * 72)
        if size >= 5:
            self.ax.text(left + 0.06 * tile, bottom + 0.92 * tile, area.lstrip('@'), ha='left',
                         va='center', fontsize=size, color=DETAIL_COLOR, zorder=6)

    def _draw_legend(self):
        (width, _) = self._figure_size
        entries = [('ranger', 'Ranger'), ('tracks', 'Earlier patrols'),
                   ('threat', 'Risk of attack'), ('elephant', 'More at stake'),
                   ('poacher', 'Poacher'), ('caught', 'Caught')]
        if len(self._steps) < 2:
            entries.pop(1)
        widths = [0.45 + 0.066 * len(label) + LEGEND_GAP for (_, label) in entries]
        shrink = min(1.0, (width - 0.2) / sum(widths))
        ax = self._inch_axes(0, CAPTION_HEIGHT, width, LEGEND_HEIGHT)
        x = (width - shrink * sum(widths)) / 2
        for ((kind, label), entry_width) in zip(entries, widths):
            if kind in ('ranger', 'poacher', 'caught'):
                self._person(ax, x + 0.18, 0.03, 0.36, 'ranger' if kind == 'ranger' else 'poacher')
                if kind == 'caught':
                    for artist in self._bars(ax, x + 0.18, 0.01, 0.36):
                        artist.set_visible(True)
            elif kind == 'tracks':
                for (k, alpha) in enumerate(TRACK_ALPHAS[:3]):
                    self._track(ax, x + 0.32 - 0.12 * k, 0.12 + 0.04 * k, 0.09, alpha)
            elif kind == 'threat':
                for (k, chance) in enumerate((0.15, 0.5, 1.0)):
                    ax.add_patch(patches.Rectangle((x + 0.02 + 0.11 * k, 0.08), 0.11, 0.22,
                                                   facecolor=SAVANNA_COLOR, edgecolor='none'))
                    ax.add_patch(patches.Rectangle(
                        (x + 0.02 + 0.11 * k, 0.08), 0.11, 0.22, facecolor=THREAT_COLOR,
                        edgecolor='none', alpha=MIN_THREAT_ALPHA + THREAT_ALPHA_RANGE * chance))
            elif kind == 'elephant':
                self._elephant(ax, x + 0.13, 0.08, 0.26)
            ax.text(x + 0.42, 0.19, label, ha='left', va='center', fontsize=8.5, color=DETAIL_COLOR)
            x += shrink * entry_width

    def _weight(self, count, poacher, area):
        name = self._weight_names.get(count)
        return 0.0 if name is None else self._nonfluents[RDDLPlanningModel.ground_var(name, [poacher, area])]

    def _attack_chances(self, state, poacher):
        '''The Discrete distribution of poacher-attacks in the domain.'''
        fluent = RDDLPlanningModel.ground_var
        weights = np.zeros(len(self._areas))
        for (i, area) in enumerate(self._areas):
            count = sum(bool(state[fluent('was-defended', [area, step])])
                        for step in self._memory[poacher])
            weights[i] = self._weight(count, poacher, area)
        total = np.sum(weights)
        return weights / total if total > 0 else weights

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var

        # the chance that some free poacher attacks each area
        safe = np.ones(len(self._areas))
        caught_count = 0
        for poacher in self._poachers:
            caught = bool(state[fluent('poacher-caught', [poacher])])
            for artist in self._cages[poacher]:
                artist.set_visible(caught)
            if caught:
                caught_count += 1
            else:
                safe *= 1 - self._attack_chances(state, poacher)

        for (area, threat) in zip(self._areas, 1 - safe):
            self._threats[area].set_alpha(MIN_THREAT_ALPHA + THREAT_ALPHA_RANGE * threat
                                          if threat > 0 else 0)
            defended = [bool(state[fluent('was-defended', [area, step])]) for step in self._steps]
            for artist in self._rangers_at[area]:
                artist.set_visible(defended[0])
            for (track, was) in zip(self._tracks[area], defended[1:]):
                for artist in track:
                    artist.set_visible(was)

        total = len(self._poachers)
        if caught_count == total:
            title = f'All {total} poachers caught' if total > 1 else 'Poacher caught'
            color = WINNING_COLOR
        else:
            title = f'{caught_count} / {total} poacher{"" if total == 1 else "s"} caught'
            color = 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
