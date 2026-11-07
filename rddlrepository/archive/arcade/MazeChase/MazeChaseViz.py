import math

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Patch, Polygon, Rectangle, Wedge
from matplotlib.transforms import Affine2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

# the gauges and text around the maze, on white as in the other domains
TRACK_COLOR = '#eceff1'
TEXT_COLOR = '#37474f'
GOOD_COLOR = '#2e7d32'
BAD_COLOR = '#c62828'

# the garden
HEDGE_COLOR = '#2f6b3a'
HEDGE_DARK_COLOR = '#25572f'
HEDGE_LIGHT_COLOR = '#3e7f48'
HEDGE_EDGE_COLOR = '#1c4424'
PATH_COLOR = '#e6d8b5'
TUNNEL_COLOR = '#6d4c41'
DEN_COLOR = '#3e2723'
BERRY_COLOR = '#c62828'
BERRY_EDGE_COLOR = '#7f1d1d'
BERRY_SHINE_COLOR = '#ef9a9a'
STICK_COLOR = '#6d4c41'
FLAME_COLOR = '#ff8f00'
FLAME_CORE_COLOR = '#ffe082'

# the forager
CLOAK_COLOR = '#1565c0'
SKIN_COLOR = '#f2c9a1'
HAT_COLOR = '#a1742f'
BASKET_COLOR = '#8d6e63'
OUTLINE_COLOR = '#263238'

# the wolves
WOLF_COLOR = '#546e7a'
WOLF_AFRAID_COLOR = '#b3c5ce'
SNOUT_COLOR = '#cfd8dc'
WOLF_EYE_COLOR = '#ffca28'
WOLF_AFRAID_EYE_COLOR = '#ffffff'
FEAR_COLOR = '#ff8f00'

BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
SIDE_MARGIN = 0.15

# the maze has the same scale on both axes
MIN_SCENE_HEIGHT = 3.2
MIN_SCENE_WIDTH = 3.2
PANEL_WIDTH = 2.2
PANEL_GAP = 0.2

# the hedge is only given leaves up to this number of tiles
MAX_LEAF_TILES = 4000

HEADINGS = {'north': (0, 1), 'east': (1, 0), 'south': (0, -1), 'west': (-1, 0)}
EYE_SHIFT = 0.03

# the shape of a wolf, whose ears lie flat when it is frightened
WOLF_HEAD = [(-0.30, 0.04), (-0.30, 0.42), (-0.11, 0.20), (0.11, 0.20), (0.30, 0.42), (0.30, 0.04),
             (0.17, -0.18), (0.08, -0.38), (-0.08, -0.38), (-0.17, -0.18)]
WOLF_AFRAID_HEAD = [(-0.30, 0.04), (-0.42, 0.10), (-0.11, 0.20), (0.11, 0.20), (0.42, 0.10), (0.30, 0.04),
                    (0.17, -0.18), (0.08, -0.38), (-0.08, -0.38), (-0.17, -0.18)]


def _disc(x, y, radius, sides=12):
    angles = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    return np.column_stack((x + radius * np.cos(angles), y + radius * np.sin(angles)))


class MazeChaseVisualizer(BaseViz):
    '''Draws the maze from above as a garden, with the hedges and paths, the berries and torches
    that are left, the dens, the forager and the wolves, and gauges of what is left and of the fear.

    The maze is read from the instance. Cells are placed by X and Y, and everything else on the grid
    is hedge. Two cells that are side by side with no way between them have a hedge drawn between
    them, and a heading that leads to a cell further away is drawn as a dotted tunnel.

    A wolf between cells is drawn halfway between them, and the ring around a frightened wolf shows
    how much of its fear is left. The state does not hold the time, so the step is the number of
    different states seen, which requires render to be called on every step.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_size=(9.0, 5.5),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._horizon = model.horizon

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._fear_time = int(nonfluents['FEAR-TIME'])
        self._load_maze(model, nonfluents)

        self._init_state = self._get_state(model.ground_vars_with_values(model.state_fluents))
        self._num_berries = len(self._init_state[3])
        self._num_torches = len(self._init_state[4])
        self._last_state = None
        self._step = 0

        self._scene_width, self._scene_height = self._get_scene_size(max_size)
        self._draw_figure()

    def _load_maze(self, model, nonfluents):
        self._cells = list(model.type_to_objects['cell'])
        self._wolves = list(model.type_to_objects['wolf'])
        self._headings = list(model.type_to_objects['heading'])
        self._where = {c: (int(nonfluents[RDDLPlanningModel.ground_var('X', [c])]),
                           int(nonfluents[RDDLPlanningModel.ground_var('Y', [c])])) for c in self._cells}
        self._next = {}
        for c in self._cells:
            for h in self._headings:
                value = nonfluents[RDDLPlanningModel.ground_var('NEXT', [c, h])]
                self._next[(c, h.lstrip('@'))] = self._object_name(value, self._cells)
        self._dens = [self._where[self._object_name(nonfluents[RDDLPlanningModel.ground_var('DEN', [w])], self._cells)]
                      for w in self._wolves]

        xs, ys = zip(*self._where.values())
        # a border of hedge goes around the cells
        self._bounds = (min(xs) - 1.5, max(xs) + 1.5, min(ys) - 1.5, max(ys) + 1.5)

    @staticmethod
    def _object_name(value, objects):
        '''An object can come as its name, with or without the @, or as its index.'''
        value = np.ravel(value)[0]
        if isinstance(value, (str, np.str_)):
            return str(value).lstrip('@')
        return objects[int(value)].lstrip('@')

    def _get_state(self, state):
        '''(forager, one (at, going, between, fear) per wolf, finished, berries, torches)'''
        forager = self._object_name(state['forager'], self._cells)
        wolves = []
        for w in self._wolves:
            at = self._object_name(state[RDDLPlanningModel.ground_var('at', [w])], self._cells)
            going = self._object_name(state[RDDLPlanningModel.ground_var('going', [w])], self._headings)
            between = bool(np.ravel(state[RDDLPlanningModel.ground_var('between', [w])])[0])
            fear = int(np.ravel(state[RDDLPlanningModel.ground_var('fear', [w])])[0])
            wolves.append((at, going, between, fear))
        berries = tuple(c for c in self._cells if bool(np.ravel(state[RDDLPlanningModel.ground_var('berry', [c])])[0]))
        torches = tuple(c for c in self._cells if bool(np.ravel(state[RDDLPlanningModel.ground_var('torch', [c])])[0]))
        return (forager, tuple(wolves), bool(np.ravel(state['finished'])[0]), berries, torches)

    # ==========================================================================
    # VIEW
    # ==========================================================================

    def _get_scene_size(self, max_size):
        left, right, bottom, top = self._bounds
        height = max(max_size[1] - BOTTOM_MARGIN - TOP_MARGIN, MIN_SCENE_HEIGHT)
        max_width = max(max_size[0] - PANEL_WIDTH - PANEL_GAP - 2 * SIDE_MARGIN, MIN_SCENE_WIDTH)
        width = height * (right - left) / (top - bottom)
        if width > max_width:
            width, height = max_width, max(max_width * (top - bottom) / (right - left), MIN_SCENE_HEIGHT)
        return max(width, MIN_SCENE_WIDTH), height

    def _get_view(self):

        # both axes use the same scale, with any spare room around the maze filled with hedge
        left, right, bottom, top = self._bounds
        scale = max((right - left) / self._scene_width, (top - bottom) / self._scene_height)
        middle_x, middle_y = (left + right) / 2, (bottom + top) / 2
        return (middle_x - scale * self._scene_width / 2, middle_x + scale * self._scene_width / 2,
                middle_y - scale * self._scene_height / 2, middle_y + scale * self._scene_height / 2)

    # ==========================================================================
    # DRAWING THE PARTS THAT DO NOT CHANGE
    # ==========================================================================

    def _draw_figure(self):
        fig_width = self._scene_width + PANEL_GAP + PANEL_WIDTH + 2 * SIDE_MARGIN
        fig_height = self._scene_height + BOTTOM_MARGIN + TOP_MARGIN
        self.fig = plt.figure(figsize=(fig_width, fig_height))
        self.ax = self.fig.add_axes([
            SIDE_MARGIN / fig_width, BOTTOM_MARGIN / fig_height,
            self._scene_width / fig_width, self._scene_height / fig_height])
        self.ax.set_autoscale_on(False)
        self.ax.set_facecolor(HEDGE_COLOR)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')
        view = self._get_view()
        self.ax.set_xlim(view[0], view[1])
        self.ax.set_ylim(view[2], view[3])

        self._draw_hedges(view)
        self._draw_paths()
        self._draw_dens()
        self._draw_items()
        self._draw_wolves()
        self._draw_forager()
        self._draw_panel(fig_width, fig_height)

        self.fig.legend(handles=[
            Line2D([], [], marker='o', linestyle='none', markerfacecolor=CLOAK_COLOR,
                   markeredgecolor=OUTLINE_COLOR, markersize=8, label='Forager'),
            Line2D([], [], marker='o', linestyle='none', markerfacecolor=BERRY_COLOR,
                   markeredgecolor=BERRY_EDGE_COLOR, markersize=6, label='Berry'),
            Line2D([], [], marker='^', linestyle='none', markerfacecolor=FLAME_COLOR,
                   markeredgecolor=STICK_COLOR, markersize=8, label='Torch'),
            Patch(facecolor=WOLF_COLOR, edgecolor=OUTLINE_COLOR, label='Wolf'),
            Patch(facecolor=WOLF_AFRAID_COLOR, edgecolor=FEAR_COLOR, label='Frightened'),
            Patch(facecolor=DEN_COLOR, label='Den')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / fig_height), ncol=6, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)
        self.fig.text(0.5, 0.1 / fig_height,
                      'A torch slows the wolves for a while, and touching one then drives it to its den',
                      ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)
        self._title = self.fig.text(0.5, 1 - 0.12 / fig_height, '', ha='center', va='top',
                                    fontsize=13, fontweight='bold')

    def _draw_hedges(self, view):

        # leaves on every tile in the view that is not a cell, the same on every render
        occupied = set(self._where.values())
        tiles = [(x, y) for x in range(math.floor(view[0]), math.ceil(view[1]) + 1)
                 for y in range(math.floor(view[2]), math.ceil(view[3]) + 1) if (x, y) not in occupied]
        if len(tiles) > MAX_LEAF_TILES:
            return
        rng = np.random.default_rng(0)
        for (color, count) in ((HEDGE_DARK_COLOR, 3), (HEDGE_LIGHT_COLOR, 3)):
            leaves = []
            for (x, y) in tiles:
                for (dx, dy) in rng.uniform(-0.42, 0.42, size=(count, 2)):
                    leaves.append(_disc(x + dx, y + dy, 0.17, sides=7))
            self.ax.add_collection(PolyCollection(leaves, facecolors=color, edgecolors='none', zorder=1))

    def _draw_paths(self):
        self.ax.add_collection(PolyCollection(
            [[(x - 0.5, y - 0.5), (x + 0.5, y - 0.5), (x + 0.5, y + 0.5), (x - 0.5, y + 0.5)]
             for (x, y) in self._where.values()],
            facecolors=PATH_COLOR, edgecolors=PATH_COLOR, linewidths=0.3, zorder=2))

        by_place = {xy: c for (c, xy) in self._where.items()}
        outline, inner, tunnels = [], [], []
        for (c, (x, y)) in self._where.items():
            for (h, (dx, dy)) in HEADINGS.items():
                side = [(x + 0.5 * (dx - dy), y + 0.5 * (dy - dx)), (x + 0.5 * (dx + dy), y + 0.5 * (dy + dx))]
                other = by_place.get((x + dx, y + dy))
                if other is None:
                    outline.append(side)

                # each side between two cells is looked at from both of them, so drawn once
                elif (dx, dy) in ((0, 1), (1, 0)) and other not in self._reachable(c) and c not in self._reachable(other):
                    inner.append(side)
                target = self._next[(c, h)]
                if target != c and target not in {by_place.get((x + ax, y + ay)) for (ax, ay) in HEADINGS.values()}:
                    tunnels.append([(x, y), self._where[target]])
        self.ax.add_collection(LineCollection(outline, colors=HEDGE_EDGE_COLOR, linewidths=1, zorder=3))
        self.ax.add_collection(LineCollection(inner, colors=HEDGE_COLOR, linewidths=3, zorder=3))
        self.ax.add_collection(LineCollection(tunnels, colors=TUNNEL_COLOR, linewidths=1.2, linestyles=':', zorder=3))
        mouths = list({xy for segment in tunnels for xy in segment})
        self.ax.add_collection(PolyCollection([_disc(x, y, 0.22) for (x, y) in mouths],
                                              facecolors=TUNNEL_COLOR, edgecolors='none', zorder=3))

    def _reachable(self, cell):
        return {self._next[(cell, h)] for h in HEADINGS} - {cell}

    def _draw_dens(self):
        arches = []
        for (x, y) in set(self._dens):
            angles = np.linspace(0, math.pi, 12)
            arches.append(np.column_stack((x + 0.38 * np.cos(angles), y - 0.3 + 0.62 * np.sin(angles))))
        self.ax.add_collection(PolyCollection(arches, facecolors=DEN_COLOR, edgecolors='none', zorder=3))

    def _draw_items(self):
        self._berries = PolyCollection([], facecolors=BERRY_COLOR, edgecolors=BERRY_EDGE_COLOR, linewidths=0.5, zorder=4)
        self._shines = PolyCollection([], facecolors=BERRY_SHINE_COLOR, edgecolors='none', zorder=5)
        self._sticks = PolyCollection([], facecolors=STICK_COLOR, edgecolors='none', zorder=4)
        self._flames = PolyCollection([], facecolors=FLAME_COLOR, edgecolors='none', zorder=5)
        self._cores = PolyCollection([], facecolors=FLAME_CORE_COLOR, edgecolors='none', zorder=6)
        for collection in (self._berries, self._shines, self._sticks, self._flames, self._cores):
            self.ax.add_collection(collection)
        self._shown_items = None

    def _add_figure(self, patches):
        '''Parts drawn around the origin, which move together. Returns what moves them.'''
        moved = Affine2D()
        for patch in patches:
            patch.set_transform(moved + self.ax.transData)
            self.ax.add_patch(patch)
        return moved

    def _draw_wolves(self):
        self._wolf_figures = []
        for _ in self._wolves:
            head = Polygon(WOLF_HEAD, closed=True, facecolor=WOLF_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.5, zorder=8)
            snout = Polygon([(-0.13, -0.12), (0.13, -0.12), (0.07, -0.36), (-0.07, -0.36)], closed=True,
                            facecolor=SNOUT_COLOR, edgecolor='none', zorder=9)
            nose = Circle((0, -0.31), 0.055, facecolor=OUTLINE_COLOR, edgecolor='none', zorder=10)
            eyes = [Circle((side * 0.13, 0.05), 0.075, facecolor=WOLF_EYE_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.4, zorder=9)
                    for side in (-1, 1)]
            pupils = [Circle((side * 0.13, 0.05), 0.032, facecolor=OUTLINE_COLOR, edgecolor='none', zorder=10)
                      for side in (-1, 1)]
            ring = Wedge((0, 0), 0.5, 90, 90, width=0.07, facecolor=FEAR_COLOR, edgecolor='none', zorder=7)
            moved = self._add_figure([ring, head, snout, nose] + eyes + pupils)
            self._wolf_figures.append({'moved': moved, 'head': head, 'eyes': eyes, 'pupils': pupils, 'ring': ring})

    def _draw_forager(self):
        cloak = Polygon([(-0.27, -0.42), (0.27, -0.42), (0.13, 0.1), (-0.13, 0.1)], closed=True,
                        facecolor=CLOAK_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.5, zorder=11)
        basket = Rectangle((0.14, -0.34), 0.22, 0.17, facecolor=BASKET_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.5, zorder=12)
        head = Circle((0, 0.14), 0.16, facecolor=SKIN_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.5, zorder=12)
        crown = Polygon([(-0.13, 0.22), (0.13, 0.22), (0.08, 0.42), (-0.08, 0.42)], closed=True,
                        facecolor=HAT_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.5, zorder=13)
        brim = Ellipse((0, 0.22), 0.54, 0.12, facecolor=HAT_COLOR, edgecolor=OUTLINE_COLOR, linewidth=0.5, zorder=13)
        self._forager_moved = self._add_figure([cloak, basket, head, crown, brim])

    def _draw_panel(self, fig_width, fig_height):
        panel = self.fig.add_axes([
            (SIDE_MARGIN + self._scene_width + PANEL_GAP) / fig_width, BOTTOM_MARGIN / fig_height,
            PANEL_WIDTH / fig_width, self._scene_height / fig_height])
        panel.set_xlim(0, PANEL_WIDTH)
        panel.set_ylim(0, self._scene_height)
        panel.axis('off')
        self._panel = panel

        pitch = min(self._scene_height / 5.0, 0.62)
        top = self._scene_height
        self._berry_gauge = self._draw_gauge('Berries left', top, BERRY_COLOR)
        self._torch_gauge = self._draw_gauge('Torches left', top - pitch, FLAME_COLOR)
        self._afraid_gauge = self._draw_gauge('Wolves frightened', top - 2 * pitch, WOLF_COLOR)
        self._fear_gauge = self._draw_gauge('Longest fear left', top - 3 * pitch, FEAR_COLOR)
        self._time = panel.text(0, top - 4 * pitch, '', ha='left', va='top', fontsize=9, color=TEXT_COLOR)

    def _draw_gauge(self, label, top, color):
        panel = self._panel
        panel.text(0, top, label, ha='left', va='top', fontsize=9, color=TEXT_COLOR)
        value = panel.text(PANEL_WIDTH, top, '', ha='right', va='top', fontsize=9, color=TEXT_COLOR)
        bar_y = top - 0.36
        panel.add_patch(Rectangle((0, bar_y), PANEL_WIDTH, 0.14, facecolor=TRACK_COLOR, edgecolor='none', zorder=1))
        fill = Rectangle((0, bar_y), 0, 0.14, facecolor=color, edgecolor='none', zorder=2)
        panel.add_patch(fill)
        return {'value': value, 'fill': fill}

    # ==========================================================================
    # UPDATING
    # ==========================================================================

    def _update_history(self, state):

        # rendering the same state again must not change the step
        if state == self._last_state:
            return
        last_state, self._last_state = self._last_state, state

        # an episode can end, and can be restarted at any time
        if last_state is None or state == self._init_state or last_state[2]:
            self._step = 0
        else:
            self._step += 1

    def _update_items(self, state):
        items = (state[3], state[4])
        if items == self._shown_items:
            return
        self._shown_items = items
        berries, torches = items
        places = [self._where[c] for c in berries]
        self._berries.set_verts([_disc(x, y, 0.17) for (x, y) in places])
        self._shines.set_verts([_disc(x - 0.05, y + 0.06, 0.05, sides=6) for (x, y) in places])
        places = [self._where[c] for c in torches]
        self._sticks.set_verts([[(x - 0.05, y - 0.42), (x + 0.05, y - 0.42), (x + 0.07, y - 0.02), (x - 0.07, y - 0.02)]
                                for (x, y) in places])
        self._flames.set_verts([[(x, y + 0.44), (x + 0.2, y + 0.12), (x + 0.12, y - 0.08), (x - 0.12, y - 0.08), (x - 0.2, y + 0.12)]
                                for (x, y) in places])
        self._cores.set_verts([[(x, y + 0.22), (x + 0.09, y + 0.06), (x, y - 0.05), (x - 0.09, y + 0.06)]
                               for (x, y) in places])

    def _update_wolves(self, state):
        for (figure, (at, going, between, fear)) in zip(self._wolf_figures, state[1]):
            x, y = self._where[at]
            if between:
                x2, y2 = self._where[self._next[(at, going)]]
                x, y = (x + x2) / 2, (y + y2) / 2
            figure['moved'].clear().translate(x, y)

            afraid = fear > 0
            figure['head'].set_xy(WOLF_AFRAID_HEAD if afraid else WOLF_HEAD)
            figure['head'].set_facecolor(WOLF_AFRAID_COLOR if afraid else WOLF_COLOR)
            shift_x, shift_y = HEADINGS.get(going, (0, 0))
            for (eye, pupil, side) in zip(figure['eyes'], figure['pupils'], (-1, 1)):
                eye.set_radius(0.09 if afraid else 0.075)
                eye.set_facecolor(WOLF_AFRAID_EYE_COLOR if afraid else WOLF_EYE_COLOR)
                pupil.set_center((side * 0.13 + EYE_SHIFT * shift_x, 0.05 + EYE_SHIFT * shift_y))

            figure['ring'].set_visible(afraid)
            if afraid:
                # the fear can be longer than FEAR-TIME if an instance starts with it so
                fraction = min(fear / max(self._fear_time, 1), 1.0)
                figure['ring'].set_theta2(90 + 360 * fraction)

    def _set_gauge(self, gauge, fraction, text):
        gauge['fill'].set_width(max(min(fraction, 1.0), 0.0) * PANEL_WIDTH)
        gauge['value'].set_text(text)

    def _update_panel(self, state):
        berries, torches = len(state[3]), len(state[4])
        fears = [wolf[3] for wolf in state[1]]
        afraid, longest = sum(fear > 0 for fear in fears), max(fears)
        self._set_gauge(self._berry_gauge, berries / max(self._num_berries, 1), f'{berries} of {self._num_berries}')
        self._set_gauge(self._torch_gauge, torches / max(self._num_torches, 1), f'{torches} of {self._num_torches}')
        self._set_gauge(self._afraid_gauge, afraid / len(fears), f'{afraid} of {len(fears)}')
        self._set_gauge(self._fear_gauge, longest / max(self._fear_time, 1), f'{longest} of {self._fear_time}')
        self._time.set_text(f'Step {self._step} of {self._horizon}')

    def _update_title(self, state):
        berries = len(state[3])
        left = f'{berries} berry left' if berries == 1 else f'{berries} berries left'
        longest = max(wolf[3] for wolf in state[1])
        if state[2] and berries == 0 and self._num_berries > 0:
            text, color = f'All berries gathered at step {self._step}', GOOD_COLOR
        elif state[2]:
            text, color = f'Bitten at step {self._step} with {left}', BAD_COLOR
        elif longest > 0:
            moves = '1 more move' if longest == 1 else f'{longest} more moves'
            text, color = f'{left.capitalize()}, wolves frightened for up to {moves}', 'black'
        else:
            text, color = left.capitalize(), 'black'
        self._title.set_text(text)
        self._title.set_color(color)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        state = self._get_state(state)
        self._update_history(state)
        self._update_items(state)
        self._update_wolves(state)
        self._forager_moved.clear().translate(*self._where[state[0]])
        self._update_panel(state)
        self._update_title(state)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
