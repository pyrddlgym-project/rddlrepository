import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

NIGHT_COLOR = '#2f3d34'
GRASS_COLOR = '#3c4d42'
LIGHT_COLOR = '#ffe082'
PAVING_COLOR = '#8a8274'
PAVING_LINE_COLOR = '#a39a8a'
FENCE_COLOR = '#d7ccc8'
SAFE_COLOR = '#546e7a'
SAFE_DARK_COLOR = '#37474f'
LANTERN_DARK_COLOR = '#4e342e'
LANTERN_GLASS_COLOR = '#fff59d'
FLAME_COLOR = '#ff9800'
SKIN_COLOR = '#d7b49e'
DARK_COLOR = '#212121'
SACK_COLOR = '#e6cfa3'
TRAIL_COLOR = '#cfd8dc'
TEXT_COLOR = '#37474f'
STEALING_COLOR = '#c62828'
SAFE_TITLE_COLOR = '#2e7d32'

# sizes on screen in inches
LANTERN_SIZE = 0.3
BURGLAR_SIZE = 0.42
SAFE_SIZE = 0.3
PAVING_STEP = 0.25
PLOT_SIZE = 5.0
SIDE_MARGIN = 0.3
BOTTOM_MARGIN = 0.75
TOP_MARGIN = 0.45


# ---------- pictures: each takes the axes, a place and a size, and returns its artists

def _glow(ax, x, y, radius):

    # lantern light, brighter towards the middle, to the edge of what it reaches
    artists = [ax.add_patch(plt.Circle((x, y), radius * k, facecolor=LIGHT_COLOR, alpha=alpha,
                                       edgecolor='none', zorder=1.5))
               for (k, alpha) in ((1.0, 0.22), (0.7, 0.2), (0.4, 0.22))]
    artists.append(ax.add_patch(plt.Circle((x, y), radius, facecolor='none', edgecolor=LIGHT_COLOR,
                                           alpha=0.7, linewidth=1, zorder=1.6)))
    return artists


def _lantern(ax, x, y, s):

    # an oil lantern centered on the given point: a ring, a cap, glass with a flame and a base
    return [
        ax.add_patch(plt.Circle((x, y + 0.46 * s), 0.08 * s, facecolor='none',
                                edgecolor=LANTERN_DARK_COLOR, linewidth=1, zorder=5)),
        ax.add_patch(patches.Polygon([(x - 0.24 * s, y + 0.25 * s), (x + 0.24 * s, y + 0.25 * s),
                                      (x + 0.12 * s, y + 0.38 * s), (x - 0.12 * s, y + 0.38 * s)],
                                     closed=True, facecolor=LANTERN_DARK_COLOR, edgecolor='none',
                                     zorder=5)),
        ax.add_patch(patches.FancyBboxPatch((x - 0.18 * s, y - 0.25 * s), 0.36 * s, 0.5 * s,
                                            boxstyle=f'round,pad=0,rounding_size={0.06 * s}',
                                            facecolor=LANTERN_GLASS_COLOR, edgecolor=LANTERN_DARK_COLOR,
                                            linewidth=1, zorder=5)),
        ax.add_patch(patches.Ellipse((x, y - 0.02 * s), 0.13 * s, 0.24 * s, facecolor=FLAME_COLOR,
                                     edgecolor='none', zorder=5.1)),
        ax.add_patch(patches.Rectangle((x - 0.24 * s, y - 0.32 * s), 0.48 * s, 0.07 * s,
                                       facecolor=LANTERN_DARK_COLOR, edgecolor='none', zorder=5.1))]


def _burglar(ax, x, y, s, pose):

    # a burglar in a striped shirt, beanie and eye mask, centered on the given point:
    # walking with arms down, stealing with a sack over the shoulder, or caught with hands up
    bottom = y - 0.5 * s

    def box(x0, y0, w, h, color, zorder=6, angle=0.0):
        return ax.add_patch(patches.Rectangle((x + x0 * s, bottom + y0 * s), w * s, h * s,
                                              angle=angle, facecolor=color, edgecolor='none',
                                              zorder=zorder))

    artists = []
    if pose == 'stealing':
        artists.append(ax.add_patch(patches.Ellipse((x - 0.2 * s, bottom + 0.72 * s), 0.34 * s,
                                                    0.4 * s, angle=20, facecolor=SACK_COLOR,
                                                    edgecolor='#6d4c41', linewidth=0.6, zorder=5.9)))
    artists += [box(-0.12, 0, 0.1, 0.4, DARK_COLOR), box(0.02, 0, 0.1, 0.4, DARK_COLOR),
                box(-0.17, 0.38, 0.34, 0.34, 'white', 6.1)]
    artists += [box(-0.17, 0.42 + 0.1 * k, 0.34, 0.05, DARK_COLOR, 6.2) for k in range(3)]
    if pose == 'caught':
        artists += [box(-0.25, 0.62, 0.08, 0.32, DARK_COLOR), box(0.17, 0.62, 0.08, 0.32, DARK_COLOR)]
    elif pose == 'stealing':
        artists += [box(-0.25, 0.62, 0.08, 0.22, DARK_COLOR, 6.3), box(0.17, 0.4, 0.08, 0.3, DARK_COLOR)]
    else:
        artists += [box(-0.25, 0.4, 0.08, 0.3, DARK_COLOR), box(0.17, 0.4, 0.08, 0.3, DARK_COLOR)]
    artists += [
        ax.add_patch(plt.Circle((x, bottom + 0.83 * s), 0.11 * s, facecolor=SKIN_COLOR,
                                edgecolor='none', zorder=6.3)),
        ax.add_patch(patches.Wedge((x, bottom + 0.87 * s), 0.12 * s, 0, 180, facecolor=DARK_COLOR,
                                   edgecolor='none', zorder=6.4)),
        box(-0.12, 0.8, 0.24, 0.06, DARK_COLOR, 6.4)]
    artists += [ax.add_patch(plt.Circle((x + dx * s, bottom + 0.83 * s), 0.018 * s,
                                        facecolor='white', edgecolor='none', zorder=6.5))
                for dx in (-0.05, 0.05)]
    return artists


def _protected_area(ax, left, bottom, width, height, step, safe):

    # stone paving inside a fence, with a safe in the middle
    artists = [ax.add_patch(patches.Rectangle((left, bottom), width, height, facecolor=PAVING_COLOR,
                                              edgecolor='none', zorder=0.5))]
    for x in np.arange(left + step, left + width, step):
        artists += ax.plot([x, x], [bottom, bottom + height], color=PAVING_LINE_COLOR,
                           linewidth=0.6, zorder=0.6)
    for y in np.arange(bottom + step, bottom + height, step):
        artists += ax.plot([left, left + width], [y, y], color=PAVING_LINE_COLOR, linewidth=0.6,
                           zorder=0.6)
    s = min(safe, 0.6 * width, 0.6 * height)
    (x, y) = (left + width / 2, bottom + height / 2)
    return artists + [
        ax.add_patch(patches.Rectangle((left, bottom), width, height, facecolor='none',
                                       edgecolor=FENCE_COLOR, linewidth=2, linestyle=(0, (3, 2)),
                                       zorder=0.7)),
        ax.add_patch(patches.FancyBboxPatch((x - s / 2, y - s / 2), s, s,
                                            boxstyle=f'round,pad=0,rounding_size={0.08 * s}',
                                            facecolor=SAFE_COLOR, edgecolor=SAFE_DARK_COLOR,
                                            linewidth=1, zorder=0.8)),
        ax.add_patch(plt.Circle((x, y), 0.18 * s, facecolor='#cfd8dc', edgecolor=SAFE_DARK_COLOR,
                                linewidth=0.8, zorder=0.9)),
        ax.add_patch(patches.Rectangle((x + 0.3 * s, y - 0.05 * s), 0.12 * s, 0.1 * s,
                                       facecolor='#cfd8dc', edgecolor='none', zorder=0.9))]


def _draw_legend(fig, figure_size, lantern):

    # the same pictures as the grounds, on a patch of night ground each
    (width, height) = figure_size
    ax = fig.add_axes([0, 0.05 / height, 1, 0.55 / height])
    ax.set_xlim(0, width)
    ax.set_ylim(0, 0.55)
    ax.set_aspect('equal')
    ax.axis('off')
    entries = [('lantern', 'Lantern'), ('area', 'Critical area'), ('walking', 'Burglar'),
               ('stealing', 'Stealing'), ('caught', 'Caught')]
    widths = [0.42 + 0.062 * len(label) + 0.12 for (_, label) in entries]
    shrink = min(1.0, (width - 0.2) / sum(widths))
    x = (width - shrink * sum(widths)) / 2
    for ((kind, label), entry_width) in zip(entries, widths):
        ax.add_patch(patches.Rectangle((x, 0.1), 0.36, 0.36, facecolor=NIGHT_COLOR, edgecolor='none',
                                       zorder=0))
        (cx, cy) = (x + 0.18, 0.28)
        if kind == 'lantern':
            lantern(ax, cx, cy)
        elif kind == 'area':
            _protected_area(ax, x + 0.03, 0.13, 0.3, 0.3, 0.1, 0.17)
        else:
            if kind == 'caught':
                _glow(ax, cx, cy, 0.17)
            _burglar(ax, cx, cy, 0.32, kind)
        ax.text(x + 0.42, cy, label, ha='left', va='center', fontsize=8.5, color=TEXT_COLOR)
        x += shrink * entry_width


class IntrudersVisualizer(BaseViz):
    '''Draws the grounds at night: each danger zone is a paved area with a safe behind a
    fence, each camera a lantern with the circle it lights, and each intruder a burglar,
    who carries off a sack while in a danger zone out of the light and raises their hands
    when caught in it, as the reward treats them. Faint trails show recent motion.

    Nothing limits where intruders and cameras can go, so unless lb and ub give a
    fixed view, the view starts around everything in the first state and grows to
    keep everything in sight. This requires render to be called on every step.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 tail_steps=5,
                 lb=None, ub=None,
                 wait_time=100) -> None:
        self._model = model
        self._tail_steps = tail_steps
        self._lb = lb
        self._ub = ub
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._init_state = model.ground_vars_with_values(model.state_fluents)
        self._cameras = model.type_to_objects['camera']
        self._intruders = model.type_to_objects['intruder']
        self._dt = self._nonfluents['DT']
        self._radius = {c: self._nonfluents[RDDLPlanningModel.ground_var('CAMERA-RADIUS', [c])]
                        for c in self._cameras}

        self._last_state = None
        self._bounds = None

        # title above, then the grounds and the legend
        width = PLOT_SIZE + 2 * SIDE_MARGIN
        height = BOTTOM_MARGIN + PLOT_SIZE + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([SIDE_MARGIN / width, BOTTOM_MARGIN / height,
                                     PLOT_SIZE / width, PLOT_SIZE / height])
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(NIGHT_COLOR)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color(NIGHT_COLOR)

        # a zone with its edges the wrong way around can never contain anything in the reward
        self._zones = []
        for z in model.type_to_objects['zone']:
            left, right, top, bottom = [
                self._nonfluents[RDDLPlanningModel.ground_var(var, [z])]
                for var in ('DANGER-L', 'DANGER-R', 'DANGER-T', 'DANGER-B')]
            if left <= right and bottom <= top:
                self._zones.append((left, right, bottom, top))

        if lb is not None and ub is not None:
            self.ax.set_xlim(lb, ub)
            self.ax.set_ylim(lb, ub)
        _draw_legend(self.fig, self._figure_size,
                     lambda ax, x, y: _lantern(ax, x, y, 0.26))

    def _get(self, state, var, obj):
        return float(state[RDDLPlanningModel.ground_var(var, [obj])])

    def _get_extent(self, state):

        # the zones, the area of each camera and the intruders
        extent = list(self._zones)
        for c in self._cameras:
            x, y = self._get(state, 'camera-x', c), self._get(state, 'camera-y', c)
            extent.append((x - self._radius[c], x + self._radius[c],
                           y - self._radius[c], y + self._radius[c]))
        for i in self._intruders:
            x, y = self._get(state, 'intruder-x', i), self._get(state, 'intruder-y', i)
            extent.append((x, x, y, y))
        return (min(e[0] for e in extent), max(e[1] for e in extent),
                min(e[2] for e in extent), max(e[3] for e in extent))

    def _update_view(self, state):

        # a fixed view was requested
        if self._lb is not None and self._ub is not None:
            return

        # rendering the same state again must not change the view, and only a new
        # episode can return to the initial state as the intruders move randomly
        if state == self._last_state:
            return
        new_episode = self._last_state is None or state == self._init_state
        self._last_state = dict(state)

        extent = self._get_extent(state)
        if new_episode:
            self._bounds = extent
        else:
            self._bounds = (min(self._bounds[0], extent[0]), max(self._bounds[1], extent[1]),
                            min(self._bounds[2], extent[2]), max(self._bounds[3], extent[3]))

        # both axes use the same scale, with a margin around everything for the pictures
        left, right, bottom, top = self._bounds
        span = (1.1 * max(right - left, top - bottom) or 1.0) / (1 - 1.2 * BURGLAR_SIZE / PLOT_SIZE)
        x_center, y_center = (left + right) / 2, (bottom + top) / 2
        self.ax.set_xlim(x_center - span / 2, x_center + span / 2)
        self.ax.set_ylim(y_center - span / 2, y_center + span / 2)

    def _draw_tail(self, x, y, vx, vy):
        length = self._tail_steps * self._dt
        return self.ax.plot([x, x - length * vx], [y, y - length * vy], color=TRAIL_COLOR,
                            linewidth=2, alpha=0.35, solid_capstyle='round', zorder=2)

    def _count_zones(self, x, y):
        return sum(left <= x <= right and bottom <= y <= top
                   for (left, right, bottom, top) in self._zones)

    def _is_detected(self, state, x, y):
        return any(
            (self._get(state, 'camera-x', c) - x) ** 2 +
            (self._get(state, 'camera-y', c) - y) ** 2 <= self._radius[c] ** 2
            for c in self._cameras)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        self._update_view(state)
        (left, right) = self.ax.get_xlim()
        per_inch = (right - left) / PLOT_SIZE

        # the zones are drawn again as the view changes the size of their paving and safe
        artists = []
        for (zone_left, zone_right, zone_bottom, zone_top) in self._zones:
            artists += _protected_area(self.ax, zone_left, zone_bottom, zone_right - zone_left,
                                       zone_top - zone_bottom, PAVING_STEP * per_inch,
                                       SAFE_SIZE * per_inch)
        for c in self._cameras:
            x, y = self._get(state, 'camera-x', c), self._get(state, 'camera-y', c)
            artists += _glow(self.ax, x, y, self._radius[c])
            artists += self._draw_tail(x, y, self._get(state, 'camera-vx', c),
                                       self._get(state, 'camera-vy', c))
            artists += _lantern(self.ax, x, y, LANTERN_SIZE * per_inch)

        # same rule as the reward, which counts each zone an intruder is in
        stealing = 0
        for i in self._intruders:
            x, y = self._get(state, 'intruder-x', i), self._get(state, 'intruder-y', i)
            zones = self._count_zones(x, y)
            if zones == 0:
                pose = 'walking'
            elif self._is_detected(state, x, y):
                pose = 'caught'
            else:
                pose = 'stealing'
                stealing += 1
            artists += self._draw_tail(x, y, self._get(state, 'intruder-vx', i),
                                       self._get(state, 'intruder-vy', i))
            artists += _burglar(self.ax, x, y, BURGLAR_SIZE * per_inch, pose)

        self.ax.set_title('Nothing stolen' if stealing == 0 else
                          f'{stealing} burglar{"" if stealing == 1 else "s"} stealing',
                          fontsize=13, fontweight='bold',
                          color=STEALING_COLOR if stealing else SAFE_TITLE_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
