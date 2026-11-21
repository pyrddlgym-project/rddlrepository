import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

SPACE_COLOR = '#0b1426'
STAR_COLOR = '#c9d6ea'
RING_COLOR = '#3a4a66'
LABEL_COLOR = '#8796b0'
EARTH_COLOR = '#1f5fa8'
LAND_COLOR = '#3f8a4a'
GLOW_COLOR = '#7fb8ff'
DOCK_COLOR = '#2e7d32'
LOST_COLOR = '#c62828'
BODY_COLOR = '#d9dee6'
FOIL_COLOR = '#d4a017'
PANEL_COLOR = '#2b4f8c'
PANEL_LINE = '#9db6e0'
SAFE_COLOR = '#66bb6a'
FAST_COLOR = '#ef5350'
PENALTY_COLOR = '#ffa726'
TIME_COLOR = '#90a4ae'
BAR_BACK = '#1d2a42'
TEXT_COLOR = '#e3e9f3'

TITLES = {
    'none': ('Approaching', TEXT_COLOR),
    'docked': ('Docked', SAFE_COLOR),
    'crash': ('Crashed', FAST_COLOR),
    'lost': ('Lost', FAST_COLOR),
    'timeout': ('Timed out', PENALTY_COLOR),
    'over-budget': ('Over the speeding budget', PENALTY_COLOR),
}

# the two views: name, the axis drawn to the right, and the name of that direction;
# the radial axis x (away from the Earth) is always up
VIEWS = (('orbit plane', 1, 'orbit'),
         ('along the orbit', 2, 'out of plane'))


class SpacecraftDockingVisualizer(BaseViz):
    '''Draws the chief and the deputy in two views of the chief's orbital frame, both
    with the radial direction up (away from the Earth, drawn below): the plane of the
    orbit, with the orbit direction to the right, and the view along the orbit, with
    the direction out of the plane to the right. In each view, the distance from the
    chief is on a log scale, from the docking region (green) to the distance at which
    the deputy is lost (red), so that both fit; the direction is drawn as it is. The
    arrow on the deputy shows its velocity, green under the speed limit and red above
    it. Bars below show the speed against the limit, the speeding penalty against its
    budget, and the time against the timeout.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_width=10.0,
                 wait_time=20) -> None:
        self._model = model
        self._wait_time = wait_time

        nf = model.ground_vars_with_values(model.non_fluents)
        self._n = float(nf['MEAN-MOTION'])
        self._dock = max(float(nf['DOCK-RADIUS']), 1e-6)
        self._lost = max(float(nf['MAX-DISTANCE']), 10 * self._dock)
        self._limit_min = float(nf['SPEED-LIMIT-MIN'])
        self._limit_slope = float(nf['SPEED-LIMIT-SLOPE'])
        self._budget = float(nf['VIOLATION-BUDGET'])
        self._timeout = float(nf['TIMEOUT'])

        # the docking region has radius 1 and the lost distance radius self._outer
        self._outer = self._radius(self._lost)
        R = self._outer
        width, height = 2 * R + 0.8, 2 * R + 1.6

        bars_height = 1.3
        panel_width = 0.48 * figure_width
        fig_height = panel_width * height / width + bars_height + 0.5
        self.fig = plt.figure(figsize=(figure_width, fig_height))
        self.fig.patch.set_facecolor(SPACE_COLOR)

        self._axes = []
        for i, (name, _, direction) in enumerate(VIEWS):
            ax = self.fig.add_axes([0.01 + 0.5 * i, bars_height / fig_height, 0.48,
                                    1 - (bars_height + 0.5) / fig_height])
            self._draw_scene(ax, name, direction)
            self._axes.append(ax)
        self._title = self.fig.text(0.5, 1 - 0.2 / fig_height, '', ha='center', va='center',
                                    fontsize=14, fontweight='bold')

        # bars for the speed, the speeding penalty and the time
        self._bars = []
        for i, (label, color) in enumerate((('speed', SAFE_COLOR),
                                            ('speeding', PENALTY_COLOR),
                                            ('time', TIME_COLOR))):
            y = (1.0 - 0.33 * i) / fig_height
            bar = self.fig.add_axes([0.15, y - 0.09 / fig_height, 0.8, 0.18 / fig_height])
            bar.set_xlim(0, 1)
            bar.set_ylim(0, 1)
            bar.axis('off')
            bar.add_patch(patches.Rectangle((0, 0), 1, 1, color=BAR_BACK))
            fill = bar.add_patch(patches.Rectangle((0, 0), 0, 1, color=color))
            self.fig.text(0.13, y, label, color=TEXT_COLOR, fontsize=9, ha='right', va='center')
            self._bars.append((bar, fill))

        # the speed bar has the limit at its middle
        self._bars[0][0].plot([0.5, 0.5], [-0.2, 1.2], color=TEXT_COLOR, linewidth=1.5, clip_on=False)

    def _radius(self, distance):
        return 1.0 + math.log10(max(distance, self._dock) / self._dock)

    @staticmethod
    def _distance_label(distance):
        if distance >= 1000:
            return f'{distance / 1000:g} km'
        if distance >= 1:
            return f'{distance:g} m'
        return f'{distance * 100:g} cm'

    def _draw_scene(self, ax, name, direction):
        '''Stars, the Earth, distance rings, the docking region and the chief, which
        never change.'''
        R = self._outer
        bottom = -R - 1.1
        ax.set_xlim(-R - 0.4, R + 0.4)
        ax.set_ylim(bottom, R + 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        ax.set_facecolor(SPACE_COLOR)

        rng = np.random.default_rng(7)
        stars = rng.uniform([-R - 0.4, bottom], [R + 0.4, R + 0.5], size=(120, 2))
        ax.scatter(stars[:, 0], stars[:, 1], s=rng.uniform(0.5, 4, 120), color=STAR_COLOR,
                   alpha=0.6, linewidths=0, zorder=0)
        earth = 6.0 * R
        center = (0.0, -R - 0.75 - earth)
        ax.add_patch(plt.Circle(center, earth + 0.12, color=GLOW_COLOR, alpha=0.25, zorder=1))
        ax.add_patch(plt.Circle(center, earth, color=EARTH_COLOR, zorder=1))
        for cx, w in ((-0.55 * R, 0.5 * R), (0.45 * R, 0.35 * R)):
            ax.add_patch(patches.Ellipse((cx, -R - 1.05), w, 0.35, color=LAND_COLOR, zorder=1))

        decade = 10 ** math.floor(math.log10(self._dock) + 1)
        while decade < self._lost:
            r = self._radius(decade)
            ax.add_patch(plt.Circle((0, 0), r, fill=False, edgecolor=RING_COLOR,
                                    linewidth=0.8, linestyle=(0, (2, 3)), zorder=2))
            ax.text(r * math.cos(math.radians(60)), r * math.sin(math.radians(60)),
                    self._distance_label(decade), color=LABEL_COLOR, fontsize=7,
                    ha='left', va='bottom', zorder=2)
            decade *= 10
        ax.add_patch(plt.Circle((0, 0), R, fill=False, edgecolor=LOST_COLOR,
                                linewidth=1.5, linestyle=(0, (5, 3)), zorder=2))
        ax.add_patch(plt.Circle((0, 0), 1.0, facecolor=DOCK_COLOR, alpha=0.35,
                                edgecolor=DOCK_COLOR, linewidth=1.5, zorder=2))
        ax.annotate('', xy=(R + 0.3, R + 0.2), xytext=(R - 0.9, R + 0.2),
                    arrowprops=dict(arrowstyle='->', color=LABEL_COLOR, linewidth=1.2))
        ax.text(R + 0.3, R + 0.3, direction, color=LABEL_COLOR, fontsize=8, ha='right', va='bottom')
        ax.text(-R - 0.3, R + 0.3, name, color=TEXT_COLOR, fontsize=10, ha='left', va='bottom')
        self._draw_spacecraft(ax, (0, 0), 0.55, zorder=4)

    def _draw_spacecraft(self, ax, center, size, zorder):
        '''A box covered in gold foil with a solar panel on each side.'''
        cx, cy = center
        artists = []
        for side in (-1, 1):
            x0 = cx + side * 0.35 * size + (0 if side > 0 else -1.1 * size)
            artists.append(ax.add_patch(patches.Rectangle(
                (x0, cy - 0.22 * size), 1.1 * size, 0.44 * size,
                facecolor=PANEL_COLOR, edgecolor=PANEL_LINE, linewidth=0.8, zorder=zorder)))
            for k in (1, 2, 3):
                xk = x0 + k * 1.1 * size / 4
                artists.extend(ax.plot([xk, xk], [cy - 0.22 * size, cy + 0.22 * size],
                                       color=PANEL_LINE, linewidth=0.5, zorder=zorder))
        artists.append(ax.add_patch(patches.FancyBboxPatch(
            (cx - 0.35 * size, cy - 0.35 * size), 0.7 * size, 0.7 * size,
            boxstyle=f'round,pad=0,rounding_size={0.1 * size}',
            facecolor=FOIL_COLOR, edgecolor=BODY_COLOR, linewidth=1.0, zorder=zorder + 0.1)))
        return artists

    def _draw_deputy(self, ax, position, velocity, ratio, fast):
        '''The deputy and its velocity in one view: position and velocity are (right, up).'''
        right, up = position
        distance = math.hypot(right, up)
        r = min(self._radius(distance), self._outer)
        if distance > 0:
            px, py = r * right / distance, r * up / distance
        else:
            px, py = 0.0, 0.0
        artists = []
        speed = math.hypot(*velocity)
        if speed > 1e-12 and ratio > 0:
            length = 0.4 + 0.8 * min(ratio, 2.0)
            artists.append(ax.annotate(
                '', xy=(px + length * velocity[0] / speed, py + length * velocity[1] / speed),
                xytext=(px, py),
                arrowprops=dict(arrowstyle='-|>', color=FAST_COLOR if fast else SAFE_COLOR,
                                linewidth=2.0, mutation_scale=14), zorder=6))
        artists.extend(self._draw_spacecraft(ax, (px, py), 0.3, zorder=7))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        pos = [float(state[v]) for v in ('x', 'y', 'z')]
        vel = [float(state[v]) for v in ('vx', 'vy', 'vz')]
        penalty, time = float(state['penalty']), float(state['time'])
        docked = bool(state['docked'])
        failure = str(state['failure']).lstrip('@')

        # the speed limit at the deputy's distance
        distance = math.sqrt(sum(p * p for p in pos))
        speed = math.sqrt(sum(v * v for v in vel))
        limit = self._limit_min + self._limit_slope * self._n * max(0.0, distance - self._dock)
        ratio = speed / max(limit, 1e-9)
        fast = speed > limit

        self.fig.canvas.draw()
        artists = []
        for ax, (_, axis, _) in zip(self._axes, VIEWS):
            artists.extend(self._draw_deputy(ax, (pos[axis], pos[0]), (vel[axis], vel[0]), ratio, fast))

        outcome = 'docked' if docked else failure if failure in TITLES else 'none'
        title, color = TITLES[outcome]
        self._title.set_text(title)
        self._title.set_color(color)

        # bars: speed (limit at the middle), speeding penalty, time
        fills = (min(ratio, 2.0) / 2.0,
                 min(penalty / self._budget, 1.0) if self._budget > 0 else 0.0,
                 min(time / self._timeout, 1.0) if self._timeout > 0 else 0.0)
        for (bar, fill), value in zip(self._bars, fills):
            fill.set_width(value)
        self._bars[0][1].set_color(FAST_COLOR if fast else SAFE_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
