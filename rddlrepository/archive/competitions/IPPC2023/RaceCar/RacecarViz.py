import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

GRASS_COLOR = '#8bc34a'
ASPHALT_COLOR = '#686c70'
KERB_RED, KERB_WHITE = '#e53935', '#ffffff'
CAR_COLOR = '#d32f2f'
CAR_DARK_COLOR = '#8e1b1b'
TIRE_COLOR = '#212121'
HELMET_COLOR = '#fdd835'
TRAIL_COLOR = '#eceff1'
START_COLOR = '#ffffff'
DIAL_COLOR = '#37474f'
TEXT_COLOR = '#37474f'
OK_COLOR = '#2e7d32'

# sizes on screen in inches
CAR_LENGTH = 0.46
PLOT_SIZE = 5.0
SIDE_MARGIN = 0.3
BOTTOM_MARGIN = 0.8
TOP_MARGIN = 0.45

# the speedometer shows half way at the speed that crosses the track in this many steps
HALF_DIAL_STEPS = 20


class RacecarVisualizer(BaseViz):
    '''Draws the race track: the closed walls around the start enclose the asphalt and the
    other closed walls are grass, and every wall is lined with red and white kerbs. The
    finish is a chequered circle with a flag, the start a grid box where the car returns when it hits
    a wall, and the car points the way it moves, with a trail of its recent motion. A
    speedometer shows how fast it goes. The view covers the track and the current position
    of the car, and the drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 tail_steps=5,
                 wait_time=100) -> None:
        self._model = model
        self._tail_steps = tail_steps
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._dt = nonfluents['DT']
        self._goal = (nonfluents['GX'], nonfluents['GY'])
        self._radius = nonfluents['RADIUS']
        self._start = (nonfluents['X0'], nonfluents['Y0'])
        fluent = RDDLPlanningModel.ground_var
        self._walls = [((nonfluents[fluent('X1', [b])], nonfluents[fluent('Y1', [b])]),
                        (nonfluents[fluent('X2', [b])], nonfluents[fluent('Y2', [b])]))
                       for b in model.type_to_objects['b']]

        points = [point for wall in self._walls for point in wall] + [self._start]
        points.extend([(self._goal[0] - self._radius, self._goal[1] - self._radius),
                       (self._goal[0] + self._radius, self._goal[1] + self._radius)])
        xs, ys = zip(*points)
        self._bounds = (min(xs), max(xs), min(ys), max(ys))
        self._half_speed = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9) / (HALF_DIAL_STEPS * self._dt)

        # title above, then the track and the legend with the speedometer
        width = PLOT_SIZE + 2 * SIDE_MARGIN
        height = BOTTOM_MARGIN + PLOT_SIZE + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([SIDE_MARGIN / width, BOTTOM_MARGIN / height,
                                     PLOT_SIZE / width, PLOT_SIZE / height])
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color('#558b2f')

        # the track, finish and start never change, so they are only drawn once
        self._draw_track()
        self._chequered(self.ax, self._goal, self._radius)
        self._update_view(*self._start)
        per_inch = self._per_inch()
        self._grid_box(self.ax, self._start, 0.18 * per_inch)
        if self._radius < 0.15 * per_inch:
            self._flag(self.ax, self._goal, per_inch)
        self._needle = self._draw_legend()

    # ---------- the track

    def _loops(self):

        # walls whose ends meet two by two form closed loops, found by walking along them
        def key(point):
            return (round(point[0], 9), round(point[1], 9))

        neighbors = {}
        for (a, b) in self._walls:
            neighbors.setdefault(key(a), []).append(key(b))
            neighbors.setdefault(key(b), []).append(key(a))
        loops, seen = [], set()
        for start in neighbors:
            if start in seen:
                continue
            component, queue = [], [start]
            seen.add(start)
            while queue:
                point = queue.pop()
                component.append(point)
                for other in neighbors[point]:
                    if other not in seen:
                        seen.add(other)
                        queue.append(other)
            if len(component) < 3 or any(len(neighbors[p]) != 2 for p in component):
                continue
            loop, previous, point = [start], None, start
            while True:
                (a, b) = neighbors[point]
                nxt = b if a == previous else a
                if nxt == start:
                    break
                (previous, point) = (point, nxt)
                loop.append(point)
            loops.append(loop)
        return loops

    def _draw_track(self):

        # larger loops go first so that the ones inside them are drawn on top: a loop
        # around the start is asphalt and any other is grass, and without a loop around the
        # start everything is asphalt
        loops = sorted(self._loops(), key=lambda loop: -abs(self._area(loop)))
        around_start = [Path(loop).contains_point(self._start) for loop in loops]
        self.ax.set_facecolor(GRASS_COLOR if any(around_start) else ASPHALT_COLOR)
        for (loop, asphalt) in zip(loops, around_start):
            self.ax.add_patch(patches.Polygon(loop, closed=True, edgecolor='none', zorder=0.5,
                                              facecolor=ASPHALT_COLOR if asphalt else GRASS_COLOR))
        for (a, b) in self._walls:
            (xs, ys) = zip(a, b)
            self.ax.plot(xs, ys, color=KERB_RED, linewidth=5, solid_capstyle='projecting', zorder=2)
            self.ax.plot(xs, ys, color=KERB_WHITE, linewidth=5, dashes=(1.5, 1.5), zorder=2.1)

    @staticmethod
    def _area(loop):
        xs, ys = np.array(loop).T
        return 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

    # ---------- pictures: each takes the axes, a place and a size, and returns its artists

    @staticmethod
    def _chequered(ax, center, radius):

        # a round finish of black and white squares inside a white ring
        circle = patches.Circle(center, radius, facecolor='white', edgecolor='none', zorder=1)
        ax.add_patch(circle)
        side = radius / 2.5
        count = 6
        for i in range(-count // 2, count // 2):
            for j in range(-count // 2, count // 2):
                if (i + j) % 2 == 0:
                    square = ax.add_patch(patches.Rectangle(
                        (center[0] + i * side, center[1] + j * side), side, side,
                        facecolor='#212121', edgecolor='none', zorder=1.1))
                    square.set_clip_path(circle)
        ax.add_patch(patches.Circle(center, radius, facecolor='none', edgecolor='white',
                                    linewidth=2, zorder=1.2))

    @staticmethod
    def _flag(ax, base, per_inch):

        # a chequered flag on a pole planted at a finish too small to see on its own
        (x, y) = base
        (pole, w, h) = (0.32 * per_inch, 0.2 * per_inch, 0.14 * per_inch)
        ax.plot([x, x], [y, y + pole], color='#eceff1', linewidth=1.5, zorder=6)
        ax.add_patch(patches.Rectangle((x, y + pole - h), w, h, facecolor='white',
                                       edgecolor='#212121', linewidth=0.6, zorder=6))
        for i in range(4):
            for j in range(3):
                if (i + j) % 2 == 0:
                    ax.add_patch(patches.Rectangle((x + i * w / 4, y + pole - h + j * h / 3), w / 4,
                                                   h / 3, facecolor='#212121', edgecolor='none',
                                                   zorder=6.1))

    @staticmethod
    def _grid_box(ax, center, size):

        # the white bracket of a starting grid slot, open towards the front
        (x, y) = center
        return ax.plot([x + 0.6 * size, x - 0.6 * size, x - 0.6 * size, x + 0.6 * size],
                       [y - 0.5 * size, y - 0.5 * size, y + 0.5 * size, y + 0.5 * size],
                       color=START_COLOR, linewidth=2, solid_joinstyle='miter', zorder=1.5)

    @staticmethod
    def _car(ax, x, y, length, heading):

        # a single-seater seen from above, its nose pointing along the heading
        rotation = np.array([[math.cos(heading), -math.sin(heading)],
                             [math.sin(heading), math.cos(heading)]])

        def shape(points):
            return np.array(points) * length @ rotation.T + (x, y)

        artists = []
        for (cx, cy) in ((0.27, 0.17), (0.27, -0.17), (-0.3, 0.18), (-0.3, -0.18)):
            artists.append(ax.add_patch(patches.Polygon(
                shape([(cx - 0.09, cy - 0.05), (cx + 0.09, cy - 0.05), (cx + 0.09, cy + 0.05),
                       (cx - 0.09, cy + 0.05)]), facecolor=TIRE_COLOR, edgecolor='none', zorder=5)))
        artists += [
            ax.add_patch(patches.Polygon(
                shape([(-0.5, -0.11), (0.25, -0.08), (0.5, -0.03), (0.5, 0.03), (0.25, 0.08),
                       (-0.5, 0.11)]), facecolor=CAR_COLOR, edgecolor=CAR_DARK_COLOR,
                linewidth=0.6, zorder=5.1)),
            ax.add_patch(patches.Polygon(
                shape([(0.4, -0.2), (0.46, -0.2), (0.46, 0.2), (0.4, 0.2)]),
                facecolor=CAR_DARK_COLOR, edgecolor='none', zorder=5.2)),
            ax.add_patch(patches.Polygon(
                shape([(-0.56, -0.19), (-0.46, -0.19), (-0.46, 0.19), (-0.56, 0.19)]),
                facecolor=CAR_DARK_COLOR, edgecolor='none', zorder=5.2)),
            ax.add_patch(patches.Circle(shape([(-0.02, 0)])[0], 0.055 * length,
                                        facecolor=HELMET_COLOR, edgecolor=CAR_DARK_COLOR,
                                        linewidth=0.5, zorder=5.3))]
        return artists

    def _draw_legend(self):

        # the same pictures as the track, and a speedometer whose needle is returned
        (width, height) = self._figure_size
        ax = self.fig.add_axes([0, 0.05 / height, 1, 0.62 / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, 0.62)
        ax.set_aspect('equal')
        ax.axis('off')
        x = 0.35
        for (kind, label) in (('car', 'Car'), ('start', 'Start'), ('finish', 'Finish'),
                              ('wall', 'Wall')):
            if kind == 'car':
                self._car(ax, x + 0.12, 0.31, 0.36, 0)
            else:
                ax.add_patch(patches.Rectangle((x - 0.06, 0.15), 0.36, 0.32, facecolor=ASPHALT_COLOR,
                                               edgecolor='none'))
                if kind == 'start':
                    self._grid_box(ax, (x + 0.12, 0.31), 0.2)
                elif kind == 'finish':
                    self._chequered(ax, (x + 0.12, 0.31), 0.13)
                else:
                    ax.plot([x - 0.06, x + 0.3], [0.31, 0.31], color=KERB_RED, linewidth=5)
                    ax.plot([x - 0.06, x + 0.3], [0.31, 0.31], color=KERB_WHITE, linewidth=5,
                            dashes=(1.5, 1.5))
            ax.text(x + 0.38, 0.31, label, ha='left', va='center', fontsize=8.5, color=TEXT_COLOR)
            x += 0.45 + 0.07 * len(label) + 0.25

        # a dial from slow on the left to fast on the right, green to red
        (cx, cy, r) = (width - 0.7, 0.2, 0.36)
        for (start, end, color) in ((120, 180, '#66bb6a'), (60, 120, '#fdd835'), (0, 60, '#e53935')):
            ax.add_patch(patches.Wedge((cx, cy), r, start, end, width=0.08, facecolor=color,
                                       edgecolor='none'))
        ax.add_patch(patches.Circle((cx, cy), 0.04, facecolor=DIAL_COLOR, edgecolor='none', zorder=3))
        ax.text(cx, cy - 0.07, 'Speed', ha='center', va='top', fontsize=8.5, color=TEXT_COLOR)
        (needle,) = ax.plot([cx, cx], [cy, cy + r - 0.06], color=DIAL_COLOR, linewidth=2,
                            solid_capstyle='round', zorder=2.9)
        self._dial = (cx, cy, r - 0.06)
        return needle

    # ---------- the state

    def _update_view(self, x, y):

        # both axes use the same scale, with a small margin around everything
        left, right, bottom, top = self._bounds
        left, right = min(left, x), max(right, x)
        bottom, top = min(bottom, y), max(top, y)
        span = 1.1 * max(right - left, top - bottom) or 1.0
        x_center, y_center = (left + right) / 2, (bottom + top) / 2
        self.ax.set_xlim(x_center - span / 2, x_center + span / 2)
        self.ax.set_ylim(y_center - span / 2, y_center + span / 2)

    def _per_inch(self):
        (left, right) = self.ax.get_xlim()
        return (right - left) / PLOT_SIZE

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        x, y = float(state['x']), float(state['y'])
        vx, vy = float(state['vx']), float(state['vy'])
        self._update_view(x, y)
        per_inch = self._per_inch()

        # a car standing still faces the finish
        speed = math.hypot(vx, vy)
        heading = (math.atan2(vy, vx) if speed > 1e-9
                   else math.atan2(self._goal[1] - y, self._goal[0] - x))
        length = self._tail_steps * self._dt
        artists = self.ax.plot([x, x - length * vx], [y, y - length * vy], color=TRAIL_COLOR,
                               linewidth=5, alpha=0.45, solid_capstyle='round', zorder=3)
        artists += self._car(self.ax, x, y, CAR_LENGTH * per_inch, heading)

        # the needle sweeps from the left, and reaches the top at half the dial speed
        (cx, cy, r) = self._dial
        angle = math.pi * (1 - speed / (speed + self._half_speed))
        self._needle.set_data([cx, cx + r * math.cos(angle)], [cy, cy + r * math.sin(angle)])

        # same rule as the termination condition
        reached = math.hypot(x - self._goal[0], y - self._goal[1]) <= self._radius
        self.ax.set_title('Finished' if reached else 'Race to the finish', fontsize=13,
                          fontweight='bold', color=OK_COLOR if reached else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
