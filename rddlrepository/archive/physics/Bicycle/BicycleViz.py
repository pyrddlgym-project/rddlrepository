import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BIKE_COLOR = '#1565c0'
WARNING_COLOR = '#ffa000'
FALLEN_COLOR = '#c62828'
FRONT_WHEEL_COLOR = '#ef6c00'
OK_COLOR = '#2e7d32'

# the bicycle falls over beyond this lean, from the termination condition
LEAN_LIMIT = math.radians(12.0)

# the lean at which the bicycle is drawn as being close to falling, as a fraction
WARNING_FRACTION = 0.75


class BicycleVisualizer(BaseViz):
    '''Draws the bicycle from behind to show its lean against the limit, from above
    to show its heading and steering, and the history of lean and steering.

    A positive lean is drawn to the right and a positive steering angle turns left,
    following the dynamics: steering into a lean is what corrects it.

    The history is built from the states seen so far, so it requires render to be
    called on every step. The state does not hold the time, so it is the number of
    different states seen multiplied by the time step.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(9.0, 7.0),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._dt = nonfluents['DT']
        self._height = nonfluents['H']
        self._rider_offset = nonfluents['DCM']
        self._wheelbase = nonfluents['L']
        self._radius = nonfluents['R']
        self._init_state = self._get_state(model.ground_vars_with_values(model.state_fluents))

        self._last_state = None
        self._times = []
        self._leans = []
        self._steers = []

        self.fig = plt.figure(figsize=figure_size)
        grid = self.fig.add_gridspec(3, 2, height_ratios=[3.0, 1.0, 1.0], left=0.09, right=0.97,
                                     top=0.89, bottom=0.08, hspace=0.35, wspace=0.12)
        self._rear_ax = self.fig.add_subplot(grid[0, 0])
        self._top_ax = self.fig.add_subplot(grid[0, 1])
        self._lean_ax = self.fig.add_subplot(grid[1, :])
        self._steer_ax = self.fig.add_subplot(grid[2, :], sharex=self._lean_ax)

        self._setup_rear_view()
        self._setup_top_view()
        self._setup_history()

    def _get_state(self, state):
        return tuple(float(state[var]) for var in ('omega', 'domega', 'theta', 'dtheta', 'psi'))

    def _setup_rear_view(self):

        # the lean is drawn at its true angle, about the point touching the ground
        ax = self._rear_ax
        self._rear_size = max(self._height + self._rider_offset, 2 * self._radius) or 1.0
        size = self._rear_size
        ax.set_xlim(-0.75 * size, 0.75 * size)
        ax.set_ylim(-0.08 * size, 1.2 * size)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0, color='#616161', linewidth=2, zorder=0)
        reach = 1.15 * size
        for side in (-1, 1):
            ax.plot([0, side * reach * math.sin(LEAN_LIMIT)],
                    [0, reach * math.cos(LEAN_LIMIT)],
                    color=FALLEN_COLOR, linestyle='--', linewidth=1.2, zorder=1)

    def _setup_top_view(self):

        # the bicycle turns in place: the fixed line is the direction of zero heading
        ax = self._top_ax
        self._top_limit = 1.35 * (self._wheelbase / 2 + self._radius) or 1.0
        ax.set_xlim(-self._top_limit, self._top_limit)
        ax.set_ylim(-self._top_limit, self._top_limit)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([0, 0], [-self._top_limit, self._top_limit], color='#bdbdbd',
                linestyle=':', linewidth=1, zorder=0)

    def _setup_history(self):
        limit = math.degrees(LEAN_LIMIT)
        self._lean_ax.set_ylim(-1.25 * limit, 1.25 * limit)
        for side in (-1, 1):
            self._lean_ax.axhline(side * limit, color=FALLEN_COLOR, linestyle='--', linewidth=1)
        self._lean_ax.axhline(0, color='#bdbdbd', linewidth=1)
        self._lean_line, = self._lean_ax.plot([], [], color=BIKE_COLOR, linewidth=1.3)
        self._lean_ax.set_ylabel('lean (deg)', fontsize=9)
        self._lean_ax.tick_params(labelsize=8, labelbottom=False)

        self._steer_ax.axhline(0, color='#bdbdbd', linewidth=1)
        self._steer_line, = self._steer_ax.plot([], [], color=FRONT_WHEEL_COLOR, linewidth=1.3)
        self._steer_ax.set_ylabel('steering (deg)', fontsize=9)
        self._steer_ax.set_xlabel('time', fontsize=9)
        self._steer_ax.tick_params(labelsize=8)

    def _update_history(self, state):

        # rendering the same state again must not change the history
        if state == self._last_state:
            return
        last_state, self._last_state = self._last_state, state

        # an episode ends when the bicycle falls, and can be restarted at any time
        if last_state is None or state == self._init_state or abs(last_state[0]) > LEAN_LIMIT:
            self._times = []
            self._leans = []
            self._steers = []
        self._times.append(len(self._times) * self._dt)
        self._leans.append(math.degrees(state[0]))
        self._steers.append(math.degrees(state[2]))

    def _lean_color(self, lean):
        if abs(lean) > LEAN_LIMIT:
            return FALLEN_COLOR
        return WARNING_COLOR if abs(lean) > WARNING_FRACTION * LEAN_LIMIT else BIKE_COLOR

    def _draw_rear_view(self, lean):
        ax = self._rear_ax
        size = self._rear_size
        color = self._lean_color(lean)
        direction = (math.sin(lean), math.cos(lean))

        def point(distance):
            return (distance * direction[0], distance * direction[1])

        def segment(start, end, **kwargs):
            (x1, y1), (x2, y2) = point(start), point(end)
            return ax.plot([x1, x2], [y1, y2], solid_capstyle='round', **kwargs)

        wheel_top = 2 * self._radius
        rider_top = self._height + self._rider_offset
        artists = []
        artists.extend(segment(0, wheel_top, color='#212121', linewidth=7, zorder=3))
        artists.extend(segment(wheel_top, self._height, color=color, linewidth=5, zorder=3))
        artists.extend(segment(self._height, rider_top, color=color, linewidth=9, zorder=3))
        artists.append(ax.add_patch(plt.Circle(
            point(rider_top + 0.07 * size), 0.07 * size, facecolor=color, zorder=4)))
        ax.set_title(f'Rear view: lean {math.degrees(lean):+.1f} deg '
                     f'(limit {math.degrees(LEAN_LIMIT):g})', fontsize=10)
        return artists

    def _draw_top_view(self, steer, heading):
        ax = self._top_ax

        # a heading of zero points up, and larger headings turn to the left
        forward = (-math.sin(heading), math.cos(heading))
        wheel = (-math.sin(heading + steer), math.cos(heading + steer))
        half = self._wheelbase / 2
        rear = (-half * forward[0], -half * forward[1])
        front = (half * forward[0], half * forward[1])

        def wheel_line(center, axis, color):
            return ax.plot(
                [center[0] - self._radius * axis[0], center[0] + self._radius * axis[0]],
                [center[1] - self._radius * axis[1], center[1] + self._radius * axis[1]],
                color=color, linewidth=6, solid_capstyle='round', zorder=3)

        artists = []
        artists.extend(ax.plot([rear[0], front[0]], [rear[1], front[1]],
                               color=BIKE_COLOR, linewidth=3, zorder=2))
        artists.extend(wheel_line(rear, forward, '#212121'))
        artists.extend(wheel_line(front, wheel, FRONT_WHEEL_COLOR))

        # the initial heading may be outside of the range the domain wraps it to
        wrapped = math.degrees(math.atan2(math.sin(heading), math.cos(heading)))
        ax.set_title(f'Top view: heading {wrapped:+.0f} deg, steering {math.degrees(steer):+.1f} deg',
                     fontsize=10)
        return artists

    def _update_history_plots(self):
        self._lean_line.set_data(self._times, self._leans)
        self._steer_line.set_data(self._times, self._steers)
        self._lean_ax.set_xlim(0, max(self._times[-1], self._dt) or 1.0)
        steer_limit = max(1.0, 1.1 * max(abs(steer) for steer in self._steers))
        self._steer_ax.set_ylim(-steer_limit, steer_limit)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        state = self._get_state(state)
        self._update_history(state)
        lean, _, steer, _, heading = state

        artists = self._draw_rear_view(lean)
        artists.extend(self._draw_top_view(steer, heading))
        self._update_history_plots()

        fallen = abs(lean) > LEAN_LIMIT
        self.fig.suptitle(
            f'FELL OVER at time {self._times[-1]:.2f}' if fallen
            else f'Balanced, time {self._times[-1]:.2f}',
            fontsize=13, fontweight='bold', color=FALLEN_COLOR if fallen else OK_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
