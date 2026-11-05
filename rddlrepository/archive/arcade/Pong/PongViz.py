import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

COURT_COLOR = '#c8e6c9'
OUT_COLOR = '#ffcdd2'
WALL_COLOR = '#37474f'
LINE_COLOR = '#66bb6a'
BALL_COLOR = '#37474f'
MISSED_COLOR = '#90a4ae'
PADDLE_COLOR = '#1565c0'
HIT_COLOR = '#ff8f00'
LOST_COLOR = '#c62828'

# the walls and paddle enclose the unit square, as the dynamics assume, and the
# width of the area beyond the paddle where missed balls are shown
OUT_WIDTH = 0.12


class PongVisualizer(BaseViz):
    '''Draws the court with its three walls, the paddle along the open side, and the
    balls with a tail showing their direction of motion.

    A missed ball keeps moving away from the court forever, so it is held in the
    area beyond the paddle. The paddle is highlighted when it hits a ball: only this
    highlight depends on the previous render, as it needs the previous velocities.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(5.2, 5.0),
                 ball_radius=0.02,
                 tail_steps=3,
                 wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._ball_radius = ball_radius
        self._tail_steps = tail_steps
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._init_state = model.ground_vars_with_values(model.state_fluents)
        self._balls = model.type_to_objects['ball']

        self._last_state = None
        self._hit = False

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0.04, 0.04, 0.92, 0.86])
        self.ax.set_xlim(-0.02, 1.0 + OUT_WIDTH)
        self.ax.set_ylim(-0.02, 1.02)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # the court never changes, so it is only drawn once
        self._court = self.ax.add_patch(patches.Rectangle(
            (0, 0), 1.0 + OUT_WIDTH, 1.0, facecolor=COURT_COLOR, edgecolor='none', zorder=0))
        self.ax.add_patch(patches.Rectangle(
            (1.0, 0), OUT_WIDTH, 1.0, facecolor=OUT_COLOR, edgecolor='none', zorder=0))
        self.ax.plot([0.5, 0.5], [0, 1], color=LINE_COLOR, linestyle='--', linewidth=1, zorder=1)
        self.ax.plot([1.0 + OUT_WIDTH, 0, 0, 1.0 + OUT_WIDTH], [1, 1, 0, 0],
                     color=WALL_COLOR, linewidth=4, solid_capstyle='butt',
                     solid_joinstyle='miter', zorder=2)

    def _get_ball(self, state, var, ball):
        return float(state[RDDLPlanningModel.ground_var(var, [ball])])

    def _update_hit(self, state):

        # rendering the same state again must not change the highlight
        if state == self._last_state:
            return
        last_state, self._last_state = self._last_state, dict(state)

        # only the paddle can turn a ball moving right around, as the left wall
        # does the opposite: a new episode can also change the velocities
        if last_state is None or state == self._init_state:
            self._hit = False
        else:
            self._hit = any(
                self._get_ball(last_state, 'vel-x', b) > 0 and self._get_ball(state, 'vel-x', b) < 0
                for b in self._balls)

    def _draw_ball(self, state, ball):
        x = self._get_ball(state, 'ball-x', ball)
        y = self._get_ball(state, 'ball-y', ball)
        missed = x > 1.0
        color = MISSED_COLOR if missed else BALL_COLOR

        artists = []
        if missed:
            x = min(x, 1.0 + OUT_WIDTH / 2)

        # the tail stays inside the court when the ball has just bounced
        else:
            tail, = self.ax.plot(
                [x, x - self._tail_steps * self._get_ball(state, 'vel-x', ball)],
                [y, y - self._tail_steps * self._get_ball(state, 'vel-y', ball)],
                color=color, linewidth=2, alpha=0.4, solid_capstyle='round', zorder=3)
            tail.set_clip_path(self._court)
            artists.append(tail)
        artists.append(self.ax.add_patch(plt.Circle(
            (x, y), self._ball_radius, facecolor=color, edgecolor='white',
            linewidth=0.8, zorder=4)))
        return artists

    def _draw_paddle(self, state):
        bottom = float(state['paddle-y'])
        return self.ax.plot(
            [1.0, 1.0], [bottom, bottom + self._nonfluents['PADDLE-H']],
            color=HIT_COLOR if self._hit else PADDLE_COLOR,
            linewidth=6, solid_capstyle='butt', zorder=5)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        self._update_hit(state)

        artists = []
        for b in self._balls:
            artists.extend(self._draw_ball(state, b))
        artists.extend(self._draw_paddle(state))

        missed = sum(self._get_ball(state, 'ball-x', b) > 1.0 for b in self._balls)
        all_missed = missed == len(self._balls)
        self.ax.set_title(
            'All balls missed' if all_missed
            else f'{len(self._balls) - missed} / {len(self._balls)} balls in play',
            fontsize=13, fontweight='bold', color=LOST_COLOR if all_missed else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
