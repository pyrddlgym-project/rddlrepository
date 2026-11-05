import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

GUIDE_COLOR = '#cfd8dc'
LINK1_COLOR = '#1565c0'
LINK2_COLOR = '#42a5f5'
PIVOT_COLOR = '#37474f'
GOAL_COLOR = '#2e7d32'

# the arm reaches its goal above this height, from the termination condition
GOAL_HEIGHT = 1.0


class AcrobotVisualizer(BaseViz):
    '''Draws the arm to scale, with its goal and a tail showing the recent motion of
    its tip. The drawing only depends on the current state.

    The goal in the domain does not depend on the lengths of the links, so it only
    corresponds to a line in the drawing when both links have the same length. The
    title always follows the rule in the domain.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(5.4, 5.8),
                 tail_steps=3,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._tail_steps = tail_steps

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._length1 = nonfluents['L1']
        self._length2 = nonfluents['L2']
        self._delta = nonfluents['DELTA']
        self._total = (self._length1 + self._length2) or 1.0

        fig_width, fig_height = figure_size
        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_axes([0.02, 0.45 / fig_height, 0.96, 1 - 0.95 / fig_height])
        limit = 1.15 * self._total
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # the reach of the arm and its goal never change, so they are only drawn once
        self.ax.add_patch(plt.Circle((0, 0), self._total, facecolor='none', edgecolor=GUIDE_COLOR,
                                     linestyle='--', linewidth=1, zorder=0))
        if self._length1 == self._length2:
            height = GOAL_HEIGHT * self._length1
            self.ax.plot([-limit, limit], [height, height], color=GOAL_COLOR, linestyle='--',
                         linewidth=1.5, zorder=1)
        self._info = self.fig.text(0.5, 0.1 / fig_height, '', ha='center', va='bottom', fontsize=9, color='#37474f')

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def _elbow_and_tip(self, theta1, theta2):

        # angles of zero hang straight down
        elbow = (self._length1 * np.sin(theta1), -self._length1 * np.cos(theta1))
        tip = (elbow[0] + self._length2 * np.sin(theta1 + theta2),
               elbow[1] - self._length2 * np.cos(theta1 + theta2))
        return elbow, tip

    def render(self, state):
        theta1, theta2 = float(state['theta1']), float(state['theta2'])
        speed1, speed2 = float(state['theta1v']), float(state['theta2v'])
        elbow, tip = self._elbow_and_tip(theta1, theta2)

        # the tail follows the path of the tip if the speeds were held constant
        steps = np.linspace(0, self._tail_steps * self._delta, 20)
        _, tail = self._elbow_and_tip(theta1 - steps * speed1, theta2 - steps * speed2)
        artists = self.ax.plot(tail[0], tail[1], color=LINK2_COLOR, linewidth=4, alpha=0.35,
                               solid_capstyle='round', zorder=2)

        artists.extend(self.ax.plot([0, elbow[0]], [0, elbow[1]], color=LINK1_COLOR, linewidth=8,
                                    solid_capstyle='round', zorder=3))
        artists.extend(self.ax.plot([elbow[0], tip[0]], [elbow[1], tip[1]], color=LINK2_COLOR,
                                    linewidth=8, solid_capstyle='round', zorder=4))
        for (center, radius, color) in (((0, 0), 0.03, PIVOT_COLOR),
                                        (elbow, 0.025, PIVOT_COLOR),
                                        (tip, 0.035, LINK2_COLOR)):
            artists.append(self.ax.add_patch(plt.Circle(
                center, radius * self._total, facecolor=color, edgecolor='white', zorder=5)))

        # same rule as the termination condition
        height = -math.cos(theta1) - math.cos(theta1 + theta2)
        reached = height > GOAL_HEIGHT
        self.ax.set_title('Goal reached' if reached else 'Swinging up', fontsize=13,
                          fontweight='bold', color=GOAL_COLOR if reached else 'black')
        self._info.set_text(f'Goal measure {height:+.2f} (must exceed {GOAL_HEIGHT:g})')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
