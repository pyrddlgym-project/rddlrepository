import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

GUIDE_COLOR = '#cfd8dc'
ROD_COLOR = '#1565c0'
PIVOT_COLOR = '#37474f'
TARGET_COLOR = '#2e7d32'


class PendulumVisualizer(BaseViz):
    '''Draws the pendulum to scale, with its upright target and a tail along the
    circle showing its recent motion. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(5.4, 5.8),
                 tail_steps=3,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._tail_steps = tail_steps

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._length = nonfluents['LENGTH'] or 1.0
        self._dt = nonfluents['DT']

        fig_width, fig_height = figure_size
        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_axes([0.02, 0.45 / fig_height, 0.96, 1 - 0.95 / fig_height])
        limit = 1.25 * self._length
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # the circle and target never change, so they are only drawn once
        self.ax.add_patch(plt.Circle((0, 0), self._length, facecolor='none', edgecolor=GUIDE_COLOR,
                                     linestyle='--', linewidth=1, zorder=0))
        self.ax.plot([0, 0], [0, 1.12 * self._length], color=TARGET_COLOR, linestyle='--',
                     linewidth=1.2, zorder=1)
        self.ax.plot([0], [self._length], marker='o', markersize=8, color='none',
                     markeredgecolor=TARGET_COLOR, markeredgewidth=2, zorder=1)
        self._info = self.fig.text(0.5, 0.1 / fig_height, '', ha='center', va='bottom', fontsize=9, color='#37474f')

    def _point(self, angle):

        # an angle of zero is upright, and larger angles turn to the left
        return -self._length * np.sin(angle), self._length * np.cos(angle)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        theta, speed = float(state['theta']), float(state['theta-dot'])
        tip_x, tip_y = self._point(theta)

        tail = theta - np.linspace(0, self._tail_steps * self._dt * speed, 20)
        artists = self.ax.plot(*self._point(tail), color=ROD_COLOR, linewidth=5, alpha=0.3,
                               solid_capstyle='round', zorder=2)
        artists.extend(self.ax.plot([0, tip_x], [0, tip_y], color=ROD_COLOR, linewidth=8,
                                    solid_capstyle='round', zorder=3))
        artists.append(self.ax.add_patch(plt.Circle(
            (tip_x, tip_y), 0.07 * self._length, facecolor=ROD_COLOR, edgecolor='white', zorder=4)))
        artists.append(self.ax.add_patch(plt.Circle(
            (0, 0), 0.05 * self._length, facecolor=PIVOT_COLOR, edgecolor='white', zorder=5)))

        # same angle as the reward, wrapped around the upright position
        error = math.degrees((theta + math.pi) % (2 * math.pi) - math.pi)
        self.ax.set_title(f'{abs(error):.1f} deg from upright', fontsize=13, fontweight='bold')
        self._info.set_text(f'Angular speed {speed:+.2f}')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
