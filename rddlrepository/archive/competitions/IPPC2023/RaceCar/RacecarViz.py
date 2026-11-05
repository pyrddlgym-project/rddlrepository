import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import collections as mc
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BOUNDARY_COLOR = '#37474f'
GOAL_COLOR = '#c8e6c9'
GOAL_EDGE_COLOR = '#2e7d32'
START_COLOR = '#90a4ae'
CAR_COLOR = '#1565c0'
OK_COLOR = '#2e7d32'


class RacecarVisualizer(BaseViz):
    '''Draws the boundaries of the track, the goal and the point the car returns to when
    it hits a boundary, and the car with a tail showing its recent motion. The view
    covers the track and the current position of the car, and the drawing only depends
    on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(5.4, 5.9),
                 tail_steps=5,
                 wait_time=100) -> None:
        self._model = model
        self._tail_steps = tail_steps
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._dt = nonfluents['DT']
        self._goal = (nonfluents['GX'], nonfluents['GY'])
        self._radius = nonfluents['RADIUS']
        start = (nonfluents['X0'], nonfluents['Y0'])

        lines = []
        for b in model.type_to_objects['b']:
            lines.append([(nonfluents[RDDLPlanningModel.ground_var('X1', [b])],
                           nonfluents[RDDLPlanningModel.ground_var('Y1', [b])]),
                          (nonfluents[RDDLPlanningModel.ground_var('X2', [b])],
                           nonfluents[RDDLPlanningModel.ground_var('Y2', [b])])])

        # everything that never changes
        points = [point for line in lines for point in line] + [start]
        points.extend([(self._goal[0] - self._radius, self._goal[1] - self._radius),
                       (self._goal[0] + self._radius, self._goal[1] + self._radius)])
        xs, ys = zip(*points)
        self._bounds = (min(xs), max(xs), min(ys), max(ys))

        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_axes([0.1, 0.15, 0.86, 0.76])
        self.ax.set_aspect('equal')
        self.ax.tick_params(labelsize=8, colors='#616161')
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')

        # the track, goal and start never change, so they are only drawn once
        self.ax.add_collection(mc.LineCollection(lines, linewidths=2.5, colors=BOUNDARY_COLOR, zorder=2))
        self.ax.add_patch(plt.Circle(self._goal, self._radius, facecolor=GOAL_COLOR,
                                     edgecolor=GOAL_EDGE_COLOR, linewidth=1.2, zorder=1))
        self.ax.plot(*start, marker='x', markersize=8, markeredgewidth=2, color=START_COLOR, zorder=1)

        self._info = self.fig.text(0.5, 0.1 / figure_size[1], '', ha='center', va='bottom', fontsize=9, color='#37474f')
        self.fig.legend(handles=[
            Line2D([], [], color=BOUNDARY_COLOR, linewidth=2.5, label='Boundary'),
            patches.Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_EDGE_COLOR, label='Goal'),
            Line2D([], [], marker='x', color='none', markeredgecolor=START_COLOR,
                   markeredgewidth=2, label='Start'),
            Line2D([], [], marker='o', color='none', markerfacecolor=CAR_COLOR,
                   markeredgecolor='none', label='Car')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / figure_size[1]), ncol=4,
           frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _update_view(self, x, y):

        # both axes use the same scale, with a small margin around everything
        left, right, bottom, top = self._bounds
        left, right = min(left, x), max(right, x)
        bottom, top = min(bottom, y), max(top, y)
        span = 1.1 * max(right - left, top - bottom) or 1.0
        x_center, y_center = (left + right) / 2, (bottom + top) / 2
        self.ax.set_xlim(x_center - span / 2, x_center + span / 2)
        self.ax.set_ylim(y_center - span / 2, y_center + span / 2)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        x, y = float(state['x']), float(state['y'])
        vx, vy = float(state['vx']), float(state['vy'])
        self._update_view(x, y)

        length = self._tail_steps * self._dt
        artists = self.ax.plot([x, x - length * vx], [y, y - length * vy], color=CAR_COLOR,
                               linewidth=3, alpha=0.35, solid_capstyle='round', zorder=3)
        artists.extend(self.ax.plot(x, y, marker='o', markersize=11, color=CAR_COLOR,
                                    markeredgecolor='white', zorder=4))

        # same rule as the termination condition
        distance = math.hypot(x - self._goal[0], y - self._goal[1])
        reached = distance <= self._radius
        self.ax.set_title('Goal reached' if reached else f'Distance to goal: {distance:.2f}',
                          fontsize=13, fontweight='bold', color=OK_COLOR if reached else 'black')
        self._info.set_text(f'Speed {math.hypot(vx, vy):.2f}')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
