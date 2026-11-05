import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

HILL_COLOR = '#c8e6c9'
GROUND_COLOR = '#2e7d32'
CAR_COLOR = '#1565c0'
WHEEL_COLOR = '#37474f'
FLAG_COLOR = '#ff8f00'
OK_COLOR = '#2e7d32'


class MountainCarVisualizer(BaseViz):
    '''Draws the car on the terrain to scale, tilted to match the slope, with the goal
    and a tail along the terrain showing its recent motion. The drawing only depends on
    the current state.

    The terrain is made of the segments in the domain, and is joined by straight lines
    wherever they leave gaps.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_width=7.0,
                 max_height=8.0,
                 tail_steps=5,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._tail_steps = tail_steps

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._min_pos = nonfluents['MIN-POS']
        self._max_pos = nonfluents['MAX-POS']
        self._goal_pos = nonfluents['GOAL-MIN']
        self._goal_vel = nonfluents['VEL-MIN']

        # the segments, in order along the terrain
        self._segments = sorted(
            [tuple(nonfluents[RDDLPlanningModel.ground_var(var, [s])]
                   for var in ('X-START', 'X-END', 'Y-START', 'Y-END'))
             for s in model.type_to_objects['segment']])
        self._terrain_x = [x for (x1, x2, _, _) in self._segments for x in (x1, x2)]
        self._terrain_y = [y for (_, _, y1, y2) in self._segments for y in (y1, y2)]

        world = (self._max_pos - self._min_pos) or 1.0
        self._car_width = 0.08 * world
        self._car_height = 0.035 * world
        self._wheel = 0.012 * world

        # the view covers the bounds of the terrain, with room above for the car and flag
        xs = np.array([self._min_pos] + self._terrain_x + [self._max_pos])
        xs = xs[(xs >= self._min_pos) & (xs <= self._max_pos)]
        ys = np.array([self._height(x) for x in xs])
        bottom = float(np.min(ys)) - 0.08 * world
        top = float(np.max(ys)) + 0.22 * world
        left, right = self._min_pos - 0.02 * world, self._max_pos + 0.02 * world
        fig_height = min(max(figure_width * (top - bottom) / (right - left), 2.5), max_height) + 0.9

        self.fig = plt.figure(figsize=(figure_width, fig_height))
        self.ax = self.fig.add_axes([0.02, 0.45 / fig_height, 0.96, 1 - 0.95 / fig_height])
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # the terrain and goal never change, so they are only drawn once
        self.ax.fill_between(xs, bottom, ys, facecolor=HILL_COLOR, edgecolor='none', zorder=0)
        self.ax.plot(xs, ys, color=GROUND_COLOR, linewidth=2, zorder=1)
        for x in (self._min_pos, self._max_pos):
            self.ax.plot([x, x], [bottom, top], color='#b0bec5', linestyle='--', linewidth=1, zorder=1)
        self._draw_flag(world)
        self._info = self.fig.text(0.5, 0.1 / fig_height, '', ha='center', va='bottom', fontsize=9, color='#37474f')

    def _height(self, x):

        # the segment holding this position, as in the domain, or else the line across a gap
        for (x1, x2, y1, y2) in self._segments:
            if x1 <= x < x2:
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        return float(np.interp(x, self._terrain_x, self._terrain_y)) if self._terrain_x else 0.0

    def _slope_angle(self, x):

        # same rule as the domain: the slope of the segments holding this position
        slope = sum((y2 - y1) / (x2 - x1) for (x1, x2, y1, y2) in self._segments if x1 <= x < x2)
        return math.atan(slope)

    def _draw_flag(self, world):
        x = self._goal_pos
        base = self._height(x)
        pole = 0.13 * world
        self.ax.plot([x, x], [base, base + pole], color=WHEEL_COLOR, linewidth=1.5, zorder=2)
        self.ax.fill([x, x, x + 0.05 * world],
                     [base + pole, base + 0.72 * pole, base + 0.86 * pole],
                     facecolor=FLAG_COLOR, edgecolor='none', zorder=2)

    def _draw_car(self, pos, vel):
        angle = self._slope_angle(pos)
        tangent = (math.cos(angle), math.sin(angle))
        normal = (-math.sin(angle), math.cos(angle))
        base = (pos, self._height(pos))

        def point(along, up):
            return (base[0] + along * tangent[0] + up * normal[0],
                    base[1] + along * tangent[1] + up * normal[1])

        # the tail follows the terrain, and stops at the bounds of the position
        tail_x = np.linspace(pos, pos - self._tail_steps * vel, 20)
        tail_x = np.clip(tail_x, self._min_pos, self._max_pos)
        tail_y = np.array([self._height(x) for x in tail_x]) + 0.6 * self._wheel
        artists = self.ax.plot(tail_x, tail_y, color=CAR_COLOR, linewidth=4, alpha=0.3,
                               solid_capstyle='round', zorder=2)

        half = self._car_width / 2
        low = 1.4 * self._wheel
        high = low + self._car_height
        corners = [point(-half, low), point(half, low), point(half, high), point(-half, high)]
        artists.extend(self.ax.fill(*zip(*corners), facecolor=CAR_COLOR, edgecolor=CAR_COLOR,
                                    linewidth=1.5, joinstyle='round', zorder=3))
        for side in (-1, 1):
            artists.append(self.ax.add_patch(plt.Circle(
                point(side * 0.28 * self._car_width, self._wheel), self._wheel,
                facecolor=WHEEL_COLOR, edgecolor='white', linewidth=0.8, zorder=4)))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        pos, vel = float(state['pos']), float(state['vel'])
        artists = self._draw_car(pos, vel)

        # same rule as the termination condition
        reached = pos >= self._goal_pos and vel >= self._goal_vel
        self.ax.set_title('Goal reached' if reached else 'Climbing to the goal', fontsize=13,
                          fontweight='bold', color=OK_COLOR if reached else 'black')
        self._info.set_text(f'Position {pos:+.3f} (goal {self._goal_pos:g}), velocity {vel:+.4f}')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
