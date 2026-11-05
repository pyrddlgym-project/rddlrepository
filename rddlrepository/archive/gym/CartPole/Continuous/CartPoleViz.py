import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

TRACK_COLOR = '#37474f'
CART_COLOR = '#37474f'
POLE_COLOR = '#1565c0'
WARNING_COLOR = '#ff8f00'
LIMIT_COLOR = '#c62828'
OK_COLOR = '#2e7d32'

# the pole is drawn as being close to its limit at this fraction of the limit
WARNING_FRACTION = 0.75


class CartPoleVisualizer(BaseViz):
    '''Draws the cart and pole to scale with the limits on their position and angle,
    with a tail on the pole and cart showing their recent motion. The drawing only
    depends on the current state.
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
        self._pos_limit = nonfluents['POS-LIMIT']
        self._ang_limit = nonfluents['ANG-LIMIT']
        self._time_step = nonfluents['TIME-STEP']

        # POLE-LEN is half of the length of the pole
        self._pole = 2 * nonfluents['POLE-LEN']
        world = 2 * self._pos_limit
        self._cart_width = min(0.1 * world, 0.5 * self._pole) or 0.1 * world or 1.0
        self._cart_height = 0.5 * self._cart_width
        self._wheel = 0.14 * self._cart_width
        self._pivot = 2 * self._wheel + self._cart_height

        # the limits are a little inside the view, and the ground is at zero
        half_width = 1.1 * self._pos_limit + 0.6 * self._cart_width
        bottom = -0.8 * self._cart_height
        top = self._pivot + 1.15 * self._pole
        fig_height = min(max(figure_width * (top - bottom) / (2 * half_width), 2.5), max_height) + 0.9

        self.fig = plt.figure(figsize=(figure_width, fig_height))
        self.ax = self.fig.add_axes([0.02, 0.45 / fig_height, 0.96, 1 - 0.95 / fig_height])
        self.ax.set_xlim(-half_width, half_width)
        self.ax.set_ylim(bottom, top)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # the track and its limits never change, so they are only drawn once
        self.ax.axhline(0, color=TRACK_COLOR, linewidth=2, zorder=1)
        for side in (-1, 1):
            self.ax.plot([side * self._pos_limit] * 2, [bottom, top], color=LIMIT_COLOR,
                         linestyle='--', linewidth=1.2, zorder=1)
        self._info = self.fig.text(0.5, 0.1 / fig_height, '', ha='center', va='bottom', fontsize=9, color='#37474f')

    def _pole_direction(self, angle):
        return math.sin(angle), math.cos(angle)

    def _pole_color(self, angle):
        if abs(angle) > self._ang_limit:
            return LIMIT_COLOR
        return WARNING_COLOR if abs(angle) > WARNING_FRACTION * self._ang_limit else POLE_COLOR

    def _draw_cart(self, pos, vel):
        artists = []
        artists.append(self.ax.add_patch(patches.FancyBboxPatch(
            (pos - self._cart_width / 2, 2 * self._wheel), self._cart_width, self._cart_height,
            boxstyle=f'round,pad=0,rounding_size={0.2 * self._cart_height}',
            facecolor=CART_COLOR, edgecolor='none', zorder=3)))
        for side in (-1, 1):
            artists.append(self.ax.add_patch(plt.Circle(
                (pos + side * 0.3 * self._cart_width, self._wheel), self._wheel,
                facecolor='#90a4ae', edgecolor=CART_COLOR, zorder=4)))

        # the tail lies under the cart, along the track
        artists.extend(self.ax.plot(
            [pos, pos - self._tail_steps * self._time_step * vel],
            [-0.35 * self._cart_height] * 2, color=CART_COLOR, linewidth=3, alpha=0.4,
            solid_capstyle='round', zorder=2))
        return artists

    def _draw_pole(self, pos, angle, ang_vel):
        artists = []
        pivot = (pos, self._pivot)

        # the limits on the angle move with the cart
        for side in (-1, 1):
            dx, dy = self._pole_direction(side * self._ang_limit)
            artists.extend(self.ax.plot(
                [pivot[0], pivot[0] + 1.1 * self._pole * dx],
                [pivot[1], pivot[1] + 1.1 * self._pole * dy],
                color=LIMIT_COLOR, linestyle='--', linewidth=1, zorder=1))

        dx, dy = self._pole_direction(angle)
        color = self._pole_color(angle)
        tip = (pivot[0] + self._pole * dx, pivot[1] + self._pole * dy)
        artists.extend(self.ax.plot(
            [pivot[0], tip[0]], [pivot[1], tip[1]], color=color, linewidth=6,
            solid_capstyle='round', zorder=5))
        artists.append(self.ax.add_patch(plt.Circle(
            pivot, 0.5 * self._wheel, facecolor='white', edgecolor=CART_COLOR, zorder=6)))

        # the tail follows the circle the tip moves along
        angles = angle - np.linspace(0, self._tail_steps * self._time_step * ang_vel, 12)
        artists.extend(self.ax.plot(
            pivot[0] + self._pole * np.sin(angles), pivot[1] + self._pole * np.cos(angles),
            color=color, linewidth=3, alpha=0.35, solid_capstyle='round', zorder=4))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        pos, vel = float(state['pos']), float(state['vel'])
        angle, ang_vel = float(state['ang-pos']), float(state['ang-vel'])

        artists = self._draw_cart(pos, vel)
        artists.extend(self._draw_pole(pos, angle, ang_vel))

        # same rule as the termination condition
        out = abs(pos) > self._pos_limit
        fallen = abs(angle) > self._ang_limit
        title = 'Out of bounds' if out else ('Fell over' if fallen else 'Balanced')
        self.ax.set_title(title, fontsize=13, fontweight='bold',
                          color=LIMIT_COLOR if out or fallen else OK_COLOR)
        self._info.set_text(
            f'Angle {math.degrees(angle):+.1f} deg (limit {math.degrees(self._ang_limit):.1f}), '
            f'position {pos:+.2f} (limit {self._pos_limit:g})')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
