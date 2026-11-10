import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

SKY_COLOR = '#eef3f7'
FLOOR_COLOR = '#cfd8dc'
RAIL_COLOR = '#78909c'
STOP_COLOR = '#c62828'
CART_COLOR = '#37474f'
WHEEL_COLOR = '#90a4ae'
POLE_COLOR = '#1565c0'
UP_COLOR = '#2e7d32'
TEXT_COLOR = '#37474f'

# the pole is drawn green when it is within this angle of upright
UPRIGHT_DEGREES = 12.0


class CartPoleSwingUpVisualizer(BaseViz):
    '''Draws the cart on its rail, with stops at the ends of the rail, and the pole
    on its hinge, to scale. The pole turns green when it is near upright.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_width=8.0,
                 max_height=7.0,
                 wait_time=20) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._rail_min = float(nonfluents['RAIL-MIN'])
        self._rail_max = float(nonfluents['RAIL-MAX'])
        self._pole = abs(float(nonfluents['POLE-LEN']))
        span = max(self._rail_max - self._rail_min, 1e-6)
        scale = self._pole or 0.25 * span
        self._cart_width = min(0.4 * scale, 0.5 * span)
        self._cart_height = 0.5 * self._cart_width
        self._pole_width = 0.45 * self._cart_height
        self._wheel = 0.22 * self._cart_height

        # the rail is at height 0, with room for the pole above and below it
        # and room for a level pole beyond the ends of the rail
        reach = self._pole + 0.6 * self._pole_width + 0.1 * scale
        side = max(reach, 0.5 * self._cart_width + 0.1 * scale)
        left, right = self._rail_min - side, self._rail_max + side
        bottom = -max(reach, self._cart_height + 2 * self._wheel) - 0.15 * scale
        top = max(reach, self._cart_height) + 0.1 * scale
        axes_height = 0.96 * figure_width * (top - bottom) / (right - left)
        fig_height = min(max(axes_height, 1.5), max_height) + 0.8

        self.fig = plt.figure(figsize=(figure_width, fig_height))
        self.ax = self.fig.add_axes([0.02, 0.4 / fig_height, 0.96, 1 - 0.8 / fig_height])
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')
        self.ax.set_facecolor(SKY_COLOR)
        self.fig.patch.set_facecolor(SKY_COLOR)

        # the floor, rail and stops never change, so they are only drawn once
        self.ax.add_patch(patches.Rectangle(
            (left - span, bottom - span), right - left + 2 * span, span + 0.1 * scale,
            facecolor=FLOOR_COLOR, edgecolor='none', zorder=0))
        rail_y = -0.5 * self._cart_height - 2 * self._wheel
        self.ax.plot([self._rail_min - 0.5 * self._cart_width, self._rail_max + 0.5 * self._cart_width],
                     [rail_y] * 2, color=RAIL_COLOR, linewidth=4, solid_capstyle='round', zorder=1)
        for x, side in ((self._rail_min, -1), (self._rail_max, 1)):
            stop_x = x + side * 0.5 * self._cart_width
            self.ax.plot([stop_x] * 2, [rail_y - 0.6 * self._cart_height, rail_y + 1.2 * self._cart_height],
                         color=STOP_COLOR, linewidth=4, solid_capstyle='round', zorder=1)
        self._rail_y = rail_y
        self._info = self.fig.text(0.5, 0.12 / fig_height, '', ha='center', va='bottom',
                                   fontsize=10, color=TEXT_COLOR)

    def _draw_cart(self, pos):
        artists = []
        w, h = self._cart_width, self._cart_height
        artists.append(self.ax.add_patch(patches.FancyBboxPatch(
            (pos - w / 2, -h / 2), w, h,
            boxstyle=f'round,pad=0,rounding_size={0.2 * h}',
            facecolor=CART_COLOR, edgecolor='none', zorder=3)))
        for side in (-1, 1):
            artists.append(self.ax.add_patch(plt.Circle(
                (pos + side * 0.3 * w, self._rail_y + self._wheel), self._wheel,
                facecolor=WHEEL_COLOR, edgecolor=CART_COLOR, linewidth=1.5, zorder=2)))
        return artists

    def _draw_pole(self, pos, angle, upright):
        tip = (pos + self._pole * math.sin(angle), self._pole * math.cos(angle))
        color = UP_COLOR if upright else POLE_COLOR

        # the pole is a capsule: a thick line with round caps, in data units
        points_per_unit = self.ax.transData.transform([(1, 0)])[0][0] - self.ax.transData.transform([(0, 0)])[0][0]
        width = max(self._pole_width * points_per_unit * 72 / self.fig.dpi, 2.0)
        artists = self.ax.plot([pos, tip[0]], [0, tip[1]], color=color, linewidth=width,
                               solid_capstyle='round', zorder=4)
        artists.append(self.ax.add_patch(plt.Circle(
            (pos, 0), 0.25 * self._cart_height, facecolor='white', edgecolor=CART_COLOR,
            linewidth=1.5, zorder=5)))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        pos, vel = float(state['pos']), float(state['vel'])
        angle, ang_vel = float(state['ang']), float(state['ang-vel'])

        # angle from upright in (-180, 180] degrees
        degrees = (math.degrees(angle) + 180.0) % 360.0 - 180.0
        upright = abs(degrees) <= UPRIGHT_DEGREES
        self.fig.canvas.draw()
        artists = self._draw_cart(pos)
        artists.extend(self._draw_pole(pos, angle, upright))

        past_end = pos > self._rail_max or pos < self._rail_min
        self.ax.set_title('Upright' if upright else 'Swinging', fontsize=13, fontweight='bold',
                          color=UP_COLOR if upright else POLE_COLOR)
        self._info.set_text(
            f'pole {degrees:+.0f} deg from upright, {ang_vel:+.2f} rad/s   |   '
            f'cart {pos:+.2f} m, {vel:+.2f} m/s' + ('   |   at the rail end' if past_end else ''))

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
