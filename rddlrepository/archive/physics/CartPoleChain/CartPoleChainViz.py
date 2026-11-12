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
POLE_COLORS = ['#1565c0', '#6a1b9a', '#00838f', '#ad1457', '#4527a0', '#00695c']
UP_COLOR = '#2e7d32'
TEXT_COLOR = '#37474f'

# a pole is drawn green when it is within this angle of upright
UPRIGHT_DEGREES = 12.0


class CartPoleChainVisualizer(BaseViz):
    '''Draws the cart on its rail, with stops at the ends of the rail, and the chain of poles on their hinges, to
    scale. A pole turns green when it is near upright.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_width=8.0,
                 max_height=7.0,
                 wait_time=20) -> None:
        self._model = model
        self._wait_time = wait_time

        nf = model.ground_vars_with_values(model.non_fluents)
        self._rail_min = float(nf['RAIL-MIN'])
        self._rail_max = float(nf['RAIL-MAX'])
        poles = list(model.type_to_objects['pole'])
        poles.sort(key=lambda p: int(nf[RDDLPlanningModel.ground_var('INDEX', [p])]))
        self._poles = [dict(name=p, ang=RDDLPlanningModel.ground_var('ang', [p]),
                            vel=RDDLPlanningModel.ground_var('ang-vel', [p]),
                            len=abs(float(nf[RDDLPlanningModel.ground_var('POLE-LEN', [p])])))
                       for p in poles]
        self._reach = sum(p['len'] for p in self._poles)
        span = max(self._rail_max - self._rail_min, 1e-6)
        first = self._poles[0]['len'] if self._poles else 0.25 * span
        scale = first or 0.25 * span
        self._cart_width = min(0.4 * scale, 0.5 * span)
        self._cart_height = 0.5 * self._cart_width
        self._pole_width = 0.45 * self._cart_height
        self._wheel = 0.22 * self._cart_height

        # the rail is at height 0, with room for the chain above and below it and beyond the ends of the rail
        reach = self._reach + 0.6 * self._pole_width + 0.1 * scale
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
        for x, s in ((self._rail_min, -1), (self._rail_max, 1)):
            stop_x = x + s * 0.5 * self._cart_width
            self.ax.plot([stop_x] * 2, [rail_y - 0.6 * self._cart_height, rail_y + 1.2 * self._cart_height],
                         color=STOP_COLOR, linewidth=4, solid_capstyle='round', zorder=1)
        self._rail_y = rail_y
        self._info = self.fig.text(0.5, 0.12 / fig_height, '', ha='center', va='bottom',
                                   fontsize=10, color=TEXT_COLOR)

    def _draw_cart(self, pos):
        artists = []
        w, h = self._cart_width, self._cart_height
        artists.append(self.ax.add_patch(patches.FancyBboxPatch(
            (pos - w / 2, -h / 2), w, h, boxstyle=f'round,pad=0,rounding_size={0.2 * h}',
            facecolor=CART_COLOR, edgecolor='none', zorder=3)))
        for s in (-1, 1):
            artists.append(self.ax.add_patch(plt.Circle(
                (pos + s * 0.3 * w, self._rail_y + self._wheel), self._wheel,
                facecolor=WHEEL_COLOR, edgecolor=CART_COLOR, linewidth=1.5, zorder=2)))
        return artists

    def _draw_chain(self, pos, angles, upright):
        points_per_unit = self.ax.transData.transform([(1, 0)])[0][0] - self.ax.transData.transform([(0, 0)])[0][0]
        width = max(self._pole_width * points_per_unit * 72 / self.fig.dpi, 2.0)
        artists = []
        x, y = pos, 0.0
        hinges = []
        for i, (pole, angle) in enumerate(zip(self._poles, angles)):
            tip = (x + pole['len'] * math.sin(angle), y + pole['len'] * math.cos(angle))
            color = UP_COLOR if upright[i] else POLE_COLORS[i % len(POLE_COLORS)]
            artists += self.ax.plot([x, tip[0]], [y, tip[1]], color=color, linewidth=width,
                                    solid_capstyle='round', zorder=4 + i)
            hinges.append((x, y))
            x, y = tip
        for i, (hx, hy) in enumerate(hinges):
            artists.append(self.ax.add_patch(plt.Circle(
                (hx, hy), 0.25 * self._cart_height, facecolor='white', edgecolor=CART_COLOR,
                linewidth=1.5, zorder=5 + len(hinges))))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        pos, vel = float(state['pos']), float(state['vel'])
        angles = [float(state[p['ang']]) for p in self._poles]

        # angles from upright in (-180, 180] degrees
        degrees = [(math.degrees(a) + 180.0) % 360.0 - 180.0 for a in angles]
        upright = [abs(d) <= UPRIGHT_DEGREES for d in degrees]
        self.fig.canvas.draw()
        artists = self._draw_cart(pos)
        artists.extend(self._draw_chain(pos, angles, upright))

        n, up = len(self._poles), sum(upright)
        if n and up == n:
            title, color = 'Upright', UP_COLOR
        else:
            title, color = f'Swinging: {up} of {n} poles upright', POLE_COLORS[0]
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)
        shown = ', '.join(f'{d:+.0f}' for d in degrees[:6]) + (', ...' if n > 6 else '')
        past_end = pos > self._rail_max or pos < self._rail_min
        self._info.set_text(f'poles {shown} deg from upright   |   cart {pos:+.2f} m, {vel:+.2f} m/s'
                            + ('   |   at the rail end' if past_end else ''))

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
