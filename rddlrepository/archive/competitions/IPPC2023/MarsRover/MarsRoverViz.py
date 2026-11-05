import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

MINERAL_COLOR = '#c8e6c9'
MINERAL_EDGE_COLOR = '#2e7d32'
HARVESTED_EDGE_COLOR = '#90a4ae'
ROVER_COLOR = '#1565c0'
IN_REACH_COLOR = '#ff8f00'
OK_COLOR = '#2e7d32'

# values are left out for minerals smaller than this on screen, in inches
MIN_LABEL_SIZE = 0.35


class MarsRoverVisualizer(BaseViz):
    '''Draws the area and value of each mineral, and each rover with a tail showing its
    recent motion. A rover is highlighted while it can harvest a mineral. The view
    covers the minerals and the current positions of the rovers, and the drawing only
    depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(5.4, 5.9),
                 tail_steps=5,
                 wait_time=100) -> None:
        self._model = model
        self._tail_steps = tail_steps
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._rovers = model.type_to_objects['rover']
        self._minerals = model.type_to_objects['mineral']
        self._time_scale = self._nonfluents['SCALE-FACTOR']
        self._centers = {m: (self._get(self._nonfluents, 'MINERAL-POS-X', m),
                             self._get(self._nonfluents, 'MINERAL-POS-Y', m))
                         for m in self._minerals}
        self._areas = {m: self._get(self._nonfluents, 'MINERAL-AREA', m) for m in self._minerals}
        self._values = {m: self._get(self._nonfluents, 'MINERAL-VALUE', m) for m in self._minerals}
        self._plot_size = 0.86 * figure_size[0]

        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_axes([0.1, 0.12, 0.86, 0.79])
        self.ax.set_aspect('equal')
        self.ax.tick_params(labelsize=8, colors='#616161')
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')

        self.fig.legend(handles=[
            patches.Patch(facecolor=MINERAL_COLOR, edgecolor=MINERAL_EDGE_COLOR, label='Mineral'),
            patches.Patch(facecolor='none', edgecolor=HARVESTED_EDGE_COLOR, linestyle='--',
                          label='Harvested'),
            Line2D([], [], marker='s', color='none', markerfacecolor=ROVER_COLOR,
                   markeredgecolor='none', label='Rover'),
            Line2D([], [], marker='s', color='none', markerfacecolor=IN_REACH_COLOR,
                   markeredgecolor='none', label='Can harvest')
        ], loc='lower center', ncol=4, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _get(self, values, var, obj):
        return float(values[RDDLPlanningModel.ground_var(var, [obj])])

    def _update_view(self, positions):

        # both axes use the same scale, with a small margin around everything
        boxes = [(x, x, y, y) for (x, y) in positions.values()]
        boxes.extend((x - self._areas[m], x + self._areas[m], y - self._areas[m], y + self._areas[m])
                     for (m, (x, y)) in self._centers.items())
        left, right = min(b[0] for b in boxes), max(b[1] for b in boxes)
        bottom, top = min(b[2] for b in boxes), max(b[3] for b in boxes)
        span = 1.1 * max(right - left, top - bottom) or 1.0
        x_center, y_center = (left + right) / 2, (bottom + top) / 2
        self.ax.set_xlim(x_center - span / 2, x_center + span / 2)
        self.ax.set_ylim(y_center - span / 2, y_center + span / 2)
        return span

    def _draw_mineral(self, mineral, harvested, span):
        x, y = self._centers[mineral]
        area = self._areas[mineral]
        artists = [self.ax.add_patch(plt.Circle(
            (x, y), area, facecolor='none' if harvested else MINERAL_COLOR,
            edgecolor=HARVESTED_EDGE_COLOR if harvested else MINERAL_EDGE_COLOR,
            linestyle='--' if harvested else '-', linewidth=1.2, zorder=1))]
        if 2 * area * self._plot_size / span >= MIN_LABEL_SIZE:
            artists.append(self.ax.text(
                x, y, f'{self._values[mineral]:g}', ha='center', va='center', fontsize=9,
                color=HARVESTED_EDGE_COLOR if harvested else MINERAL_EDGE_COLOR,
                fontweight='bold', zorder=2))
        return artists

    def _can_harvest(self, position, available):

        # same rule as the domain
        return any((position[0] - self._centers[m][0]) ** 2 + (position[1] - self._centers[m][1]) ** 2
                   < self._areas[m] ** 2 for m in available)

    def _draw_rover(self, rover, position, state, available):
        x, y = position
        color = IN_REACH_COLOR if self._can_harvest(position, available) else ROVER_COLOR
        length = self._tail_steps * self._time_scale
        artists = self.ax.plot(
            [x, x - length * self._get(state, 'vel-x', rover)],
            [y, y - length * self._get(state, 'vel-y', rover)],
            color=color, linewidth=2.5, alpha=0.4, solid_capstyle='round', zorder=3)
        artists.extend(self.ax.plot(x, y, marker='s', markersize=9, color=color,
                                    markeredgecolor='white', zorder=4))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        positions = {r: (self._get(state, 'pos-x', r), self._get(state, 'pos-y', r))
                     for r in self._rovers}
        harvested = {m: bool(state[RDDLPlanningModel.ground_var('mineral-harvested', [m])])
                     for m in self._minerals}
        available = [m for m in self._minerals if not harvested[m]]
        span = self._update_view(positions)

        artists = []
        for m in self._minerals:
            artists.extend(self._draw_mineral(m, harvested[m], span))
        for r in self._rovers:
            artists.extend(self._draw_rover(r, positions[r], state, available))

        collected = sum(self._values[m] for m in self._minerals if harvested[m])
        total = sum(self._values.values())
        self.ax.set_title(
            f'{len(self._minerals) - len(available)} / {len(self._minerals)} minerals harvested, '
            f'value {collected:g} / {total:g}', fontsize=13, fontweight='bold',
            color=OK_COLOR if not available else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
