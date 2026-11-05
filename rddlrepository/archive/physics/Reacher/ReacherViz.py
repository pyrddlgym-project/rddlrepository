import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

GUIDE_COLOR = '#cfd8dc'
SEGMENT_COLORS = ('#1565c0', '#42a5f5')
PIVOT_COLOR = '#37474f'
TARGET_COLOR = '#2e7d32'
TIP_COLOR = '#ff8f00'


class ReacherVisualizer(BaseViz):
    '''Draws the arm in the order given by POS, how far it can reach, the target, and
    the tips that the reward measures against the target. The drawing only depends
    on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(5.4, 5.9),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._segments = sorted(
            model.type_to_objects['segment'],
            key=lambda s: nonfluents[RDDLPlanningModel.ground_var('POS', [s])])
        self._target_tips = [s for s in self._segments if nonfluents[
            RDDLPlanningModel.ground_var('TARGET-SEGMENT', [s])]]
        self._target = (nonfluents['TARGET-X'], nonfluents['TARGET-Y'])
        reach = sum(nonfluents[RDDLPlanningModel.ground_var('LENGTH', [s])] for s in self._segments)

        # the view covers the reach of the arm and the target
        self._limit = 1.12 * max(reach, math.hypot(*self._target)) or 1.0

        # arms with many segments use thinner lines to stay readable
        self._line_width = min(7, 70 / len(self._segments))
        self._marker_size = min(9, 90 / len(self._segments))

        _, fig_height = figure_size
        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_axes([0.02, 0.5 / fig_height, 0.96, 1 - 1.0 / fig_height])
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_xlim(-self._limit, self._limit)
        self.ax.set_ylim(-self._limit, self._limit)

        # the reach and target never change, so they are only drawn once
        self.ax.add_patch(plt.Circle((0, 0), reach, facecolor='none', edgecolor=GUIDE_COLOR,
                                     linestyle='--', linewidth=1, zorder=0))
        self.ax.plot(*self._target, marker='*', markersize=17, color=TARGET_COLOR,
                     markeredgecolor='white', zorder=2)
        self.fig.legend(handles=[
            Line2D([], [], marker='*', color='none', markerfacecolor=TARGET_COLOR,
                   markeredgecolor='none', markersize=11, label='Target'),
            Line2D([], [], marker='o', color='none', markerfacecolor=TIP_COLOR,
                   markeredgecolor='white', label='Tip measured by the reward')
        ], loc='lower center', ncol=2, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _get_tip(self, state, segment):
        return (float(state[RDDLPlanningModel.ground_var('tip-x', [segment])]),
                float(state[RDDLPlanningModel.ground_var('tip-y', [segment])]))

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        tips = {s: self._get_tip(state, s) for s in self._segments}

        # draw the arm outwards from the origin
        artists = []
        start = (0.0, 0.0)
        for (i, s) in enumerate(self._segments):
            end = tips[s]
            artists.extend(self.ax.plot(
                [start[0], end[0]], [start[1], end[1]], linewidth=self._line_width,
                color=SEGMENT_COLORS[i % len(SEGMENT_COLORS)], solid_capstyle='round', zorder=3))
            artists.extend(self.ax.plot(
                *end, marker='o', markersize=self._marker_size, color=PIVOT_COLOR,
                markeredgecolor='white', zorder=4))
            start = end
        artists.extend(self.ax.plot(
            0, 0, marker='o', markersize=1.3 * self._marker_size, color=PIVOT_COLOR,
            markeredgecolor='white', zorder=5))

        # same rule as the reward, which sums over all tips measured against the target
        penalty = 0.0
        for s in self._target_tips:
            x, y = tips[s]
            penalty += (x - self._target[0]) ** 2 + (y - self._target[1]) ** 2
            artists.extend(self.ax.plot(
                [x, self._target[0]], [y, self._target[1]], color=TIP_COLOR, linestyle='--',
                linewidth=1.2, zorder=2))
            artists.extend(self.ax.plot(
                x, y, marker='o', markersize=1.4 * self._marker_size, color=TIP_COLOR,
                markeredgecolor='white', zorder=6))

        self.ax.set_title(f'Squared distance to target: {penalty:.3f}',
                          fontsize=13, fontweight='bold')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
