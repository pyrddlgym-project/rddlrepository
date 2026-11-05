import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BELIEF_COLOR = '#64b5f6'
TRUTH_COLOR = '#212121'
MEAN_COLOR = '#0d47a1'
WIN_COLOR = '#c8e6c9'
LOSS_COLOR = '#ffcdd2'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class BernoulliBanditVisualizer(BaseViz):
    '''Draws the current belief about each arm, as the density of its Beta posterior
    next to the true probability of success, and the results of the pulls so far.

    Each density is scaled to the height of its row, so that its width shows how
    uncertain the belief is. The counts come from the state alone: only the
    highlight on the most recently pulled arm depends on the previous render.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_width=9.0,
                 row_height=0.8,
                 max_height=10.0,
                 num_points=800,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._arms = model.type_to_objects['arm']
        self._theta = {a: nonfluents[RDDLPlanningModel.ground_var('TRUE_THETA', [a])]
                       for a in self._arms}
        self._best = max(self._theta.values())
        self._init_state = self._get_state(model.ground_vars_with_values(model.state_fluents))

        # the density is infinite at the ends when a parameter is below one
        self._grid = (np.arange(num_points) + 0.5) / num_points

        self._last_state = None
        self._newest = {}

        # the margins hold the arm names on the left and the results on the right,
        # and many arms use shorter rows to fit on screen
        num_arms = len(self._arms)
        left, right, bottom, top = 1.3, 1.3, 1.05, 0.5
        row_height = min(row_height, (max_height - bottom - top) / num_arms)
        self._row_height = row_height
        fig_height = num_arms * row_height + bottom + top

        self.fig = plt.figure(figsize=(figure_width, fig_height))
        self.ax = self.fig.add_axes(
            [left / figure_width, bottom / fig_height,
             (figure_width - left - right) / figure_width, num_arms * row_height / fig_height])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, num_arms)
        self.ax.set_yticks([])
        self.ax.set_xlabel('probability of success', fontsize=10)
        self.ax.tick_params(axis='x', labelsize=9)

        # the first arm is on the top row: rows and their names never change
        self._rows = {a: num_arms - 1 - i for (i, a) in enumerate(self._arms)}
        self._text_size = min(10, 72 * 0.75 * row_height)
        for a in self._arms:
            row = self._rows[a]
            if row > 0:
                self.ax.axhline(row, color='#e0e0e0', linewidth=0.8, zorder=1)
            self.ax.plot([self._theta[a]] * 2, [row, row + 0.92], color=TRUTH_COLOR,
                         linewidth=1.5, zorder=4)
            font_size = min(self._text_size, 72 * 1.15 / (0.62 * len(a)))
            if font_size >= MIN_FONT_SIZE:
                self.ax.text(-0.015, row + 0.5, a, ha='right', va='center', fontsize=font_size,
                             fontweight='bold' if self._theta[a] == self._best else 'normal')

        self.ax.legend(handles=[
            patches.Patch(facecolor=BELIEF_COLOR, label='Belief'),
            Line2D([], [], color=TRUTH_COLOR, linewidth=1.5, label='True probability'),
            Line2D([], [], marker='o', color='none', markerfacecolor=MEAN_COLOR,
                   markeredgecolor='none', label='Belief mean'),
            patches.Patch(facecolor=WIN_COLOR, label='Last pull won'),
            patches.Patch(facecolor=LOSS_COLOR, label='Last pull lost')
        ], loc='upper center', bbox_to_anchor=(0.5, -0.5 / (num_arms * row_height)), ncol=5,
           frameon=False, fontsize=8.5, columnspacing=1.2, handlelength=1.5)
        self.fig.text(0.985, 0.02, 'Bold name: best arm', ha='right', va='bottom', fontsize=8)

    def _get_state(self, state):
        return {a: (float(state[RDDLPlanningModel.ground_var('alpha', [a])]),
                    float(state[RDDLPlanningModel.ground_var('beta', [a])]))
                for a in self._arms}

    def _update_newest(self, state):

        # rendering the same state again must not change the highlight
        if state == self._last_state:
            return
        last_state, self._last_state = self._last_state, state

        # the counts only ever grow during an episode, so a smaller one means a new one
        if last_state is None or state == self._init_state or any(
                state[a][0] < last_state[a][0] or state[a][1] < last_state[a][1]
                for a in self._arms):
            self._newest = {}
            return
        self._newest = {a: state[a][0] > last_state[a][0] for a in self._arms
                        if state[a] != last_state[a]}

    def _belief_density(self, alpha, beta):
        if alpha <= 0 or beta <= 0:
            return None
        log_density = ((alpha - 1) * np.log(self._grid) + (beta - 1) * np.log1p(-self._grid))
        return np.exp(log_density - np.max(log_density))

    def _draw_arm(self, arm, alpha, beta):
        row = self._rows[arm]
        artists = []
        if arm in self._newest:
            artists.append(self.ax.add_patch(patches.Rectangle(
                (0, row), 1, 1, facecolor=WIN_COLOR if self._newest[arm] else LOSS_COLOR,
                edgecolor='none', zorder=0)))

        density = self._belief_density(alpha, beta)
        if density is not None:
            artists.append(self.ax.fill_between(
                self._grid, row, row + 0.88 * density, facecolor=BELIEF_COLOR,
                edgecolor=MEAN_COLOR, linewidth=0.6, alpha=0.8, zorder=2))
            mean = alpha / (alpha + beta)
            artists.extend(self.ax.plot(
                [mean], [row + 0.1], marker='o', markersize=min(6, 12 * self._row_height),
                color=MEAN_COLOR, markeredgecolor='white', markeredgewidth=0.5, zorder=5))

        # results since the start of the episode
        wins = alpha - self._init_state[arm][0]
        losses = beta - self._init_state[arm][1]
        text = f'{wins:g} / {wins + losses:g} won'
        font_size = min(self._text_size, 72 * 1.15 / (0.62 * len(text)))
        if font_size >= MIN_FONT_SIZE:
            artists.append(self.ax.text(
                1.015, row + 0.5, text, ha='left', va='center', fontsize=font_size))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        state = self._get_state(state)
        self._update_newest(state)

        artists = []
        for a in self._arms:
            artists.extend(self._draw_arm(a, *state[a]))

        wins = sum(state[a][0] - self._init_state[a][0] for a in self._arms)
        pulls = {a: state[a][0] + state[a][1] - sum(self._init_state[a]) for a in self._arms}
        regret = sum(pulls[a] * (self._best - self._theta[a]) for a in self._arms)
        self.ax.set_title(f'{sum(pulls.values()):g} pulls, {wins:g} won, '
                          f'expected regret {regret:.1f}', fontsize=13, fontweight='bold')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
