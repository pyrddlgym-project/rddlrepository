import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

AIRCRAFT_COLORS = ('#1565c0', '#ef6c00', '#2e7d32', '#6a1b9a', '#00838f', '#ad1457', '#5d4037')
OTHER_COLOR = '#90a4ae'
LEGEND_COLOR = '#37474f'

# the domain does not give aircraft a size, so they are drawn at this fraction of the view
AIRCRAFT_FRACTION = 0.05

# names are left out with more aircraft than this, as they would overlap
MAX_LABELS = 12


class UAVsVisualizer(BaseViz):
    '''Draws each aircraft with its heading and roll, its goal in the same color, and
    its path during the episode. Only controllable aircraft have a goal in the reward,
    so the others are drawn in grey without one.

    The view covers the goals and the paths and grows as aircraft leave it. The paths
    and view are built from the states seen so far, so they require render to be
    called on every step.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(6.6, 6.8),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._init_state = model.ground_vars_with_values(model.state_fluents)
        self._aircraft = model.type_to_objects['aircraft']
        self._controlled = [a for a in self._aircraft if self._nonfluents[
            RDDLPlanningModel.ground_var('CONTROLLABLE', [a])]]
        self._colors = {a: OTHER_COLOR for a in self._aircraft}
        for (i, a) in enumerate(self._controlled):
            self._colors[a] = AIRCRAFT_COLORS[i % len(AIRCRAFT_COLORS)]
        self._goals = {a: np.array([self._get(self._nonfluents, var, a)
                                    for var in ('GOAL-X', 'GOAL-Y', 'GOAL-Z')])
                       for a in self._controlled}

        self._paths = {a: [] for a in self._aircraft}
        self._last_state = None
        self._bounds = None
        self._span = 1.0

        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_axes([0.02, 0.07, 0.96, 0.85], projection='3d')
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlabel('x', fontsize=9)
        self.ax.set_ylabel('y', fontsize=9)
        self.ax.set_zlabel('z', fontsize=9)
        self.ax.tick_params(labelsize=8, colors='#616161')
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0.96, 0.97, 0.98, 1.0))

        self.fig.legend(handles=[
            Line2D([], [], marker='o', color=LEGEND_COLOR, markerfacecolor=LEGEND_COLOR,
                   label='Controllable'),
            Line2D([], [], marker='o', color=OTHER_COLOR, markerfacecolor=OTHER_COLOR,
                   label='Not controllable'),
            Line2D([], [], marker='*', color='none', markerfacecolor=LEGEND_COLOR,
                   markeredgecolor='none', markersize=11, label='Goal'),
            Line2D([], [], color=LEGEND_COLOR, alpha=0.45, linewidth=2, label='Path')
        ], loc='lower center', ncol=4, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _get(self, values, var, aircraft):
        return float(values[RDDLPlanningModel.ground_var(var, [aircraft])])

    def _position(self, state, aircraft):
        return np.array([self._get(state, var, aircraft) for var in ('pos-x', 'pos-y', 'pos-z')])

    def _body_axes(self, state, aircraft):
        psi = self._get(state, 'psi', aircraft)
        theta = self._get(state, 'theta', aircraft)
        phi = self._get(state, 'phi', aircraft)

        # the direction of motion in the domain, and the wings from the roll: a positive
        # roll turns the aircraft to its left, so its right wing is the one raised
        forward = np.array([math.cos(psi) * math.cos(theta),
                            math.sin(psi) * math.cos(theta),
                            math.sin(theta)])
        right = np.array([math.sin(psi), -math.cos(psi), 0.0])
        up = np.cross(right, forward)
        return forward, math.cos(phi) * right + math.sin(phi) * up

    def _update_history(self, state):

        # rendering the same state again must not change the paths or view
        if state == self._last_state:
            return
        new_episode = self._last_state is None or state == self._init_state
        self._last_state = dict(state)

        positions = {a: self._position(state, a) for a in self._aircraft}
        current = np.array(list(positions.values()))
        if new_episode:
            self._paths = {a: [] for a in self._aircraft}
            points = np.vstack([current] + [goal.reshape((1, 3)) for goal in self._goals.values()])
            self._bounds = (points.min(axis=0), points.max(axis=0))

        for a in self._aircraft:
            self._paths[a].append(positions[a])
        self._bounds = (np.minimum(self._bounds[0], current.min(axis=0)),
                        np.maximum(self._bounds[1], current.max(axis=0)))

    def _update_view(self):

        # all axes use the same scale, with a margin for the size of the aircraft
        low, high = self._bounds
        span = 1.2 * float(np.max(high - low))
        self._span = span if span > 0 else 1.0
        center = (low + high) / 2
        self.ax.set_xlim3d(center[0] - self._span / 2, center[0] + self._span / 2)
        self.ax.set_ylim3d(center[1] - self._span / 2, center[1] + self._span / 2)
        self.ax.set_zlim3d(center[2] - self._span / 2, center[2] + self._span / 2)

    def _draw_aircraft(self, state, aircraft):
        color = self._colors[aircraft]
        position = self._position(state, aircraft)
        forward, wing = self._body_axes(state, aircraft)
        size = AIRCRAFT_FRACTION * self._span

        path = np.array(self._paths[aircraft])
        artists = self.ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color,
                               alpha=0.45, linewidth=1.5)

        # the body points in the direction of motion, and the wings show the roll
        nose = position + size * forward
        tail = position - 0.6 * size * forward
        artists.extend(self.ax.plot(*zip(tail, nose), color=color, linewidth=3))
        artists.extend(self.ax.plot(
            *zip(position - 0.8 * size * wing, position + 0.8 * size * wing),
            color=color, linewidth=2))
        artists.extend(self.ax.plot(*nose, marker='o', markersize=5, color=color,
                                    markeredgecolor='white', linestyle='none'))

        # the goal and how far the aircraft is from it
        if aircraft in self._goals:
            goal = self._goals[aircraft]
            artists.extend(self.ax.plot(*zip(position, goal), color=color, linestyle=':', linewidth=1))
            artists.extend(self.ax.plot(*goal, marker='*', markersize=13, color=color,
                                        markeredgecolor='white', linestyle='none'))
        if len(self._aircraft) <= MAX_LABELS:
            artists.append(self.ax.text(*(position + 1.4 * size * np.array((0, 0, 1))),
                                        aircraft, fontsize=8))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        self._update_history(state)
        self._update_view()

        artists = []
        for a in self._aircraft:
            artists.extend(self._draw_aircraft(state, a))

        # same rule as the reward
        distance = sum(np.linalg.norm(self._position(state, a) - self._goals[a])
                       for a in self._controlled)
        self.ax.set_title(f'Total distance to goals: {distance:.2f}', fontsize=13, fontweight='bold')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
