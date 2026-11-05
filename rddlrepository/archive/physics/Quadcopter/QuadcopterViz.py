import itertools

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

DRONE_COLORS = ('#1565c0', '#ef6c00', '#2e7d32', '#6a1b9a', '#00838f', '#ad1457', '#5d4037', '#37474f')
COLLISION_COLOR = '#c62828'

# drones are small compared to the distances they fly, so that their tilt can still be
# seen their arms are drawn at least at this fraction of the size of the view
MIN_ARM_FRACTION = 0.04

# names are left out with more drones than this, as they would overlap
MAX_LABELS = 12


class QuadcopterVisualizer(BaseViz):
    '''Draws each drone with its propeller arms and thrust direction, its target in the
    same color, and its path during the episode.

    The view covers the targets and the paths and grows as drones leave it. The paths
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
        self._drones = model.type_to_objects['drone']
        self._colors = {d: DRONE_COLORS[i % len(DRONE_COLORS)]
                        for (i, d) in enumerate(self._drones)}
        self._targets = {d: np.array([self._get(self._nonfluents, var, d)
                                      for var in ('TX', 'TY', 'TZ')]) for d in self._drones}

        self._paths = {d: [] for d in self._drones}
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
            Line2D([], [], marker='o', color='#37474f', markerfacecolor='#37474f', label='Drone'),
            Line2D([], [], marker='*', color='none', markerfacecolor='#37474f',
                   markeredgecolor='none', markersize=11, label='Target'),
            Line2D([], [], color='#37474f', alpha=0.45, linewidth=2, label='Path')
        ], loc='lower center', ncol=3, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _get(self, values, var, drone):
        return float(values[RDDLPlanningModel.ground_var(var, [drone])])

    def _position(self, state, drone):
        return np.array([self._get(state, var, drone) for var in ('x', 'y', 'z')])

    def _rotation(self, state, drone):
        phi, theta, psi = [self._get(state, var, drone) for var in ('phi', 'theta', 'psi')]
        cph, cth, cps = np.cos(phi), np.cos(theta), np.cos(psi)
        sph, sth, sps = np.sin(phi), np.sin(theta), np.sin(psi)

        # the last column is the direction of thrust in the domain
        return np.array([
            [cph * cps - cth * sph * sps, -cps * sph - cph * cth * sps, sth * sps],
            [cth * cps * sph + cph * sps, cph * cth * cps - sph * sps, -cps * sth],
            [sph * sth, cph * sth, cth]
        ])

    def _update_history(self, state):

        # rendering the same state again must not change the paths or view
        if state == self._last_state:
            return
        new_episode = self._last_state is None or state == self._init_state
        self._last_state = dict(state)

        if new_episode:
            self._paths = {d: [] for d in self._drones}
            targets = np.array(list(self._targets.values()))
            self._bounds = (targets.min(axis=0), targets.max(axis=0))

        positions = {d: self._position(state, d) for d in self._drones}
        for d in self._drones:
            self._paths[d].append(positions[d])
        current = np.array(list(positions.values()))
        self._bounds = (np.minimum(self._bounds[0], current.min(axis=0)),
                        np.maximum(self._bounds[1], current.max(axis=0)))

    def _update_view(self):

        # all axes use the same scale, with a margin for the arms of the drones
        low, high = self._bounds
        span = 1.2 * float(np.max(high - low))
        self._span = span if span > 0 else 1.0
        center = (low + high) / 2
        self.ax.set_xlim3d(center[0] - self._span / 2, center[0] + self._span / 2)
        self.ax.set_ylim3d(center[1] - self._span / 2, center[1] + self._span / 2)
        self.ax.set_zlim3d(center[2] - self._span / 2, center[2] + self._span / 2)

    def _draw_drone(self, state, drone, colliding):
        color = self._colors[drone]
        position = self._position(state, drone)
        rotation = self._rotation(state, drone)
        arm = max(self._nonfluents['L'], MIN_ARM_FRACTION * self._span)

        path = np.array(self._paths[drone])
        artists = self.ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color,
                               alpha=0.45, linewidth=1.5)

        # each propeller is at the end of an arm along the body axes
        for axis in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
            motor = position + rotation @ (arm * np.array(axis))
            artists.extend(self.ax.plot(*zip(position, motor), color=color, linewidth=2))
            artists.extend(self.ax.plot(*motor, marker='o', markersize=4, color=color))

        # the direction of thrust
        up = position + rotation @ (2 * arm * np.array((0, 0, 1)))
        artists.extend(self.ax.plot(*zip(position, up), color=color, linewidth=1.5, alpha=0.5))

        artists.extend(self.ax.plot(
            *position, marker='o', markersize=9 if colliding else 7, color=color,
            markeredgecolor=COLLISION_COLOR if colliding else 'white', markeredgewidth=2))

        # the target and how far the drone is from it
        target = self._targets[drone]
        artists.extend(self.ax.plot(*zip(position, target), color=color, linestyle=':', linewidth=1))
        artists.extend(self.ax.plot(*target, marker='*', markersize=13, color=color,
                                    markeredgecolor='white', linestyle='none'))
        if len(self._drones) <= MAX_LABELS:
            artists.append(self.ax.text(*(position + 2.2 * arm * rotation[:, 2]), drone, fontsize=8))
        return artists

    def _get_collisions(self, state):

        # same rule as the reward: drones with the same id are never in collision
        limit = 2 * self._nonfluents['L'] * self._nonfluents['DF']
        collisions = []
        for (d1, d2) in itertools.combinations(self._drones, 2):
            if self._nonfluents[RDDLPlanningModel.ground_var('ID', [d1])] != \
                    self._nonfluents[RDDLPlanningModel.ground_var('ID', [d2])] and \
                    np.sum((self._position(state, d1) - self._position(state, d2)) ** 2) <= limit ** 2:
                collisions.append((d1, d2))
        return collisions

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        self._update_history(state)
        self._update_view()

        collisions = self._get_collisions(state)
        colliding = {d for pair in collisions for d in pair}
        artists = []
        for d in self._drones:
            artists.extend(self._draw_drone(state, d, d in colliding))

        distance = np.mean([np.linalg.norm(self._position(state, d) - self._targets[d])
                            for d in self._drones])
        title = f'Mean distance to target: {distance:.2f}'
        if collisions:
            title += f', {len(collisions)} collision{"s" if len(collisions) != 1 else ""}'
        self.ax.set_title(title, fontsize=13, fontweight='bold',
                          color=COLLISION_COLOR if collisions else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
