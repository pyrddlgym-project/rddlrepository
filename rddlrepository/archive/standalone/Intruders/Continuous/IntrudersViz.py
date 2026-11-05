import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

ZONE_COLOR = '#ffe0b2'
ZONE_EDGE_COLOR = '#ff8f00'
CAMERA_COLOR = '#1565c0'
INTRUDER_COLOR = '#37474f'
UNDETECTED_COLOR = '#c62828'
DETECTED_COLOR = '#2e7d32'


class IntrudersVisualizer(BaseViz):
    '''Draws the danger zones, the area each camera can see, and the intruders colored
    by how the reward treats them, with a tail showing recent motion.

    Nothing limits where intruders and cameras can go, so unless lb and ub give a
    fixed view, the view starts around everything in the first state and grows to
    keep everything in sight. This requires render to be called on every step.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(5.4, 5.9),
                 tail_steps=5,
                 lb=None, ub=None,
                 wait_time=100) -> None:
        self._model = model
        self._tail_steps = tail_steps
        self._lb = lb
        self._ub = ub
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._init_state = model.ground_vars_with_values(model.state_fluents)
        self._cameras = model.type_to_objects['camera']
        self._intruders = model.type_to_objects['intruder']
        self._dt = self._nonfluents['DT']
        self._radius = {c: self._nonfluents[RDDLPlanningModel.ground_var('CAMERA-RADIUS', [c])]
                        for c in self._cameras}

        self._last_state = None
        self._bounds = None

        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_axes([0.1, 0.12, 0.86, 0.79])
        self.ax.set_aspect('equal')
        self.ax.tick_params(labelsize=8, colors='#616161')
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')

        # the zones never change, so they are only drawn once: a zone with its edges
        # the wrong way around can never contain anything in the reward
        self._zones = []
        for z in model.type_to_objects['zone']:
            left, right, top, bottom = [
                self._nonfluents[RDDLPlanningModel.ground_var(var, [z])]
                for var in ('DANGER-L', 'DANGER-R', 'DANGER-T', 'DANGER-B')]
            if left <= right and bottom <= top:
                self._zones.append((left, right, bottom, top))
                self.ax.add_patch(patches.Rectangle(
                    (left, bottom), right - left, top - bottom, facecolor=ZONE_COLOR,
                    edgecolor=ZONE_EDGE_COLOR, linewidth=1, zorder=0))

        if lb is not None and ub is not None:
            self.ax.set_xlim(lb, ub)
            self.ax.set_ylim(lb, ub)

        self.fig.legend(handles=[
            patches.Patch(facecolor=ZONE_COLOR, edgecolor=ZONE_EDGE_COLOR, label='Danger zone'),
            patches.Patch(facecolor=CAMERA_COLOR, alpha=0.25, label='Camera'),
            Line2D([], [], marker='o', color='none', markerfacecolor=INTRUDER_COLOR,
                   markeredgecolor='none', label='Intruder'),
            Line2D([], [], marker='o', color='none', markerfacecolor=UNDETECTED_COLOR,
                   markeredgecolor='none', label='Undetected'),
            Line2D([], [], marker='o', color='none', markerfacecolor=DETECTED_COLOR,
                   markeredgecolor='none', label='Detected')
        ], loc='lower center', ncol=5, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _get(self, state, var, obj):
        return float(state[RDDLPlanningModel.ground_var(var, [obj])])

    def _get_extent(self, state):

        # the zones, the area of each camera and the intruders
        extent = list(self._zones)
        for c in self._cameras:
            x, y = self._get(state, 'camera-x', c), self._get(state, 'camera-y', c)
            extent.append((x - self._radius[c], x + self._radius[c],
                           y - self._radius[c], y + self._radius[c]))
        for i in self._intruders:
            x, y = self._get(state, 'intruder-x', i), self._get(state, 'intruder-y', i)
            extent.append((x, x, y, y))
        return (min(e[0] for e in extent), max(e[1] for e in extent),
                min(e[2] for e in extent), max(e[3] for e in extent))

    def _update_view(self, state):

        # a fixed view was requested
        if self._lb is not None and self._ub is not None:
            return

        # rendering the same state again must not change the view, and only a new
        # episode can return to the initial state as the intruders move randomly
        if state == self._last_state:
            return
        new_episode = self._last_state is None or state == self._init_state
        self._last_state = dict(state)

        extent = self._get_extent(state)
        if new_episode:
            self._bounds = extent
        else:
            self._bounds = (min(self._bounds[0], extent[0]), max(self._bounds[1], extent[1]),
                            min(self._bounds[2], extent[2]), max(self._bounds[3], extent[3]))

        # both axes use the same scale, with a small margin around everything
        left, right, bottom, top = self._bounds
        span = 1.1 * max(right - left, top - bottom) or 1.0
        x_center, y_center = (left + right) / 2, (bottom + top) / 2
        self.ax.set_xlim(x_center - span / 2, x_center + span / 2)
        self.ax.set_ylim(y_center - span / 2, y_center + span / 2)

    def _draw_tail(self, x, y, vx, vy, color):
        length = self._tail_steps * self._dt
        return self.ax.plot([x, x - length * vx], [y, y - length * vy], color=color,
                            linewidth=2, alpha=0.4, solid_capstyle='round', zorder=2)

    def _draw_camera(self, state, camera):
        x, y = self._get(state, 'camera-x', camera), self._get(state, 'camera-y', camera)
        artists = [self.ax.add_patch(plt.Circle(
            (x, y), self._radius[camera], facecolor=CAMERA_COLOR, alpha=0.25,
            edgecolor='none', zorder=1))]
        artists.extend(self._draw_tail(
            x, y, self._get(state, 'camera-vx', camera), self._get(state, 'camera-vy', camera),
            CAMERA_COLOR))
        artists.extend(self.ax.plot(x, y, marker='o', markersize=5, color=CAMERA_COLOR,
                                    markeredgecolor='white', zorder=3))
        return artists

    def _count_zones(self, x, y):
        return sum(left <= x <= right and bottom <= y <= top
                   for (left, right, bottom, top) in self._zones)

    def _is_detected(self, state, x, y):
        return any(
            (self._get(state, 'camera-x', c) - x) ** 2 +
            (self._get(state, 'camera-y', c) - y) ** 2 <= self._radius[c] ** 2
            for c in self._cameras)

    def _draw_intruder(self, state, intruder):
        x, y = self._get(state, 'intruder-x', intruder), self._get(state, 'intruder-y', intruder)

        # same rule as the reward, which counts each zone an intruder is in
        zones = self._count_zones(x, y)
        detected = self._is_detected(state, x, y)
        penalty = zones if not detected else 0
        if zones == 0:
            color = INTRUDER_COLOR
        else:
            color = DETECTED_COLOR if detected else UNDETECTED_COLOR

        artists = self._draw_tail(
            x, y, self._get(state, 'intruder-vx', intruder), self._get(state, 'intruder-vy', intruder),
            color)
        artists.extend(self.ax.plot(x, y, marker='o', markersize=8, color=color,
                                    markeredgecolor='white', zorder=4))
        return artists, penalty

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        self._update_view(state)

        artists = []
        for c in self._cameras:
            artists.extend(self._draw_camera(state, c))
        undetected = 0
        for i in self._intruders:
            intruder_artists, penalty = self._draw_intruder(state, i)
            artists.extend(intruder_artists)
            undetected += penalty

        self.ax.set_title(f'Undetected in danger zone: {undetected}', fontsize=13,
                          fontweight='bold', color=UNDETECTED_COLOR if undetected else DETECTED_COLOR)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
