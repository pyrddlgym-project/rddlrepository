import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class IntrudersVisualizer(BaseViz):
    '''Draws the danger zones, the cameras and the intruders.

    Nothing limits where intruders and cameras can go, so unless lb and ub give a
    fixed view, the view starts around everything in the first state and grows to
    keep everything in sight. This requires render to be called on every step.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(4, 4),
                 vector_len=0.15,
                 lb=None, ub=None,
                 wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._vector_len = vector_len
        self._lb = lb
        self._ub = ub
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._init_state = model.ground_vars_with_values(model.state_fluents)
        self._last_state = None
        self._bounds = None
        self._span = 1.0

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = plt.gca()
        self.ax.set_aspect('equal')

        # draw zones
        FLUENT_SEP = RDDLPlanningModel.FLUENT_SEP
        obj = model.type_to_objects['zone']
        self._zone_bounds = []
        for o in obj:
            l = self._nonfluents['DANGER-L' + FLUENT_SEP + o]
            r = self._nonfluents['DANGER-R' + FLUENT_SEP + o]
            t = self._nonfluents['DANGER-T' + FLUENT_SEP + o]
            b = self._nonfluents['DANGER-B' + FLUENT_SEP + o]
            zone = Rectangle((l, b), r - l, t - b, color='orange', alpha=0.4)
            self.ax.add_patch(zone)
            self._zone_bounds.append((min(l, r), max(l, r), min(t, b), max(t, b)))

        if lb is not None and ub is not None:
            self._span = ub - lb
            self.ax.set_xlim(lb, ub)
            self.ax.set_ylim(lb, ub)

    def convert2img(self, canvas):
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def _extent(self, state):
        FLUENT_SEP = RDDLPlanningModel.FLUENT_SEP

        # everything drawn for the cameras and intruders, including their arrows
        extent = list(self._zone_bounds)
        for camera in self._model.type_to_objects['camera']:
            x = state['camera-x' + FLUENT_SEP + camera]
            y = state['camera-y' + FLUENT_SEP + camera]
            vx = state['camera-vx' + FLUENT_SEP + camera]
            vy = state['camera-vy' + FLUENT_SEP + camera]
            radius = self._nonfluents['CAMERA-RADIUS' + FLUENT_SEP + camera]
            extent.append((min(x - radius, x + vx), max(x + radius, x + vx),
                           min(y - radius, y + vy), max(y + radius, y + vy)))
        for intr in self._model.type_to_objects['intruder']:
            x = state['intruder-x' + FLUENT_SEP + intr]
            y = state['intruder-y' + FLUENT_SEP + intr]
            vx = state['intruder-vx' + FLUENT_SEP + intr]
            vy = state['intruder-vy' + FLUENT_SEP + intr]
            extent.append((min(x, x + vx), max(x, x + vx), min(y, y + vy), max(y, y + vy)))
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

        extent = self._extent(state)
        if new_episode:
            self._bounds = extent
        else:
            self._bounds = (min(self._bounds[0], extent[0]), max(self._bounds[1], extent[1]),
                            min(self._bounds[2], extent[2]), max(self._bounds[3], extent[3]))

        # both axes use the same scale, with a small margin around everything
        left, right, bottom, top = self._bounds
        span = 1.1 * max(right - left, top - bottom)
        self._span = span if span > 0 else 1.0
        x_center, y_center = (left + right) / 2, (bottom + top) / 2
        self.ax.set_xlim(x_center - self._span / 2, x_center + self._span / 2)
        self.ax.set_ylim(y_center - self._span / 2, y_center + self._span / 2)

    def render(self, state):
        FLUENT_SEP = RDDLPlanningModel.FLUENT_SEP
        self._update_view(state)

        camera_circs, camera_arrows = [], []
        for camera in self._model.type_to_objects['camera']:
            x = state['camera-x' + FLUENT_SEP + camera]
            y = state['camera-y' + FLUENT_SEP + camera]
            vx = state['camera-vx' + FLUENT_SEP + camera]
            vy = state['camera-vy' + FLUENT_SEP + camera]
            circ = plt.Circle((x, y),
                              self._nonfluents['CAMERA-RADIUS' + FLUENT_SEP + camera],
                              color='green', alpha=0.4)
            camera_circs.append(self.ax.add_patch(circ))
            arrow = Arrow(x, y, vx, vy, width=0.1 * self._span, color='green')
            camera_arrows.append(self.ax.add_patch(arrow))

        intr_circs, intr_arrows = [], []
        for intr in self._model.type_to_objects['intruder']:
            x = state['intruder-x' + FLUENT_SEP + intr]
            y = state['intruder-y' + FLUENT_SEP + intr]
            vx = state['intruder-vx' + FLUENT_SEP + intr]
            vy = state['intruder-vy' + FLUENT_SEP + intr]
            circ = plt.Circle((x, y), 0.01 * self._span, color='red')
            intr_circs.append(self.ax.add_patch(circ))
            arrow = Arrow(x, y, vx, vy, width=0.05 * self._span, color='red')
            intr_arrows.append(self.ax.add_patch(arrow))

        self.fig.canvas.draw()
        img = self.convert2img(self.fig.canvas)

        for obj in camera_circs + camera_arrows + intr_circs + intr_arrows:
            obj.remove()
        del camera_circs, camera_arrows, intr_circs, intr_arrows

        return img
