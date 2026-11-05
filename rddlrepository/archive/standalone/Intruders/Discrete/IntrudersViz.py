import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 4


class IntrudersVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(4, 4),
                 wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)

        # moving right and down define the geometry of the grid
        self._xs = self._get_axis_order('TARGET-X', 'right', model.type_to_objects['x-pos'])
        self._ys = self._get_axis_order('TARGET-Y', 'down', model.type_to_objects['y-pos'])
        self._columns = {x: i for (i, x) in enumerate(self._xs)}
        self._rows = {y: len(self._ys) - 1 - i for (i, y) in enumerate(self._ys)}
        self._danger = {(x, y) for x in self._xs for y in self._ys if self._nonfluents[
            RDDLPlanningModel.ground_var('IS-DANGER-ZONE', [x, y])]}

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = plt.gca()
        self.ax.set_xlim(0, len(self._xs))
        self.ax.set_ylim(0, len(self._ys))
        self.ax.set_aspect('equal')

        # draw the object names on the axes
        self._set_labels(self.ax.set_xticks, self.ax.set_xticklabels, self._xs)
        self._set_labels(self.ax.set_yticks, self.ax.set_yticklabels, self._ys[::-1])

        # draw grid
        for i in range(1, len(self._xs)):
            self.ax.axvline(i, color='lightgrey', linewidth=0.5, zorder=0)
        for i in range(1, len(self._ys)):
            self.ax.axhline(i, color='lightgrey', linewidth=0.5, zorder=0)

        # draw zones
        for (x, y) in self._danger:
            self.ax.add_patch(Rectangle((self._columns[x], self._rows[y]), 1, 1,
                                        color='orange', alpha=0.4))

    def _get_axis_order(self, fluent, direction, objects):

        # follow the moves from the object that no other move reaches: moves that do
        # not form a single line or loop have no clear order, so use the declared one
        following = {}
        for start in objects:
            targets = [end for end in objects if end != start and self._nonfluents[
                RDDLPlanningModel.ground_var(fluent, [direction, start, end])]]
            if len(targets) > 1:
                return objects
            if targets:
                following[start] = targets[0]
        first = [o for o in objects if o not in following.values()]
        order = [first[0] if len(first) == 1 else objects[0]]
        while order[-1] in following and following[order[-1]] not in order:
            order.append(following[order[-1]])
        return order if len(order) == len(objects) else objects

    def _set_labels(self, set_ticks, set_labels, names):
        set_ticks([i + 0.5 for i in range(len(names))])
        font_size = min(9, 72 * 3 / (len(names) * 0.62 * max(len(n) for n in names)))
        set_labels(names if font_size >= MIN_FONT_SIZE else [],
                   fontsize=max(font_size, MIN_FONT_SIZE))
        if font_size < MIN_FONT_SIZE:
            set_ticks([])

    def _get_cells(self, state, fluent, objects):
        cells = {}
        for o in objects:
            for x in self._xs:
                for y in self._ys:
                    if state[RDDLPlanningModel.ground_var(fluent, [o, x, y])]:
                        cells.setdefault((x, y), []).append(o)
        return cells

    def convert2img(self, canvas):
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        cameras = self._get_cells(state, 'camera-at', self._model.type_to_objects['camera'])
        intruders = self._get_cells(state, 'intruder-at', self._model.type_to_objects['intruder'])

        artists = []
        for (x, y) in cameras:
            circ = plt.Circle((self._columns[x] + 0.5, self._rows[y] + 0.5), 0.45,
                              color='green', alpha=0.4)
            artists.append(self.ax.add_patch(circ))

        # intruders in the same cell are spread around its center so all can be seen
        for ((x, y), names) in intruders.items():
            count = len(names)
            spread = 0.22 if count > 1 else 0.0
            radius = min(0.12, 0.9 * spread * math.sin(math.pi / count)) if count > 1 else 0.12
            for i in range(count):
                angle = 2 * math.pi * i / count
                circ = plt.Circle((self._columns[x] + 0.5 + spread * math.sin(angle),
                                   self._rows[y] + 0.5 + spread * math.cos(angle)),
                                  radius, color='red')
                artists.append(self.ax.add_patch(circ))

        # same rule as the reward
        missed = sum(len(names) for (cell, names) in intruders.items()
                     if cell in self._danger and cell not in cameras)
        self.ax.set_title(f'Undetected in danger zone: {missed}', fontsize=9)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig.canvas)

        for obj in artists:
            obj.remove()
        del artists

        return img
