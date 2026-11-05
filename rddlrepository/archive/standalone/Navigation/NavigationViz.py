import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

PATH_COLOR = '#0d47a1'
GOAL_COLOR = '#2e7d32'
ZONE_COLOR = '#212121'
FIELD_COLORMAP = 'YlOrRd_r'

# the slowdown of a zone is mostly gone at this many multiples of one over its decay
ZONE_REACH = 3.0


class NavigationVisualizer(BaseViz):
    '''Draws the path of the agent and the goal over a map of how much of each move
    takes effect, which is the product of the slowdown of all zones.

    The map shows the first two dimensions, and any others are shown as a slice
    through the current location of the agent. The view covers the goal, the zones
    and the path, growing as the agent leaves it.

    The path and view are built from the states seen so far, so they require render
    to be called on every step.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(7.2, 6.6),
                 resolution=200,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._dims = model.type_to_objects['dim']
        self._zones = model.type_to_objects['zone']
        self._goal = np.array([nonfluents[RDDLPlanningModel.ground_var('GOAL', [d])]
                               for d in self._dims], dtype=float)
        self._centers = np.array(
            [[nonfluents[RDDLPlanningModel.ground_var('DECELERATION_ZONE_CENTER', [z, d])]
              for d in self._dims] for z in self._zones], dtype=float).reshape(
                  (len(self._zones), len(self._dims)))
        self._decays = np.array(
            [nonfluents[RDDLPlanningModel.ground_var('DECELERATION_ZONE_DECAY', [z])]
             for z in self._zones], dtype=float)
        self._init_state = self._get_location(model.ground_vars_with_values(model.state_fluents))

        self._resolution = resolution
        self._path = []
        self._last_state = None
        self._bounds = None
        self._view = None

        # the map is square, with room for the labels, title and color scale
        fig_width, fig_height = figure_size
        size = fig_height - 1.1
        self.fig = plt.figure(figsize=figure_size)
        self.ax = self.fig.add_axes(
            [0.7 / fig_width, 0.6 / fig_height, size / fig_width, size / fig_height])
        self.ax.set_autoscale_on(False)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel(self._dims[0])
        if len(self._dims) > 1:
            self.ax.set_ylabel(self._dims[1])
        else:
            self.ax.set_yticks([])

        self._image = self.ax.imshow(
            np.ones((2, 2)), origin='lower', extent=(0, 1, 0, 1), cmap=FIELD_COLORMAP,
            vmin=0.0, vmax=1.0, interpolation='bilinear', zorder=0)
        self.ax.set_autoscale_on(False)
        bar_axis = self.fig.add_axes(
            [(0.95 + size) / fig_width, 0.6 / fig_height, 0.15 / fig_width, size / fig_height])
        bar = self.fig.colorbar(self._image, cax=bar_axis)
        bar.set_label('fraction of each move taken', fontsize=9)
        bar.ax.tick_params(labelsize=8)

        # the goal and zones never change, so they are only drawn once
        goal_x, goal_y = self._project(self._goal)
        self.ax.plot([goal_x], [goal_y], marker='*', markersize=18, color=GOAL_COLOR,
                     markeredgecolor='white', zorder=4)
        for (z, center) in zip(self._zones, self._centers):
            x, y = self._project(center)
            self.ax.plot([x], [y], marker='x', markersize=7, markeredgewidth=1.5,
                         color=ZONE_COLOR, zorder=2)
            self.ax.annotate(z, (x, y), xytext=(6, 4), textcoords='offset points',
                             fontsize=8, color=ZONE_COLOR, zorder=2)

    def _get_location(self, state):
        return tuple(float(state[RDDLPlanningModel.ground_var('location', [d])])
                     for d in self._dims)

    def _project(self, location):
        return location[0], (location[1] if len(self._dims) > 1 else 0.0)

    def _speed_factor(self, distances_squared):

        # same rule as the domain: the product of the deceleration of all zones
        factor = 1.0
        for (i, decay) in enumerate(self._decays):
            distance = np.sqrt(distances_squared[i])
            factor = factor * (2.0 / (1.0 + np.exp(-decay * distance)) - 1.0)
        return factor

    def _current_factor(self, location):
        return float(self._speed_factor(
            np.sum((self._centers - np.array(location)) ** 2, axis=1)))

    def _get_static_bounds(self):

        # the goal and all zones, including the area each zone slows down
        goal_x, goal_y = self._project(self._goal)
        left, right, bottom, top = goal_x, goal_x, goal_y, goal_y
        for (center, decay) in zip(self._centers, self._decays):
            x, y = self._project(center)
            reach = ZONE_REACH / decay if decay > 0 else 0.0
            left, right = min(left, x - reach), max(right, x + reach)
            bottom, top = min(bottom, y - reach), max(top, y + reach)
        return (left, right, bottom, top)

    def _update_history(self, state):

        # rendering the same state again must not change the path or view
        if state == self._last_state:
            return
        new_episode = self._last_state is None or state == self._init_state
        self._last_state = state

        if new_episode:
            self._path = []
            self._bounds = self._get_static_bounds()
        self._path.append(state)

        x, y = self._project(state)
        left, right, bottom, top = self._bounds
        self._bounds = (min(left, x), max(right, x), min(bottom, y), max(top, y))

    def _update_view(self):

        # both axes use the same scale, with a small margin around everything
        left, right, bottom, top = self._bounds
        span = 1.1 * max(right - left, top - bottom)
        span = span if span > 0 else 1.0
        x_center, y_center = (left + right) / 2, (bottom + top) / 2
        view = (x_center - span / 2, x_center + span / 2,
                y_center - span / 2, y_center + span / 2)

        # the map only changes when the view does, or with the slice of other dimensions
        if view != self._view or len(self._dims) > 2:
            self._view = view
            self._image.set_extent(view)
            self._image.set_data(self._get_field(view))
            self.ax.set_xlim(view[0], view[1])
            self.ax.set_ylim(view[2], view[3])

    def _get_field(self, view):
        n = self._resolution
        step_x, step_y = (view[1] - view[0]) / n, (view[3] - view[2]) / n
        xs = view[0] + step_x * (np.arange(n) + 0.5)
        ys = view[2] + step_y * (np.arange(n) + 0.5)
        grid_x, grid_y = np.meshgrid(xs, ys)

        distances_squared = []
        for center in self._centers:
            total = (grid_x - center[0]) ** 2
            if len(self._dims) > 1:
                total = total + (grid_y - center[1]) ** 2
            for i in range(2, len(self._dims)):
                total = total + (self._last_state[i] - center[i]) ** 2
            distances_squared.append(total)
        return self._speed_factor(distances_squared) * np.ones_like(grid_x)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        state = self._get_location(state)
        self._update_history(state)
        self._update_view()

        path = [self._project(location) for location in self._path]
        xs, ys = zip(*path)
        artists = []
        artists.extend(self.ax.plot(
            xs, ys, color=PATH_COLOR, linewidth=1.5, marker='o', markersize=3, zorder=3))
        artists.extend(self.ax.plot(
            [xs[-1]], [ys[-1]], marker='o', markersize=11, color=PATH_COLOR,
            markeredgecolor='white', zorder=5))

        # any other dimensions are held at the current location in this map
        others = ', '.join(f'{d} = {state[i]:.2f}' for (i, d) in enumerate(self._dims) if i >= 2)
        if others:
            artists.append(self.ax.text(
                0.01, 0.99, f'Slice at {others}', transform=self.ax.transAxes, ha='left',
                va='top', fontsize=8, zorder=6,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5, 'alpha': 0.8}))

        distance = float(np.sqrt(np.sum((self._goal - np.array(state)) ** 2)))
        self.ax.set_title(f'Distance to goal: {distance:.2f}, '
                          f'move taken: {self._current_factor(state):.0%}',
                          fontsize=12, fontweight='bold')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
