import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

# the scenery
GRASS_COLOR = '#7cb342'
SHOULDER_COLOR = '#9e9e9e'
ROAD_COLOR = '#455a64'
MARKING_COLOR = '#f5f5f5'
POST_COLOR = '#eceff1'
TEXT_COLOR = '#37474f'

# the cars
EGO_COLOR = '#43a047'
CRASHED_COLOR = '#e53935'
OTHER_COLORS = ['#1e88e5', '#90a4ae', '#fdd835', '#8e24aa', '#f4f4f4', '#3949ab', '#6d4c41', '#00acc1']
GLASS_COLOR = '#263238'
WHEEL_COLOR = '#212121'
SIGNAL_COLOR = '#ff9800'
OUTLINE_COLOR = '#212121'

# the window that follows the ego car, in meters behind and ahead of it
VIEW_BEHIND = 35.0
VIEW_AHEAD = 85.0

# the shoulders and the grass on each side of the road, in meters
SHOULDER = 1.0
VERGE = 3.0

# the dashes between lanes, and the posts along the road, in meters
DASH_LENGTH = 3.0
DASH_PERIOD = 12.0
POST_PERIOD = 50.0

# the figure, in inches
FIGURE_WIDTH = 12.0
MAP_HEIGHT = 0.7
TITLE_HEIGHT = 0.45
GAP = 0.3
SIDE_MARGIN = 0.15
BOTTOM_MARGIN = 0.15


class HighwayVisualizer(BaseViz):
    '''Draws the highway from above, in a window that follows the ego car, with the lanes, their
    markings, posts every 50 meters along the road and every car in the window at its true size
    and heading. Below it, a map of the stretch of road that holds all the cars shows where the
    window is in the traffic.

    The ego car is green, crashed cars are red and the other cars take a color each. A car whose
    target lane differs from its lane shows its turn signals on the side it is moving to. The
    text above the road gives the time, the speed and lane of the ego car and, depending on the
    action space, its target speed and target lane or the acceleration and steering it holds.
    Everything is read from the state and the instance, so any number of lanes and cars works.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_width=FIGURE_WIDTH,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._lanes = int(nonfluents['NUM-LANES'])
        self._lane_width = float(nonfluents['LANE-WIDTH'])
        self._length = float(nonfluents['VEHICLE-LENGTH'])
        self._width = float(nonfluents['VEHICLE-WIDTH'])
        self._frequency = float(nonfluents['SIMULATION-FREQUENCY'])
        self._cars = list(model.type_to_objects['car'])
        egos = [c for c in self._cars if bool(nonfluents[RDDLPlanningModel.ground_var('EGO', [c])])]
        self._ego = egos[0] if egos else self._cars[0]
        order = {c: int(nonfluents[RDDLPlanningModel.ground_var('ORDER', [c])]) for c in self._cars}
        others = sorted((c for c in self._cars if c != self._ego), key=lambda c: order[c])
        self._colors = {c: OTHER_COLORS[i % len(OTHER_COLORS)] for (i, c) in enumerate(others)}
        self._colors[self._ego] = EGO_COLOR

        # the ego car of the discrete version has targets, that of the continuous one holds actions
        self._continuous = 'held-acc' in model.state_fluents

        # the road runs from the leftmost lane at the top to the rightmost at the bottom
        self._road_top = -self._lane_width / 2.0
        self._road_bottom = (self._lanes - 0.5) * self._lane_width
        self._scene_top = self._road_top - SHOULDER - VERGE
        self._scene_bottom = self._road_bottom + SHOULDER + VERGE
        self._figure_width = figure_width
        self._draw_figure()

    # ==========================================================================
    # FIGURE
    # ==========================================================================

    def _draw_figure(self):
        view_width = self._figure_width - 2 * SIDE_MARGIN
        scale = view_width / (VIEW_BEHIND + VIEW_AHEAD)
        view_height = scale * (self._scene_bottom - self._scene_top)
        height = BOTTOM_MARGIN + MAP_HEIGHT + GAP + view_height + TITLE_HEIGHT
        self.fig = plt.figure(figsize=(self._figure_width, height))
        self.ax = self.fig.add_axes([
            SIDE_MARGIN / self._figure_width, (BOTTOM_MARGIN + MAP_HEIGHT + GAP) / height,
            view_width / self._figure_width, view_height / height])
        self.map_ax = self.fig.add_axes([
            SIDE_MARGIN / self._figure_width, BOTTOM_MARGIN / height,
            view_width / self._figure_width, MAP_HEIGHT / height])
        self._title = self.fig.text(0.5, 1 - 0.1 / height, '', ha='center', va='top',
                                    fontsize=11, color=TEXT_COLOR)
        self.fig.text(SIDE_MARGIN / self._figure_width, (BOTTOM_MARGIN + MAP_HEIGHT + 0.05) / height,
                      'all traffic', fontsize=8, color=TEXT_COLOR, va='bottom')

    # ==========================================================================
    # STATE
    # ==========================================================================

    def _get_cars(self, state):
        cars = {}
        for c in self._cars:
            get = lambda var: state[RDDLPlanningModel.ground_var(var, [c])]
            y = float(get('y'))
            lane = max(0, min(self._lanes - 1, math.ceil(y / self._lane_width - 0.5)))
            cars[c] = {'x': float(get('x')), 'y': y, 'heading': float(get('heading')),
                       'speed': float(get('speed')), 'crashed': bool(get('crashed')),
                       'lane': lane, 'target-lane': int(get('target-lane')),
                       'target-speed': float(get('target-speed'))}
        return cars

    # ==========================================================================
    # DRAWING
    # ==========================================================================

    def _local_to_world(self, car, points):
        c, s = math.cos(car['heading']), math.sin(car['heading'])
        return [(car['x'] + c * px - s * py, car['y'] + s * px + c * py) for (px, py) in points]

    def _draw_scenery(self, left, right):
        ax = self.ax
        ax.add_patch(Rectangle((left, self._scene_top), right - left, self._scene_bottom - self._scene_top,
                               facecolor=GRASS_COLOR, edgecolor='none', zorder=0))
        ax.add_patch(Rectangle((left, self._road_top - SHOULDER), right - left,
                               self._road_bottom - self._road_top + 2 * SHOULDER,
                               facecolor=SHOULDER_COLOR, edgecolor='none', zorder=1))
        ax.add_patch(Rectangle((left, self._road_top), right - left, self._road_bottom - self._road_top,
                               facecolor=ROAD_COLOR, edgecolor='none', zorder=2))

        # solid lines on the edges and dashes between lanes, fixed to the road
        for y in (self._road_top, self._road_bottom):
            ax.plot([left, right], [y, y], color=MARKING_COLOR, linewidth=1.6, zorder=3)
        first = math.floor(left / DASH_PERIOD) * DASH_PERIOD
        for i in range(1, self._lanes):
            y = (i - 0.5) * self._lane_width
            x = first
            while x < right:
                ax.plot([x, x + DASH_LENGTH], [y, y], color=MARKING_COLOR, linewidth=1.2,
                        solid_capstyle='butt', zorder=3)
                x += DASH_PERIOD

        # posts on the verge, so that the motion along the road shows
        post = math.ceil(left / POST_PERIOD) * POST_PERIOD
        while post < right:
            y = self._road_bottom + SHOULDER + 0.4
            ax.add_patch(Rectangle((post - 0.25, y), 0.5, 1.0, facecolor=POST_COLOR,
                                   edgecolor=OUTLINE_COLOR, linewidth=0.5, zorder=3))
            ax.text(post + 0.8, y + 0.5, f'{post:.0f} m', fontsize=7, color='white',
                    va='center', ha='left', zorder=3)
            post += POST_PERIOD

    def _draw_car(self, name, car):
        half_l, half_w = self._length / 2.0, self._width / 2.0
        cut = 0.12 * self._width
        color = CRASHED_COLOR if car['crashed'] else self._colors[name]
        outline = 2.0 if name == self._ego else 0.8

        # wheels peeking out under the body
        for sx in (-1, 1):
            for sy in (-1, 1):
                cx, cy = sx * 0.62 * half_l, sy * 0.92 * half_w
                wheel = [(cx - 0.18 * half_l, cy - 0.12 * half_w), (cx + 0.18 * half_l, cy - 0.12 * half_w),
                         (cx + 0.18 * half_l, cy + 0.12 * half_w), (cx - 0.18 * half_l, cy + 0.12 * half_w)]
                self.ax.add_patch(Polygon(self._local_to_world(car, wheel), closed=True,
                                          facecolor=WHEEL_COLOR, edgecolor='none', zorder=4))

        # the body with cut corners, the windshield and the rear window
        body = [(-half_l + cut, -half_w), (half_l - cut, -half_w), (half_l, -half_w + cut),
                (half_l, half_w - cut), (half_l - cut, half_w), (-half_l + cut, half_w),
                (-half_l, half_w - cut), (-half_l, -half_w + cut)]
        self.ax.add_patch(Polygon(self._local_to_world(car, body), closed=True, facecolor=color,
                                  edgecolor=OUTLINE_COLOR, linewidth=outline, zorder=5))
        windshield = [(0.18 * half_l, -0.78 * half_w), (0.5 * half_l, -0.86 * half_w),
                      (0.5 * half_l, 0.86 * half_w), (0.18 * half_l, 0.78 * half_w)]
        rear = [(-0.82 * half_l, -0.74 * half_w), (-0.62 * half_l, -0.7 * half_w),
                (-0.62 * half_l, 0.7 * half_w), (-0.82 * half_l, 0.74 * half_w)]
        for glass in (windshield, rear):
            self.ax.add_patch(Polygon(self._local_to_world(car, glass), closed=True,
                                      facecolor=GLASS_COLOR, edgecolor='none', zorder=6))

        # turn signals toward the target lane; lanes grow to the right, which is +y here
        has_target = not (self._continuous and name == self._ego)
        if has_target and car['target-lane'] != car['lane'] and not car['crashed']:
            side = 1 if car['target-lane'] > car['lane'] else -1
            for px in (-half_l + 0.15, half_l - 0.15):
                (sx, sy), = self._local_to_world(car, [(px, side * (half_w - 0.15))])
                self.ax.plot(sx, sy, 'o', markersize=4, color=SIGNAL_COLOR,
                             markeredgecolor=OUTLINE_COLOR, markeredgewidth=0.5, zorder=7)

    def _draw_map(self, cars, left, right):
        ax = self.map_ax
        ax.clear()
        xs = [car['x'] for car in cars.values()]
        start = min(xs + [left]) - 20.0
        end = max(xs + [right]) + 20.0
        ax.add_patch(Rectangle((start, self._road_top), end - start, self._road_bottom - self._road_top,
                               facecolor=ROAD_COLOR, edgecolor='none'))
        for i in range(1, self._lanes):
            y = (i - 0.5) * self._lane_width
            ax.plot([start, end], [y, y], color=MARKING_COLOR, linewidth=0.5, linestyle=(0, (4, 4)))
        for name, car in cars.items():
            color = CRASHED_COLOR if car['crashed'] else self._colors[name]
            length = max(self._length, 0.004 * (end - start))
            ax.add_patch(Rectangle((car['x'] - length / 2, car['y'] - self._width / 2), length, self._width,
                                   facecolor=color, edgecolor=OUTLINE_COLOR,
                                   linewidth=1.0 if name == self._ego else 0.3, zorder=3))
        ax.add_patch(Rectangle((left, self._road_top - 0.6), right - left,
                               self._road_bottom - self._road_top + 1.2,
                               facecolor='none', edgecolor='white', linewidth=1.5, zorder=4))
        ax.set_xlim(start, end)
        ax.set_ylim(self._road_bottom + 0.8, self._road_top - 0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _update_title(self, state, cars):
        ego = cars[self._ego]
        time = int(state['frame']) / self._frequency
        text = (f't = {time:.1f} s    speed {ego["speed"]:.1f} m/s    '
                f'lane {ego["lane"] + 1} of {self._lanes} from the left')
        if self._continuous:
            text += (f'    acceleration {float(state["held-acc"]):+.2f} m/s²'
                     f'    steering {math.degrees(float(state["held-steering"])):+.1f}°')
        else:
            text += (f'    target speed {ego["target-speed"]:.0f} m/s'
                     f'    target lane {ego["target-lane"] + 1}')
        crashed = sum(car['crashed'] for car in cars.values())
        if ego['crashed']:
            text = 'CRASHED    ' + text
        elif crashed:
            text += f'    {crashed} cars crashed'
        self._title.set_text(text)
        self._title.set_color(CRASHED_COLOR if ego['crashed'] else TEXT_COLOR)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        cars = self._get_cars(state)
        ego = cars[self._ego]
        left, right = ego['x'] - VIEW_BEHIND, ego['x'] + VIEW_AHEAD

        self.ax.clear()
        self._draw_scenery(left, right)
        reach = self._length + self._width
        for name, car in cars.items():
            if left - reach <= car['x'] <= right + reach and name != self._ego:
                self._draw_car(name, car)
        self._draw_car(self._ego, ego)
        self.ax.text(ego['x'], ego['y'] - self._width / 2.0 - 0.4, 'ego', fontsize=8, ha='center',
                     va='bottom', color='white', fontweight='bold', zorder=8)
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(self._scene_bottom, self._scene_top)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        self._draw_map(cars, left, right)
        self._update_title(state, cars)
        self.fig.canvas.draw()
        return self.convert2img(self.fig)
