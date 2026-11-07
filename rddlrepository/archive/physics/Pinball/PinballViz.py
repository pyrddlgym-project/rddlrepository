import math

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

# the gauges and text around the table, on white as in the other domains
GAUGE_COLOR = '#1565c0'
GOOD_COLOR = '#2e7d32'
TRACK_COLOR = '#eceff1'
TEXT_COLOR = '#37474f'
PAD_OFF_COLOR = '#cfd8dc'

# the table itself is dark, so that the ball and the target stand out
TABLE_COLOR = '#0d1b2a'
OBSTACLE_COLOR = '#3949ab'
OBSTACLE_EDGE_COLOR = '#c5cae9'
BALL_COLOR = '#eceff1'
BALL_EDGE_COLOR = '#ffffff'
BALL_SHINE_COLOR = '#ffffff'
BALL_REACHED_COLOR = '#66bb6a'
TARGET_COLOR = '#66bb6a'
PUSH_COLOR = '#ff8f00'
PATH_COLOR = '#64b5f6'
START_COLOR = '#90a4ae'

BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
SIDE_MARGIN = 0.15

# the table has the same scale on both axes
MIN_SCENE_HEIGHT = 3.2
MIN_SCENE_WIDTH = 3.2
PANEL_WIDTH = 2.2
PANEL_GAP = 0.2

# how far around the table the view extends, as a fraction of its size
VIEW_PADDING = 0.02

# the exhaust of a push leaves the ball on the side opposite to the push
PUSH_DIRECTIONS = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}
PUSH_LENGTH = 2.2
PUSH_WIDTH = 0.7


class PinballVisualizer(BaseViz):
    '''Draws the table from above, with the obstacles, the target, the ball at its true size and
    the path it took, and gauges of its velocity and distance to the target, with a pad of the
    four directions that shows the last push.

    The obstacles and target are read from the instance, and the view holds them, the start and
    anywhere the ball has been. A ring of NEXT that does not close is drawn as lines only.

    The push is the one in the state, which is the push that happened in the last step, including
    those replaced by noise. The path and step are built from the states seen so far, so they
    require render to be called on every step. The state does not hold the time, so the step is
    the number of different states seen, which does not count coasting while at a complete stop.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_size=(9.0, 5.5),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._horizon = model.horizon

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._ball_radius = float(nonfluents['BALL-RADIUS'])
        self._target = (float(nonfluents['TARGET-X']), float(nonfluents['TARGET-Y']))
        self._target_radius = float(nonfluents['TARGET-RADIUS'])
        self._max_speed = float(nonfluents['MAX-SPEED'])
        self._load_obstacles(model, nonfluents)

        self._init_state = self._get_state(model.ground_vars_with_values(model.state_fluents))
        self._start_distance = self._distance(self._init_state)
        self._last_state = None
        self._path = []
        self._bounds = self._get_static_bounds()
        self._view = None

        self._scene_width, self._scene_height = self._get_scene_size(max_size)
        self._draw_figure()

    def _load_obstacles(self, model, nonfluents):
        points = model.type_to_objects['point']
        where = {p: (float(nonfluents[RDDLPlanningModel.ground_var('X', [p])]),
                     float(nonfluents[RDDLPlanningModel.ground_var('Y', [p])])) for p in points}
        follows = {p: self._object_name(nonfluents[RDDLPlanningModel.ground_var('NEXT', [p])], points)
                   for p in points}
        self._corners = list(where.values())
        self._edges = [[where[p], where[q]] for (p, q) in follows.items()]

        # the rings, whichever point each one is listed from
        self._rings, seen = [], set()
        for start in points:
            walk, p = [], start
            while p not in seen and p not in walk:
                walk.append(p)
                p = follows[p]
            if p in walk:
                self._rings.append([where[q] for q in walk[walk.index(p):]])
            seen.update(walk)

    @staticmethod
    def _object_name(value, objects):
        '''An object can come as its name, with or without the @, or as its index.'''
        if isinstance(value, str):
            return value.lstrip('@')
        return objects[int(value)]

    def _get_state(self, state):
        push = state['last-push']
        directions = self._model.type_to_objects['direction']
        return (float(state['x']), float(state['y']), float(state['vx']), float(state['vy']),
                bool(state['reached']), self._object_name(push, directions))

    def _distance(self, state):
        return math.hypot(state[0] - self._target[0], state[1] - self._target[1])

    # ==========================================================================
    # VIEW
    # ==========================================================================

    def _get_static_bounds(self):
        x, y = self._init_state[:2]
        reach = max(self._ball_radius, self._target_radius)
        xs = [p[0] for p in self._corners] + [x - reach, x + reach,
                                              self._target[0] - reach, self._target[0] + reach]
        ys = [p[1] for p in self._corners] + [y - reach, y + reach,
                                              self._target[1] - reach, self._target[1] + reach]
        return (min(xs), max(xs), min(ys), max(ys))

    def _get_scene_size(self, max_size):
        left, right, bottom, top = self._bounds
        height = max(max_size[1] - BOTTOM_MARGIN - TOP_MARGIN, MIN_SCENE_HEIGHT)
        max_width = max(max_size[0] - PANEL_WIDTH - PANEL_GAP - 2 * SIDE_MARGIN, MIN_SCENE_WIDTH)
        width = height * (right - left) / max(top - bottom, 1e-9)
        if width > max_width:
            width, height = max_width, max(max_width * (top - bottom) / (right - left), MIN_SCENE_HEIGHT)
        return max(width, MIN_SCENE_WIDTH), height

    def _update_view(self):

        # both axes use the same scale, with any spare room around the table
        left, right, bottom, top = self._bounds
        pad = VIEW_PADDING * max(right - left, top - bottom)
        scale = max((right - left + 2 * pad) / self._scene_width,
                    (top - bottom + 2 * pad) / self._scene_height)
        middle_x, middle_y = (left + right) / 2, (bottom + top) / 2
        view = (middle_x - scale * self._scene_width / 2, middle_x + scale * self._scene_width / 2,
                middle_y - scale * self._scene_height / 2, middle_y + scale * self._scene_height / 2)
        if view != self._view:
            self._view = view
            self.ax.set_xlim(view[0], view[1])
            self.ax.set_ylim(view[2], view[3])

    # ==========================================================================
    # DRAWING THE PARTS THAT DO NOT CHANGE
    # ==========================================================================

    def _draw_figure(self):
        fig_width = self._scene_width + PANEL_GAP + PANEL_WIDTH + 2 * SIDE_MARGIN
        fig_height = self._scene_height + BOTTOM_MARGIN + TOP_MARGIN
        self.fig = plt.figure(figsize=(fig_width, fig_height))
        self.ax = self.fig.add_axes([
            SIDE_MARGIN / fig_width, BOTTOM_MARGIN / fig_height,
            self._scene_width / fig_width, self._scene_height / fig_height])
        self.ax.set_autoscale_on(False)
        self.ax.set_facecolor(TABLE_COLOR)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')
        self._update_view()

        self._draw_table()
        self._draw_ball()
        self._draw_panel(fig_width, fig_height)

        self.fig.legend(handles=[
            Line2D([], [], marker='o', linestyle='none', markerfacecolor=BALL_COLOR,
                   markeredgecolor=TEXT_COLOR, markersize=7, label='Ball'),
            Patch(facecolor=PUSH_COLOR, label='Push'),
            Line2D([], [], color=PATH_COLOR, linewidth=1.2, label='Path'),
            Line2D([], [], marker='o', linestyle='none', markerfacecolor='none',
                   markeredgecolor=START_COLOR, markersize=7, label='Start'),
            Line2D([], [], marker='o', linestyle='none', markerfacecolor=TARGET_COLOR,
                   markeredgecolor=GOOD_COLOR, markersize=9, label='Target'),
            Patch(facecolor=OBSTACLE_COLOR, edgecolor=TEXT_COLOR, label='Obstacle')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / fig_height), ncol=6, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)
        self.fig.text(0.5, 0.1 / fig_height,
                      'A hit on one edge mirrors the ball, a hit on two at once sends it back',
                      ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)
        self._title = self.fig.text(0.5, 1 - 0.12 / fig_height, '', ha='center', va='top',
                                    fontsize=13, fontweight='bold')

    def _draw_table(self):
        self.ax.add_collection(PolyCollection(
            self._rings, facecolors=OBSTACLE_COLOR, edgecolors='none', zorder=2))
        self.ax.add_collection(LineCollection(
            self._edges, colors=OBSTACLE_EDGE_COLOR, linewidths=1, zorder=3))

        # rings around the target, so that it reads as a target and its edge can be seen
        for (fraction, alpha) in ((1.0, 0.25), (0.6, 0.45), (0.25, 1.0)):
            self.ax.add_patch(Circle(self._target, fraction * self._target_radius,
                                     facecolor=TARGET_COLOR, edgecolor='none', alpha=alpha, zorder=1))
        self.ax.add_patch(Circle(self._target, self._target_radius, facecolor='none',
                                 edgecolor=TARGET_COLOR, linewidth=1.2, zorder=1))
        self.ax.add_patch(Circle(self._init_state[:2], self._ball_radius, facecolor='none',
                                 edgecolor=START_COLOR, linewidth=1, linestyle=':', zorder=1))

    def _draw_ball(self):
        self._path_line, = self.ax.plot([], [], color=PATH_COLOR, linewidth=1.0, alpha=0.8, zorder=4)
        self._push_flame = Polygon([(0, 0), (0, 0), (0, 0)], closed=True, facecolor=PUSH_COLOR,
                                   edgecolor='none', zorder=5)
        self._ball = Circle((0, 0), self._ball_radius, facecolor=BALL_COLOR,
                            edgecolor=BALL_EDGE_COLOR, linewidth=0.8, zorder=6)
        self._shine = Circle((0, 0), 0.3 * self._ball_radius, facecolor=BALL_SHINE_COLOR,
                             edgecolor='none', zorder=7)
        for patch in (self._push_flame, self._ball, self._shine):
            self.ax.add_patch(patch)

    def _draw_panel(self, fig_width, fig_height):
        panel = self.fig.add_axes([
            (SIDE_MARGIN + self._scene_width + PANEL_GAP) / fig_width, BOTTOM_MARGIN / fig_height,
            PANEL_WIDTH / fig_width, self._scene_height / fig_height])
        panel.set_xlim(0, PANEL_WIDTH)
        panel.set_ylim(0, self._scene_height)
        panel.axis('off')
        self._panel = panel

        pitch = min(self._scene_height / 6.0, 0.62)
        top = self._scene_height
        self._horizontal = self._draw_gauge('Left or right', top, centered=True)
        self._vertical = self._draw_gauge('Down or up', top - pitch, centered=True)
        self._away = self._draw_gauge('From the target', top - 2 * pitch)
        self._time = panel.text(0, top - 3 * pitch, '', ha='left', va='top', fontsize=9, color=TEXT_COLOR)
        self._draw_push_pad(top - 3 * pitch - 0.25)

    def _draw_gauge(self, label, top, centered=False):
        panel = self._panel
        panel.text(0, top, label, ha='left', va='top', fontsize=9, color=TEXT_COLOR)
        value = panel.text(PANEL_WIDTH, top, '', ha='right', va='top', fontsize=9, color=TEXT_COLOR)
        bar_y = top - 0.36
        panel.add_patch(Rectangle((0, bar_y), PANEL_WIDTH, 0.14, facecolor=TRACK_COLOR, edgecolor='none', zorder=1))
        fill = Rectangle((0, bar_y), 0, 0.14, facecolor=GAUGE_COLOR, edgecolor='none', zorder=2)
        panel.add_patch(fill)
        if centered:
            panel.plot([PANEL_WIDTH / 2] * 2, [bar_y - 0.03, bar_y + 0.17], color='#b0bec5', linewidth=1, zorder=3)
        return {'value': value, 'fill': fill, 'centered': centered}

    def _draw_push_pad(self, top):

        # four arrows around the middle, the size of which follows the room that is left
        panel = self._panel
        room = max(top - 0.25, 0.5)
        size = min(room / 3.2, 0.26)
        middle_x, middle_y = PANEL_WIDTH / 2, top - 1.6 * size - 0.02
        self._push_label = panel.text(0, middle_y, '', ha='left', va='center', fontsize=9, color=TEXT_COLOR)
        self._push_arrows = {}
        for (name, (dx, dy)) in PUSH_DIRECTIONS.items():
            tip = (middle_x + dx * 1.5 * size, middle_y + dy * 1.5 * size)
            base = (middle_x + dx * 0.55 * size, middle_y + dy * 0.55 * size)
            wing = (0.6 * size * dy, 0.6 * size * dx)
            arrow = Polygon([tip, (base[0] + wing[0], base[1] + wing[1]), (base[0] - wing[0], base[1] - wing[1])],
                            closed=True, facecolor=PAD_OFF_COLOR, edgecolor='none', zorder=2)
            panel.add_patch(arrow)
            self._push_arrows[name] = arrow

    # ==========================================================================
    # UPDATING
    # ==========================================================================

    def _update_history(self, state):

        # rendering the same state again must not change the path or view
        if state == self._last_state:
            return
        last_state, self._last_state = self._last_state, state

        # an episode ends at the target, and can be restarted at any time
        if last_state is None or state == self._init_state or last_state[4]:
            self._path = []
            self._bounds = self._get_static_bounds()
        self._path.append(state[:2])

        x, y = state[:2]
        r = self._ball_radius
        left, right, bottom, top = self._bounds
        self._bounds = (min(left, x - r), max(right, x + r), min(bottom, y - r), max(top, y + r))

    def _set_gauge(self, gauge, fraction, text):
        fraction = max(min(fraction, 1.0), -1.0)
        if gauge['centered']:
            middle = PANEL_WIDTH / 2
            gauge['fill'].set_x(middle + min(fraction, 0.0) * middle)
            gauge['fill'].set_width(abs(fraction) * middle)
        else:
            gauge['fill'].set_width(max(fraction, 0.0) * PANEL_WIDTH)
        gauge['value'].set_text(text)

    def _update_panel(self, state):
        _, _, vx, vy, _, push = state
        limit = self._max_speed if self._max_speed > 0 else 1.0
        self._set_gauge(self._horizontal, vx / limit, f'{vx:+.2f}')
        self._set_gauge(self._vertical, vy / limit, f'{vy:+.2f}')
        distance = self._distance(state)
        scale = self._start_distance if self._start_distance > 0 else 1.0
        self._set_gauge(self._away, distance / scale, f'{distance:.2f}')
        self._time.set_text(f'Step {len(self._path) - 1} of {self._horizon}')

        for (name, arrow) in self._push_arrows.items():
            arrow.set_facecolor(PUSH_COLOR if name == push else PAD_OFF_COLOR)
        self._push_label.set_text('Coasting' if push not in PUSH_DIRECTIONS else f'Pushed {push}')

    def _update_ball(self, state):
        x, y, _, _, reached, push = state
        r = self._ball_radius
        self._ball.set_center((x, y))
        self._ball.set_facecolor(BALL_REACHED_COLOR if reached else BALL_COLOR)
        self._shine.set_center((x - 0.35 * r, y + 0.35 * r))

        self._push_flame.set_visible(push in PUSH_DIRECTIONS)
        if push in PUSH_DIRECTIONS:
            dx, dy = PUSH_DIRECTIONS[push]
            wing = (PUSH_WIDTH * r * dy, PUSH_WIDTH * r * dx)
            self._push_flame.set_xy([(x - dx * PUSH_LENGTH * r, y - dy * PUSH_LENGTH * r),
                                     (x + wing[0], y + wing[1]), (x - wing[0], y - wing[1])])

        self._path_line.set_data(*zip(*self._path))

    def _update_title(self, state):
        if state[4]:
            text, color = f'Reached the target at step {len(self._path) - 1}', GOOD_COLOR
        else:
            speed = math.hypot(state[2], state[3])
            text, color = f'Speed {speed:.2f}, {self._distance(state):.2f} from the target', 'black'
        self._title.set_text(text)
        self._title.set_color(color)

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        state = self._get_state(state)
        self._update_history(state)
        self._update_view()
        self._update_ball(state)
        self._update_panel(state)
        self._update_title(state)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
