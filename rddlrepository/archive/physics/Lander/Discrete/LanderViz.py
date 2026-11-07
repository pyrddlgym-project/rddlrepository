import math

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

# the gauges and text around the scene, on white as in the other domains
LANDER_COLOR = '#1565c0'
GOOD_COLOR = '#2e7d32'
WARNING_COLOR = '#ff8f00'
BAD_COLOR = '#c62828'
TRACK_COLOR = '#eceff1'
TEXT_COLOR = '#37474f'

# the scene is in space, so its colors must stand out on black
SPACE_COLOR = '#000000'
HULL_COLOR = '#42a5f5'
HULL_LANDED_COLOR = '#66bb6a'
HULL_OFF_PAD_COLOR = '#ffb300'
HULL_CRASHED_COLOR = '#ef5350'
LEG_COLOR = '#eceff1'
FLAME_COLOR = '#ff8f00'
PAD_COLOR = '#66bb6a'
GROUND_COLOR = '#78909c'
GROUND_EDGE_COLOR = '#b0bec5'
PATH_COLOR = '#90a4ae'
SCENE_TEXT_COLOR = '#eceff1'

BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
SIDE_MARGIN = 0.15

# the scene has the same scale on both axes, so any spare height is sky
MAX_SCENE_WIDTH = 6.2
MIN_SCENE_WIDTH = 3.6
MIN_SCENE_HEIGHT = 3.2
PANEL_WIDTH = 2.2
PANEL_GAP = 0.2

# terrain wider than this many bodies is shown around the start and the pads instead of whole
MAX_FULL_VIEW = 60.0
VIEW_MARGIN = 6.0

# the view is a little wider than what it holds, which also shows where the terrain ends
VIEW_PADDING = 1.06

# the lander is drawn larger if it would be smaller than this when the view has grown
MIN_BODY_INCHES = 0.14

# the gauges show the descent and drift up to this many times the break speed
SPEED_GAUGE_RANGE = 1.5
WARNING_FRACTION = 0.75
MAX_TILT_GAUGE = 90.0

# an engine below this level is shown as off
VISIBLE_LEVEL = 0.005


class LanderVisualizer(BaseViz):
    '''Draws the lander over the terrain, with the pads, the path flown and the flames of the
    engines, and gauges of the descent against the break speed, the drift, the tilt against
    where the legs would tip over, and the engines.

    The terrain, pads and body are all read from the instance. The view holds the whole terrain
    unless it is much wider than the lander, in which case it holds the start and the pads, and
    it grows as the lander leaves it. Legs that compress on contact go into the ground, which is
    drawn over them.

    The engines are drawn from the levels in the state, which are those of the last step. The
    path and time are built from the states seen so far, so they require render to be called on
    every step. The state does not hold the time, so it is the number of different states seen
    multiplied by the time step. The tipping angle is for level ground with the center of mass
    over the origin of the body.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_size=(9.0, 5.5),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._dt = nonfluents['DT']
        self._break_speed = nonfluents['BREAK-SPEED']
        self._side_arm = nonfluents['SIDE-ARM']
        self._duration = model.horizon * self._dt

        self._segments = self._load_terrain(model, nonfluents)
        self._load_body(model, nonfluents)
        self._load_pads()

        self._init_state = self._get_state(model.ground_vars_with_values(model.state_fluents))
        self._last_state = None
        self._path = []
        self._bounds = self._get_static_bounds()
        self._view = None

        self._scene_width, self._scene_height = self._get_scene_size(max_size)
        self._draw_figure()

    def _load_terrain(self, model, nonfluents):
        points = model.type_to_objects['point']
        segments = []
        for p in points:
            for q in points:
                if nonfluents[RDDLPlanningModel.ground_var('NEXT', [p, q])]:
                    segments.append((
                        float(nonfluents[RDDLPlanningModel.ground_var('X', [p])]),
                        float(nonfluents[RDDLPlanningModel.ground_var('Y', [p])]),
                        float(nonfluents[RDDLPlanningModel.ground_var('X', [q])]),
                        float(nonfluents[RDDLPlanningModel.ground_var('Y', [q])]),
                        bool(nonfluents[RDDLPlanningModel.ground_var('PAD', [p, q])])))
        return segments

    def _load_body(self, model, nonfluents):
        hull, legs = [], []
        for r in model.type_to_objects['probe']:
            point = (float(nonfluents[RDDLPlanningModel.ground_var('PX', [r])]),
                     float(nonfluents[RDDLPlanningModel.ground_var('PY', [r])]))
            is_leg = nonfluents[RDDLPlanningModel.ground_var('IS-LEG', [r])]
            (legs if is_leg else hull).append(point)
        self._legs = legs

        # the outline of the hull, whichever order the probes are in
        if len(hull) >= 3:
            center_x = sum(p[0] for p in hull) / len(hull)
            center_y = sum(p[1] for p in hull) / len(hull)
            hull = sorted(hull, key=lambda p: math.atan2(p[1] - center_y, p[0] - center_x))
        self._hull = hull

        # the size of the body, from which everything else that is drawn on it is scaled
        extent = [abs(c) for p in hull + legs for c in p]
        self._body = 2 * max(extent) if extent and max(extent) > 0 else 1.0

        # a body with too few hull probes to have an outline is drawn as a small box
        outline = hull if len(hull) >= 3 else [
            (0.3 * self._body * a, 0.3 * self._body * b)
            for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        self._outline = outline
        self._hull_left = min(p[0] for p in outline)
        self._hull_right = max(p[0] for p in outline)
        self._hull_bottom = min(p[1] for p in outline)
        self._hull_top = max(p[1] for p in outline)

        # each leg joins the hull at the nearest corner of its outline
        self._leg_roots = [min(outline, key=lambda p: (p[0] - leg[0]) ** 2 + (p[1] - leg[1]) ** 2)
                           for leg in legs]

        # the body tips over a leg once the center of mass passes over it
        right = [leg for leg in legs if leg[0] > 0 and leg[1] < 0]
        left = [leg for leg in legs if leg[0] < 0 and leg[1] < 0]
        self._tip_right = max((math.degrees(math.atan2(p[0], -p[1])) for p in right), default=None)
        self._tip_left = max((math.degrees(math.atan2(-p[0], -p[1])) for p in left), default=None)

    def _load_pads(self):
        pads = [s for s in self._segments if s[4]]
        self._pad_middles = [((s[0] + s[2]) / 2, (s[1] + s[3]) / 2) for s in pads]

        # pads that share a point are one pad with its flags at the outer ends, flying over the pad
        ends = {}
        for s in pads:
            ends.setdefault((s[0], s[1]), []).append(1)
            ends.setdefault((s[2], s[3]), []).append(-1)
        self._flags = [(x, y, sides[0]) for ((x, y), sides) in ends.items() if len(sides) == 1]

    def _get_state(self, state):
        return tuple(float(state[var]) for var in (
            'x', 'y', 'vx', 'vy', 'ang', 'omega', 'crashed', 'settled', 'on-pad',
            'main-level', 'side-level'))

    # ==========================================================================
    # VIEW
    # ==========================================================================

    def _heights_between(self, left, right):
        heights = []
        for (x0, y0, x1, y1, _) in self._segments:
            if x1 >= left and x0 <= right:
                slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
                for x in (max(x0, left), min(x1, right)):
                    heights.append(y0 + slope * (x - x0))
        return heights

    def _get_static_bounds(self):
        start_x, start_y = self._init_state[:2]
        body = self._body
        xs = [start_x - body, start_x + body]
        ys = [start_y - body, start_y + body]

        if self._segments:
            left = min(s[0] for s in self._segments)
            right = max(s[2] for s in self._segments)
            if right - left <= MAX_FULL_VIEW * body:
                xs += [left, right]
            else:
                middles = [m[0] for m in self._pad_middles] + [start_x]
                xs += [min(middles) - VIEW_MARGIN * body, max(middles) + VIEW_MARGIN * body]
            ys += self._heights_between(min(xs), max(xs))
        return (min(xs), max(xs), min(ys) - 0.3 * body, max(ys) + 0.5 * body)

    def _get_scene_size(self, max_size):

        # use the shape of what is shown as far as the figure and the gauges allow
        left, right, bottom, top = self._bounds
        max_height = max(max_size[1] - BOTTOM_MARGIN - TOP_MARGIN, MIN_SCENE_HEIGHT)
        max_width = min(MAX_SCENE_WIDTH,
                        max_size[0] - PANEL_WIDTH - PANEL_GAP - 2 * SIDE_MARGIN)
        width, height = max_width, max_width * (top - bottom) / (right - left)
        if height > max_height:
            width, height = width * max_height / height, max_height
        return max(width, MIN_SCENE_WIDTH), max(height, MIN_SCENE_HEIGHT)

    def _update_view(self):

        # both axes use the same scale, and any spare height goes to the sky
        left, right, bottom, top = self._bounds
        scale = max(VIEW_PADDING * (right - left) / self._scene_width,
                    (top - bottom) / self._scene_height)
        middle = (left + right) / 2
        view = (middle - scale * self._scene_width / 2, middle + scale * self._scene_width / 2,
                bottom, bottom + scale * self._scene_height)
        if view != self._view:
            self._view = view
            self.ax.set_xlim(view[0], view[1])
            self.ax.set_ylim(view[2], view[3])
        return scale

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
        self.ax.set_facecolor(SPACE_COLOR)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')
        self._update_view()

        self._draw_terrain()
        self._draw_lander()
        self._draw_panel(fig_width, fig_height)

        self.fig.legend(handles=[
            Patch(facecolor=HULL_COLOR, label='Lander'),
            Patch(facecolor=FLAME_COLOR, label='Engine'),
            Line2D([], [], color=PATH_COLOR, linewidth=1.2, label='Path'),
            Line2D([], [], color=PAD_COLOR, linewidth=4, solid_capstyle='butt', label='Pad'),
            Patch(facecolor=GROUND_COLOR, edgecolor=GROUND_EDGE_COLOR, label='Ground')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / fig_height), ncol=5, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)
        self.fig.text(0.5, 0.1 / fig_height,
                      f'A leg arriving faster than {self._break_speed:g}, the hull touching '
                      f'the ground or leaving the terrain is a crash',
                      ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)
        self._title = self.fig.text(0.5, 1 - 0.12 / fig_height, '', ha='center', va='top',
                                    fontsize=13, fontweight='bold')

    def _draw_terrain(self):

        # drawn over the lander so that compressed legs go into the ground
        view_height = self._view[3] - self._view[2]
        reach = max([self._view[1] - self._view[0], view_height] +
                    [abs(c) for s in self._segments for c in s[:4]])
        base = min([s[1] for s in self._segments] + [s[3] for s in self._segments] +
                   [self._view[2]]) - 1000 * reach
        self.ax.add_collection(PolyCollection(
            [[(s[0], s[1]), (s[2], s[3]), (s[2], base), (s[0], base)] for s in self._segments],
            facecolors=GROUND_COLOR, edgecolors='none', antialiaseds=False, zorder=4))
        self.ax.add_collection(LineCollection(
            [[(s[0], s[1]), (s[2], s[3])] for s in self._segments],
            colors=GROUND_EDGE_COLOR, linewidths=1.5, zorder=5))
        self.ax.add_collection(LineCollection(
            [[(s[0], s[1]), (s[2], s[3])] for s in self._segments if s[4]],
            colors=PAD_COLOR, linewidths=4, capstyle='butt', zorder=6))

        # a flag at each end of a pad, about as tall as the lander when the view allows
        pole = min(0.09 * view_height, 1.5 * self._body)
        for (x, y, side) in self._flags:
            self.ax.plot([x, x], [y, y + pole], color=LEG_COLOR, linewidth=1, zorder=6)
            self.ax.add_patch(Polygon(
                [(x, y + pole), (x, y + 0.6 * pole), (x + side * 0.5 * pole, y + 0.8 * pole)],
                closed=True, facecolor=PAD_COLOR, edgecolor='none', zorder=6))

    def _draw_lander(self):
        empty = [(0, 0), (0, 0), (0, 0)]
        self._path_line, = self.ax.plot([], [], color=PATH_COLOR, linewidth=1.2, zorder=1)
        self._main_flame = Polygon(empty, closed=True, facecolor=FLAME_COLOR, edgecolor='none', zorder=2)
        self._side_flame = Polygon(empty, closed=True, facecolor=FLAME_COLOR, edgecolor='none', zorder=2)
        self._hull_patch = Polygon(empty, closed=True, facecolor=HULL_COLOR, edgecolor='white',
                                   linewidth=0.8, zorder=3)
        self._leg_lines = LineCollection([], colors=LEG_COLOR, linewidths=2, capstyle='round', zorder=3)
        self._foot_lines = LineCollection([], colors=LEG_COLOR, linewidths=3, capstyle='round', zorder=3)
        for patch in (self._main_flame, self._side_flame, self._hull_patch):
            self.ax.add_patch(patch)
        self.ax.add_collection(self._leg_lines)
        self.ax.add_collection(self._foot_lines)
        self._scale_note = self.ax.text(
            0.01, 0.99, '', transform=self.ax.transAxes, ha='left', va='top', fontsize=8,
            color=SCENE_TEXT_COLOR, zorder=7)

    def _draw_panel(self, fig_width, fig_height):
        panel = self.fig.add_axes([
            (SIDE_MARGIN + self._scene_width + PANEL_GAP) / fig_width, BOTTOM_MARGIN / fig_height,
            PANEL_WIDTH / fig_width, self._scene_height / fig_height])
        panel.set_xlim(0, PANEL_WIDTH)
        panel.set_ylim(0, self._scene_height)
        panel.axis('off')
        self._panel = panel

        # the rows spread out with the height of the scene, within reason
        pitch = min(self._scene_height / 5.5, 0.72)
        top = self._scene_height
        limit = self._break_speed if self._break_speed > 0 else None
        self._speed_range = SPEED_GAUGE_RANGE * limit if limit else 1.0
        self._descent = self._draw_gauge('Descent', top, [limit / self._speed_range] if limit else [])
        self._drift = self._draw_gauge('Drift', top - pitch, [])

        # centered gauges with leaning or pushing to the right being on the right
        marks = []
        if self._tip_right is not None:
            marks.append(0.5 + 0.5 * min(self._tip_right / MAX_TILT_GAUGE, 1.0))
        if self._tip_left is not None:
            marks.append(0.5 - 0.5 * min(self._tip_left / MAX_TILT_GAUGE, 1.0))
        self._tilt = self._draw_gauge('Tilt', top - 2 * pitch, marks, centered=True)
        self._main_gauge = self._draw_gauge('Main engine', top - 3 * pitch, [])
        self._side_gauge = self._draw_gauge('Side engine', top - 4 * pitch, [], centered=True)
        self._time = panel.text(0, top - 5 * pitch, '', ha='left', va='top', fontsize=9, color=TEXT_COLOR)

    def _draw_gauge(self, label, top, marks, centered=False):
        panel = self._panel
        panel.text(0, top, label, ha='left', va='top', fontsize=9, color=TEXT_COLOR)
        value = panel.text(PANEL_WIDTH, top, '', ha='right', va='top', fontsize=9, color=TEXT_COLOR)
        bar_y = top - 0.36
        panel.add_patch(Rectangle((0, bar_y), PANEL_WIDTH, 0.14, facecolor=TRACK_COLOR, edgecolor='none', zorder=1))
        fill = Rectangle((0, bar_y), 0, 0.14, facecolor=LANDER_COLOR, edgecolor='none', zorder=2)
        panel.add_patch(fill)
        if centered:
            panel.plot([PANEL_WIDTH / 2] * 2, [bar_y - 0.03, bar_y + 0.17], color='#b0bec5', linewidth=1, zorder=3)
        for mark in marks:
            panel.plot([mark * PANEL_WIDTH] * 2, [bar_y - 0.04, bar_y + 0.18], color=BAD_COLOR, linewidth=1.5, zorder=3)
        return {'value': value, 'fill': fill, 'centered': centered}

    # ==========================================================================
    # UPDATING
    # ==========================================================================

    def _update_history(self, state):

        # rendering the same state again must not change the path or view
        if state == self._last_state:
            return
        last_state, self._last_state = self._last_state, state

        # an episode ends with a crash or by settling, and can be restarted at any time
        if last_state is None or state == self._init_state or last_state[6] or last_state[7]:
            self._path = []
            self._bounds = self._get_static_bounds()
        self._path.append(state[:2])

        x, y = state[:2]
        left, right, bottom, top = self._bounds
        body = self._body
        self._bounds = (min(left, x - body), max(right, x + body),
                        min(bottom, y - body), max(top, y + body))

    def _set_gauge(self, gauge, fraction, text, color=LANDER_COLOR):
        fraction = max(min(fraction, 1.0), -1.0)
        if gauge['centered']:
            middle = PANEL_WIDTH / 2
            gauge['fill'].set_x(middle + min(fraction, 0.0) * middle)
            gauge['fill'].set_width(abs(fraction) * middle)
        else:
            gauge['fill'].set_width(max(fraction, 0.0) * PANEL_WIDTH)
        gauge['fill'].set_facecolor(color)
        gauge['value'].set_text(text)

    def _update_gauges(self, state):
        vx, vy, ang = state[2], state[3], state[4]
        main_level, side_level = state[9], state[10]

        descent = -vy if vy < 0 else 0.0
        color = LANDER_COLOR
        if self._break_speed > 0:
            if descent > self._break_speed:
                color = BAD_COLOR
            elif descent > WARNING_FRACTION * self._break_speed:
                color = WARNING_COLOR
        text = f'{self._number(descent)}' if vy <= 0 else f'climbing {self._number(vy)}'
        self._set_gauge(self._descent, descent / self._speed_range, text, color)

        side = 'right' if vx >= 0 else 'left'
        self._set_gauge(self._drift, abs(vx) / self._speed_range, f'{self._number(abs(vx))} {side}')

        # turning counter-clockwise leans to the left
        tilt = math.degrees(math.atan2(math.sin(ang), math.cos(ang)))
        limit = self._tip_left if tilt > 0 else self._tip_right
        color = BAD_COLOR if limit is not None and abs(tilt) > limit else LANDER_COLOR
        side = 'left' if tilt > 0 else 'right'
        self._set_gauge(self._tilt, -tilt / MAX_TILT_GAUGE, f'{abs(tilt):.0f} deg {side}', color)

        text = f'{main_level:.0%}' if main_level >= VISIBLE_LEVEL else 'off'
        self._set_gauge(self._main_gauge, main_level, text, FLAME_COLOR)
        side = 'push right' if side_level >= 0 else 'push left'
        text = f'{side} {abs(side_level):.0%}' if abs(side_level) >= VISIBLE_LEVEL else 'off'
        self._set_gauge(self._side_gauge, side_level, text, FLAME_COLOR)

        time = (len(self._path) - 1) * self._dt
        self._time.set_text(f'Time {time:.2f} of {self._duration:g}')

    def _update_lander(self, state, view_scale):
        x, y, _, _, ang, _, crashed, settled, on_pad, main_level, side_level = state

        # the lander keeps a size that can be seen as the view grows
        scale = max(MIN_BODY_INCHES * view_scale / self._body, 1.0)
        cos, sin = math.cos(ang), math.sin(ang)

        def world(px, py):
            return (x + scale * (px * cos - py * sin), y + scale * (px * sin + py * cos))

        color = HULL_COLOR
        if crashed:
            color = HULL_CRASHED_COLOR
        elif settled:
            color = HULL_LANDED_COLOR if on_pad or not self._pad_middles else HULL_OFF_PAD_COLOR
        self._hull_patch.set_facecolor(color)
        self._hull_patch.set_xy([world(*p) for p in self._outline])

        # a crash can bury the hull, which must still be seen
        self._hull_patch.set_zorder(7 if crashed else 3)

        self._leg_lines.set_segments([[world(*root), world(*leg)]
                                      for (root, leg) in zip(self._leg_roots, self._legs)])
        foot = 0.05 * self._body
        self._foot_lines.set_segments([[world(leg[0] - foot, leg[1]), world(leg[0] + foot, leg[1])]
                                       for leg in self._legs])

        # the exhaust of the main engine leaves through the bottom of the hull
        middle = (self._hull_left + self._hull_right) / 2
        half = 0.25 * (self._hull_right - self._hull_left)
        length = main_level * 0.8 * self._body
        self._main_flame.set_visible(main_level >= VISIBLE_LEVEL)
        self._main_flame.set_xy([world(middle - half, self._hull_bottom),
                                 world(middle + half, self._hull_bottom),
                                 world(middle, self._hull_bottom - length)])

        # pushing to the right takes exhaust leaving to the left, at the height of the engine
        height = max(min(self._side_arm, self._hull_top), self._hull_bottom)
        edge = self._hull_left if side_level >= 0 else self._hull_right
        direction = -1 if side_level >= 0 else 1
        half = 0.08 * self._body
        length = abs(side_level) * 0.4 * self._body
        self._side_flame.set_visible(abs(side_level) >= VISIBLE_LEVEL)
        self._side_flame.set_xy([world(edge, height - half), world(edge, height + half),
                                 world(edge + direction * length, height)])

        self._path_line.set_data(*zip(*self._path))
        self._scale_note.set_text(f'Lander drawn {scale:.0f}x larger' if scale >= 1.5 else '')

    def _ground_under(self, x):
        for (x0, y0, x1, y1, _) in self._segments:
            if x0 <= x < x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return None

    def _clearance(self, state):
        '''How far the lowest point of the body is above the terrain under it.'''
        x, y, ang = state[0], state[1], state[4]
        cos, sin = math.cos(ang), math.sin(ang)
        gaps = []
        for (px, py) in self._legs + self._hull:
            ground = self._ground_under(x + px * cos - py * sin)
            if ground is not None:
                gaps.append(y + px * sin + py * cos - ground)
        return min(gaps) if gaps else None

    def _update_title(self, state):
        x, y, crashed, settled, on_pad = state[0], state[1], state[6], state[7], state[8]
        time = (len(self._path) - 1) * self._dt
        if crashed:
            text, color = f'Crashed at time {time:.2f}', BAD_COLOR
        elif settled and not self._pad_middles:
            text, color = f'Landed at time {time:.2f}', GOOD_COLOR
        elif settled and on_pad:
            text, color = f'Landed on the pad at time {time:.2f}', GOOD_COLOR
        elif settled:
            text, color = f'Landed off the pad at time {time:.2f}', WARNING_COLOR
        else:
            parts = []
            clearance = self._clearance(state)
            if clearance is not None:
                parts.append('touching down' if clearance <= 0 else f'{self._number(clearance)} above ground')
            if self._pad_middles:
                distance = min(math.hypot(x - m[0], y - m[1]) for m in self._pad_middles)
                parts.append(f'{self._number(distance)} from the pad')
            text, color = ', '.join(['Flying'] + parts), 'black'
        self._title.set_text(text)
        self._title.set_color(color)

    @staticmethod
    def _number(value):
        return f'{value:.2f}' if abs(value) < 100 else f'{value:.0f}'

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        state = self._get_state(state)
        self._update_history(state)
        view_scale = self._update_view()
        self._update_lander(state, view_scale)
        self._update_gauges(state)
        self._update_title(state)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
