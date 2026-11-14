import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BEZEL_COLOR = '#37474f'
SCREEN_COLOR = '#4fc3f7'
OFF_BEZEL_COLOR = '#cfd8dc'
OFF_SCREEN_COLOR = '#b0bec5'
OFF_LIGHT_COLOR = '#e0e6e9'
FEED_UP_COLOR = '#66bb6a'
FEED_DOWN_COLOR = '#e53935'
WARNING_COLOR = '#ffa000'
RUNNING_TITLE_COLOR = '#2e7d32'
DOWN_TITLE_COLOR = '#c62828'
TEXT_COLOR = '#37474f'

# a computer is designed for a cell of this size in inches, and shrinks with its cell
CELL = (1.05, 0.8)

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class ComplexSysAdminVisualizer(BaseViz):
    '''Draws the computers as monitors on a desk, in their declared order.

    A running computer has a lit screen, and one that is down is switched off and greyed
    out. The network shows as lights on the monitors rather than as lines between them:
    each light is a feed of the computer, green while that feed is up and red while it is
    down. A running computer can only stay up for sure if it has feeds and they are all
    up, so any other running computer shows a warning. Reboots are not part of the state,
    so they only show as the computer running afterwards. This shows which computers are
    really running rather than only what is observed. The drawing only depends on the
    current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_size=(9.0, 5.5),
                 max_cell_size=CELL,
                 min_width=7.8,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._computers = list(model.type_to_objects['computer'])
        self._feeds = {b: [a for a in self._computers
                           if nonfluents[RDDLPlanningModel.ground_var('CONNECTED', [a, b])]]
                       for b in self._computers}

        # a desk wider than it is deep, as the figure is
        count = len(self._computers)
        num_cols = math.ceil(math.sqrt(1.8 * count))
        num_rows = math.ceil(count / num_cols)
        cell_width = min(max_cell_size[0], max_size[0] / num_cols)
        cell_height = min(max_cell_size[1], max_size[1] / num_rows)
        scale = min(cell_width / CELL[0], cell_height / CELL[1])

        # title above, then the computers, the legend and the caption: the axes are in
        # inches, so that the computers keep their shape
        bottom, top = 0.85, 0.45
        grid_width, grid_height = num_cols * cell_width, num_rows * cell_height
        width = max(grid_width + 0.4, min_width)
        height = grid_height + bottom + top
        self._figure_size = (width, height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width), bottom / height,
             grid_width / width, grid_height / height])
        self.ax.set_xlim(0, grid_width)
        self.ax.set_ylim(grid_height, 0)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # the computers never move, so they are only drawn once and then switched on
        # and off; names go below them where they are readable
        name_fit = 72 * 0.95 * cell_width / 0.62
        self._parts = {}
        for (i, computer) in enumerate(self._computers):
            row, col = divmod(i, num_cols)
            left = col * cell_width + (cell_width - CELL[0] * scale) / 2
            top_edge = row * cell_height + (cell_height - CELL[1] * scale) / 2
            self._parts[computer] = self._draw_computer(
                self.ax, left, top_edge, scale, len(self._feeds[computer]))
            if 8.5 * scale >= MIN_FONT_SIZE:
                name = self._shorten(computer, int(name_fit / MIN_FONT_SIZE))
                self.ax.text(left + 0.525 * scale, top_edge + 0.715 * scale, name,
                             ha='center', va='center', color=TEXT_COLOR,
                             fontsize=max(min(8.5 * scale, name_fit / len(name)), MIN_FONT_SIZE))

        self._draw_legend(width, height, 'Can fail')
        restart = nonfluents[RDDLPlanningModel.ground_var('REBOOT-PROB', [])]
        self.fig.text(0.5, 0.08 / height,
                      f'Computers that are off restart by themselves with chance {restart:g}, '
                      'and a reboot costs 1',
                      ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)

    @staticmethod
    def _draw_computer(ax, left, top, scale, num_feeds):

        # a monitor on a stand, laid out in a cell of the usual size and scaled: the lights
        # on its chin are its feeds
        def at(x, y):
            return (left + x * scale, top + y * scale)

        def box(x0, y0, x1, y1, radius=0.0, zorder=1, **kwargs):
            return ax.add_patch(patches.FancyBboxPatch(
                at(x0, y0), (x1 - x0) * scale, (y1 - y0) * scale,
                boxstyle=f'round,pad=0,rounding_size={radius * scale}', edgecolor='none',
                zorder=zorder, **kwargs))

        (cx, cy) = (0.525, 0.25)
        spacing = min(0.085, 0.65 / max(num_feeds, 1))
        first = cx - spacing * (num_feeds - 1) / 2
        return {
            'body': [box(0.495, 0.5, 0.555, 0.6), box(0.38, 0.59, 0.67, 0.63, 0.015),
                     box(0.12, 0.04, 0.93, 0.52, 0.04)],
            'screen': box(0.16, 0.08, 0.89, 0.42, zorder=2),
            'lines': [box(0.21, y, 0.21 + length, y + 0.025, 0.0125, zorder=3,
                          facecolor='white', alpha=0.75)
                      for (y, length) in ((0.13, 0.42), (0.19, 0.52), (0.25, 0.3), (0.31, 0.46))],
            'warning': [
                ax.add_patch(patches.Polygon(
                    [at(cx - 0.14, cy + 0.12), at(cx + 0.14, cy + 0.12), at(cx, cy - 0.13)],
                    closed=True, facecolor=WARNING_COLOR, edgecolor=BEZEL_COLOR,
                    linewidth=0.6, joinstyle='round', zorder=4)),
                box(cx - 0.013, cy - 0.05, cx + 0.013, cy + 0.04, 0.01, zorder=5,
                    facecolor=BEZEL_COLOR),
                ax.add_patch(patches.Circle(at(cx, cy + 0.08), 0.017 * scale,
                                            facecolor=BEZEL_COLOR, zorder=5))],
            'lights': [ax.add_patch(patches.Circle(
                at(first + k * spacing, 0.47), min(0.028, 0.35 * spacing) * scale,
                edgecolor='none', zorder=3)) for k in range(num_feeds)]
        }

    @staticmethod
    def _show(parts, status, feeds_up):

        # a computer that is off is greyed out, and its lights with it
        on = status != 'off'
        for artist in parts['body']:
            artist.set_facecolor(BEZEL_COLOR if on else OFF_BEZEL_COLOR)
        parts['screen'].set_facecolor(SCREEN_COLOR if on else OFF_SCREEN_COLOR)
        for artist in parts['lines']:
            artist.set_visible(on)
        for artist in parts['warning']:
            artist.set_visible(status == 'risk')
        for (light, up) in zip(parts['lights'], feeds_up):
            light.set_facecolor(OFF_LIGHT_COLOR if not on else
                                FEED_UP_COLOR if up else FEED_DOWN_COLOR)

    def _draw_legend(self, width, height, risk_label):

        # the legend shows the same computers as the desk, each with two feeds
        ax = self.fig.add_axes([0, 0.3 / height, 1, 0.42 / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0.42, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        entries = (('running', (True, True), 'Running'), ('risk', (True, False), risk_label),
                   ('off', (True, True), 'Off'))
        widths = [0.6 + 0.075 * len(label) + 0.35 for (_, _, label) in entries]
        x = (width - sum(widths)) / 2
        for ((status, feeds_up, label), entry_width) in zip(entries, widths):
            self._show(self._draw_computer(ax, x - 0.07, 0.0, 0.6, 2), status, feeds_up)
            ax.text(x + 0.55, 0.19, label, va='center', fontsize=8.5, color=TEXT_COLOR)
            x += entry_width

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        running = {computer for computer in self._computers
                   if state[RDDLPlanningModel.ground_var('running', [computer])]}

        for computer in self._computers:
            feeds_up = [feed in running for feed in self._feeds[computer]]
            if computer not in running:
                status = 'off'
            else:
                status = 'running' if feeds_up and all(feeds_up) else 'risk'
            self._show(self._parts[computer], status, feeds_up)

        if len(running) == len(self._computers):
            title, color = 'All computers running', RUNNING_TITLE_COLOR
        elif not running:
            title, color = 'All computers down', DOWN_TITLE_COLOR
        else:
            title = f'{len(running)} / {len(self._computers)} computers running'
            color = 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
