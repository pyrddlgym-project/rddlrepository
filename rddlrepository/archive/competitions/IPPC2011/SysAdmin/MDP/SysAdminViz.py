import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

RUNNING_COLOR = '#c8e6c9'
RUNNING_EDGE_COLOR = '#2e7d32'
RISK_COLOR = '#ffe0b2'
RISK_EDGE_COLOR = '#ff8f00'
DOWN_COLOR = '#ffcdd2'
DOWN_EDGE_COLOR = '#c62828'
DETAIL_COLOR = '#37474f'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class SysAdminVisualizer(BaseViz):
    '''Draws the computers as a rack of tiles in their declared order.

    The network shows on the tiles rather than as lines between them: each running computer
    has the number of its feeds that are up, and names those that are down, since these
    make it more likely to fail. Reboots are not part of the state, so they only show as the
    computer running afterwards. The POMDP shows which computers are really running rather
    than only what is observed. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_size=(9.0, 5.5),
                 max_cell_size=(1.05, 0.8),
                 min_width=7.8,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._computers = list(model.type_to_objects['computer'])
        self._feeds = {b: [a for a in self._computers
                           if nonfluents[RDDLPlanningModel.ground_var('CONNECTED', [a, b])]]
                       for b in self._computers}

        # a rack wider than it is tall, as the figure is
        count = len(self._computers)
        num_cols = math.ceil(math.sqrt(1.8 * count))
        num_rows = math.ceil(count / num_cols)
        cell_width = min(max_cell_size[0], max_size[0] / num_cols)
        cell_height = min(max_cell_size[1], max_size[1] / num_rows)
        scale = min(1.0, cell_width / max_cell_size[0], cell_height / max_cell_size[1])

        # title above, then the rack, the legend and the caption
        bottom, top = 0.62, 0.45
        grid_width, grid_height = num_cols * cell_width, num_rows * cell_height
        width = max(grid_width + 0.4, min_width)
        height = grid_height + bottom + top
        self._figure_size = (width, height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width), bottom / height,
             grid_width / width, grid_height / height])
        self.ax.set_xlim(0, num_cols)
        self.ax.set_ylim(num_rows, 0)
        self.ax.axis('off')

        # the tiles never move, so they are only drawn once and then recolored
        gap = 0.08
        tile_width, tile_height = 1 - gap / cell_width, 1 - gap / cell_height
        name_fit = 0.9 * (cell_width - gap) * 72 / 0.62
        self._detail_size = 7.5 * scale
        self._has_details = self._detail_size >= MIN_FONT_SIZE
        self._name_limit = int(0.9 * (cell_width - gap) * 72 / (0.56 * self._detail_size))
        self._tiles, self._counts, self._down_names = {}, {}, {}
        for (i, computer) in enumerate(self._computers):
            row, col = divmod(i, num_cols)
            self._tiles[computer] = self.ax.add_patch(patches.FancyBboxPatch(
                (col + (1 - tile_width) / 2, row + (1 - tile_height) / 2), tile_width, tile_height,
                boxstyle='round,pad=0,rounding_size=0.07', mutation_aspect=cell_width / cell_height,
                linewidth=1.2))

            name = self._shorten(computer, int(name_fit / MIN_FONT_SIZE))
            self.ax.text(
                col + 0.5, row + (0.25 if self._has_details else 0.5), name, ha='center', va='center',
                fontsize=max(min(10 * scale, name_fit / len(name)), MIN_FONT_SIZE), fontweight='bold')
            if self._has_details:
                self._counts[computer] = self.ax.text(
                    col + 0.5, row + 0.52, '', ha='center', va='center',
                    fontsize=self._detail_size, color=DETAIL_COLOR)
                self._down_names[computer] = self.ax.text(
                    col + 0.5, row + 0.76, '', ha='center', va='center',
                    fontsize=self._detail_size, color=DOWN_EDGE_COLOR)

        self.fig.legend(handles=[
            patches.Patch(facecolor=RUNNING_COLOR, edgecolor=RUNNING_EDGE_COLOR, label='Running'),
            patches.Patch(facecolor=RISK_COLOR, edgecolor=RISK_EDGE_COLOR, label='Running, feed down'),
            patches.Patch(facecolor=DOWN_COLOR, edgecolor=DOWN_EDGE_COLOR, label='Down')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / height), ncol=3, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)

        restart = nonfluents[RDDLPlanningModel.ground_var('REBOOT-PROB', [])]
        penalty = nonfluents[RDDLPlanningModel.ground_var('REBOOT-PENALTY', [])]
        self.fig.text(0.5, 0.1 / height,
                      f'Red names are feeds that are down. Down computers restart alone with '
                      f'chance {restart:g}, and a reboot costs {penalty:g}',
                      ha='center', va='bottom', fontsize=9, color=DETAIL_COLOR)

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _list_names(self, names):

        # as many names as fit on the tile, then how many more there are
        text = ''
        for (i, name) in enumerate(names):
            longer = f'{text} {name}' if text else name
            rest = len(names) - i - 1
            if len(longer) + (len(f' +{rest}') if rest else 0) > self._name_limit:
                return f'{text} +{len(names) - i}' if text else f'+{len(names)}'
            text = longer
        return text

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        running = {computer for computer in self._computers
                   if state[RDDLPlanningModel.ground_var('running', [computer])]}

        for computer in self._computers:
            feeds = self._feeds[computer]
            feeds_down = [feed for feed in feeds if feed not in running]

            if computer not in running:
                fill, edge, count, names = DOWN_COLOR, DOWN_EDGE_COLOR, 'Down', ''
            else:
                names = self._list_names(feeds_down)
                count = f'{len(feeds) - len(feeds_down)} / {len(feeds)} feeds up' if feeds else 'No feeds'
                if feeds_down:
                    fill, edge = RISK_COLOR, RISK_EDGE_COLOR
                else:
                    fill, edge = RUNNING_COLOR, RUNNING_EDGE_COLOR

            self._tiles[computer].set_facecolor(fill)
            self._tiles[computer].set_edgecolor(edge)
            if self._has_details:
                self._counts[computer].set_text(count)
                self._down_names[computer].set_text(names)

        if len(running) == len(self._computers):
            title, color = 'All computers running', RUNNING_EDGE_COLOR
        elif not running:
            title, color = 'All computers down', DOWN_EDGE_COLOR
        else:
            title = f'{len(running)} / {len(self._computers)} computers running'
            color = 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
