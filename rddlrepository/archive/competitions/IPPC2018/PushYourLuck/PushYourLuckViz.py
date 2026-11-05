import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

OPEN_COLOR = '#ffffff'
OPEN_EDGE_COLOR = '#90a4ae'
MARKED_COLOR = '#c8e6c9'
MARKED_EDGE_COLOR = '#2e7d32'
NEW_COLORS = ('#e3f2fd', '#90caf9')
BUST_COLORS = ('#ffebee', '#ef9a9a')
NEVER_COLOR = '#f5f7f8'
LOSING_COLOR = '#c62828'
DETAIL_COLOR = '#37474f'

# how much a chance colors its cell at the least, so that small chances still show
MIN_SHADE = 0.3

# sizes in inches
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
MAX_SIZE = (9.0, 5.5)
MIN_WIDTH = 8.6
MIN_HEIGHT = 2.8
LABEL_WIDTH = 0.7
BUST_WIDTH = 0.55
ROLLED_WIDTH = 0.85
MAX_CELL_WIDTH = 0.62
FACE_HEIGHT = 0.62
MAX_ROW_HEIGHT = 0.42
ROW_GAP = 0.05
CELL_GAP = 0.03

# text does not fit in smaller rows, and a chance does not fit in narrower cells
MIN_TEXT_ROW = 0.16
LARGE_CELL = 0.5
CHAR_WIDTH = 0.7


class PushYourLuckVisualizer(BaseViz):
    '''Draws the faces on top with their values and whether they are marked, and below them
    the chance of each die to come up each face, which is red for a face that would clear
    the marks. At the side of each die is its chance to clear them, and the expected reward
    of rolling it and then cashing out, to compare with what cashing out pays now.

    The rolled dice are not part of the state, so they only show in the marks, and clearing
    the marks by bad luck looks the same as cashing out. The drawing only depends on the
    current state.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        initial = model.ground_vars_with_values(model.state_fluents)
        fluent = RDDLPlanningModel.ground_var

        self._dice = list(model.type_to_objects['die'])
        self._numbers = list(model.type_to_objects['number'])

        # the type has every face that any die in any instance can have, so only those that
        # can come up or are already marked are drawn
        self._faces = [number for number in self._numbers
                       if initial[fluent('die-value-seen', [number])]
                       or any(self._nonfluent('PROB', die, number) > 0 for die in self._dice)]
        self._max_chance = max((self._nonfluent('PROB', die, number)
                                for die in self._dice for number in self._faces), default=0)

        # everything is laid out in inches
        room = MAX_SIZE[0] - LABEL_WIDTH - BUST_WIDTH - ROLLED_WIDTH
        self._cell_width = min(MAX_CELL_WIDTH, room / max(len(self._faces), 1))
        self._row_height = min(MAX_ROW_HEIGHT,
                               (MAX_SIZE[1] - FACE_HEIGHT - ROW_GAP) / max(len(self._dice), 1))
        grid_width = (LABEL_WIDTH + len(self._faces) * self._cell_width
                      + BUST_WIDTH + ROLLED_WIDTH)
        grid_height = FACE_HEIGHT + ROW_GAP + len(self._dice) * self._row_height

        width = max(grid_width + 0.4, MIN_WIDTH)
        height = max(grid_height + BOTTOM_MARGIN + TOP_MARGIN, MIN_HEIGHT)
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, BOTTOM_MARGIN / height, 1,
                                     (height - BOTTOM_MARGIN - TOP_MARGIN) / height])
        extra = height - BOTTOM_MARGIN - TOP_MARGIN - grid_height
        self.ax.set_xlim(-(width - grid_width) / 2, (width + grid_width) / 2)
        self.ax.set_ylim(-extra / 2, grid_height + extra / 2)
        self.ax.axis('off')

        self._left = LABEL_WIDTH
        self._grid_height = grid_height
        self._draw_faces()
        self._draw_dice()

        handles = [
            patches.Patch(facecolor=MARKED_COLOR, edgecolor=MARKED_EDGE_COLOR, label='Marked'),
            patches.Patch(facecolor=OPEN_COLOR, edgecolor=OPEN_EDGE_COLOR, label='Not marked'),
            patches.Patch(facecolor=NEW_COLORS[1], edgecolor='none', label='Chance of a new mark'),
            patches.Patch(facecolor=BUST_COLORS[1], edgecolor='none', label='Chance to clear the marks')
        ]
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.25 / height),
                        ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                        handlelength=1.4)

        if len(self._dice) > 1:
            caption = ('Rolling a marked face or two equal faces clears all marks. '
                       'If rolled is the expected cash-out after rolling only that die.')
        else:
            caption = ('Rolling a marked face clears all marks. '
                       'If rolled is the expected cash-out after rolling the die once.')
        self.fig.text(0.5, 0.1 / height, caption, ha='center', va='bottom', fontsize=9,
                      color=DETAIL_COLOR)

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _amount(self, amount):

        # rewards grow very fast with the marks, and keep only what can be told apart
        if abs(amount) < 1000:
            return f'{amount:.3g}'
        if abs(amount) < 1e9:
            return f'{amount:,.0f}'
        return f'{amount:.2e}'

    def _cell_x(self, i):
        return self._left + i * self._cell_width

    def _draw_faces(self):
        top = self._grid_height
        number_y, value_y = top - 0.22, top - 0.46
        large = self._cell_width >= LARGE_CELL
        self.ax.text(LABEL_WIDTH / 2, number_y, 'Face', ha='center', va='center', fontsize=8,
                     color=DETAIL_COLOR)
        self.ax.text(LABEL_WIDTH / 2, value_y, 'Value', ha='center', va='center', fontsize=8,
                     color=DETAIL_COLOR)

        self._face_boxes = {}
        for (i, number) in enumerate(self._faces):
            x = self._cell_x(i)
            self._face_boxes[number] = self.ax.add_patch(patches.FancyBboxPatch(
                (x + CELL_GAP / 2, top - FACE_HEIGHT), self._cell_width - CELL_GAP, FACE_HEIGHT,
                boxstyle='round,pad=0,rounding_size=0.05', linewidth=1.4, zorder=1))
            center = x + self._cell_width / 2
            self.ax.text(center, number_y, str(number), ha='center', va='center',
                         fontsize=11 if large else 9.5, fontweight='bold', zorder=2)

            # a value below one makes the reward smaller, so it is not worth marking
            value = self._nonfluent('VALUE', number)
            self.ax.text(center, value_y, f'×{value:g}', ha='center', va='center',
                         fontsize=8 if large else 6.5, zorder=2,
                         color=LOSING_COLOR if value < 1 else DETAIL_COLOR)

        if self._row_height >= MIN_TEXT_ROW:
            right = self._cell_x(len(self._faces))
            self.ax.text(right + BUST_WIDTH / 2, value_y, 'Clears', ha='center', va='center',
                         fontsize=8, color=DETAIL_COLOR)
            self.ax.text(right + BUST_WIDTH + ROLLED_WIDTH / 2, value_y, 'If rolled', ha='center',
                         va='center', fontsize=8, color=DETAIL_COLOR)

    def _draw_dice(self):
        has_text = self._row_height >= MIN_TEXT_ROW
        font = 8 if self._cell_width >= LARGE_CELL else 6.5
        right = self._cell_x(len(self._faces))
        self._cells, self._clears, self._rolled = {}, {}, {}
        for (row, die) in enumerate(self._dice):
            y = self._grid_height - FACE_HEIGHT - ROW_GAP - (row + 0.5) * self._row_height
            if has_text:
                limit = int(LABEL_WIDTH * 72 / (CHAR_WIDTH * 8))
                self.ax.text(LABEL_WIDTH / 2, y, self._shorten(die, limit), ha='center',
                             va='center', fontsize=8, fontweight='bold')
            for (i, number) in enumerate(self._faces):
                chance = self._nonfluent('PROB', die, number)
                box = self.ax.add_patch(patches.Rectangle(
                    (self._cell_x(i) + CELL_GAP / 2, y - self._row_height / 2 + CELL_GAP / 2),
                    self._cell_width - CELL_GAP, self._row_height - CELL_GAP,
                    facecolor=NEVER_COLOR, edgecolor='none', zorder=1))
                if chance > 0:
                    self._cells[(die, number)] = box
                    if has_text:
                        self.ax.text(self._cell_x(i) + self._cell_width / 2, y,
                                     f'{chance:.0%}' if chance >= 0.005 else '<1%', ha='center',
                                     va='center', fontsize=font, zorder=2)
            if has_text:
                self._clears[die] = self.ax.text(right + BUST_WIDTH / 2, y, '', ha='center',
                                                 va='center', fontsize=8.5, fontweight='bold')
                self._rolled[die] = self.ax.text(right + BUST_WIDTH + ROLLED_WIDTH / 2, y, '',
                                                 ha='center', va='center', fontsize=8.5)

    def _shade(self, colors, chance):
        (light, strong) = (np.array(to_rgb(color)) for color in colors)
        amount = MIN_SHADE + (1 - MIN_SHADE) * chance / self._max_chance
        return tuple(light + amount * (strong - light))

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        marked = {number for number in self._numbers
                  if bool(state[fluent('die-value-seen', [number])])}
        product = math.prod(self._nonfluent('VALUE', number) for number in marked)

        for (number, box) in self._face_boxes.items():
            box.set(facecolor=MARKED_COLOR if number in marked else OPEN_COLOR,
                    edgecolor=MARKED_EDGE_COLOR if number in marked else OPEN_EDGE_COLOR)

        for ((die, number), box) in self._cells.items():
            chance = self._nonfluent('PROB', die, number)
            box.set_facecolor(self._shade(BUST_COLORS if number in marked else NEW_COLORS, chance))

        for die in self._clears:
            chances = {number: self._nonfluent('PROB', die, number) for number in self._numbers}
            clears = sum(chance for (number, chance) in chances.items() if number in marked)
            rolled = product * sum(chance * self._nonfluent('VALUE', number)
                                   for (number, chance) in chances.items() if number not in marked)
            self._clears[die].set(text=f'{clears:.0%}', color=LOSING_COLOR if clears > 0 else DETAIL_COLOR)
            self._rolled[die].set_text(self._amount(rolled))

        total = len(self._faces)
        done = sum(number in marked for number in self._faces)
        title = f'{done} / {total} face{"" if total == 1 else "s"} marked, '
        title += f'cash out for {self._amount(product)}' if marked else 'nothing to cash out'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color='black')

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
