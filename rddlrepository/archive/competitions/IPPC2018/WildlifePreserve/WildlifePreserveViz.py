import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

EMPTY_COLOR = '#f5f7f8'
ATTACK_COLORS = ('#ffebee', '#e57373')
CAUGHT_COLORS = ('#eceff1', '#b0bec5')
DEFENDED_COLOR = '#64b5f6'
WINNING_COLOR = '#2e7d32'
LOSING_COLOR = '#c62828'
DETAIL_COLOR = '#37474f'

# how much a chance colors its cell at the least, so that small chances still show
MIN_SHADE = 0.1

# sizes in inches
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
MAX_SIZE = (9.0, 5.5)
MIN_WIDTH = 8.6
MIN_HEIGHT = 2.8
LABEL_WIDTH = 1.3
MAX_CELL_WIDTH = 0.7
HEADER_HEIGHT = 0.3
SECTION_HEIGHT = 0.24
MAX_ROW_HEIGHT = 0.32
CELL_GAP = 0.03

# text does not fit in anything smaller
MIN_TEXT_ROW = 0.16
MIN_TEXT_CELL = 0.3
MIN_NAME_CELL = 0.18
LARGE_ROW = 0.22
LARGE_CELL = 0.42
CHAR_WIDTH = 0.7


class WildlifePreserveVisualizer(BaseViz):
    '''Draws a table with one column per area. It shows the chance of each poacher to attack
    each area next, the chance that any poacher who can do harm attacks it, the steps in which
    rangers defended it, and the reward it pays when safe or attacked.

    The attacks and the rangers' next action are not part of the state, so the attacks only
    show as chances. The drawing only depends on the current state.

    All versions of the domain share this class, and the size of the table, the number of
    remembered steps and the attack weights all come from the model.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._areas = list(model.type_to_objects['area'])
        self._poachers = list(model.type_to_objects['poacher'])
        self._steps = list(model.type_to_objects['number'])
        self._rangers = list(model.type_to_objects['ranger'])

        # the fluent ATTACK-WEIGHT_k applies to an area remembered as defended k times
        self._weight_names = {}
        for name in model.non_fluents:
            match = re.fullmatch(r'ATTACK-WEIGHT_(\d+)', name)
            if match:
                self._weight_names[int(match.group(1))] = name

        self._memory = {poacher: [step for step in self._steps
                                  if self._nonfluent('POACHER-REMEMBERS', poacher, step)]
                        for poacher in self._poachers}
        self._same_memory = len({tuple(steps) for steps in self._memory.values()}) <= 1

        # everything is laid out in inches
        self._cell_width = min(MAX_CELL_WIDTH,
                               (MAX_SIZE[0] - LABEL_WIDTH - 0.4) / max(len(self._areas), 1))
        room = MAX_SIZE[1] - BOTTOM_MARGIN - TOP_MARGIN - HEADER_HEIGHT - 3 * SECTION_HEIGHT
        others = 3 + len(self._steps)
        count = others + len(self._poachers)
        self._row_height = self._poacher_height = min(MAX_ROW_HEIGHT, room / count)

        # with very many poachers their rows shrink first, to keep the other rows readable
        if self._row_height < MIN_TEXT_ROW + 0.02:
            self._row_height = min(MIN_TEXT_ROW + 0.02, room / (others + 1))
            self._poacher_height = (room - others * self._row_height) / max(len(self._poachers), 1)

        grid_width = LABEL_WIDTH + len(self._areas) * self._cell_width
        grid_height = (HEADER_HEIGHT + 3 * SECTION_HEIGHT + others * self._row_height
                       + len(self._poachers) * self._poacher_height)
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

        self._y = grid_height
        self._draw_header()
        self._draw_attacks()
        self._draw_defended()
        self._draw_rewards()

        handles = [
            patches.Patch(facecolor=ATTACK_COLORS[1], edgecolor='none', label='Chance of attack'),
            patches.Patch(facecolor=CAUGHT_COLORS[1], edgecolor='none', label='Chance of a caught poacher, no harm'),
            patches.Patch(facecolor=DEFENDED_COLOR, edgecolor='none', label='Defended'),
            patches.Patch(facecolor=EMPTY_COLOR, edgecolor='#cfd8dc', label='Not defended')
        ]
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.25 / height),
                        ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                        handlelength=1.4)
        self.fig.text(0.5, 0.1 / height, self._caption(), ha='center', va='bottom', fontsize=9,
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

    def _caption(self):
        count = len(self._rangers)
        text = '1 ranger defends 1 area per step. ' if count == 1 else f'{count} rangers defend 1 area each per step. '
        if not self._same_memory:
            text += 'Brackets show the steps remembered. '
        elif self._poachers:
            steps = next(iter(self._memory.values()))
            if not steps:
                text += 'Poachers remember no steps. '
            elif len(steps) == 1 and steps == self._steps[:1]:
                text += 'Poachers remember the last step. '
            elif steps == self._steps[:len(steps)]:
                text += f'Poachers remember the last {len(steps)} steps. '
            else:
                text += f'Poachers remember the steps {", ".join(steps)} ago. '
        return text + 'A caught poacher does no harm this step.'

    def _cell_x(self, i):
        return LABEL_WIDTH + i * self._cell_width

    def _font_size(self, row_height):
        return 8 if row_height >= LARGE_ROW and self._cell_width >= LARGE_CELL else 6.5

    def _has_text(self, row_height):
        return row_height >= MIN_TEXT_ROW and self._cell_width >= MIN_TEXT_CELL

    def _draw_header(self):
        self._y -= HEADER_HEIGHT
        if self._cell_width < MIN_NAME_CELL:
            return
        limit = max(int(self._cell_width * 72 / (CHAR_WIDTH * 8.5)), 2)
        for (i, area) in enumerate(self._areas):
            self.ax.text(self._cell_x(i) + self._cell_width / 2, self._y + HEADER_HEIGHT / 2,
                         self._shorten(area, limit), ha='center', va='center',
                         fontsize=8.5 if self._cell_width >= LARGE_CELL else 6.5, fontweight='bold')

    def _draw_section(self, title):
        self._y -= SECTION_HEIGHT
        self.ax.text(0, self._y + SECTION_HEIGHT * 0.42, title, ha='left', va='center',
                     fontsize=8, fontweight='bold', color=DETAIL_COLOR)

    def _draw_row(self, label, height, bold=False, with_text=True):
        '''Starts a row of empty cells and returns its boxes, and its texts if they fit.'''
        self._y -= height
        center = self._y + height / 2
        if height >= MIN_TEXT_ROW and label:
            self.ax.text(0.06, center, label, ha='left', va='center', fontsize=8,
                         fontweight='bold' if bold else 'normal')
        boxes, texts = [], []
        font = self._font_size(height)

        # the gap between rows cannot take up most of a very thin row
        row_gap = min(CELL_GAP, height * 0.15)
        for i in range(len(self._areas)):
            boxes.append(self.ax.add_patch(patches.Rectangle(
                (self._cell_x(i) + CELL_GAP / 2, self._y + row_gap / 2),
                self._cell_width - CELL_GAP, height - row_gap,
                facecolor=EMPTY_COLOR, edgecolor='none', zorder=1)))
            if with_text and self._has_text(height):
                texts.append(self.ax.text(self._cell_x(i) + self._cell_width / 2, center, '',
                                          ha='center', va='center', fontsize=font, zorder=2,
                                          fontweight='bold' if bold else 'normal'))
        return (boxes, texts)

    def _draw_attacks(self):
        self._draw_section('Chance to attack this step')
        self._attack_cells, self._caught_labels = {}, {}
        for poacher in self._poachers:
            label = self._shorten(poacher, 8)
            if not self._same_memory:
                steps = ','.join(self._memory[poacher]) or 'none'
                label = f'{self._shorten(poacher, 5)} ({self._shorten(steps, 5)})'
            self._attack_cells[poacher] = self._draw_row(label, self._poacher_height)
            if self._poacher_height >= MIN_TEXT_ROW:
                self._caught_labels[poacher] = self.ax.text(
                    LABEL_WIDTH - 0.06, self._y + self._poacher_height / 2, '', ha='right',
                    va='center', fontsize=7.5, fontweight='bold', color=WINNING_COLOR)
        self._any_cells = self._draw_row('Any free poacher', self._row_height, bold=True)

    def _draw_defended(self):
        self._draw_section('Defended by a ranger')
        self._defended_cells = {}
        for step in self._steps:
            unit = 'step' if step == '1' else 'steps'
            self._defended_cells[step] = self._draw_row(f'{self._shorten(step, 6)} {unit} ago', self._row_height,
                                                        with_text=False)[0]

    def _draw_rewards(self):
        self._draw_section('Reward this step')
        for (label, var, color) in (('If safe', 'DEFENDER-REWARD', WINNING_COLOR),
                                    ('If attacked', 'DEFENDER-PENALTY', LOSING_COLOR)):
            (_, texts) = self._draw_row(label, self._row_height)
            for (text, area) in zip(texts, self._areas):
                text.set(text=f'{self._nonfluent(var, area):+.4g}', color=color)

    def _weight(self, count, poacher, area):
        name = self._weight_names.get(count)
        return 0.0 if name is None else self._nonfluents[RDDLPlanningModel.ground_var(name, [poacher, area])]

    def _attack_chances(self, state, poacher):
        '''The Discrete distribution of poacher-attacks in the domain.'''
        fluent = RDDLPlanningModel.ground_var
        weights = np.zeros(len(self._areas))
        for (i, area) in enumerate(self._areas):
            count = sum(bool(state[fluent('was-defended', [area, step])])
                        for step in self._memory[poacher])
            weights[i] = self._weight(count, poacher, area)
        total = np.sum(weights)
        return weights / total if total > 0 else weights

    def _chance_text(self, chance):
        if chance == 0:
            return '0%'
        if chance < 0.005:
            return '<1%'
        if chance > 0.995:
            return '>99%'
        return f'{chance:.0%}'

    def _shade(self, colors, chance):
        if chance <= 0:
            return EMPTY_COLOR
        (light, strong) = (np.array(to_rgb(color)) for color in colors)
        return tuple(light + (MIN_SHADE + (1 - MIN_SHADE) * chance) * (strong - light))

    def _fill(self, cells, chances, colors):
        (boxes, texts) = cells
        for (box, chance) in zip(boxes, chances):
            box.set_facecolor(self._shade(colors, chance))
        for (text, chance) in zip(texts, chances):
            text.set_text(self._chance_text(chance))

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        safe = np.ones(len(self._areas))
        caught_count = 0
        for poacher in self._poachers:
            caught = bool(state[fluent('poacher-caught', [poacher])])
            chances = self._attack_chances(state, poacher)
            self._fill(self._attack_cells[poacher], chances, CAUGHT_COLORS if caught else ATTACK_COLORS)
            if poacher in self._caught_labels:
                self._caught_labels[poacher].set_text('Caught' if caught else '')
            if caught:
                caught_count += 1
            else:
                safe *= 1 - chances
        threat = 1 - safe
        self._fill(self._any_cells, threat, ATTACK_COLORS)

        for (step, boxes) in self._defended_cells.items():
            for (box, area) in zip(boxes, self._areas):
                defended = bool(state[fluent('was-defended', [area, step])])
                box.set_facecolor(DEFENDED_COLOR if defended else EMPTY_COLOR)

        total = len(self._poachers)
        if caught_count == total:
            title = f'All {total} poachers caught' if total > 1 else 'Poacher caught'
            title += ', no attack can do harm this step'
            color = WINNING_COLOR
        else:
            title = f'{caught_count} / {total} poacher{"" if total == 1 else "s"} caught'
            color = 'black'
            worst = int(np.argmax(threat))
            if threat[worst] > 0:
                title += f', biggest threat {self._areas[worst]} at {self._chance_text(threat[worst])}'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
