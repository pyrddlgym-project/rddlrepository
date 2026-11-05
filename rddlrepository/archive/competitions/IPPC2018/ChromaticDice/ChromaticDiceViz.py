from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

# the face and the edge and pips of a die of each color
DIE_COLORS = {
    'red': ('#ffcdd2', '#c62828'),
    'green': ('#c8e6c9', '#2e7d32'),
    'blue': ('#bbdefb', '#1565c0'),
    'yellow': ('#fff9c4', '#f9a825'),
    'purple': ('#e1bee7', '#6a1b9a')
}
PANEL_COLOR = '#f5f7f8'
OPEN_COLOR = '#ffffff'
OPEN_EDGE_COLOR = '#90a4ae'
TAKEN_COLOR = '#eceff1'
TAKEN_TEXT_COLOR = '#90a4ae'
CURRENT_COLOR = '#bbdefb'
CURRENT_EDGE_COLOR = '#1565c0'
POSSIBLE_COLOR = '#c8e6c9'
RING_COLOR = '#263238'
GOOD_COLOR = '#2e7d32'
BAD_COLOR = '#c62828'
DETAIL_COLOR = '#37474f'

# the categories and phases of the domain, which every instance has
VALUE_CATEGORIES = ('ones', 'twos', 'threes', 'fours', 'fives', 'sixes')
COLOR_CATEGORIES = {'reds': 'red', 'greens': 'green', 'blues': 'blue',
                    'yellows': 'yellow', 'purples': 'purple'}
VALUE_HANDS = ('two-pairs', 'three-of-a-kind', 'four-of-a-kind', 'full-house',
               'small-straight', 'large-straight', 'five-of-a-kind', 'chance')
COLOR_HANDS = ('three-of-a-color', 'four-of-a-color', 'color-full-house', 'flush', 'rainbow')
PHASES = {'roll1': 'Roll 1', 'roll2': 'Roll 2', 'roll3': 'Roll 3', 'assign': 'Assign',
          'receive-boni': 'Boni', 'finish': 'Finish'}

# the dice only count for something in these phases, as the first roll replaces all of them
LIVE_PHASES = ('roll2', 'roll3', 'assign')

# where the pips of each value of a die are, as a fraction of its size from its center
PIPS = ((-0.27, 0.27), (0.27, 0.27), (-0.27, 0), (0, 0), (0.27, 0), (-0.27, -0.27), (0.27, -0.27))
VALUE_PIPS = {1: (3,), 2: (1, 5), 3: (1, 3, 5), 4: (0, 1, 5, 6), 5: (0, 1, 3, 5, 6),
              6: (0, 1, 2, 4, 5, 6)}

# how far a bonus level moves with this many matching dice, and it stays with any other number
LEVEL_MOVES = {0: -3, 1: -2, 2: -1, 4: 1, 5: 2}
MIN_LEVEL, MAX_LEVEL = 1, 6

# sizes in inches
SIDE_MARGIN = 0.2
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
CHIP_WIDTH = 1.0
CHIP_HEIGHT = 0.28
CHIP_GAP = 0.26
MAX_DIE_SIZE = 0.72
DIE_GAP = 0.16
DIE_NAME_HEIGHT = 0.18
SECTION_GAP = 0.12
PANEL_GAP = 0.12
PANEL_PAD = 0.06
HEADER_HEIGHT = 0.26
ROW_HEIGHT = 0.22
ROW_GAP = 0.03
PANEL_ROWS = 8
LEVEL_CELL = 0.24
MIN_DIE_NAME_SIZE = 0.3


class ChromaticDiceVisualizer(BaseViz):
    '''Draws the dice and the phase of the turn above a scorecard of all categories. Each
    open category shows what assigning the current dice to it would score, and how that
    would move the bonus level of its section, whose gauge shows the chance and size of
    the bonus at each level.

    Entries are turned into reward as they are assigned, so the state does not remember
    them and categories already taken only show as taken. The drawing only depends on the
    current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 width=9.4,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._dice = list(model.type_to_objects['die'])

        section_height = HEADER_HEIGHT + PANEL_ROWS * (ROW_HEIGHT + ROW_GAP) + PANEL_PAD
        die_size = min(MAX_DIE_SIZE, (width - 2 * SIDE_MARGIN) / max(len(self._dice), 1) - DIE_GAP)
        content = (CHIP_HEIGHT + SECTION_GAP + die_size + DIE_NAME_HEIGHT + SECTION_GAP
                   + section_height)
        height = content + BOTTOM_MARGIN + TOP_MARGIN
        self._figure_size = (width, height)

        # everything is laid out in inches
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, BOTTOM_MARGIN / height, 1, content / height])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, content)
        self.ax.axis('off')

        self._draw_phases(width, content - CHIP_HEIGHT / 2)
        self._draw_dice(width, content - CHIP_HEIGHT - SECTION_GAP - die_size / 2, die_size)
        self._draw_scorecard(width, section_height)

        self.fig.legend(handles=[
            patches.Patch(facecolor=OPEN_COLOR, edgecolor=OPEN_EDGE_COLOR, label='Open'),
            patches.Patch(facecolor=TAKEN_COLOR, edgecolor=OPEN_EDGE_COLOR, label='Taken'),
            patches.Patch(facecolor=POSSIBLE_COLOR, edgecolor=OPEN_EDGE_COLOR, label='Level with a chance of bonus'),
            patches.Patch(facecolor='none', edgecolor=RING_COLOR, linewidth=2, label='Current level')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / height), ncol=4, frameon=False,
           fontsize=8.5, columnspacing=1.0, handlelength=1.4)

        self.fig.text(0.5, 0.1 / height,
                      'Scores are for assigning the dice now, ▲▼ is the move in bonus level, and '
                      'percentages are how often a value or color is rolled',
                      ha='center', va='bottom', fontsize=9, color=DETAIL_COLOR)

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _draw_phases(self, width, y):
        total = len(PHASES) * CHIP_WIDTH + (len(PHASES) - 1) * CHIP_GAP
        left = (width - total) / 2
        self._chips = {}
        for (i, (phase, label)) in enumerate(PHASES.items()):
            x = left + i * (CHIP_WIDTH + CHIP_GAP)
            box = self.ax.add_patch(patches.FancyBboxPatch(
                (x, y - CHIP_HEIGHT / 2), CHIP_WIDTH, CHIP_HEIGHT,
                boxstyle='round,pad=0,rounding_size=0.06', linewidth=1.2, zorder=2))
            text = self.ax.text(x + CHIP_WIDTH / 2, y, label, ha='center', va='center',
                                fontsize=9, zorder=3)
            self._chips[phase] = (box, text)
            if i > 0:
                self.ax.annotate(
                    '', xy=(x - 0.03, y), xytext=(x - CHIP_GAP + 0.03, y),
                    arrowprops=dict(arrowstyle='-|>', color=OPEN_EDGE_COLOR, linewidth=1,
                                    shrinkA=0, shrinkB=0, mutation_scale=8))

    def _draw_dice(self, width, y, size):
        total = len(self._dice) * size + (len(self._dice) - 1) * DIE_GAP
        left = (width - total) / 2
        self._faces, self._pips, self._die_names = {}, {}, {}
        for (i, die) in enumerate(self._dice):
            x = left + i * (size + DIE_GAP) + size / 2
            self._faces[die] = self.ax.add_patch(patches.FancyBboxPatch(
                (x - size / 2, y - size / 2), size, size,
                boxstyle=f'round,pad=0,rounding_size={size / 6:g}', linewidth=1.6, zorder=2))
            self._pips[die] = [self.ax.add_patch(patches.Circle(
                (x + dx * size, y + dy * size), 0.085 * size, linewidth=0, zorder=3))
                for (dx, dy) in PIPS]
            if size >= MIN_DIE_NAME_SIZE:
                self._die_names[die] = self.ax.text(
                    x, y - size / 2 - DIE_NAME_HEIGHT / 2,
                    self._shorten(die, int((size + DIE_GAP) * 72 / (0.6 * 7.5))),
                    ha='center', va='center', fontsize=7.5, color=DETAIL_COLOR)

    def _draw_scorecard(self, width, height):
        self._rows, self._gauges = {}, {}
        panel_width = (width - 2 * SIDE_MARGIN - 3 * PANEL_GAP) / 4
        sections = (('Values', VALUE_CATEGORIES, 'VALUE'), ('Colors', tuple(COLOR_CATEGORIES), 'COLOR'),
                    ('Value hands', VALUE_HANDS, None), ('Color hands', COLOR_HANDS, None))
        for (i, (title, categories, part)) in enumerate(sections):
            left = SIDE_MARGIN + i * (panel_width + PANEL_GAP)
            self.ax.add_patch(patches.FancyBboxPatch(
                (left, 0), panel_width, height, boxstyle='round,pad=0,rounding_size=0.05',
                facecolor=PANEL_COLOR, edgecolor='none', zorder=0))
            self.ax.text(left + panel_width / 2, height - HEADER_HEIGHT / 2, title, ha='center',
                         va='center', fontsize=9, fontweight='bold', zorder=1)
            for (row, category) in enumerate(categories):
                y = height - HEADER_HEIGHT - row * (ROW_HEIGHT + ROW_GAP) - ROW_HEIGHT / 2
                self._add_row(category, left + PANEL_PAD, y, panel_width - 2 * PANEL_PAD)
            if part is not None:
                self._add_gauge(part, left, panel_width)

    def _add_row(self, category, left, y, width):
        box = self.ax.add_patch(patches.FancyBboxPatch(
            (left, y - ROW_HEIGHT / 2), width, ROW_HEIGHT,
            boxstyle='round,pad=0,rounding_size=0.04', linewidth=0.8, zorder=1))
        label = category.replace('-', ' ').capitalize()
        name = self.ax.text(left + 0.07, y, label, ha='left', va='center', fontsize=8, zorder=2)

        # the rolls that a category of a value or color counts on are not always equally likely
        chance = None
        if category in VALUE_CATEGORIES:
            chance = self._nonfluent('VALUE_PROB', str(VALUE_CATEGORIES.index(category) + 1))
        elif category in COLOR_CATEGORIES:
            chance = self._nonfluent('COLOR_PROB', COLOR_CATEGORIES[category])
        odds = None
        if chance is not None:
            odds = self.ax.text(left + 0.98, y, f'{chance:.0%}', ha='right', va='center',
                                fontsize=7, zorder=2)
        move = self.ax.text(left + width - 0.32, y, '', ha='right', va='center', fontsize=7, zorder=2)
        score = self.ax.text(left + width - 0.07, y, '', ha='right', va='center', fontsize=8, zorder=2)
        self._rows[category] = (box, name, odds, move, score)

    def _add_gauge(self, part, left, panel_width):
        total = (MAX_LEVEL - MIN_LEVEL + 1) * LEVEL_CELL
        start = left + (panel_width - total) / 2
        y = 0.3
        cells = {}
        for level in range(MIN_LEVEL, MAX_LEVEL + 1):
            x = start + (level - MIN_LEVEL) * LEVEL_CELL
            possible = self._nonfluent(f'BONUS_PROB_{part}_PART', str(level)) > 0
            cells[level] = self.ax.add_patch(patches.Rectangle(
                (x + 0.015, y - 0.09), LEVEL_CELL - 0.03, 0.18,
                facecolor=POSSIBLE_COLOR if possible else OPEN_COLOR,
                edgecolor=OPEN_EDGE_COLOR, linewidth=0.8, zorder=1))
            self.ax.text(x + LEVEL_CELL / 2, y, str(level), ha='center', va='center',
                         fontsize=7.5, zorder=2)
        text = self.ax.text(left + panel_width / 2, 0.1, '', ha='center', va='center',
                            fontsize=7.5, color=DETAIL_COLOR, zorder=2)
        self._gauges[part] = (cells, text)

    def _has_full_house(self, counts):

        # three of a kind and a pair among the other dice, which may be of the same kind
        return any(n >= 3 and (n >= 5 or any(m >= 2 for (other, m) in counts.items() if other != kind))
                   for (kind, n) in counts.items())

    def _scores(self, dice):

        # the reward of the domain for assigning these dice to each of the categories
        counts = Counter(value for (value, _) in dice)
        colors = Counter(color for (_, color) in dice)
        total = sum(value for (value, _) in dice)
        most = max(counts.values(), default=0)
        most_color = max(colors.values(), default=0)

        scores = {category: (i + 1) * counts[i + 1] for (i, category) in enumerate(VALUE_CATEGORIES)}
        for (category, shade) in COLOR_CATEGORIES.items():
            scores[category] = sum(value for (value, color) in dice if color == shade)
        scores['two-pairs'] = total if sum(n // 2 for n in counts.values()) >= 2 else 0
        scores['three-of-a-kind'] = total if most >= 3 else 0
        scores['four-of-a-kind'] = total if most >= 4 else 0
        scores['full-house'] = self._nonfluent('SCORE_FULL_HOUSE') if self._has_full_house(counts) else 0
        straight = lambda length, start: all(counts[start + i] for i in range(length))
        scores['small-straight'] = (self._nonfluent('SCORE_SMALL_STRAIGHT')
                                    if any(straight(4, start) for start in (1, 2, 3)) else 0)
        scores['large-straight'] = (self._nonfluent('SCORE_LARGE_STRAIGHT')
                                    if any(straight(5, start) for start in (1, 2)) else 0)
        scores['chance'] = total
        scores['five-of-a-kind'] = self._nonfluent('SCORE_FIVE_OF_A_KIND') if most >= 5 else 0
        scores['three-of-a-color'] = sum(value for (value, color) in dice if colors[color] >= 3)
        scores['four-of-a-color'] = (sum(value for (value, color) in dice if colors[color] >= 2)
                                     if most_color >= 4 else 0)
        scores['color-full-house'] = (self._nonfluent('SCORE_COLOR_FULL_HOUSE')
                                      if self._has_full_house(colors) else 0)
        scores['flush'] = total if most_color >= 5 else 0

        # the domain looks for two dice of the same color without asking for them to be
        # different dice, so it finds every die matching itself and a rainbow never scores
        scores['rainbow'] = self._nonfluent('SCORE_RAINBOW') if not dice else 0
        return scores

    def _level_move(self, level, matching):
        return np.clip(level + LEVEL_MOVES.get(matching, 0), MIN_LEVEL, MAX_LEVEL) - level

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def _render_dice(self, dice, live):
        for (die, (value, color)) in zip(self._dice, dice):
            (face, edge) = DIE_COLORS[color]
            alpha = 1 if live else 0.35
            self._faces[die].set(facecolor=face, edgecolor=edge, alpha=alpha)
            for (i, pip) in enumerate(self._pips[die]):
                pip.set(facecolor=edge, alpha=alpha, visible=i in VALUE_PIPS[value])

    def _render_row(self, category, taken, score, move):
        (box, name, odds, move_text, score_text) = self._rows[category]
        box.set(facecolor=TAKEN_COLOR if taken else OPEN_COLOR,
                edgecolor='none' if taken else OPEN_EDGE_COLOR)
        muted = TAKEN_TEXT_COLOR if taken else 'black'
        name.set_color(muted)
        if odds is not None:
            odds.set_color(TAKEN_TEXT_COLOR if taken else DETAIL_COLOR)

        if taken:
            score_text.set(text='✓', color=TAKEN_TEXT_COLOR, fontweight='normal')
        elif score is None:
            score_text.set(text='')
        else:
            score_text.set(text=f'{score:g}', color='black' if score > 0 else TAKEN_TEXT_COLOR,
                           fontweight='bold' if score > 0 else 'normal')

        if taken or not move:
            move_text.set(text='')
        else:
            move_text.set(text=f'{"▲" if move > 0 else "▼"}{abs(move):d}',
                          color=GOOD_COLOR if move > 0 else BAD_COLOR)

    def _render_gauge(self, part, level, received):
        (cells, text) = self._gauges[part]
        for (other, cell) in cells.items():
            current = other == level
            cell.set(edgecolor=RING_COLOR if current else OPEN_EDGE_COLOR,
                     linewidth=2 if current else 0.8, zorder=1.5 if current else 1)
        bonus = self._nonfluent(f'BONUS_{part}_PART', str(level))
        chance = self._nonfluent(f'BONUS_PROB_{part}_PART', str(level))
        if received:
            text.set(text=f'Bonus of {bonus:g} received', color=GOOD_COLOR, fontweight='bold')
        else:
            text.set(text=f'Level {level}: {chance:.0%} chance of {bonus:g}',
                     color=DETAIL_COLOR, fontweight='normal')

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        dice = [(int(str(state[fluent('die-value', [die])])), str(state[fluent('die-color', [die])]))
                for die in self._dice]
        phase = str(state['current-phase'])
        live = phase in LIVE_PHASES

        for (name, (box, text)) in self._chips.items():
            current = name == phase
            box.set(facecolor=CURRENT_COLOR if current else OPEN_COLOR,
                    edgecolor=CURRENT_EDGE_COLOR if current else OPEN_EDGE_COLOR)
            text.set(fontweight='bold' if current else 'normal',
                     color='black' if current else DETAIL_COLOR)
        self._render_dice(dice, live)

        # the matching dice of a section are what decide how far its bonus level would move
        levels = {'VALUE': int(str(state['bonus-level-value-part'])),
                  'COLOR': int(str(state['bonus-level-color-part']))}
        counts = Counter(value for (value, _) in dice)
        colors = Counter(color for (_, color) in dice)
        scores = self._scores(dice)
        remaining = 0
        for category in self._rows:
            taken = bool(state[fluent('taken', [category])])
            remaining += not taken
            move = 0
            if live and category in VALUE_CATEGORIES:
                matching = counts[VALUE_CATEGORIES.index(category) + 1]
                move = self._level_move(levels['VALUE'], matching)
            elif live and category in COLOR_CATEGORIES:
                matching = colors[COLOR_CATEGORIES[category]]
                move = self._level_move(levels['COLOR'], matching)
            self._render_row(category, taken, scores[category] if live else None, int(move))

        value_bonus = bool(state['receive-value-bonus'])
        color_bonus = bool(state['receive-color-bonus'])
        finished = phase == 'finish'
        self._render_gauge('VALUE', levels['VALUE'], finished and value_bonus)
        self._render_gauge('COLOR', levels['COLOR'], finished and color_bonus)

        # the boni only show in the state for the one step that they are paid, so a finished
        # game without any showing may have received them on an earlier step
        if finished and value_bonus and color_bonus:
            title, color = 'Both boni received', GOOD_COLOR
        elif finished and value_bonus:
            title, color = 'Value bonus received', GOOD_COLOR
        elif finished and color_bonus:
            title, color = 'Color bonus received', GOOD_COLOR
        elif finished:
            title, color = 'Game finished', 'black'
        elif remaining == 0:
            title, color = 'All categories assigned', 'black'
        else:
            title, color = f'{remaining} categor{"y" if remaining == 1 else "ies"} left', 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
