import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

PAVING_COLOR = '#fafafa'
GRID_COLOR = '#b0bec5'
CURB_COLOR = '#90a4ae'
ROAD_COLOR = '#cfd8dc'
ROAD_LINE_COLOR = '#ffffff'
GRASS_COLOR = '#e8f5e9'
POLE_COLOR = '#546e7a'
GOAL_COLOR = '#c8e6c9'
GOAL_EDGE_COLOR = '#2e7d32'
WRONG_COLOR = '#c62828'
DETAIL_COLOR = '#37474f'

# a color tells people apart at a glance, and the names still do when the colors run out
PERSON_COLORS = ['#1565c0', '#ef6c00', '#6a1b9a', '#00838f', '#ad1457', '#5d4037', '#283593', '#f9a825']

# sizes in inches
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
MIN_WIDTH = 7.8
MAX_WIDTH = 9.0
MAX_CELL_WIDTH = 1.3
MAX_CELL_HEIGHT = 1.0
MIN_CELL_HEIGHT = 0.45
FLAG_ROW = 0.46
FLAG_WIDTH = 0.34
MIN_PENNANT_WIDTH = 0.1
MIN_PERSON_RADIUS = 0.08
MIN_NAME_CHARS = 3
CURB_HEIGHT = 0.05
ROAD_HEIGHT = 0.26
NUMBER_ROW = 0.2
LANES = 2

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5
NAME_FONT_SIZE = 7.5
FLAG_FONT_SIZE = 6.5
NUMBER_FONT_SIZE = 7.5
CHAR_WIDTH = 0.62


class SidewalkVisualizer(BaseViz):
    '''Draws the sidewalk from above with its two lanes of squares, between the road and the
    grass. Each person has a color and a flag in the grass over the column of their goal,
    since their goal only depends on how far along the sidewalk they are. People have a green
    ring when they are in that column in either lane, and otherwise an arrow towards it.

    The state invariants do not stop an instance from starting with people on the same square
    or off the sidewalk, so those people are side by side or at the nearest edge with a red
    ring. The actions are not part of the state, so a blocked step only shows as the person
    staying where they are. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._people = list(model.type_to_objects['person'])
        self._size = max(int(nonfluents[RDDLPlanningModel.ground_var('SIDEWALK-SIZE', [])]), 1)
        self._goals = {person: int(nonfluents[RDDLPlanningModel.ground_var('GOAL', [person])])
                       for person in self._people}
        self._colors = {person: PERSON_COLORS[i % len(PERSON_COLORS)]
                        for (i, person) in enumerate(self._people)}

        # the sidewalk fills the figure when it is long, and squares stay wide when it is short
        room = MAX_WIDTH - 0.4
        self._cell_width = min(MAX_CELL_WIDTH, room / self._size)
        self._cell_height = max(min(MAX_CELL_HEIGHT, self._cell_width), MIN_CELL_HEIGHT)
        grid_width = self._size * self._cell_width
        self._sidewalk_bottom = NUMBER_ROW + ROAD_HEIGHT + CURB_HEIGHT
        self._sidewalk_top = self._sidewalk_bottom + LANES * self._cell_height
        grid_height = self._sidewalk_top + FLAG_ROW

        # title above, then the grass, the sidewalk, the road, the legend and the caption
        width = max(grid_width + 0.4, MIN_WIDTH)
        height = grid_height + BOTTOM_MARGIN + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width), BOTTOM_MARGIN / height,
             grid_width / width, grid_height / height])
        self.ax.set_xlim(0, grid_width)
        self.ax.set_ylim(0, grid_height)
        self.ax.axis('off')

        # the street never changes, so it is only drawn once
        self._draw_street(grid_width, grid_height)
        self._draw_flags(grid_height)

        handles = [
            Line2D([], [], linestyle='', marker='o', markersize=8, markerfacecolor=PERSON_COLORS[0],
                   markeredgecolor='none', label='Person'),
            Line2D([], [], linestyle='', marker='o', markersize=9, markerfacecolor=GOAL_COLOR,
                   markeredgecolor=GOAL_EDGE_COLOR, markeredgewidth=1.5, label='At their goal'),
            Line2D([], [], linestyle='', marker='>', markersize=7, markerfacecolor=PERSON_COLORS[0],
                   markeredgecolor='none', label='Flag over their goal'),
            Line2D([], [], color=PERSON_COLORS[0], marker='>', markersize=4, label='Way to their goal')
        ]
        self.fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.25 / height),
                        ncol=len(handles), frameon=False, fontsize=8.5, columnspacing=1.0,
                        handlelength=1.4)

        caption = 'Each person at their goal earns 1 per step. People stepping onto the same square stay put'
        off = sum(not 0 <= goal < self._size for goal in self._goals.values())
        if off:
            caption += f'. {off} goal{"" if off == 1 else "s"} off the sidewalk'
        self.fig.text(0.5, 0.1 / height, caption, ha='center', va='bottom', fontsize=9,
                      color=DETAIL_COLOR)

    def _shorten(self, name, limit):

        # names that cannot fit keep both ends, since numbered names differ at the end
        if len(name) <= limit:
            return name
        head = max(limit, 2) // 2
        tail = max(limit - head - 1, 0)
        return f'{name[:head]}…{name[len(name) - tail:]}'

    def _text_limit(self, width, font):
        return int(width * 72 / (CHAR_WIDTH * font))

    def _name_fits(self, name, limit):

        # a name cut down to almost nothing would only be clutter
        return limit >= min(len(name), MIN_NAME_CHARS)

    def _draw_street(self, width, height):
        road_bottom = NUMBER_ROW
        self.ax.add_patch(patches.Rectangle((0, road_bottom), width, ROAD_HEIGHT, facecolor=ROAD_COLOR,
                                            edgecolor='none', zorder=1))
        self.ax.plot([0, width], [road_bottom + ROAD_HEIGHT / 2] * 2, color=ROAD_LINE_COLOR, linewidth=1.5,
                     linestyle=(0, (5, 5)), zorder=2)
        self.ax.add_patch(patches.Rectangle((0, road_bottom + ROAD_HEIGHT), width, CURB_HEIGHT,
                                            facecolor=CURB_COLOR, edgecolor='none', zorder=2))
        self.ax.add_patch(patches.Rectangle((0, self._sidewalk_top), width, height - self._sidewalk_top,
                                            facecolor=GRASS_COLOR, edgecolor='none', zorder=1))

        # the squares, whose lines would only blur together on a very long sidewalk
        self.ax.add_patch(patches.Rectangle((0, self._sidewalk_bottom), width, self._sidewalk_top - self._sidewalk_bottom,
                                            facecolor=PAVING_COLOR, edgecolor=GRID_COLOR, linewidth=1.0, zorder=1))
        self.ax.plot([0, width], [self._sidewalk_bottom + self._cell_height] * 2, color=GRID_COLOR,
                     linewidth=1.0, zorder=2)
        if self._cell_width >= 0.05:
            for col in range(1, self._size):
                self.ax.plot([col * self._cell_width] * 2, [self._sidewalk_bottom, self._sidewalk_top],
                             color=GRID_COLOR, linewidth=1.0, zorder=2)

        # as many column numbers as can be read, so that the goals can be looked up
        digits = len(str(self._size - 1))
        every = max(math.ceil((digits + 1) * CHAR_WIDTH * NUMBER_FONT_SIZE / 72 / self._cell_width), 1)
        for col in range(0, self._size, every):
            self.ax.text((col + 0.5) * self._cell_width, NUMBER_ROW / 2, str(col), ha='center', va='center',
                         fontsize=NUMBER_FONT_SIZE, color=DETAIL_COLOR, zorder=3)

    def _draw_flags(self, height):
        columns = {}
        for person in self._people:
            if 0 <= self._goals[person] < self._size:
                columns.setdefault(self._goals[person], []).append(person)

        fits = max(int(self._cell_width / FLAG_WIDTH), 1)
        flag_width = min(FLAG_WIDTH, self._cell_width)
        for (col, people) in columns.items():

            # the column always shows a flag, and says how many more want it when they do not fit
            shown = people if len(people) <= fits else people[:max(fits - 1, 1)]
            left = col * self._cell_width + (self._cell_width - len(people[:fits]) * flag_width) / 2
            for (i, person) in enumerate(shown):
                self._draw_flag(left + i * flag_width, flag_width, person)
            if len(shown) < len(people):
                self.ax.text(left + len(shown) * flag_width + max(flag_width / 2, MIN_PENNANT_WIDTH),
                             self._sidewalk_top + 0.1, f'+{len(people) - len(shown)}',
                             ha='left' if flag_width < FLAG_WIDTH else 'center', va='center',
                             fontsize=FLAG_FONT_SIZE, color=DETAIL_COLOR, zorder=4)

    def _draw_flag(self, x, flag_width, person):

        # a pole at the edge of the sidewalk with a pennant, and the name below the pennant
        pole_x = x + 0.12 * flag_width
        bottom, top = self._sidewalk_top, self._sidewalk_top + FLAG_ROW - 0.05
        self.ax.plot([pole_x, pole_x], [bottom, top], color=POLE_COLOR, linewidth=1.2, zorder=3)
        pennant = max(0.8 * flag_width, MIN_PENNANT_WIDTH)
        self.ax.add_patch(plt.Polygon(
            [(pole_x, top), (pole_x + pennant, top - 0.08), (pole_x, top - 0.16)], closed=True,
            facecolor=self._colors[person], edgecolor='none', zorder=4, clip_on=False))
        limit = self._text_limit(0.85 * flag_width, FLAG_FONT_SIZE)
        if self._name_fits(person, limit):
            self.ax.text(pole_x + 0.02, bottom + 0.1, self._shorten(person, limit), ha='left', va='center',
                         fontsize=FLAG_FONT_SIZE, color=DETAIL_COLOR, zorder=4)

    def _draw_person(self, x, y, room, person, at_goal, is_wrong, col, artists):

        # the figure as in the other domains with people, standing on the square it is on, and
        # it stays visible on a very long sidewalk even if that makes neighbors overlap
        def add(patch):
            patch.set_clip_on(False)
            artists.append(self.ax.add_patch(patch))

        cell_height = self._cell_height
        radius = max(0.5 * min(room, cell_height * 0.9), MIN_PERSON_RADIUS)
        font = NAME_FONT_SIZE
        limit = self._text_limit(room, font)
        has_name = cell_height >= 0.6 and self._name_fits(person, limit)
        middle = y + (0.06 * cell_height if has_name else 0)
        center = (x, middle)

        if at_goal or is_wrong:
            add(patches.Circle(center, 0.84 * radius,
                               facecolor=GOAL_COLOR if at_goal and not is_wrong else 'none',
                               edgecolor=WRONG_COLOR if is_wrong else GOAL_EDGE_COLOR,
                               linewidth=2.2 if is_wrong else 1.5, zorder=4))

        color = self._colors[person]
        body = 0.55 * radius
        foot = middle - 0.62 * radius
        add(patches.Wedge((x, foot), body, 0, 180, facecolor=color, edgecolor='none', zorder=5))
        add(patches.Circle((x, foot + body + 0.32 * radius), 0.3 * radius, facecolor=color,
                           edgecolor='none', zorder=5))

        if has_name:
            artists.append(self.ax.text(
                x, y - 0.4 * cell_height, self._shorten(person, limit), ha='center', va='center',
                fontsize=font, fontweight='bold', zorder=6))

        # an arrow beside the head towards the goal, when there is room for it
        goal = self._goals[person]
        if not at_goal and 0 <= goal < self._size and room >= 0.4:
            side = 1 if goal > col else -1
            start = x + side * 0.42 * radius
            add(patches.FancyArrowPatch(
                (start, foot + body + 0.32 * radius), (start + side * 0.55 * radius, foot + body + 0.32 * radius),
                arrowstyle='-|>', mutation_scale=8, linewidth=1.5, color=color, shrinkA=0, shrinkB=0, zorder=5))

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        positions = {person: (int(state[fluent('xPos', [person])]), int(state[fluent('yPos', [person])]))
                     for person in self._people}

        # people off the sidewalk stand at its nearest edge, and people on a square side by side
        squares = {}
        for person in self._people:
            (x, y) = positions[person]
            nearest = (min(max(x, 0), self._size - 1), min(max(y, 0), LANES - 1))
            squares.setdefault(nearest, []).append(person)

        artists = []
        num_wrong = 0
        for ((col, lane), people) in squares.items():
            room = self._cell_width / len(people)
            for (i, person) in enumerate(people):
                is_wrong = len(people) > 1 or positions[person] != (col, lane)
                num_wrong += is_wrong
                self._draw_person(
                    (col + (i + 0.5) / len(people)) * self._cell_width,
                    self._sidewalk_bottom + (lane + 0.5) * self._cell_height,
                    room, person, positions[person][0] == self._goals[person], is_wrong, col, artists)

        at_goal = sum(positions[person][0] == self._goals[person] for person in self._people)
        total = len(self._people)
        if num_wrong:
            title, color = f'{num_wrong} / {total} people on a shared square or off the sidewalk', WRONG_COLOR
        elif total and at_goal == total:
            title, color = 'Everyone at their goal', GOAL_EDGE_COLOR
        else:
            title, color = f'{at_goal} / {total} people at their goal', 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
