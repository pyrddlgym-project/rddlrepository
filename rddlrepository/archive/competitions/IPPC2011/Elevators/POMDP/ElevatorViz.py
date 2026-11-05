import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

FLOOR_COLORS = ('#fafafa', '#eceff1')
EXIT_COLOR = '#c8e6c9'
SHAFT_COLOR = '#cfd8dc'
CLOSED_COLOR = '#1565c0'
OPEN_COLOR = '#bbdefb'
DELIVER_COLOR = '#2e7d32'
WAITING_COLOR = '#ff8f00'
EMPTY_COLOR = '#90a4ae'
TEXT_COLOR = 'black'

# the columns of the building: floor names, then one for each elevator, then waiting
NAME_WIDTH = 1.1
WAITING_WIDTH = 2.6
HEADER_HEIGHT = 0.4

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class ElevatorVisualizer(BaseViz):
    '''Draws the building with its floors in the order given by ADJACENT-UP, each
    elevator with its door, direction and passengers, and the people waiting on each
    floor. A passenger is an arrow pointing towards their destination, which is the
    top or bottom floor, and an elevator is green while passengers get off.

    The POMDP shows which way each person is going, rather than the single button
    that is observed. The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 column_width=0.8,
                 row_height=0.7,
                 max_size=(14.0, 9.0),
                 min_width=4.2,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._elevators = model.type_to_objects['elevator']
        self._floors = self._get_floor_order(model.type_to_objects['floor'])
        self._rows = {f: i for (i, f) in enumerate(self._floors)}
        self._top = {f for f in self._floors if self._nonfluents[
            RDDLPlanningModel.ground_var('TOP-FLOOR', [f])]}
        self._bottom = {f for f in self._floors if self._nonfluents[
            RDDLPlanningModel.ground_var('BOTTOM-FLOOR', [f])]}

        num_columns = NAME_WIDTH + len(self._elevators) + WAITING_WIDTH
        num_rows = len(self._floors) + HEADER_HEIGHT
        self._waiting_x = NAME_WIDTH + len(self._elevators)

        # large buildings use smaller rows and columns to fit on screen
        row_height = min(row_height, max_size[1] / num_rows)
        column_width = min(column_width, max_size[0] / num_columns)
        margin, top, bottom = 0.1, 0.5, 0.45
        fig_width = max(num_columns * column_width + 2 * margin, min_width)
        column_width = (fig_width - 2 * margin) / num_columns
        fig_height = num_rows * row_height + top + bottom
        self._column_width = column_width
        self._row_height = row_height

        self.fig = plt.figure(figsize=(fig_width, fig_height))
        self.ax = self.fig.add_axes(
            [margin / fig_width, bottom / fig_height,
             (fig_width - 2 * margin) / fig_width, num_rows * row_height / fig_height])
        self.ax.set_xlim(0, num_columns)
        self.ax.set_ylim(0, num_rows)
        self.ax.axis('off')

        self._draw_building(num_columns)

        self.fig.legend(handles=[
            patches.Patch(facecolor=CLOSED_COLOR, label='Door closed'),
            patches.Patch(facecolor=OPEN_COLOR, edgecolor=CLOSED_COLOR, label='Door open'),
            patches.Patch(facecolor=DELIVER_COLOR, label='Delivering'),
            Line2D([], [], marker='^', color='none', markerfacecolor=WAITING_COLOR,
                   markeredgecolor='none', label='Person')
        ], loc='lower center', ncol=4, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _get_floor_order(self, floors):

        # follow the floors upwards from the one with none below: floors that do not
        # form a single line have no clear order, so use the declared one
        above = {}
        for f1 in floors:
            targets = [f2 for f2 in floors if self._nonfluents[
                RDDLPlanningModel.ground_var('ADJACENT-UP', [f1, f2])]]
            if len(targets) > 1:
                return floors
            if targets:
                above[f1] = targets[0]
        first = [f for f in floors if f not in above.values()]
        if len(first) != 1:
            return floors
        order = first
        while order[-1] in above and above[order[-1]] not in order:
            order = order + [above[order[-1]]]
        return order if len(order) == len(floors) else floors

    def _font_size(self, text, width, font_size):

        # text shrinks to fit the given number of columns and the height of a row, and
        # is left out if unreadable
        size = min(font_size, 72 * 0.5 * self._row_height,
                   72 * width * self._column_width / (0.62 * len(text)))
        return size if size >= MIN_FONT_SIZE else None

    def _add_text(self, x, y, text, width, font_size, **kwargs):
        size = self._font_size(text, width, font_size)
        if size is None:
            return None
        return self.ax.text(x, y, text, ha='center', va='center', fontsize=size, **kwargs)

    def _draw_building(self, num_columns):

        # the building never changes, so it is only drawn once
        num_floors = len(self._floors)
        for (i, f) in enumerate(self._floors):
            is_exit = f in self._top or f in self._bottom
            color = EXIT_COLOR if is_exit else FLOOR_COLORS[i % 2]
            self.ax.add_patch(patches.Rectangle(
                (0, i), num_columns, 1, facecolor=color, edgecolor='none', zorder=0))
            self.ax.plot([0, num_columns], [i, i], color=SHAFT_COLOR, linewidth=1, zorder=1)
            self._add_text(NAME_WIDTH / 2, i + 0.5, f, 0.9 * NAME_WIDTH, 10, color=TEXT_COLOR,
                           fontweight='bold' if is_exit else 'normal', zorder=5)

        for (i, e) in enumerate(self._elevators):
            x = NAME_WIDTH + i
            self.ax.add_patch(patches.Rectangle(
                (x + 0.08, 0), 0.84, num_floors, facecolor=SHAFT_COLOR, alpha=0.45,
                edgecolor='none', zorder=1))
            self._add_text(x + 0.5, num_floors + HEADER_HEIGHT / 2, e, 0.95, 10, color=TEXT_COLOR)
        self._add_text(self._waiting_x + WAITING_WIDTH / 2, num_floors + HEADER_HEIGHT / 2,
                       'waiting', WAITING_WIDTH, 10, color=TEXT_COLOR)

    def _draw_elevator(self, index, elevator, floor, state):
        x = NAME_WIDTH + index + 0.14
        y = self._rows[floor] + 0.08
        width, height = 0.72, 0.84
        going_up_riders = state[RDDLPlanningModel.ground_var('person-in-elevator-going-up', [elevator])]
        going_down_riders = state[RDDLPlanningModel.ground_var('person-in-elevator-going-down', [elevator])]
        riders = int(going_up_riders) + int(going_down_riders)
        closed = state[RDDLPlanningModel.ground_var('elevator-closed', [elevator])]
        going_up = state[RDDLPlanningModel.ground_var('elevator-dir-up', [elevator])]

        # same rule as the state: passengers get off at the floor they are heading to
        delivering = (going_up_riders and floor in self._top) or (going_down_riders and floor in self._bottom)
        color = DELIVER_COLOR if delivering else CLOSED_COLOR
        text_color = 'white' if closed or delivering else color

        rounding = 0.06 * min(self._column_width, self._row_height)
        artists = [self.ax.add_patch(patches.FancyBboxPatch(
            (x, y), width, height, boxstyle=f'round,pad=0,rounding_size={rounding}',
            mutation_aspect=self._column_width / self._row_height,
            facecolor=color if closed or delivering else OPEN_COLOR,
            edgecolor=color, linewidth=1.5, zorder=3))]

        # a closed door has a line down the middle of it
        if closed:
            artists.extend(self.ax.plot(
                [x + width / 2] * 2, [y, y + height], color='white', linewidth=1, alpha=0.35,
                zorder=4))

        # the direction is above the number of passengers and where they are heading
        marker_size = min(9, 12 * self._row_height)
        artists.extend(self.ax.plot(
            [x + width / 2], [y + 0.72 * height], marker='^' if going_up else 'v',
            markersize=marker_size, color=text_color, markeredgecolor='none', zorder=5))
        label = self._add_text(x + 0.28 * width, y + 0.3 * height, str(riders), 0.3 * width, 10,
                               color=text_color, fontweight='bold', zorder=5)
        if label is not None:
            artists.append(label)
        for (rider, marker, position) in ((going_up_riders, '^', 0.58), (going_down_riders, 'v', 0.82)):
            if rider:
                artists.extend(self.ax.plot(
                    [x + position * width], [y + 0.3 * height], marker=marker,
                    markersize=0.8 * marker_size, color=WAITING_COLOR,
                    markeredgecolor='white', markeredgewidth=0.5, zorder=5))
        return artists

    def _draw_waiting(self, floor, state):
        waiting_up = state[RDDLPlanningModel.ground_var('person-waiting-up', [floor])]
        waiting_down = state[RDDLPlanningModel.ground_var('person-waiting-down', [floor])]
        count = int(waiting_up) + int(waiting_down)
        y = self._rows[floor] + 0.5
        artists = []
        label = self._add_text(self._waiting_x + 0.35, y, str(count), 0.6, 10, zorder=5,
                               color=WAITING_COLOR if count > 0 else EMPTY_COLOR, fontweight='bold')
        if label is not None:
            artists.append(label)

        # each direction has its own place, so a person never appears to jump
        marker_size = min(9, 12 * self._row_height)
        for (waiting, marker, x) in ((waiting_up, '^', 0.85), (waiting_down, 'v', 1.2)):
            if waiting:
                artists.extend(self.ax.plot(
                    [self._waiting_x + x], [y], marker=marker, markersize=marker_size,
                    color=WAITING_COLOR, markeredgecolor='none', zorder=4))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        artists = []
        for (i, e) in enumerate(self._elevators):
            for f in self._floors:
                if state[RDDLPlanningModel.ground_var('elevator-at-floor', [e, f])]:
                    artists.extend(self._draw_elevator(i, e, f, state))
        for f in self._floors:
            artists.extend(self._draw_waiting(f, state))

        waiting = sum(int(state[RDDLPlanningModel.ground_var(var, [f])]) for f in self._floors
                      for var in ('person-waiting-up', 'person-waiting-down'))
        riding = sum(int(state[RDDLPlanningModel.ground_var(var, [e])]) for e in self._elevators
                     for var in ('person-in-elevator-going-up', 'person-in-elevator-going-down'))
        self.ax.set_title(f'{waiting} waiting, {riding} riding', fontsize=13, fontweight='bold')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
