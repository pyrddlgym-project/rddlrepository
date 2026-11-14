import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

FLOOR_COLORS = ('#fafafa', '#eceff1')
EXIT_COLOR = '#c8e6c9'
SLAB_COLOR = '#90a4ae'
SHAFT_COLOR = '#dde3e6'
RAIL_COLOR = '#b0bec5'
CABLE_COLOR = '#37474f'
FRAME_COLOR = '#455a64'
DOOR_COLOR = '#b0bec5'
DOOR_EDGE_COLOR = '#78909c'
CAB_COLOR = '#fff3c4'
PANEL_COLOR = '#212121'
LED_COLOR = '#ffca28'
DELIVER_COLOR = '#2e7d32'
RIDER_COLOR = '#1565c0'
DESTINATION_COLOR = '#4fc3f7'
WAITING_COLOR = '#ff8f00'
EMPTY_COLOR = '#90a4ae'
TEXT_COLOR = 'black'

# the columns of the building: floor names, then one for each elevator, then waiting
NAME_WIDTH = 1.1
WAITING_WIDTH = 2.6
HEADER_HEIGHT = 0.4

# the car within its cell, and the indicator panel and doors within the car, as fractions;
# the car is never wider than CAR_ASPECT times its height
CAR_Y, CAR_WIDTH, CAR_HEIGHT, CAR_ASPECT = 0.06, 0.6, 0.88, 0.75
FRAME = 0.08
PANEL = 0.3

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


def _person_marker():
    '''A head above rounded shoulders, as one path, so that a single call to plot draws
    a whole row of people.'''
    t = np.linspace(0, np.pi, 13)
    body = np.column_stack([np.r_[0.34 * np.cos(t), -0.34, 0.34, 0.34],
                            np.r_[0.34 * np.sin(t) - 0.16, -0.75, -0.75, 0]])
    return Path.make_compound_path(Path.circle((0, 0.5), 0.24), Path(body, closed=True))


PERSON = _person_marker()


class ElevatorVisualizer(BaseViz):
    '''Draws the building with its floors in the order given by ADJACENT-UP, each
    elevator car on its cable with the panel above its doors, and the people waiting on
    each floor with arrows towards their destinations, the top or the bottom floor. The
    panel shows the direction of the car and the destination of each passenger. An open
    car shows its passengers inside, and its frame is green while they get off.

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
        margin, top, bottom = 0.1, 0.5, 0.65
        fig_width = max(num_columns * column_width + 2 * margin, min_width)
        column_width = (fig_width - 2 * margin) / num_columns
        fig_height = num_rows * row_height + top + bottom
        self._column_width = column_width
        self._row_height = row_height
        self._car_width = min(CAR_WIDTH, CAR_ASPECT * CAR_HEIGHT * row_height / column_width)

        self.fig = plt.figure(figsize=(fig_width, fig_height))
        self.ax = self.fig.add_axes(
            [margin / fig_width, bottom / fig_height,
             (fig_width - 2 * margin) / fig_width, num_rows * row_height / fig_height])
        self.ax.set_xlim(0, num_columns)
        self.ax.set_ylim(0, num_rows)
        self.ax.axis('off')

        self._draw_building(num_columns)

        person = dict(marker=PERSON, color='none', markeredgecolor='none', markersize=11)
        self.fig.legend(handles=[
            patches.Patch(facecolor=DOOR_COLOR, edgecolor=FRAME_COLOR, label='Door closed'),
            patches.Patch(facecolor=CAB_COLOR, edgecolor=FRAME_COLOR, label='Door open'),
            patches.Patch(facecolor='none', edgecolor=DELIVER_COLOR, linewidth=2,
                          label='Delivering'),
            Line2D([], [], markerfacecolor=RIDER_COLOR, label='Riding', **person),
            Line2D([], [], markerfacecolor=WAITING_COLOR, label='Waiting', **person),
            Line2D([], [], marker='^', color='none', markerfacecolor=DESTINATION_COLOR,
                   markeredgecolor='none', label='Destination')
        ], loc='lower center', ncol=3, frameon=False, fontsize=8.5, columnspacing=1.0,
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

    def _font_size(self, text, width, font_size, height=0.5):

        # text shrinks to fit the given number of columns and rows, and is left out if
        # unreadable
        size = min(font_size, 72 * height * self._row_height,
                   72 * width * self._column_width / (0.62 * len(text)))
        return size if size >= MIN_FONT_SIZE else None

    def _add_text(self, x, y, text, width, font_size, height=0.5, **kwargs):
        size = self._font_size(text, width, font_size, height)
        if size is None:
            return None
        return self.ax.text(x, y, text, ha='center', va='center', fontsize=size, **kwargs)

    def _draw_people(self, x, y, width, height, count, color):

        # one figure for each person, as many as fit in the given box
        size_inches = min(0.24, height * self._row_height)
        spacing = 0.65 * size_inches / self._column_width
        shown = min(count, int(width / spacing))
        if shown <= 0:
            return []
        return self.ax.plot(
            x + spacing * (np.arange(shown) + 0.5), [y + height / 2] * shown,
            linestyle='none', marker=PERSON, markersize=72 * size_inches,
            color=color, markeredgecolor='none', zorder=6)

    def _draw_building(self, num_columns):

        # the building never changes, so it is only drawn once: floors and their slabs,
        # then the shafts in front of them
        num_floors = len(self._floors)
        for (i, f) in enumerate(self._floors):
            is_exit = f in self._top or f in self._bottom
            color = EXIT_COLOR if is_exit else FLOOR_COLORS[i % 2]
            self.ax.add_patch(patches.Rectangle(
                (0, i), num_columns, 1, facecolor=color, edgecolor='none', zorder=0))
            self.ax.plot([0, num_columns], [i, i], color=SLAB_COLOR, linewidth=2, zorder=1)
            self._add_text(NAME_WIDTH / 2, i + 0.5, f, 0.9 * NAME_WIDTH, 10, color=TEXT_COLOR,
                           fontweight='bold' if is_exit else 'normal', zorder=5)

        for (i, e) in enumerate(self._elevators):
            x = NAME_WIDTH + i
            self.ax.add_patch(patches.Rectangle(
                (x + 0.08, 0), 0.84, num_floors, facecolor=SHAFT_COLOR, edgecolor='none',
                zorder=1))
            for rail in (x + 0.1, x + 0.9):
                self.ax.plot([rail] * 2, [0, num_floors], color=RAIL_COLOR, linewidth=1.5,
                             zorder=1)
            self._add_text(x + 0.5, num_floors + HEADER_HEIGHT / 2, e, 0.95, 10, color=TEXT_COLOR)
        self._add_text(self._waiting_x + WAITING_WIDTH / 2, num_floors + HEADER_HEIGHT / 2,
                       'waiting', WAITING_WIDTH, 10, color=TEXT_COLOR)

    def _draw_elevator(self, index, elevator, floor, state):
        car_width = self._car_width
        x = NAME_WIDTH + index + (1 - car_width) / 2
        y = self._rows[floor] + CAR_Y
        riding_up = state[RDDLPlanningModel.ground_var('person-in-elevator-going-up', [elevator])]
        riding_down = state[RDDLPlanningModel.ground_var('person-in-elevator-going-down', [elevator])]
        riders = int(riding_up) + int(riding_down)
        closed = state[RDDLPlanningModel.ground_var('elevator-closed', [elevator])]
        going_up = state[RDDLPlanningModel.ground_var('elevator-dir-up', [elevator])]

        # same rule as the state: passengers get off at the floor they are heading to
        delivering = (riding_up and floor in self._top) or (riding_down and floor in self._bottom)

        # the cable from the top of the shaft, and the car around its panel and doors
        center = x + car_width / 2
        artists = self.ax.plot([center] * 2, [y + CAR_HEIGHT, len(self._floors)],
                               color=CABLE_COLOR, linewidth=1, zorder=2)
        rounding = 0.06 * min(self._column_width, self._row_height)
        artists.append(self.ax.add_patch(patches.FancyBboxPatch(
            (x, y), car_width, CAR_HEIGHT, boxstyle=f'round,pad=0,rounding_size={rounding}',
            mutation_aspect=self._column_width / self._row_height,
            facecolor=DELIVER_COLOR if delivering else FRAME_COLOR, edgecolor='none',
            zorder=3)))
        inset_x = FRAME * car_width
        inset_y = inset_x * self._column_width / self._row_height
        inner_x, inner_width = x + inset_x, car_width - 2 * inset_x
        door_y = y + inset_y
        door_height = (1 - PANEL) * CAR_HEIGHT - 2 * inset_y
        panel_y = door_y + door_height + inset_y
        panel_height = y + CAR_HEIGHT - inset_y - panel_y

        # the panel above the doors shows the direction of the car, then where each
        # passenger is heading, each in its own place
        artists.append(self.ax.add_patch(patches.Rectangle(
            (inner_x, panel_y), inner_width, panel_height, facecolor=PANEL_COLOR,
            edgecolor='none', zorder=4)))
        panel_middle = panel_y + panel_height / 2
        arrow_size = 0.7 * 72 * panel_height * self._row_height
        artists.extend(self.ax.plot(
            [inner_x + 0.2 * inner_width], [panel_middle], marker='^' if going_up else 'v',
            markersize=arrow_size, color=LED_COLOR, markeredgecolor='none', zorder=5))
        for (rider, marker, position) in ((riding_up, '^', 0.58), (riding_down, 'v', 0.82)):
            if rider:
                artists.extend(self.ax.plot(
                    [inner_x + position * inner_width], [panel_middle], marker=marker,
                    markersize=0.8 * arrow_size, color=DESTINATION_COLOR,
                    markeredgecolor='none', zorder=5))

        # closed doors meet in the middle; open ones slide to the sides of the lit cab,
        # where the passengers stand
        if closed:
            for left in (inner_x, center):
                artists.append(self.ax.add_patch(patches.Rectangle(
                    (left, door_y), inner_width / 2, door_height, facecolor=DOOR_COLOR,
                    edgecolor=DOOR_EDGE_COLOR, linewidth=0.8, zorder=4)))
        else:
            artists.append(self.ax.add_patch(patches.Rectangle(
                (inner_x, door_y), inner_width, door_height, facecolor=CAB_COLOR,
                edgecolor='none', zorder=4)))
            leaf = 0.12 * inner_width
            for left in (inner_x, inner_x + inner_width - leaf):
                artists.append(self.ax.add_patch(patches.Rectangle(
                    (left, door_y), leaf, door_height, facecolor=DOOR_COLOR,
                    edgecolor=DOOR_EDGE_COLOR, linewidth=0.8, zorder=5)))
            artists.extend(self._draw_people(
                inner_x + leaf, door_y + 0.1 * door_height, inner_width - 2 * leaf,
                0.8 * door_height, riders, RIDER_COLOR))
        return artists

    def _draw_waiting(self, floor, state):
        waiting_up = state[RDDLPlanningModel.ground_var('person-waiting-up', [floor])]
        waiting_down = state[RDDLPlanningModel.ground_var('person-waiting-down', [floor])]
        count = int(waiting_up) + int(waiting_down)
        y = self._rows[floor]
        artists = []
        label = self._add_text(self._waiting_x + 0.35, y + 0.5, str(count), 0.6, 10, zorder=5,
                               color=WAITING_COLOR if count > 0 else EMPTY_COLOR, fontweight='bold')
        if label is not None:
            artists.append(label)

        # each direction has its own place, so a person never appears to jump, with an
        # arrow beside the person towards the destination
        size_inches = min(0.24, 0.6 * self._row_height)
        beside = 0.5 * size_inches / self._column_width
        for (waiting, marker, x) in ((waiting_up, '^', 0.95), (waiting_down, 'v', 1.75)):
            if waiting:
                x += self._waiting_x
                artists.extend(self.ax.plot(
                    [x], [y + 0.45], marker=PERSON, markersize=72 * size_inches,
                    color=WAITING_COLOR, markeredgecolor='none', zorder=6))
                artists.extend(self.ax.plot(
                    [x + beside], [y + 0.55], marker=marker,
                    markersize=0.45 * 72 * size_inches, color=WAITING_COLOR,
                    markeredgecolor='none', zorder=6))
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
