import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

VISIBILITY_COLORS = {'high': '#ffffff', 'medium': '#cfd8dc', 'low': '#90a4ae'}
GRID_COLOR = '#b0bec5'
WRAP_COLOR = '#1565c0'
FOCUS_COLOR = '#1565c0'
TARGET_COLOR = '#ff8f00'
PHOTO_COLOR = '#2e7d32'
FAILED_COLOR = '#c62828'
NAME_COLOR = '#455a64'
DETAIL_COLOR = '#37474f'

# how each direction moves the focus on the map, which is what the map is recovered from
DIRECTIONS = {'north-east': (1, 1), 'east': (1, 0), 'south-east': (1, -1)}

# sizes in inches, and in fractions of a cell for what is drawn inside of one
MAX_CELL_SIZE = 0.9
BOTTOM_MARGIN = 0.62
TOP_MARGIN = 0.45
TARGET_RADIUS = 0.24
ARROW_START = 0.16
ARROW_END = 0.42
FOCUS_INSET = 0.04

# names and arrows do not fit on smaller cells, and the rings show the moves without them
MIN_NAME_CELL = 0.45
MIN_ARROW_CELL = 0.4
NAME_SIZE = 6.5


class EarthObservationVisualizer(BaseViz):
    '''Draws the map of regions shaded by their visibility, with the targets, the region in
    focus and the regions that the focus can move to next. The map is recovered from how
    the regions connect, and wraps around from east to west if they do.

    The state only tracks the targets that remain, so the targets of the initial state
    that are gone are the ones shown as photographed. The drawing otherwise only depends
    on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_size=(9.0, 5.5),
                 min_width=8.6,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._patches = list(model.type_to_objects['patch'])
        self._links = {patch: {direction: [other for other in self._patches
                                           if self._nonfluent('CONNECTED', patch, other, direction)]
                               for direction in DIRECTIONS} for patch in self._patches}

        initial = model.ground_vars_with_values(model.state_fluents)
        self._targets = [patch for patch in self._patches
                         if initial[RDDLPlanningModel.ground_var('is-target', [patch])]]

        self._places, self._is_map, self._wraps = self._layout()
        columns = max(col for (col, _) in self._places.values()) + 1
        rows = max(row for (_, row) in self._places.values()) + 1
        self._cell = min(MAX_CELL_SIZE, max_size[0] / columns, max_size[1] / rows)
        grid_width, grid_height = columns * self._cell, rows * self._cell

        # title above, then the map, the legend and the caption
        width = max(grid_width + 0.4, min_width)
        height = grid_height + BOTTOM_MARGIN + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([(width - grid_width) / (2 * width), BOTTOM_MARGIN / height,
                                     grid_width / width, grid_height / height])
        self.ax.set_xlim(0, columns)
        self.ax.set_ylim(0, rows)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for (side, spine) in self.ax.spines.items():
            wrapped = self._wraps and side in ('left', 'right')
            spine.set(color=WRAP_COLOR if wrapped else GRID_COLOR, linewidth=1.6 if wrapped else 1,
                      linestyle='--' if wrapped else '-')

        # the regions never move, so they are only drawn once and then recolored
        self._cells, self._target_marks, self._photo_marks = {}, {}, {}
        for (patch, (col, row)) in self._places.items():
            self._cells[patch] = self.ax.add_patch(patches.Rectangle(
                (col, row), 1, 1, edgecolor=GRID_COLOR, linewidth=0.6, zorder=1))
            self._add_marks(patch, col + 0.5, row + 0.5)
            if self._cell >= MIN_NAME_CELL and NAME_SIZE * 0.6 * len(patch) / 72 <= self._cell:
                self.ax.text(col + 0.5, row + 0.13, patch, ha='center', va='center',
                             fontsize=NAME_SIZE, color=NAME_COLOR, zorder=2)

        self.fig.legend(handles=self._legend_handles(), loc='lower center',
                        bbox_to_anchor=(0.5, 0.25 / height), ncol=7, frameon=False,
                        fontsize=8.5, columnspacing=1.0, handlelength=1.4)

        wrap = ' East wraps around to west.' if self._wraps else ''
        self.fig.text(0.5, 0.1 / height,
                      f'Percentages are chances of a good photo. Steps cost 1 per target left, '
                      f'plus 1 for a photo or shift.{wrap}',
                      ha='center', va='bottom', fontsize=9, color=DETAIL_COLOR)

    def _nonfluent(self, var, *args):
        return self._nonfluents[RDDLPlanningModel.ground_var(var, list(args))]

    def _east_cycle(self, start):

        # how many steps east lead back to where they started, if they ever do
        current, steps, seen = start, 0, set()
        while current not in seen:
            seen.add(current)
            following = self._links[current]['east']
            if not following:
                return None
            current = following[0]
            steps += 1
            if current == start:
                return steps
        return None

    def _walk(self, start, seen):

        # follow the connections from one region in both directions to place all that it
        # reaches, and return None if they do not agree with each other as a map
        cycle = self._east_cycle(start)
        place = {start: (0, 0)}
        stack = [start]
        while stack:
            patch = stack.pop()
            (col, row) = place[patch]
            neighbors = []
            for (direction, (dx, dy)) in DIRECTIONS.items():
                neighbors += [(other, dx, dy) for other in self._links[patch][direction]]
                neighbors += [(other, -dx, -dy) for other in self._patches
                              if patch in self._links[other][direction]]
            for (other, dx, dy) in neighbors:
                new = (col + dx if cycle is None else (col + dx) % cycle, row + dy)
                if other not in place:
                    place[other] = new
                    stack.append(other)
                elif place[other] != new:
                    return None, cycle
        seen.update(place)
        return place, cycle

    def _layout(self):

        # regions that are not connected to each other go on top of each other
        places, seen = {}, set()
        wraps, is_map = False, True
        for patch in self._patches:
            if patch in seen:
                continue
            (place, cycle) = self._walk(patch, seen)
            if place is None:
                is_map = False
                break
            wraps = wraps or cycle is not None
            left = min(col for (col, _) in place.values())
            bottom = min(row for (_, row) in place.values())
            base = max((row for (_, row) in places.values()), default=-1) + 1
            places.update((p, (col - left, row - bottom + base)) for (p, (col, row)) in place.items())
        if is_map and len(set(places.values())) == len(self._patches):
            return places, True, wraps

        # the connections do not form a map, so the regions are only listed in order
        columns = math.ceil(math.sqrt(len(self._patches)))
        rows = math.ceil(len(self._patches) / columns)
        return ({patch: (i % columns, rows - 1 - i // columns)
                 for (i, patch) in enumerate(self._patches)}, False, False)

    def _add_marks(self, patch, x, y):
        ring = patches.Circle((x, y), TARGET_RADIUS, facecolor='none', edgecolor=TARGET_COLOR,
                              linewidth=2.2, zorder=3)
        dot = patches.Circle((x, y), TARGET_RADIUS / 3, facecolor=TARGET_COLOR, zorder=3)
        photo = patches.Circle((x, y), TARGET_RADIUS, facecolor=PHOTO_COLOR, edgecolor='white',
                               linewidth=1, zorder=3)
        tick = self.ax.text(x, y, '✓', ha='center', va='center', color='white', zorder=3.5,
                            fontsize=min(11, 0.32 * self._cell * 72), fontweight='bold')
        self._target_marks[patch] = [self.ax.add_patch(ring), self.ax.add_patch(dot)]
        self._photo_marks[patch] = [self.ax.add_patch(photo), tick]

    def _legend_handles(self):
        chances = {level: 1 - self._nonfluent(f'FAILURE_PROB_{level.upper()}_VIS')
                   for level in VISIBILITY_COLORS}
        return [
            patches.Patch(facecolor=VISIBILITY_COLORS['high'], edgecolor=GRID_COLOR,
                          label=f'High visibility {chances["high"]:.0%}'),
            patches.Patch(facecolor=VISIBILITY_COLORS['medium'], edgecolor=GRID_COLOR,
                          label=f'Medium {chances["medium"]:.0%}'),
            patches.Patch(facecolor=VISIBILITY_COLORS['low'], edgecolor=GRID_COLOR,
                          label=f'Low {chances["low"]:.0%}'),
            Line2D([], [], marker='o', color='none', markerfacecolor='none',
                   markeredgecolor=TARGET_COLOR, markeredgewidth=2, markersize=9, label='Target'),
            Line2D([], [], marker='o', color='none', markerfacecolor=PHOTO_COLOR,
                   markeredgecolor='white', markersize=10, label='Photographed'),
            patches.Patch(facecolor='none', edgecolor=FOCUS_COLOR, linewidth=2.2, label='Focus'),
            patches.Patch(facecolor='none', edgecolor=FOCUS_COLOR, linestyle='--', linewidth=1.2,
                          label='Next focus')
        ]

    def _draw_focus(self, patch):
        artists = []
        (col, row) = self._places[patch]
        inset = FOCUS_INSET
        artists.append(self.ax.add_patch(patches.Rectangle(
            (col + inset, row + inset), 1 - 2 * inset, 1 - 2 * inset, facecolor='none',
            edgecolor=FOCUS_COLOR, linewidth=2.4, zorder=4)))

        # the regions it can move to are ringed since they may be across the wrap, and arrows
        # show the direction of each when the regions form a map
        for (direction, (dx, dy)) in DIRECTIONS.items():
            following = self._links[patch][direction]
            for other in following:
                (next_col, next_row) = self._places[other]
                artists.append(self.ax.add_patch(patches.Rectangle(
                    (next_col + inset, next_row + inset), 1 - 2 * inset, 1 - 2 * inset,
                    facecolor='none', edgecolor=FOCUS_COLOR, linewidth=1.2, linestyle='--', zorder=2.5)))
            if following and self._is_map and self._cell >= MIN_ARROW_CELL:
                size = math.hypot(dx, dy)
                center_x, center_y = col + 0.5, row + 0.5
                start = (center_x + dx / size * ARROW_START, center_y + dy / size * ARROW_START)
                end = (center_x + dx * ARROW_END, center_y + dy * ARROW_END)
                artists.append(self.ax.add_patch(patches.FancyArrowPatch(
                    start, end, arrowstyle='-|>', mutation_scale=min(9, 26 * self._cell),
                    color=FOCUS_COLOR, linewidth=1.2, shrinkA=0, shrinkB=0, zorder=5)))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        fluent = RDDLPlanningModel.ground_var
        remaining = [patch for patch in self._patches if bool(state[fluent('is-target', [patch])])]
        for patch in self._patches:
            self._cells[patch].set_facecolor(
                VISIBILITY_COLORS[str(state[fluent('visibility', [patch])])])
            for mark in self._target_marks[patch]:
                mark.set_visible(patch in remaining)
            for mark in self._photo_marks[patch]:
                mark.set_visible(patch in self._targets and patch not in remaining)

        focus = [patch for patch in self._patches if bool(state[fluent('is-focal-point', [patch])])]
        artists = []
        for patch in focus:
            artists.extend(self._draw_focus(patch))

        total = len(set(self._targets) | set(remaining))
        done = total - len(remaining)
        plural = '' if total == 1 else 's'
        if total == 0:
            title, color = 'No targets', 'black'
        elif not remaining:
            title, color = f'{total} / {total} target{plural} photographed', PHOTO_COLOR
        elif not focus:

            # a step without a slew breaks the preconditions, and leaves nothing in focus for good
            title, color = f'{done} / {total} target{plural} photographed, focus lost', FAILED_COLOR
        else:
            title, color = f'{done} / {total} target{plural} photographed', 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
