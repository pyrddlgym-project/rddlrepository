import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

ALIVE_COLOR = '#1565c0'
DYING_COLOR = '#42a5f5'
BORN_COLOR = '#bbdefb'
DEAD_COLOR = '#fafafa'
GRID_COLOR = '#b0bec5'


class GameOfLifeVisualizer(BaseViz):
    '''Draws the grid of cells, coloured by whether each one is alive now and whether
    the rules of the game would keep it alive on the next step, given its neighbors.
    What actually happens can differ, because of the noise and the cells that are set.

    The POMDP shows every cell, rather than only what is observed around each one.
    The drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_cell_size=2.0,
                 max_size=5.5,
                 min_width=5.6,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)

        # each cell only needs to look at its neighbors on every step
        x_objects = model.type_to_objects['x_pos']
        y_objects = model.type_to_objects['y_pos']
        self._neighbors = {(x, y): [] for x in x_objects for y in y_objects}
        x_links, y_links = {x: set() for x in x_objects}, {y: set() for y in y_objects}
        for (name, value) in self._nonfluents.items():
            var, objects = RDDLPlanningModel.parse_grounded(name)
            if var == 'NEIGHBOR' and value:
                x, y, x2, y2 = objects
                self._neighbors[(x, y)].append(RDDLPlanningModel.ground_var('alive', [x2, y2]))
                if x != x2:
                    x_links[x].add(x2)
                    x_links[x2].add(x)
                if y != y2:
                    y_links[y].add(y2)
                    y_links[y2].add(y)

        # the positions follow the neighbors, whatever the objects are called
        self._xs = self._line_order(x_objects, x_links)
        self._ys = self._line_order(y_objects, y_links)
        self._cells = [(col, row, x, y) for (col, x) in enumerate(self._xs)
                       for (row, y) in enumerate(self._ys)]

        # title above, then the grid, its coordinates, the legend and the caption
        cell_size = min(max_cell_size, max_size / max(len(self._xs), len(self._ys)))
        grid_width = len(self._xs) * cell_size
        grid_height = len(self._ys) * cell_size
        bottom = 0.95
        width = max(grid_width + 0.8, min_width)
        height = bottom + grid_height + 0.45
        self._figure_size = (width, height)

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes(
            [(width - grid_width) / (2 * width), bottom / height,
             grid_width / width, grid_height / height])
        self.ax.set_xlim(0, len(self._xs))
        self.ax.set_ylim(0, len(self._ys))
        self.ax.set_aspect('equal')
        self.ax.set_xticks(np.arange(len(self._xs)) + 0.5, self._xs)
        self.ax.set_yticks(np.arange(len(self._ys)) + 0.5, self._ys)
        self.ax.tick_params(length=0, labelsize=9, colors='#616161')
        self.ax.set_facecolor(DEAD_COLOR)
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        self.fig.legend(handles=[
            patches.Patch(facecolor=ALIVE_COLOR, edgecolor=GRID_COLOR, label='Alive'),
            patches.Patch(facecolor=DYING_COLOR, edgecolor=GRID_COLOR, label='Alive, dies next'),
            patches.Patch(facecolor=BORN_COLOR, edgecolor=GRID_COLOR, label='Dead, born next'),
            patches.Patch(facecolor=DEAD_COLOR, edgecolor=GRID_COLOR, label='Dead')
        ], loc='lower center', bbox_to_anchor=(0.5, 0.25 / height), ncol=4,
           frameon=False, fontsize=8.5, columnspacing=1.0, handlelength=1.4)

        noise = [self._nonfluents[RDDLPlanningModel.ground_var('NOISE-PROB', [x, y])]
                 for (_, _, x, y) in self._cells]
        low, high = min(noise), max(noise)
        amount = f'{low:.2g}' if low == high else f'{low:.2g} to {high:.2g}'
        self.fig.text(0.5, 0.1 / height, f'Next step ignores actions and noise ({amount} per cell)',
                      ha='center', va='bottom', fontsize=9, color='#37474f')

    def _line_order(self, objects, links):

        # the objects along an axis form a line or loop when each links to at most two
        # others, and anything else has no clear order, so use the declared one
        if any(len(linked) > 2 for linked in links.values()):
            return objects
        ends = [a for a in objects if len(links[a]) == 1]
        current = ends[0] if ends else objects[0]
        order = [current]
        while True:
            following = [a for a in objects if a in links[current] and a not in order]
            if not following:
                break
            current = following[0]
            order.append(current)
        return order if len(order) == len(objects) else objects

    def _will_live(self, state, x, y):
        count = sum(bool(state[neighbor]) for neighbor in self._neighbors[(x, y)])
        if state[RDDLPlanningModel.ground_var('alive', [x, y])]:
            return count == 2 or count == 3
        return count == 3

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        artists = []
        num_alive = 0
        for (col, row, x, y) in self._cells:
            alive = state[RDDLPlanningModel.ground_var('alive', [x, y])]
            will_live = self._will_live(state, x, y)
            num_alive += bool(alive)
            if alive:
                color = ALIVE_COLOR if will_live else DYING_COLOR
            else:
                color = BORN_COLOR if will_live else DEAD_COLOR
            artists.append(self.ax.add_patch(patches.Rectangle(
                (col, row), 1, 1, facecolor=color, edgecolor=GRID_COLOR)))

        if num_alive == 0:
            title, color = 'All cells dead', '#c62828'
        elif num_alive == len(self._cells):
            title, color = 'All cells alive', '#2e7d32'
        else:
            title, color = f'{num_alive} / {len(self._cells)} cells alive', 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
