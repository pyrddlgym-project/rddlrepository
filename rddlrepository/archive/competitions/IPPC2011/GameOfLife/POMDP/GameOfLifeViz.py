import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

GRASS_COLOR = '#e3f1d4'
SOIL_COLOR = '#f1e8d8'
GRID_COLOR = '#d7ccc8'
TRUNK_COLOR = '#6d4c41'
LEAF_COLOR = '#43a047'
LEAF_EDGE_COLOR = '#2e7d32'
OLD_LEAF_COLORS = ('#f9a825', '#ef6c00')
SPROUT_COLOR = '#9ccc65'
DEAD_WOOD_COLOR = '#c9bfb3'

# the phases of a cell, each drawn as a plant
PHASES = ('tree', 'old', 'sprout', 'dead')


class GameOfLifeVisualizer(BaseViz):
    '''Draws the grid of cells as plants, by whether each one is alive now and whether
    the rules of the game would keep it alive on the next step, given its neighbors: a
    tree lives on, an old tree dies, a sprout is born and a dead tree stays dead. Living
    cells stand on grass and dead ones on soil. What actually happens can differ,
    because of the noise and the cells that are set.

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
        bottom = 1.2
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
        for spine in self.ax.spines.values():
            spine.set_color(GRID_COLOR)

        # every cell has its ground and all four plants, drawn once: render only changes
        # the color of the ground and which plant shows
        self._cell_size = cell_size
        self._ground, self._plants = {}, {}
        for (col, row, x, y) in self._cells:
            self._ground[(x, y)] = self.ax.add_patch(patches.Rectangle(
                (col, row), 1, 1, facecolor=SOIL_COLOR, edgecolor=GRID_COLOR, zorder=1))
            self._plants[(x, y)] = {phase: self._draw_plant(self.ax, phase, col + 0.5, row, 1.0,
                                                            cell_size)
                                    for phase in PHASES}
        self._draw_legend(width, height)

        noise = [self._nonfluents[RDDLPlanningModel.ground_var('NOISE-PROB', [x, y])]
                 for (_, _, x, y) in self._cells]
        low, high = min(noise), max(noise)
        amount = f'{low:.2g}' if low == high else f'{low:.2g} to {high:.2g}'
        self.fig.text(0.5, 0.1 / height, f'Next step ignores actions and noise ({amount} per cell)',
                      ha='center', va='bottom', fontsize=9, color='#37474f')

    @staticmethod
    def _draw_plant(ax, phase, x, y, size, inches):

        # a plant standing at the bottom middle of a square of the given size, all of it
        # hidden: lines are as thick as the square is large in inches
        width = 72 * inches * size
        artists = []

        def circle(cx, cy, r, color, edge):
            artists.append(ax.add_patch(patches.Circle(
                (x + cx * size, y + cy * size), r * size, facecolor=color, edgecolor=edge,
                linewidth=0.012 * width, zorder=3)))

        def line(xs, ys, color, thickness, zorder=2.5):
            artists.extend(ax.plot([x + a * size for a in xs], [y + b * size for b in ys],
                                   color=color, linewidth=thickness * width,
                                   solid_capstyle='round', zorder=zorder))

        if phase == 'tree':
            artists.append(ax.add_patch(patches.Rectangle(
                (x - 0.045 * size, y + 0.1 * size), 0.09 * size, 0.35 * size,
                facecolor=TRUNK_COLOR, edgecolor='none', zorder=2.5)))
            for (cx, cy, r) in ((-0.15, 0.48, 0.16), (0.15, 0.48, 0.16), (0.0, 0.62, 0.22)):
                circle(cx, cy, r, LEAF_COLOR, LEAF_EDGE_COLOR)
        elif phase == 'old':
            line([0, 0], [0.1, 0.5], TRUNK_COLOR, 0.07)
            line([0, -0.14], [0.38, 0.52], TRUNK_COLOR, 0.035)
            for (cx, cy, r, k) in ((-0.15, 0.54, 0.1, 1), (0.06, 0.6, 0.15, 0), (0.17, 0.48, 0.08, 1)):
                circle(cx, cy, r, OLD_LEAF_COLORS[k], '#bf6f00')
            artists.append(ax.add_patch(patches.Ellipse(
                (x + 0.3 * size, y + 0.22 * size), 0.09 * size, 0.045 * size, angle=-30,
                facecolor=OLD_LEAF_COLORS[1], edgecolor='none', zorder=3)))
        elif phase == 'sprout':
            artists.append(ax.add_patch(patches.Ellipse(
                (x, y + 0.1 * size), 0.34 * size, 0.08 * size, facecolor='#d7c4a5',
                edgecolor='none', zorder=2.4)))
            line([0, 0], [0.1, 0.32], LEAF_EDGE_COLOR, 0.025)
            for (side, angle) in ((-1, 30), (1, -30)):
                artists.append(ax.add_patch(patches.Ellipse(
                    (x + side * 0.08 * size, y + 0.33 * size), 0.16 * size, 0.07 * size,
                    angle=angle, facecolor=SPROUT_COLOR, edgecolor=LEAF_EDGE_COLOR,
                    linewidth=0.01 * width, zorder=3)))
        else:
            line([0, 0], [0.1, 0.5], DEAD_WOOD_COLOR, 0.025)
            line([0, -0.12], [0.32, 0.46], DEAD_WOOD_COLOR, 0.017)
            line([0, 0.11], [0.38, 0.54], DEAD_WOOD_COLOR, 0.017)
        for artist in artists:
            artist.set_visible(False)
        return artists

    def _draw_legend(self, width, height):

        # the legend shows the same plants as the grid
        ax = self.fig.add_axes([0, 0.3 / height, 1, 0.55 / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, 0.55)
        ax.set_aspect('equal')
        ax.axis('off')
        entries = (('tree', GRASS_COLOR, 'Alive'), ('old', GRASS_COLOR, 'Dies next'),
                   ('sprout', SOIL_COLOR, 'Born next'), ('dead', SOIL_COLOR, 'Dead'))
        spacing = min(1.6, (width - 0.2) / len(entries))
        for (k, (phase, ground, label)) in enumerate(entries):
            x = (width - len(entries) * spacing) / 2 + k * spacing + 0.1
            ax.add_patch(patches.Rectangle((x, 0.05), 0.45, 0.45, facecolor=ground,
                                           edgecolor=GRID_COLOR))
            for artist in self._draw_plant(ax, phase, x + 0.225, 0.05, 0.45, 0.45):
                artist.set_visible(True)
            ax.text(x + 0.55, 0.275, label, ha='left', va='center', fontsize=8.5,
                    color='#37474f')

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
        num_alive = 0
        for (col, row, x, y) in self._cells:
            alive = state[RDDLPlanningModel.ground_var('alive', [x, y])]
            will_live = self._will_live(state, x, y)
            num_alive += bool(alive)
            if alive:
                phase = 'tree' if will_live else 'old'
            else:
                phase = 'sprout' if will_live else 'dead'
            self._ground[(x, y)].set_facecolor(GRASS_COLOR if alive else SOIL_COLOR)
            for (other, plant) in self._plants[(x, y)].items():
                for artist in plant:
                    artist.set_visible(other == phase)

        if num_alive == 0:
            title, color = 'All cells dead', '#c62828'
        elif num_alive == len(self._cells):
            title, color = 'All cells alive', '#2e7d32'
        else:
            title, color = f'{num_alive} / {len(self._cells)} cells alive', 'black'
        self.ax.set_title(title, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
