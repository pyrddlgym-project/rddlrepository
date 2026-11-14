import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

GROUND_COLOR = '#e2a878'
GROUND_EDGE_COLOR = '#b7794f'
CRATER_COLOR = '#d1966a'
CRATER_RIM_COLOR = '#ecbd93'
PEBBLE_COLOR = '#b98058'
AREA_COLOR = '#f6dcc3'
AREA_EDGE_COLOR = '#c48b5c'
CRYSTAL_COLORS = ('#7e57c2', '#b39ddb')
DULL_COLORS = ('#90a4ae', '#cfd8dc')
HOLE_COLOR = '#8d5a3c'
TRACK_COLOR = '#a1683f'
BODY_COLOR = '#eceff1'
BODY_EDGE_COLOR = '#78909c'
PANEL_COLOR = '#283593'
WHEEL_COLOR = '#37474f'
ARM_COLOR = '#ff8f00'
TEXT_COLOR = '#37474f'
OK_COLOR = '#2e7d32'

# sizes on screen in inches: the rovers, the smallest and largest crystals, and the layout
ROVER_SIZE = 0.46
CRYSTAL_SIZES = (0.12, 0.36)
PLOT_SIZE = 5.0
LEFT_MARGIN = 0.25
BOTTOM_MARGIN = 0.85
TOP_MARGIN = 0.45
LEGEND_ROWS = 2
LEGEND_ROW = 0.34


class MarsRoverVisualizer(BaseViz):
    '''Draws the scene on the ground of Mars: each mineral is a cluster of crystals in its
    harvesting area, larger for a more valuable mineral and grey for one worth nothing, and
    a harvested one leaves a hole. Each rover faces the way it drives, with the tracks of
    its recent motion behind it, and reaches down with its arm while it can harvest. The
    view covers the minerals and the current positions of the rovers, and the drawing only
    depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 tail_steps=5,
                 wait_time=100) -> None:
        self._model = model
        self._tail_steps = tail_steps
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._rovers = model.type_to_objects['rover']
        self._minerals = model.type_to_objects['mineral']
        self._time_scale = self._nonfluents['SCALE-FACTOR']
        self._centers = {m: (self._get(self._nonfluents, 'MINERAL-POS-X', m),
                             self._get(self._nonfluents, 'MINERAL-POS-Y', m))
                         for m in self._minerals}
        self._areas = {m: self._get(self._nonfluents, 'MINERAL-AREA', m) for m in self._minerals}
        self._values = {m: self._get(self._nonfluents, 'MINERAL-VALUE', m) for m in self._minerals}
        self._top_value = max([v for v in self._values.values() if v > 0], default=1.0)

        # title above, then the scene and the legend
        width = PLOT_SIZE + 2 * LEFT_MARGIN
        height = BOTTOM_MARGIN + PLOT_SIZE + TOP_MARGIN
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        rect = [LEFT_MARGIN / width, BOTTOM_MARGIN / height, PLOT_SIZE / width, PLOT_SIZE / height]

        # the ground stays put on the screen, under the scene that zooms to fit
        ground = self.fig.add_axes(rect)
        self._draw_ground(ground)
        self.ax = self.fig.add_axes(rect)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('none')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color(GROUND_EDGE_COLOR)
        self._draw_legend()

    def _get(self, values, var, obj):
        return float(values[RDDLPlanningModel.ground_var(var, [obj])])

    # ---------- pictures: each takes the axes, a place and a size, and returns its artists

    @staticmethod
    def _draw_ground(ax):

        # rusty ground with the same craters and pebbles every time
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor(GROUND_COLOR)
        ax.set_xticks([])
        ax.set_yticks([])
        rng = np.random.default_rng(7)
        for _ in range(9):
            (x, y), r = rng.uniform(0.05, 0.95, 2), rng.uniform(0.03, 0.08)
            ax.add_patch(patches.Ellipse((x, y), 2 * r, 1.4 * r, facecolor=CRATER_COLOR,
                                         edgecolor=CRATER_RIM_COLOR, linewidth=1.5))
        for _ in range(60):
            (x, y), r = rng.uniform(0, 1, 2), rng.uniform(0.003, 0.008)
            ax.add_patch(patches.Ellipse((x, y), 2 * r, 1.5 * r, facecolor=PEBBLE_COLOR,
                                         edgecolor='none'))

    @staticmethod
    def _crystals(ax, x, y, size, colors):

        # a cluster of pointed crystals growing out of the ground around the given point
        artists = []
        for (dx, tall, angle) in ((-0.28, 0.7, 22), (0.3, 0.6, -24), (0.0, 1.0, 0), (-0.1, 0.45, -12)):
            w, h = 0.13 * size, tall * size
            shape = np.array([(-w, 0), (-w, 0.72 * h), (0, h), (w, 0.72 * h), (w, 0)])
            turn = np.radians(angle)
            rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
            points = shape @ rotation.T + (x + dx * size, y - 0.4 * size)
            artists.append(ax.add_patch(patches.Polygon(points, closed=True, facecolor=colors[0],
                                                        edgecolor=colors[1], linewidth=0.8,
                                                        zorder=3)))
            face = np.array([(0, 0.05 * h), (0, 0.95 * h), (w * 0.9, 0.7 * h), (w * 0.9, 0.05 * h)])
            artists.append(ax.add_patch(patches.Polygon(face @ rotation.T + (x + dx * size, y - 0.4 * size),
                                                        closed=True, facecolor=colors[1],
                                                        edgecolor='none', alpha=0.6, zorder=3.1)))
        return artists

    @staticmethod
    def _hole(ax, x, y, size):

        # where a mineral was dug out: a pit with some rubble beside it
        return [ax.add_patch(patches.Ellipse((x, y - 0.2 * size), 0.8 * size, 0.3 * size,
                                             facecolor=HOLE_COLOR, edgecolor='none', zorder=3)),
                ax.add_patch(patches.Circle((x + 0.48 * size, y - 0.25 * size), 0.06 * size,
                                            facecolor=PEBBLE_COLOR, edgecolor='none', zorder=3)),
                ax.add_patch(patches.Circle((x - 0.5 * size, y - 0.18 * size), 0.045 * size,
                                            facecolor=PEBBLE_COLOR, edgecolor='none', zorder=3))]

    @staticmethod
    def _rover(ax, x, y, size, facing, harvesting):

        # a six-wheeled rover seen from the side, centered on the given point, with a solar
        # deck, a camera mast at the front and an arm with a drill reaching down
        s, f = size, facing
        bottom = y - 0.3 * s
        artists = [ax.add_patch(patches.Rectangle((x - 0.38 * s, bottom + 0.12 * s), 0.76 * s,
                                                  0.05 * s, facecolor=WHEEL_COLOR, edgecolor='none',
                                                  zorder=5))]
        artists += [ax.add_patch(patches.Circle((x + dx * s, bottom + 0.12 * s), 0.12 * s,
                                                facecolor=WHEEL_COLOR, edgecolor='white',
                                                linewidth=0.5, zorder=5.1)) for dx in (-0.36, 0, 0.36)]
        artists += [
            ax.add_patch(patches.FancyBboxPatch(
                (x - 0.4 * s, bottom + 0.24 * s), 0.8 * s, 0.2 * s,
                boxstyle=f'round,pad=0,rounding_size={0.04 * s}', facecolor=BODY_COLOR,
                edgecolor=BODY_EDGE_COLOR, linewidth=0.8, zorder=5.2)),
            ax.add_patch(patches.Rectangle((x - 0.46 * s, bottom + 0.44 * s), 0.92 * s, 0.05 * s,
                                           facecolor=PANEL_COLOR, edgecolor='none', zorder=5.3)),
            ax.add_patch(patches.Rectangle((x + f * 0.26 * s - 0.025 * s, bottom + 0.49 * s), 0.05 * s,
                                           0.3 * s, facecolor=BODY_EDGE_COLOR, edgecolor='none',
                                           zorder=5.3)),
            ax.add_patch(patches.Rectangle((x + f * 0.26 * s - 0.09 * s, bottom + 0.78 * s), 0.18 * s,
                                           0.09 * s, facecolor=BODY_COLOR, edgecolor=BODY_EDGE_COLOR,
                                           linewidth=0.6, zorder=5.4))]
        if harvesting:
            joint = (x + f * 0.62 * s, bottom + 0.4 * s)
            tip = (x + f * 0.62 * s, bottom + 0.05 * s)
            artists += ax.plot([x + f * 0.38 * s, joint[0], tip[0]],
                               [bottom + 0.33 * s, joint[1], tip[1]], color=ARM_COLOR,
                               linewidth=2.2, solid_capstyle='round', zorder=5.5)
            artists.append(ax.add_patch(patches.Polygon(
                [(tip[0] - 0.05 * s, tip[1]), (tip[0] + 0.05 * s, tip[1]), (tip[0], tip[1] - 0.1 * s)],
                closed=True, facecolor=WHEEL_COLOR, edgecolor='none', zorder=5.6)))
        return artists

    def _draw_legend(self):
        (width, height) = self._figure_size
        ax = self.fig.add_axes([0, 0.05 / height, 1, LEGEND_ROWS * LEGEND_ROW / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, LEGEND_ROWS * LEGEND_ROW)
        ax.set_aspect('equal')
        ax.axis('off')
        entries = [('rover', 'Rover'), ('harvest', 'Can harvest'), ('area', 'Harvest area'),
                   ('mineral', 'Mineral'), ('valuable', 'More valuable'), ('hole', 'Harvested')]
        per_row = 3
        spacing = (width - 0.4) / per_row
        for (k, (kind, label)) in enumerate(entries):
            r, c = divmod(k, per_row)
            x = 0.35 + c * spacing
            y = LEGEND_ROW * (LEGEND_ROWS - r - 0.5)
            if kind in ('rover', 'harvest'):
                self._rover(ax, x, y, 0.32, 1, kind == 'harvest')
            elif kind == 'area':
                ax.add_patch(patches.Circle((x, y), 0.13, facecolor=AREA_COLOR,
                                            edgecolor=AREA_EDGE_COLOR, linestyle='--', linewidth=1))
            elif kind == 'hole':
                self._hole(ax, x, y + 0.04, 0.26)
            else:
                self._crystals(ax, x, y + 0.03, 0.16 if kind == 'mineral' else 0.27, CRYSTAL_COLORS)
            ax.text(x + 0.3, y, label, ha='left', va='center', fontsize=8.5, color=TEXT_COLOR)

    # ---------- the scene

    def _update_view(self, positions):

        # both axes use the same scale, with a small margin around everything and room for
        # the picture of a rover at the edge
        boxes = [(x, x, y, y) for (x, y) in positions.values()]
        boxes.extend((x - self._areas[m], x + self._areas[m], y - self._areas[m], y + self._areas[m])
                     for (m, (x, y)) in self._centers.items())
        left, right = min(b[0] for b in boxes), max(b[1] for b in boxes)
        bottom, top = min(b[2] for b in boxes), max(b[3] for b in boxes)
        span = (1.1 * max(right - left, top - bottom) or 1.0) / (1 - 1.2 * ROVER_SIZE / PLOT_SIZE)
        x_center, y_center = (left + right) / 2, (bottom + top) / 2
        self.ax.set_xlim(x_center - span / 2, x_center + span / 2)
        self.ax.set_ylim(y_center - span / 2, y_center + span / 2)
        return span / PLOT_SIZE

    def _draw_mineral(self, mineral, harvested, per_inch):
        x, y = self._centers[mineral]
        area = self._areas[mineral]
        artists = [self.ax.add_patch(plt.Circle(
            (x, y), area, facecolor=AREA_COLOR, edgecolor=AREA_EDGE_COLOR, alpha=0.75,
            linestyle='--', linewidth=1, zorder=1))]

        # the crystals grow with the value, but stay inside of their area
        value = self._values[mineral]
        low, high = CRYSTAL_SIZES
        size = min((low + (high - low) * np.sqrt(max(value, 0) / self._top_value)) * per_inch,
                   1.4 * area)
        if harvested:
            artists.extend(self._hole(self.ax, x, y, size))
        else:
            artists.extend(self._crystals(self.ax, x, y, size,
                                          CRYSTAL_COLORS if value > 0 else DULL_COLORS))
        return artists

    def _can_harvest(self, position, available):

        # same rule as the domain
        return any((position[0] - self._centers[m][0]) ** 2 + (position[1] - self._centers[m][1]) ** 2
                   < self._areas[m] ** 2 for m in available)

    def _draw_rover(self, rover, position, state, available, per_inch):
        x, y = position
        (vx, vy) = (self._get(state, 'vel-x', rover), self._get(state, 'vel-y', rover))
        length = self._tail_steps * self._time_scale
        artists = self.ax.plot([x, x - length * vx], [y, y - length * vy], color=TRACK_COLOR,
                               linewidth=4, alpha=0.45, solid_capstyle='round', zorder=4)
        artists += self._rover(self.ax, x, y, ROVER_SIZE * per_inch, -1 if vx < 0 else 1,
                               self._can_harvest(position, available))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        positions = {r: (self._get(state, 'pos-x', r), self._get(state, 'pos-y', r))
                     for r in self._rovers}
        harvested = {m: bool(state[RDDLPlanningModel.ground_var('mineral-harvested', [m])])
                     for m in self._minerals}
        available = [m for m in self._minerals if not harvested[m]]
        per_inch = self._update_view(positions)

        artists = []
        for m in self._minerals:
            artists.extend(self._draw_mineral(m, harvested[m], per_inch))
        for r in self._rovers:
            artists.extend(self._draw_rover(r, positions[r], state, available, per_inch))

        collected = sum(self._values[m] for m in self._minerals if harvested[m])
        total = sum(self._values.values())
        self.ax.set_title(
            f'{len(self._minerals) - len(available)} / {len(self._minerals)} minerals harvested, '
            f'value {collected:.3g} / {total:.3g}', fontsize=13, fontweight='bold',
            color=OK_COLOR if not available else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
