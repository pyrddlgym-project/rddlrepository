import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

PLASMA_COLOR = '#fdf1e4'
LENS_COLOR = '#37474f'
T_CELL_COLOR = '#90caf9'
T_CELL_DARK_COLOR = '#1e88e5'
MACROPHAGE_COLOR = '#ffe0b2'
MACROPHAGE_DARK_COLOR = '#ef6c00'
INFECTED_COLOR = '#ce93d8'
INFECTED_DARK_COLOR = '#8e24aa'
VIRUS_COLOR = '#e53935'
KILLER_COLOR = '#a5d6a7'
KILLER_DARK_COLOR = '#2e7d32'
RTI_COLOR = '#1565c0'
PI_COLOR = '#f9a825'
OFF_COLOR = '#cfd8dc'
TEXT_COLOR = '#37474f'

# how many of each population are drawn: so many per factor of ten above a floor, up to
# a largest number, since the populations span many orders of magnitude
#   (state fluent, picture, floor in log10, pictures per factor of ten, most pictures)
POPULATIONS = [('t1', 't-cell', 3.0, 5.0, 18), ('i1', 'infected-t-cell', 0.0, 1.5, 12),
               ('t2', 'macrophage', 0.0, 2.0, 8), ('i2', 'infected-macrophage', -1.0, 2.0, 6),
               ('v', 'virus', 0.0, 3.0, 20), ('e', 'killer', 0.0, 3.0, 18)]
PICTURE_SIZE = {'t-cell': 0.16, 'infected-t-cell': 0.16, 'macrophage': 0.26, 'infected-macrophage': 0.26,
                'virus': 0.09, 'killer': 0.15}

# sizes in inches: the field of view of each patient, the drugs under it, and the legend
VIEW = 2.3
DRUGS = 0.5
GAP = 0.3
MAX_COLUMNS = 5
TOP_MARGIN = 0.3
LEGEND_HEIGHT = 0.95


# ---------- pictures: each takes the axes, a centre and a size in inches, and returns its artists

def _cell(ax, x, y, s, fill, edge, spots=0, rng=None):

    # a round cell with a nucleus, and red virus inside if infected
    artists = [ax.add_patch(plt.Circle((x, y), s / 2, facecolor=fill, edgecolor=edge, linewidth=0.8, zorder=2)),
               ax.add_patch(plt.Circle((x - 0.08 * s, y + 0.06 * s), 0.2 * s, facecolor=edge, edgecolor='none',
                                       alpha=0.55, zorder=2.1))]
    for k in range(spots):
        angle = 2 * math.pi * k / spots + 0.6
        artists.append(ax.add_patch(plt.Circle((x + 0.28 * s * math.cos(angle), y + 0.28 * s * math.sin(angle)),
                                               0.07 * s, facecolor=VIRUS_COLOR, edgecolor='none', zorder=2.2)))
    return artists


def _macrophage(ax, x, y, s, fill, edge, spots=0):

    # a large cell with an uneven outline, stretching out arms
    points = []
    for k in range(14):
        angle = 2 * math.pi * k / 14
        radius = s / 2 * (1.0 if k % 3 else 0.72)
        points.append((x + radius * math.cos(angle), y + 0.85 * radius * math.sin(angle)))
    artists = [ax.add_patch(patches.Polygon(points, closed=True, facecolor=fill, edgecolor=edge, linewidth=0.8,
                                            joinstyle='round', zorder=2)),
               ax.add_patch(patches.Ellipse((x + 0.05 * s, y), 0.32 * s, 0.24 * s, facecolor=edge, edgecolor='none',
                                            alpha=0.5, zorder=2.1))]
    for k in range(spots):
        angle = 2 * math.pi * k / spots + 0.3
        artists.append(ax.add_patch(plt.Circle((x + 0.25 * s * math.cos(angle), y + 0.22 * s * math.sin(angle)),
                                               0.06 * s, facecolor=VIRUS_COLOR, edgecolor='none', zorder=2.2)))
    return artists


def _virus(ax, x, y, s):

    # a small round particle covered in spikes
    artists = []
    for k in range(8):
        angle = 2 * math.pi * k / 8
        artists += ax.plot([x + 0.3 * s * math.cos(angle), x + 0.5 * s * math.cos(angle)],
                           [y + 0.3 * s * math.sin(angle), y + 0.5 * s * math.sin(angle)],
                           color=VIRUS_COLOR, linewidth=0.8, solid_capstyle='round', zorder=2.3)
    artists.append(ax.add_patch(plt.Circle((x, y), 0.32 * s, facecolor=VIRUS_COLOR, edgecolor='none', zorder=2.4)))
    return artists


def _killer(ax, x, y, s):

    # a green killer T cell with short hairs
    artists = []
    for k in range(10):
        angle = 2 * math.pi * k / 10
        artists += ax.plot([x + 0.48 * s * math.cos(angle), x + 0.62 * s * math.cos(angle)],
                           [y + 0.48 * s * math.sin(angle), y + 0.62 * s * math.sin(angle)],
                           color=KILLER_DARK_COLOR, linewidth=0.7, zorder=2)
    return artists + _cell(ax, x, y, s, KILLER_COLOR, KILLER_DARK_COLOR)


def _picture(ax, kind, x, y, s=None):
    s = PICTURE_SIZE[kind] if s is None else s
    if kind == 't-cell':
        return _cell(ax, x, y, s, T_CELL_COLOR, T_CELL_DARK_COLOR)
    if kind == 'infected-t-cell':
        return _cell(ax, x, y, s, INFECTED_COLOR, INFECTED_DARK_COLOR, spots=4)
    if kind == 'macrophage':
        return _macrophage(ax, x, y, s, MACROPHAGE_COLOR, MACROPHAGE_DARK_COLOR)
    if kind == 'infected-macrophage':
        return _macrophage(ax, x, y, s, INFECTED_COLOR, INFECTED_DARK_COLOR, spots=5)
    if kind == 'virus':
        return _virus(ax, x, y, s)
    return _killer(ax, x, y, s)


def _pill(ax, x, y, w, color):

    # a capsule lying on its side, half white
    h = 0.42 * w
    body = ax.add_patch(patches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                               boxstyle=f'round,pad=0,rounding_size={h / 2}',
                                               facecolor='white', edgecolor=LENS_COLOR, linewidth=0.8, zorder=2))
    half = ax.add_patch(patches.FancyBboxPatch((x - w / 2, y - h / 2), w / 2 + h / 4, h,
                                               boxstyle=f'round,pad=0,rounding_size={h / 2}',
                                               facecolor=color, edgecolor='none', zorder=2.1))
    return body, half


class HIVVisualizer(BaseViz):
    '''Draws a sample of each patient's blood under a microscope, with healthy T cells
    (blue), macrophages (orange), infected cells of both kinds (purple, with virus inside),
    free virus (red) and killer T cells (green), and under it the two drugs: the RTI (blue
    capsule) and the PI (yellow capsule), coloured while they are taken and grey otherwise.

    The populations span many orders of magnitude, so the number of pictures of each grows
    with its logarithm: every factor of ten adds a few. The drawing only depends on the
    current state.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._patients = list(model.type_to_objects['patient'])

        count = max(len(self._patients), 1)
        self._columns = min(count, MAX_COLUMNS)
        rows = math.ceil(count / self._columns)
        cell_w, cell_h = VIEW + GAP, VIEW + DRUGS + GAP
        width = max(self._columns * cell_w + GAP, 6.8)
        height = rows * cell_h + TOP_MARGIN + LEGEND_HEIGHT
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.fig.patch.set_facecolor('white')
        self.ax = self.fig.add_axes([0, LEGEND_HEIGHT / height, 1, (height - LEGEND_HEIGHT) / height])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height - LEGEND_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # every picture that may be shown is drawn once, and only shown or hidden
        self._views = {}
        left0 = (width - self._columns * cell_w + GAP) / 2
        for (n, patient) in enumerate(self._patients):
            (row, col) = divmod(n, self._columns)
            cx = left0 + col * cell_w + VIEW / 2
            cy = height - LEGEND_HEIGHT - TOP_MARGIN - row * cell_h - VIEW / 2
            self._views[patient] = self._draw_view(patient, cx, cy, n)
        self._draw_legend()

    def _draw_view(self, patient, cx, cy, seed):
        ax = self.ax
        lens = plt.Circle((cx, cy), VIEW / 2, facecolor=PLASMA_COLOR, edgecolor=LENS_COLOR, linewidth=3, zorder=1)
        ax.add_patch(lens)

        # places for the pictures, scattered over the field without much overlap, the
        # largest first
        rng = np.random.default_rng(seed)
        placed, pictures = [], {}
        order = sorted(POPULATIONS, key=lambda item: -PICTURE_SIZE[item[1]])
        for (fluent, kind, _, _, most) in order:
            size = PICTURE_SIZE[kind]
            pictures[fluent] = []
            for _ in range(most):
                for attempt in range(200):
                    r = (VIEW / 2 - 0.6 * size - 0.05) * math.sqrt(rng.uniform())
                    angle = rng.uniform(0, 2 * math.pi)
                    (x, y) = (cx + r * math.cos(angle), cy + r * math.sin(angle))
                    if all(math.dist((x, y), q) > 0.55 * (size + qs) for (q, qs) in placed) or attempt == 199:
                        break
                placed.append(((x, y), size))
                artists = _picture(ax, kind, x, y)
                for artist in artists:
                    artist.set_clip_path(lens)
                    artist.set_visible(False)
                pictures[fluent].append(artists)

        # the drugs, and the name of the patient when there are several
        y = cy - VIEW / 2 - DRUGS / 2
        rti = _pill(ax, cx - 0.35, y, 0.5, RTI_COLOR)
        pi = _pill(ax, cx + 0.35, y, 0.5, PI_COLOR)
        if len(self._patients) > 1:
            ax.text(cx, cy + VIEW / 2 + 0.06, patient, ha='center', va='bottom', fontsize=9, color=TEXT_COLOR)
        return pictures, rti, pi

    def _draw_legend(self):
        (width, height) = self._figure_size
        ax = self.fig.add_axes([0, 0.02 / height, 1, (LEGEND_HEIGHT - 0.04) / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, LEGEND_HEIGHT - 0.04)
        ax.set_aspect('equal')
        ax.axis('off')
        rows = [[('t-cell', 'T cell'), ('macrophage', 'Macrophage'), ('virus', 'Virus'), ('killer', 'Killer T cell')],
                [('infected-t-cell', 'Infected T cell'), ('infected-macrophage', 'Infected macrophage'),
                 ('rti', 'RTI taken'), ('pi', 'PI taken')]]
        for (r, entries) in enumerate(rows):
            widths = [0.4 + 0.066 * len(label) + 0.3 for (_, label) in entries]
            x = (width - sum(widths)) / 2
            y = LEGEND_HEIGHT - 0.04 - 0.24 - 0.42 * r
            for ((kind, label), entry_width) in zip(entries, widths):
                if kind in ('rti', 'pi'):
                    _pill(ax, x + 0.15, y, 0.3, RTI_COLOR if kind == 'rti' else PI_COLOR)
                else:
                    _picture(ax, kind, x + 0.15, y, min(PICTURE_SIZE[kind], 0.24))
                ax.text(x + 0.4, y, label, ha='left', va='center', fontsize=8.5, color=TEXT_COLOR)
                x += entry_width

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        for (patient, (pictures, rti, pi)) in self._views.items():
            for (fluent, _, floor, per_decade, most) in POPULATIONS:
                value = float(state[RDDLPlanningModel.ground_var(fluent, [patient])])
                shown = 0
                if value > 0:
                    shown = int(min(max(round(per_decade * (math.log10(value) - floor)), 0), most))
                for (k, artists) in enumerate(pictures[fluent]):
                    for artist in artists:
                        artist.set_visible(k < shown)
            for (pill, taken) in ((rti, 'rti-on'), (pi, 'pi-on')):
                on = bool(state[RDDLPlanningModel.ground_var(taken, [patient])])
                pill[1].set_facecolor((RTI_COLOR if pill is rti else PI_COLOR) if on else OFF_COLOR)
                pill[0].set_alpha(1.0 if on else 0.6)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)
