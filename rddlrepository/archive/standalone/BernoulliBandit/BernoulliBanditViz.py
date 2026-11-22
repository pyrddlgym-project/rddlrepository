import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

CABINET_COLOR = '#c62828'
CABINET_EDGE = '#7f0000'
SCREEN_COLOR = '#fffde7'
BELIEF_COLOR = '#42a5f5'
MEAN_COLOR = '#0d47a1'
LAMP_OFF = '#9e9e9e'
LAMP_ON = '#ffd600'
COIN_COLOR = '#ffc107'
COIN_EDGE = '#ff8f00'
LOSS_COLOR = '#d32f2f'
WON_COLOR = '#ffc107'
LOST_COLOR = '#78909c'
LEFT_COLOR = '#eceff1'
TRAY_COLOR = '#37474f'

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5

# a machine fills a cell of 1 by CELL_HEIGHT
CELL_HEIGHT = 1.62
SCREEN = (0.16, 0.52, 0.70, 1.02)       # left, bottom, right, top
BAR = (0.08, -0.05, 0.76, 0.1)         # left, bottom, width, height of the bar of the machine's pulls


class BernoulliBanditVisualizer(BaseViz):
    '''Draws the arms as slot machines. The screen of each shows the belief about its chance of
    paying out, as the density of its Beta posterior over 0 to 1 with a mark at its mean, and the
    lamp on top is lit on the machine that looks best so far (highest mean). The machine pulled last
    has its lever down, and a coin in its tray if it paid out or a red cross if not. A bar at the top
    shows the pulls of the episode: won, lost and still left.

    The counts come from the state alone: only the lever and tray of the machine pulled last depend
    on the previous render.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 max_columns=10,
                 max_width=11.0,
                 num_points=200,
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._arms = model.type_to_objects['arm']
        self._horizon = max(int(model.horizon), 1)
        self._init_state = self._get_state(model.ground_vars_with_values(model.state_fluents))

        # the density is infinite at the ends when a parameter is below one
        self._grid = (np.arange(num_points) + 0.5) / num_points

        self._last_state = None
        self._newest = {}

        # the machines in a grid, larger when there are few
        num_arms = len(self._arms)
        self._cols = min(num_arms, max_columns)
        self._rows = math.ceil(num_arms / self._cols)
        cell = min(2.0, (max_width - 0.6) / self._cols)
        self._cell = cell
        head, foot, side = 1.05, 1.0, 0.3
        width = max(11.0, self._cols * cell + 2 * side)       # at least the legend's width
        grid_height = (self._rows * CELL_HEIGHT + 0.1) * cell
        height = head + grid_height + foot
        self.fig = plt.figure(figsize=(width, height))
        self.fig.patch.set_facecolor('white')

        # the machines
        grid_width = self._cols * cell
        self.ax = self.fig.add_axes([(width - grid_width) / 2 / width, foot / height,
                                     grid_width / width, grid_height / height])
        self.ax.set_xlim(-0.02, self._cols + 0.04)
        self.ax.set_ylim(-0.3, self._rows * CELL_HEIGHT - 0.2)      # the names hang below the bottom row
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self._name_size = min(10, 72 * cell * 0.13)
        self._origin = {}
        for (i, a) in enumerate(self._arms):
            x, y = i % self._cols, (self._rows - 1 - i // self._cols) * CELL_HEIGHT
            self._origin[a] = (x, y)
            self._machine(self.ax, x, y, a)

        # the pulls of the episode
        self.head = self.fig.add_axes([0.08, (height - 0.85) / height, 0.84, 0.32 / height])
        self.head.set_xlim(0, self._horizon)
        self.head.set_ylim(0, 1)
        self.head.axis('off')
        self.head.add_patch(patches.Rectangle((0, 0), self._horizon, 1, facecolor=LEFT_COLOR,
                                              edgecolor='#90a4ae', linewidth=0.8))
        self.fig.text(0.5, (height - 0.3) / height, 'Win as many pulls as possible',
                      ha='center', va='center', fontsize=13, fontweight='bold', color='#263238')
        self.fig.text(0.075, (height - 0.69) / height, 'pulls', ha='right', va='center',
                      fontsize=9, color='#455a64')

        self._legend(width, foot)

    # ===== drawing =====

    def _machine(self, ax, x, y, name=None):
        '''The static parts of a slot machine in the cell with lower left corner (x, y).'''
        ax.add_patch(patches.FancyBboxPatch(
            (x + 0.08, y + 0.08), 0.76, 1.08, boxstyle='round,pad=0,rounding_size=0.06',
            facecolor=CABINET_COLOR, edgecolor=CABINET_EDGE, linewidth=0.8, zorder=1))
        ax.add_patch(patches.Wedge((x + 0.46, y + 1.16), 0.13, 0, 180, facecolor=LAMP_OFF,
                                   edgecolor=CABINET_EDGE, linewidth=0.6, zorder=1))
        l, b, r, t = SCREEN
        ax.add_patch(patches.Rectangle((x + l, y + b), r - l, t - b, facecolor=SCREEN_COLOR,
                                       edgecolor=CABINET_EDGE, linewidth=0.8, zorder=2))
        for f in (0.0, 0.5, 1.0):
            ax.plot([x + l + f * (r - l)] * 2, [y + b, y + b + 0.04], color='#9e9e9e',
                    linewidth=0.6, zorder=3)
        ax.add_patch(patches.Rectangle((x + 0.24, y + 0.16), 0.44, 0.2, facecolor=TRAY_COLOR,
                                       edgecolor=CABINET_EDGE, linewidth=0.6, zorder=2))
        ax.add_patch(patches.Rectangle((x + 0.84, y + 0.66), 0.05, 0.12, facecolor='#757575',
                                       edgecolor='none', zorder=1))
        if name is not None:
            ax.add_patch(patches.Rectangle((x + BAR[0], y + BAR[1]), BAR[2], BAR[3], facecolor='white',
                                           edgecolor='#90a4ae', linewidth=0.6, zorder=1))
            font_size = min(self._name_size, 72 * self._cell * 0.85 / (0.62 * max(len(name), 1)))
            if font_size >= MIN_FONT_SIZE:
                ax.text(x + 0.46, y - 0.08, name, ha='center', va='top', fontsize=font_size,
                        color='#37474f')

    def _pulls(self, ax, x, y, won, lost, most):
        '''The pulls of this machine so far, the bar full for the machine pulled most.'''
        scale = BAR[2] / max(most, 1.0)
        return [ax.add_patch(patches.Rectangle((x + BAR[0], y + BAR[1]), won * scale, BAR[3],
                                               facecolor=WON_COLOR, edgecolor='none', zorder=2)),
                ax.add_patch(patches.Rectangle((x + BAR[0] + won * scale, y + BAR[1]), lost * scale, BAR[3],
                                               facecolor=LOST_COLOR, edgecolor='none', zorder=2))]

    def _lever(self, ax, x, y, down):
        tip = (x + 0.95, y + 0.4) if down else (x + 0.95, y + 1.05)
        line, = ax.plot([x + 0.89, tip[0]], [y + 0.72, tip[1]], color='#616161', linewidth=1.6,
                        solid_capstyle='round', zorder=1)
        ball = ax.add_patch(patches.Circle(tip, 0.05, facecolor=LOSS_COLOR, edgecolor=CABINET_EDGE,
                                           linewidth=0.5, zorder=2))
        return [line, ball]

    def _coin(self, ax, cx, cy, r):
        return [ax.add_patch(patches.Circle((cx, cy), r, facecolor=COIN_COLOR, edgecolor=COIN_EDGE,
                                            linewidth=0.8, zorder=4)),
                ax.add_patch(patches.Circle((cx, cy), 0.55 * r, facecolor='none', edgecolor=COIN_EDGE,
                                            linewidth=0.5, zorder=4))]

    def _cross(self, ax, cx, cy, r, width=2.0):
        outline = [patheffects.Stroke(linewidth=width + 1.6, foreground='white'), patheffects.Normal()]
        return [ax.plot([cx - r, cx + r], [cy - s * r, cy + s * r], color=LOSS_COLOR, linewidth=width,
                        solid_capstyle='round', zorder=4, path_effects=outline)[0] for s in (1, -1)]

    def _glow(self, ax, x, y, won):
        '''A halo around the machine pulled last, gold if it paid out and red if not.'''
        color = COIN_COLOR if won else LOSS_COLOR
        return [ax.add_patch(patches.FancyBboxPatch(
            (x + 0.01, y + 0.01), 0.9, 1.26, boxstyle='round,pad=0,rounding_size=0.1',
            facecolor=color, edgecolor=color, alpha=0.45, linewidth=max(1.5, 2.5 * self._cell), zorder=0))]

    def _lamp(self, ax, x, y):
        return [ax.add_patch(patches.Wedge((x + 0.46, y + 1.16), 0.13, 0, 180, facecolor=LAMP_ON,
                                           edgecolor=CABINET_EDGE, linewidth=0.6, zorder=2)),
                ax.add_patch(patches.Wedge((x + 0.46, y + 1.16), 0.22, 0, 180, facecolor=LAMP_ON,
                                           edgecolor='none', alpha=0.35, zorder=0))]

    def _belief(self, ax, x, y, alpha, beta):
        density = self._belief_density(alpha, beta)
        if density is None:
            return []
        l, b, r, t = SCREEN
        xs = x + l + self._grid * (r - l)
        artists = [ax.fill_between(xs, y + b, y + b + 0.85 * (t - b) * density, facecolor=BELIEF_COLOR,
                                   edgecolor=MEAN_COLOR, linewidth=0.5, alpha=0.85, zorder=3)]
        mean = alpha / (alpha + beta)
        mx = x + l + mean * (r - l)
        artists.append(ax.add_patch(patches.Polygon(
            [(mx, y + b + 0.07), (mx - 0.04, y + b - 0.03), (mx + 0.04, y + b - 0.03)],
            closed=True, facecolor=MEAN_COLOR, edgecolor='white', linewidth=0.4, zorder=4)))
        return artists

    def _legend(self, width, foot):
        '''Static icons along the bottom that say what the parts of a machine mean.'''
        h = foot - 0.08
        ax = self.fig.add_axes([0, 0, 1, h / self.fig.get_figheight()])
        ax.set_xlim(0, width)
        ax.set_ylim(0, h)
        ax.set_aspect('equal')
        ax.axis('off')
        s = 0.5

        def screen(x, y):
            ax.add_patch(patches.Rectangle((x, y), s, 0.22, facecolor=SCREEN_COLOR,
                                           edgecolor=CABINET_EDGE, linewidth=0.8))
            g = np.linspace(0.01, 0.99, 60)
            d = g ** 4 * (1 - g) ** 2
            ax.fill_between(x + g * s, y, y + 0.17 * d / d.max(), facecolor=BELIEF_COLOR,
                            edgecolor=MEAN_COLOR, linewidth=0.5)
            mx = x + 0.625 * s
            ax.add_patch(patches.Polygon([(mx, y + 0.05), (mx - 0.035, y - 0.02), (mx + 0.035, y - 0.02)],
                                         closed=True, facecolor=MEAN_COLOR, edgecolor='white', linewidth=0.4))
            ax.text(x, y - 0.03, '0', ha='center', va='top', fontsize=7, color='#616161')
            ax.text(x + s, y - 0.03, '1', ha='center', va='top', fontsize=7, color='#616161')

        def lamp(x, y):
            ax.add_patch(patches.Wedge((x + 0.25, y + 0.02), 0.13, 0, 180, facecolor=LAMP_ON,
                                       edgecolor=CABINET_EDGE, linewidth=0.6))

        def halo(x, y, color):
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.08, y - 0.06), 0.34, 0.32, boxstyle='round,pad=0,rounding_size=0.05',
                facecolor=color, edgecolor=color, alpha=0.45, linewidth=1.5))

        def coin(x, y):
            halo(x, y, COIN_COLOR)
            self._coin(ax, x + 0.25, y + 0.1, 0.09)

        def cross(x, y):
            halo(x, y, LOSS_COLOR)
            self._cross(ax, x + 0.25, y + 0.1, 0.08)

        def machine_bar(x, y):
            ax.add_patch(patches.Rectangle((x, y + 0.03), s, 0.12, facecolor='white', edgecolor='#90a4ae',
                                           linewidth=0.6))
            ax.add_patch(patches.Rectangle((x, y + 0.03), 0.22, 0.12, facecolor=WON_COLOR, edgecolor='none'))
            ax.add_patch(patches.Rectangle((x + 0.22, y + 0.03), 0.12, 0.12, facecolor=LOST_COLOR,
                                           edgecolor='none'))

        def bar(color):
            def draw(x, y):
                ax.add_patch(patches.Rectangle((x + 0.05, y + 0.02), 0.4, 0.16, facecolor=color,
                                               edgecolor='#90a4ae', linewidth=0.6))
            return draw

        items = [(screen, 'belief about its chance\nof paying (▲ mean)'), (lamp, 'looks best\nso far'),
                 (machine_bar, 'its pulls (full for\nthe most pulled)'), (coin, 'last pull\nwon'), (cross, 'last pull\nlost'), (bar(WON_COLOR), 'won'),
                 (bar(LOST_COLOR), 'lost'), (bar(LEFT_COLOR), 'left')]
        widths = [2.05, 1.25, 1.95, 1.05, 1.05, 0.85, 0.85, 0.8]
        x = (width - sum(widths)) / 2
        y = h / 2 - 0.05
        for (draw, label), w in zip(items, widths):
            draw(x, y)
            ax.text(x + 0.58, y + 0.09, label, ha='left', va='center', fontsize=8, color='#37474f',
                    linespacing=1.0)
            x += w

    # ===== state =====

    def _get_state(self, state):
        return {a: (float(state[RDDLPlanningModel.ground_var('alpha', [a])]),
                    float(state[RDDLPlanningModel.ground_var('beta', [a])]))
                for a in self._arms}

    def _update_newest(self, state):

        # rendering the same state again must not change the lever
        if state == self._last_state:
            return
        last_state, self._last_state = self._last_state, state

        # the counts only ever grow during an episode, so a smaller one means a new one
        if last_state is None or state == self._init_state or any(
                state[a][0] < last_state[a][0] or state[a][1] < last_state[a][1]
                for a in self._arms):
            self._newest = {}
            return
        self._newest = {a: state[a][0] > last_state[a][0] for a in self._arms
                        if state[a] != last_state[a]}

    def _belief_density(self, alpha, beta):
        if alpha <= 0 or beta <= 0:
            return None
        log_density = ((alpha - 1) * np.log(self._grid) + (beta - 1) * np.log1p(-self._grid))
        return np.exp(log_density - np.max(log_density))

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        state = self._get_state(state)
        self._update_newest(state)

        artists = []
        means = {a: al / (al + be) for a, (al, be) in state.items() if al > 0 and be > 0}
        best = max(means.values()) if means else None
        pulls = {a: (max(state[a][0] - self._init_state[a][0], 0.0), max(state[a][1] - self._init_state[a][1], 0.0))
                 for a in self._arms}
        most = max(w + l for (w, l) in pulls.values())
        for a in self._arms:
            x, y = self._origin[a]
            artists.extend(self._belief(self.ax, x, y, *state[a]))
            artists.extend(self._pulls(self.ax, x, y, *pulls[a], most))
            if best is not None and a in means and means[a] >= best - 1e-12:
                artists.extend(self._lamp(self.ax, x, y))
            artists.extend(self._lever(self.ax, x, y, a in self._newest))
            if a in self._newest:
                artists.extend(self._glow(self.ax, x, y, self._newest[a]))
                if self._newest[a]:
                    artists.extend(self._coin(self.ax, x + 0.46, y + 0.26, 0.1))
                else:
                    artists.extend(self._cross(self.ax, x + 0.46, y + 0.26, 0.08,
                                               width=max(2.0, 3.0 * self._cell)))

        # the pulls of the episode so far, in a bar of the horizon's length
        won = sum(state[a][0] - self._init_state[a][0] for a in self._arms)
        lost = sum(state[a][1] - self._init_state[a][1] for a in self._arms)
        won, lost = max(won, 0.0), max(lost, 0.0)
        artists.append(self.head.add_patch(patches.Rectangle(
            (0, 0), min(won, self._horizon), 1, facecolor=WON_COLOR, edgecolor='none')))
        artists.append(self.head.add_patch(patches.Rectangle(
            (won, 0), max(min(lost, self._horizon - won), 0), 1, facecolor=LOST_COLOR, edgecolor='none')))

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
