import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f5f7fa'
SKY = '#dcefff'
WATER_TOP = np.array([0.36, 0.71, 0.89])
WATER_BOTTOM = np.array([0.05, 0.25, 0.45])
SAND = '#e3c890'
LANE_LINE = '#ffffff'
HULL = '#8d4b2a'
CABIN = '#f4f4f4'
NET = '#5d4037'
FISH_COLORS = ['#ffb300', '#ff7043', '#c0ca33', '#ec407a', '#26c6da', '#ab47bc', '#ffee58', '#8d6e63']
PREDATOR_COLORS = ['#78909c', '#546e7a', '#90a4ae', '#607d8b']
EYE = '#1b1b1b'
TEXT = '#37474f'
DANGER = '#e53935'

# the largest scene on the screen, the narrowest lane, and the largest school (inches, icons)
MAX_SCENE = (9.0, 4.0)
MIN_LANE = 0.2
MAX_LANE = 1.6


class FisheryVisualizer(BaseViz):
    '''Draws the sea in lanes, one per stock: a school of fish whose size follows the stock, as a
    share of its carrying capacity, and sharks for the predators, as a share of the largest starting
    stock of a predator. A boat floats over each fished stock. A stock at or below the collapse
    threshold has red sand under it and a red label.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nf = model.ground_vars_with_values(model.non_fluents)
        init = model.ground_vars_with_values(model.state_fluents)
        self._threshold = float(nf['THRESHOLD'])
        self._lanes = []
        for (i, f) in enumerate(model.type_to_objects.get('fish', [])):
            full = float(nf[RDDLPlanningModel.ground_var('CAPACITY', [f])])
            fished = float(nf[RDDLPlanningModel.ground_var('MAX-CATCH', [f])]) > 0
            self._lanes.append(dict(name=f, var=RDDLPlanningModel.ground_var('fish-stock', [f]), full=full,
                                    fished=fished, predator=False, color=FISH_COLORS[i % len(FISH_COLORS)]))
        predators = model.type_to_objects.get('predator', [])
        predator_vars = [RDDLPlanningModel.ground_var('predator-stock', [z]) for z in predators]
        predator_full = max([abs(float(init[var])) for var in predator_vars], default=0.0)
        for (i, (z, var)) in enumerate(zip(predators, predator_vars)):
            self._lanes.append(dict(name=z, var=var, full=predator_full, fished=False, predator=True,
                                    color=PREDATOR_COLORS[i % len(PREDATOR_COLORS)]))
        n = max(len(self._lanes), 1)

        # lanes in inches, the sky above and the sand below
        self._lane = min(max(MAX_SCENE[0] / n, MIN_LANE), MAX_LANE)
        self._depth = MAX_SCENE[1]
        self._sky = 0.7
        self._sand = 0.25
        width = self._lane * n
        height = self._depth + self._sky + self._sand
        self._label_lanes = self._lane >= 0.45
        fig_w, fig_h = max(width, 4.0) + 0.4, height + (1.1 if self._label_lanes else 0.7)
        self.fig = plt.figure(figsize=(fig_w, fig_h))
        self.fig.patch.set_facecolor(BACKGROUND)
        bottom = (0.65 if self._label_lanes else 0.3) / fig_h
        self.ax = self.fig.add_axes([(fig_w - width) / 2 / fig_w, bottom, width / fig_w, height / fig_h])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(-self._sand, self._depth + self._sky)
        self.ax.axis('off')
        self._draw_scene()
        self._schools = [self._school_spots(i, lane) for (i, lane) in enumerate(self._lanes)]
        self._title = self.fig.text(0.5, 1 - 0.1 / fig_h, '', ha='center', va='top',
                                    fontsize=13, fontweight='bold', color=TEXT)
        self._labels = []
        for (i, lane) in enumerate(self._lanes):
            if self._label_lanes:
                x = self.ax.transData.transform(((i + 0.5) * self._lane, 0))
                fx = self.fig.transFigure.inverted().transform(x)[0]
                self.fig.text(fx, bottom - 0.08 / fig_h, lane['name'], ha='center', va='top', fontsize=9,
                              color=TEXT, clip_on=False)
                self._labels.append(self.fig.text(fx, bottom - 0.3 / fig_h, '', ha='center', va='top',
                                                  fontsize=9, fontweight='bold', color=TEXT))

    # ===== parts that never change =====

    def _draw_scene(self):
        ax, w, d = self.ax, self._lane * max(len(self._lanes), 1), self._depth
        ax.add_patch(patches.Rectangle((0, d), w, self._sky, facecolor=SKY, edgecolor='none', zorder=0))
        shade = np.linspace(0, 1, 64)[:, None, None] * (WATER_TOP - WATER_BOTTOM) + WATER_BOTTOM
        ax.imshow(shade, extent=(0, w, 0, d), origin='lower', aspect='auto', zorder=0)
        ax.add_patch(patches.Rectangle((0, -self._sand), w, self._sand, facecolor=SAND, edgecolor='none',
                                       zorder=1))
        for i in range(1, len(self._lanes)):
            ax.plot([i * self._lane] * 2, [0, d], color=LANE_LINE, alpha=0.25, linewidth=0.8, zorder=1)
        for (i, lane) in enumerate(self._lanes):
            if lane['fished']:
                self._draw_boat((i + 0.5) * self._lane)

    def _draw_boat(self, x):
        s = min(self._lane * 0.4, 0.45)
        d = self._depth
        hull = [(x - s, d + 0.12 * s), (x + s, d + 0.12 * s), (x + 0.75 * s, d - 0.18 * s), (x - 0.75 * s, d - 0.18 * s)]
        self.ax.add_patch(patches.Polygon(hull, closed=True, facecolor=HULL, edgecolor='none', zorder=3))
        self.ax.add_patch(patches.Rectangle((x - 0.45 * s, d + 0.12 * s), 0.5 * s, 0.32 * s, facecolor=CABIN,
                                            edgecolor=HULL, linewidth=0.8, zorder=3))
        self.ax.plot([x + 0.55 * s, x + 0.85 * s, x + 0.85 * s], [d + 0.12 * s, d + 0.7 * s, d - 0.5 * s],
                     color=NET, linewidth=0.8, zorder=3)

    def _school_spots(self, i, lane):
        '''Fixed places for the icons of a lane, in an order that spreads out any first few of them.'''
        size = self._icon_size(lane)
        length, height = (4.4 * size, 2.3 * size) if lane['predator'] else (2.7 * size, 1.0 * size)
        cols = max(int(self._lane / (1.2 * length)), 1)
        rows = max(int((self._depth - 0.3) / (1.8 * height)), 1)
        cell_w, cell_h = self._lane / cols, (self._depth - 0.3) / rows
        rng = np.random.default_rng(i)
        spots = []
        for r in range(rows):
            for c in range(cols):   # jitter that keeps the icon inside its cell
                x = i * self._lane + (c + 0.5) * cell_w + max(cell_w - length, 0) / 2 * rng.uniform(-1, 1)
                y = 0.15 + (r + 0.5) * cell_h + max(cell_h - height, 0) / 2 * rng.uniform(-1, 1)
                spots.append((x, y, 1 if rng.random() < 0.5 else -1))
        return [spots[k] for k in rng.permutation(len(spots))]

    def _icon_size(self, lane):
        '''Half the height of an icon (inches).'''
        return min(self._lane / (5.5 if lane['predator'] else 3.5), 0.1 if lane['predator'] else 0.07)

    # ===== parts that follow the state =====

    def _icons(self, lane, spots, count):
        s = self._icon_size(lane)
        bodies, fins, eyes = [], [], []
        for (x, y, facing) in spots[:count]:
            if lane['predator']:   # a shark: long body, a dorsal fin and a tail
                bodies.append(patches.Ellipse((x, y), 3.2 * s, 1.0 * s))
                fins.append(patches.Polygon([(x - 0.3 * facing * s, y + 0.4 * s), (x + 0.3 * facing * s, y + 0.4 * s),
                                             (x - 0.4 * facing * s, y + 1.1 * s)], closed=True))
                tail = x - 1.5 * facing * s
                fins.append(patches.Polygon([(tail, y), (tail - 0.7 * facing * s, y + 0.7 * s),
                                             (tail - 0.5 * facing * s, y - 0.5 * s)], closed=True))
                eyes.append(patches.Circle((x + 1.05 * facing * s, y + 0.12 * s), 0.11 * s))
            else:   # a fish: round body and a tail
                bodies.append(patches.Ellipse((x, y), 2.0 * s, 1.0 * s))
                tail = x - 0.9 * facing * s
                fins.append(patches.Polygon([(tail, y), (tail - 0.6 * facing * s, y + 0.45 * s),
                                             (tail - 0.6 * facing * s, y - 0.45 * s)], closed=True))
                eyes.append(patches.Circle((x + 0.55 * facing * s, y + 0.1 * s), 0.12 * s))
        return [PatchCollection(fins, facecolor=lane['color'], edgecolor='none', zorder=4),
                PatchCollection(bodies, facecolor=lane['color'], edgecolor='none', zorder=5),
                PatchCollection(eyes, facecolor=EYE, edgecolor='none', zorder=6)]

    def _draw_state(self, state):
        artists = []
        for (i, (lane, spots)) in enumerate(zip(self._lanes, self._schools)):
            stock = float(state[lane['var']])
            share = stock / lane['full'] if lane['full'] > 0 else (1.0 if stock > 0 else 0.0)

            # a full lane holds 1.5 times the carrying capacity (or largest starting stock of a predator)
            count = int(round(len(spots) * min(max(share / 1.5, 0.0), 1.0)))
            if stock > self._threshold:
                count = max(count, 1)
            for collection in self._icons(lane, spots, count):
                artists.append(self.ax.add_collection(collection))
            if stock <= self._threshold:   # red sand under a collapsed stock, for lanes too narrow for labels
                artists.append(self.ax.add_patch(patches.Rectangle(
                    (i * self._lane, -self._sand), self._lane, self._sand, facecolor=DANGER, edgecolor='none',
                    alpha=0.8, zorder=2)))
            if self._label_lanes:
                label = self._labels[i]
                label.set_text(f'{stock:.3f}')
                label.set_color(DANGER if stock <= self._threshold else TEXT)
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        artists = self._draw_state(state)
        year = int(state['year'])
        if bool(state['collapsed']):
            self._title.set_text(f'Fishery collapsed in year {year}')
        else:
            self._title.set_text(f'Fishery, year {year}')
        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
