import math

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#e3f2fd'
WATER = '#64b5f6'
WAVE = '#90caf9'
CONCRETE = '#9e9e9e'
CHORD = '#455a64'
DECK = '#37474f'
NEW = '#29b6f6'
FAILED = '#c62828'
TEXT = '#263238'
WEAR = LinearSegmentedColormap.from_list('wear', ['#43a047', '#fdd835', '#ef6c00'])

# the largest picture (inches), the largest panel of a span (inches) and the most members in a span
MAX_PICTURE = (9.0, 6.5)
MAX_PANEL = 0.6
PER_SPAN = 10
SPAN_H = 3.4        # height of a span's cell, in panel widths
TRUSS_H = 1.3       # height of the truss
DECK_Y = 1.2        # height of the deck above the water
MIN_FONT = 4.5


class StructuralMaintenanceVisualizer(BaseViz):
    '''Draws the structure as a truss bridge over water whose diagonal members are the components, in spans of up
    to ten. A member is colored by its crack, from green for none to orange near the critical size; a failed member
    is drawn broken in red, and a new one, whose crack is not drawn yet, in blue. The title says whether the
    structure stands.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nf = model.ground_vars_with_values(model.non_fluents)
        self._crit = float(nf['CRACK-CRIT'])
        self._min_working = int(nf['MIN-WORKING'])
        self._comps = [dict(name=c, crack=RDDLPlanningModel.ground_var('crack', [c]),
                            new=RDDLPlanningModel.ground_var('new', [c]))
                       for c in model.type_to_objects['component']]
        n = max(len(self._comps), 1)

        self._spans = math.ceil(n / PER_SPAN)
        self._per = math.ceil(n / self._spans)
        self._cell_w = self._per + 2.4
        self._cols = min(self._spans, max(1, math.ceil(math.sqrt(self._spans * SPAN_H / self._cell_w * 1.4))))
        self._rows = math.ceil(self._spans / self._cols)
        width, height = self._cols * self._cell_w, self._rows * SPAN_H
        unit = min(MAX_PANEL, MAX_PICTURE[0] / width, MAX_PICTURE[1] / height)
        self._unit = unit

        fig_w, fig_h = max(width * unit, 4.0) + 0.4, height * unit + 0.8
        self.fig = plt.figure(figsize=(fig_w, fig_h))
        self.fig.patch.set_facecolor(BACKGROUND)
        self.ax = self.fig.add_axes([(fig_w - width * unit) / 2 / fig_w, 0.3 / fig_h,
                                     width * unit / fig_w, height * unit / fig_h])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis('off')
        self._lw = max(0.09 * unit * 72, 0.8)
        self._members = []
        for s in range(self._spans):
            self._draw_span(s)
        self._title = self.fig.text(0.5, 1 - 0.08 / fig_h, '', ha='center', va='top',
                                    fontsize=13, fontweight='bold', color=TEXT)

    def _origin(self, s):
        c, r = s % self._cols, s // self._cols
        return c * self._cell_w, (self._rows - 1 - r) * SPAN_H

    # ===== parts that never change =====

    def _draw_span(self, s):
        ax = self.ax
        x0, y0 = self._origin(s)
        members = range(s * self._per, min((s + 1) * self._per, len(self._comps)))
        k = len(members)
        left, right = x0 + 1.2, x0 + 1.2 + k
        deck, top = y0 + DECK_Y, y0 + DECK_Y + TRUSS_H

        # water, piers, deck and top chord
        ax.add_patch(patches.Rectangle((x0, y0), self._cell_w, 0.55, facecolor=WATER, edgecolor='none', zorder=0))
        waves = [[(x, y0 + 0.3), (x + 0.3, y0 + 0.3)] for x in np.arange(x0 + 0.2, x0 + self._cell_w - 0.3, 0.9)]
        ax.add_collection(LineCollection(waves, colors=WAVE, linewidths=self._lw * 0.5, zorder=1))
        piers = [patches.Rectangle((x - 0.25, y0 + 0.15), 0.5, DECK_Y - 0.15) for x in (left - 0.45, right + 0.45)]
        ax.add_collection(PatchCollection(piers, facecolor=CONCRETE, edgecolor='none', zorder=1))
        ax.add_patch(patches.Rectangle((left - 0.75, deck - 0.12), k + 1.5, 0.14, facecolor=DECK, edgecolor='none',
                                       zorder=3))
        chords = [[(left, top), (right, top)]]
        chords += [[(x, deck), (x, top)] for x in np.arange(left, right + 0.5, 1.0)]       # verticals
        ax.add_collection(LineCollection(chords, colors=CHORD, linewidths=self._lw * 0.7, zorder=2))

        # the components: diagonals of the panels, zig-zag
        longest = max([len(self._comps[i]['name']) for i in members] + [1])
        size = min(7.0, self._unit * 72 * 0.3, 0.9 * self._unit * 72 / (0.62 * longest))
        for j, i in enumerate(members):
            a, b = left + j, left + j + 1
            ends = ((a, deck), (b, top)) if j % 2 == 0 else ((a, top), (b, deck))
            self._members.append(ends)
            if size >= MIN_FONT:
                ax.text(a + 0.5, deck - 0.35, self._comps[i]['name'], ha='center', va='center', fontsize=size,
                        color=TEXT, zorder=3)

    # ===== parts that follow the state =====

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        segments, colors, working = [], [], 0
        for comp, ((xa, ya), (xb, yb)) in zip(self._comps, self._members):
            crack = float(state[comp['crack']])
            if bool(state[comp['new']]):
                segments.append([(xa, ya), (xb, yb)])
                colors.append(NEW)
                working += 1
            elif crack > self._crit:
                for (u, v) in ((0.0, 0.38), (0.62, 1.0)):            # broken in the middle
                    segments.append([(xa + u * (xb - xa), ya + u * (yb - ya)), (xa + v * (xb - xa), ya + v * (yb - ya))])
                    colors.append(FAILED)
            else:
                segments.append([(xa, ya), (xb, yb)])
                colors.append(WEAR(min(max(crack / self._crit, 0.0), 1.0)))
                working += 1
        artist = self.ax.add_collection(LineCollection(segments, colors=colors, linewidths=self._lw,
                                                       capstyle='round', zorder=4))
        n = len(self._comps)
        status = 'stands' if working >= self._min_working else 'has failed'
        title = f'The structure {status}: {working} of {n} components work, {self._min_working} needed'
        self._title.set_text(title)
        self._title.set_fontsize(min(13.0, 0.95 * self.fig.get_figwidth() * 72 / (0.6 * len(title))))
        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        artist.remove()
        return img
