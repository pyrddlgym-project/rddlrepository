import colorsys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

FLOOR_COLOR = '#eceff1'
BELT_COLOR = '#455a64'
ROLLER_COLOR = '#607d8b'
MACHINE_COLOR = '#78909c'
MACHINE_EDGE_COLOR = '#37474f'
HEAD_COLOR = '#ffb300'
SPAN_COLOR = '#d32f2f'
LATEST_COLOR = '#212121'
TEXT_COLOR = '#263238'
MUTED_COLOR = '#607d8b'

ROW_HEIGHT = 0.36               # inches per machine
TOP = 0.55                      # inches above the rows, for the header
BOTTOM = 0.4                    # inches under the rows, for the time axis
LEFT = 1.0                      # inches of the machines at the left
WIDTH = 10.0                    # inches of the figure
BELT = 0.62                     # height of a belt, as a fraction of a row
PIECE = 0.5                     # height of a workpiece, as a fraction of a row


def _job_color(k):
    '''Distinct colours around the hue circle, alternating in lightness.'''
    hue = (k * 0.618034) % 1.0
    light = 0.5 if k % 2 == 0 else 0.62
    return colorsys.hls_to_rgb(hue, light, 0.65)


class JobShopVisualizer(BaseViz):
    '''Draws the shop floor as a schedule: a belt for each machine, with the machine at its
    left, carrying the operations dispatched to it as workpieces coloured by job and placed at
    their times, the operation dispatched last outlined, and the makespan so far.

    The start times of the operations are not part of the state, so each operation is placed
    when render sees its job's count of dispatched operations grow, from the job's new ready
    time less its duration: the schedule shows in full when render is called on every step.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._jobs = list(model.type_to_objects['job'])
        self._machines = list(model.type_to_objects['machine'])
        stages = list(model.type_to_objects['stage'])

        # the route of each job: its machine and duration at each position
        nf = model.ground_vars_with_values(model.non_fluents)
        value = lambda name, *args: nf[RDDLPlanningModel.ground_var(name, list(args))]
        order = sorted(stages, key=lambda s: int(value('ORDER', s)))
        self._routes = {}
        for j in self._jobs:
            route = []
            for s in order:
                on = [m for m in self._machines if value('ON', j, s, m)]
                route.append((on[0] if on else None, int(value('DURATION', j, s))))
            self._routes[j] = route
        self._operations = sum(len(r) for r in self._routes.values())

        # a lower bound on the makespan sets the first scale of the time axis
        work = {m: 0 for m in self._machines}
        for route in self._routes.values():
            for (m, d) in route:
                if m is not None:
                    work[m] += d
        self._bound = max([sum(d for _, d in r) for r in self._routes.values()]
                          + list(work.values()) + [1])
        self._colors = {j: _job_color(k) for (k, j) in enumerate(self._jobs)}

        rows = max(len(self._machines), 1)
        height = TOP + rows * ROW_HEIGHT + BOTTOM
        self.fig = plt.figure(figsize=(WIDTH, height))
        self.fig.patch.set_facecolor(FLOOR_COLOR)
        box = [LEFT / WIDTH, BOTTOM / height, 1 - (LEFT + 0.3) / WIDTH,
               rows * ROW_HEIGHT / height]
        self.ax = self.fig.add_axes(box)
        self.ax.set_ylim(rows - 0.5, -0.5)
        self.ax.set_facecolor(FLOOR_COLOR)
        for side in ('top', 'right', 'left'):
            self.ax.spines[side].set_visible(False)
        self.ax.spines['bottom'].set_color(MUTED_COLOR)
        self.ax.tick_params(axis='x', colors=MUTED_COLOR, labelsize=7)
        self.ax.set_yticks([])
        self.side = self.fig.add_axes([0, box[1], LEFT / WIDTH, box[3]])
        self.side.set_xlim(0, LEFT)
        self.side.set_ylim(rows - 0.5, -0.5)
        self.side.axis('off')

        self._row = {m: k for (k, m) in enumerate(self._machines)}
        for (m, k) in self._row.items():
            self._draw_machine(m, k)
        self._belts = [self._draw_belt(k) for k in range(rows)]
        self._span = self.ax.axvline(0, color=SPAN_COLOR, linewidth=1.4, linestyle='--', zorder=5)
        self._span_label = self.ax.text(0, -0.5, '', ha='center', va='bottom', fontsize=7.5,
                                        color=SPAN_COLOR, fontweight='bold', clip_on=False)
        self._header = self.fig.text(0.012, 1 - 0.32 / height, '', ha='left', va='center',
                                     fontsize=9.5, color=TEXT_COLOR, fontweight='bold')
        self.fig.text(1 - 0.3 / WIDTH, BOTTOM / height * 0.25, 'time', ha='right', va='center',
                      fontsize=7, color=MUTED_COLOR)
        self._pieces, self._done, self._latest = [], None, None
        self._set_scale(self._bound)

    # ---------- drawing

    def _draw_machine(self, m, k):
        '''A machine tool: a base, a column and a head with its spindle over the belt.'''
        add = self.side.add_patch
        style = {'facecolor': MACHINE_COLOR, 'edgecolor': MACHINE_EDGE_COLOR, 'linewidth': 0.7}
        (x, y) = (LEFT - 0.42, k)
        add(patches.Rectangle((x - 0.2, y + 0.22), 0.4, 0.12, **style))
        add(patches.Rectangle((x - 0.17, y - 0.34), 0.09, 0.58, **style))
        add(patches.FancyBboxPatch((x - 0.17, y - 0.38), 0.34, 0.16,
                                   boxstyle='round,pad=0,rounding_size=0.04',
                                   facecolor=HEAD_COLOR, edgecolor=MACHINE_EDGE_COLOR,
                                   linewidth=0.7))
        add(patches.Rectangle((x + 0.08, y - 0.22), 0.04, 0.18,
                              facecolor=MACHINE_EDGE_COLOR, edgecolor='none'))
        self.side.text(0.12, y, m.upper(), ha='left', va='center', fontsize=7.5,
                       color=TEXT_COLOR, fontweight='bold')

    def _draw_belt(self, k):
        belt = self.ax.add_patch(patches.Rectangle((0, k - BELT / 2), 1, BELT,
                                                   facecolor=BELT_COLOR, edgecolor='none',
                                                   zorder=1))
        rollers = self.ax.plot([], [], color=ROLLER_COLOR, linewidth=0.6, zorder=2)[0]
        return (belt, rollers, k)

    def _set_scale(self, end):
        '''Time axis from 0 to a little past end, with the belts and their rollers along it.'''
        end = max(end, 1) * 1.04
        self.ax.set_xlim(0, end)
        step = end / 120
        for (belt, rollers, k) in self._belts:
            belt.set_width(end)
            xs = np.arange(0, end, step)
            top, low = k - BELT / 2, k + BELT / 2
            rollers.set_data(np.repeat(xs, 3), np.tile([top, low, np.nan], len(xs)))
        self._scale = end

    # ---------- rendering

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        value = lambda name, *args: state[RDDLPlanningModel.ground_var(name, list(args))]
        done = {j: int(value('done', j)) for j in self._jobs}
        ready = {j: int(value('job-ready', j)) for j in self._jobs}

        # a new episode, or a state not following the last one, clears the floor
        if self._done is None or any(done[j] < self._done[j] for j in self._jobs):
            for piece in self._pieces:
                piece.remove()
            self._pieces, self._latest = [], None
            self._done = {j: 0 for j in self._jobs}

        # place the operations dispatched since the last render
        for j in self._jobs:
            if done[j] == self._done[j] + 1 and done[j] <= len(self._routes[j]):
                (m, d) = self._routes[j][done[j] - 1]
                if m is not None:
                    self._latest = self._place(j, m, ready[j] - d, d, done[j])
        self._done = done

        span = max(ready.values(), default=0)
        if span > self._scale / 1.04 or self._scale > 1.04 * max(self._bound, span) + 1e-9:
            self._set_scale(max(self._bound, span))
        for piece in self._pieces:
            if isinstance(piece, patches.Rectangle):
                piece.set_edgecolor('white')
                piece.set_linewidth(0.6)
        if self._latest is not None:
            self._latest.set_edgecolor(LATEST_COLOR)
            self._latest.set_linewidth(1.8)
        self._span.set_xdata([span, span])
        self._span_label.set_position((span, -0.55))
        self._span_label.set_text(f'makespan {span}')
        count = sum(done.values())
        self._header.set_text(f'Operations dispatched {count} of {self._operations}'
                              f'      Makespan so far {span}')
        self.fig.canvas.draw()
        return self.convert2img(self.fig)

    def _place(self, j, m, start, d, position):
        k = self._row[m]
        piece = patches.Rectangle((start, k - PIECE / 2), d, PIECE, facecolor=self._colors[j],
                                  edgecolor='white', linewidth=0.6, zorder=3)
        self.ax.add_patch(piece)
        self._pieces.append(piece)

        # the job on the piece when there is room for it
        pixels = d / self._scale * (WIDTH - LEFT - 0.3) * self.fig.dpi
        if pixels > 7 * (len(j) + 1):
            label = self.ax.text(start + d / 2, k, j.upper(), ha='center', va='center',
                                 fontsize=6, color='white', fontweight='bold', zorder=4,
                                 clip_on=True)
            self._pieces.append(label)
        return piece
