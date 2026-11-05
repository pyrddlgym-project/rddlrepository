import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class TowerOfHanoiVisualizer(BaseViz):
    '''Draws each rod with its disks, wider and lighter for larger disks, and the target
    rods in green. Only the highlight on the most recently moved disk depends on the
    previous render.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 rod_width=2.4,
                 disk_height=0.4,
                 max_size=(14, 8),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._rods = model.type_to_objects['rod']
        self._disks = model.type_to_objects['disk']
        self._targets = [r for r in self._rods if self._nonfluents[
            RDDLPlanningModel.ground_var('TARGET', [r])]]
        self._sizes = {d: self._nonfluents[RDDLPlanningModel.ground_var('SIZE', [d])]
                       for d in self._disks}

        self._last_rods = None
        self._newest = None

        # the height of the figure has room for all the disks on one rod, plus 0.6
        # inches below for the base and rod names and 0.5 inches above for the title:
        # many rods or disks use smaller rods and disks to fit on screen
        num_disks = len(self._disks)
        rod_width = min(rod_width, max_size[0] / len(self._rods))
        disk_height = min(disk_height, max_size[1] / (num_disks + 1))
        self._font_size = min(10, 40 * disk_height)
        self._figure_size = (len(self._rods) * rod_width,
                             (num_disks + 1) * disk_height + 1.1)
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0, 0, 1, 1 - 0.5 / self._figure_size[1]])
        self.ax.set_xlim(0, len(self._rods))
        self.ax.set_ylim(-0.6 / disk_height, num_disks + 1)
        self.ax.axis('off')

        # the rods and base never change, so they are only drawn once
        for (i, r) in enumerate(self._rods):
            target = r in self._targets
            color = '#2e7d32' if target else '#37474f'
            self.ax.add_patch(patches.Rectangle(
                (i + 0.49, 0), 0.02, num_disks + 0.7, facecolor=color, zorder=1))
            self.ax.add_patch(patches.Rectangle(
                (i + 0.03, -0.12 / disk_height), 0.94, 0.12 / disk_height,
                facecolor=color, zorder=1))
            self.ax.text(i + 0.5, -0.32 / disk_height, f'{r} (target)' if target else r, ha='center',
                         va='center', fontsize=10, color=color if target else 'black',
                         fontweight='bold' if target else 'normal')

    def _get_rods(self, state):
        rods = {r: [] for r in self._rods}
        for d in self._disks:
            for r in self._rods:
                if state[RDDLPlanningModel.ground_var('disk-on-rod', [d, r])]:
                    rods[r].append(d)

        # the disk with the lowest order is at the bottom of the rod
        for disks in rods.values():
            disks.sort(key=lambda d: state[RDDLPlanningModel.ground_var('disk-order', [d])])
        return rods

    def _update_newest(self, rods):

        # rendering the same state again must not change the highlighted disk
        if rods == self._last_rods:
            return
        changed = []
        if self._last_rods is not None:
            changed = [d for r in self._rods for d in rods[r]
                       if d not in self._last_rods[r]]

        # several disks only change together when a new episode starts
        self._newest = changed[0] if len(changed) == 1 else None
        self._last_rods = rods

    def _draw_disk(self, disk, rod_index, level):
        low, high = min(self._sizes.values()), max(self._sizes.values())
        scale = (self._sizes[disk] - low) / (high - low) if high > low else 1.0
        width = 0.9 * (0.35 + 0.65 * scale)
        newest = disk == self._newest
        return [
            self.ax.add_patch(patches.FancyBboxPatch(
                (rod_index + 0.5 - width / 2, level + 0.05), width, 0.9,
                boxstyle='round,pad=0,rounding_size=0.03', mutation_aspect=8,
                facecolor=plt.cm.viridis(0.7 * scale), zorder=2,
                edgecolor='#ff8f00' if newest else '#37474f',
                linewidth=3 if newest else 1)),
            self.ax.text(rod_index + 0.5, level + 0.5, disk, ha='center', va='center',
                         color='white', fontweight='bold', zorder=3,
                         fontsize=self._font_size * min(1, 6 / len(disk)))
        ]

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        rods = self._get_rods(state)
        self._update_newest(rods)

        # draw the disks on each rod from the bottom up
        artists = []
        for (i, r) in enumerate(self._rods):
            for (level, d) in enumerate(rods[r]):
                artists.extend(self._draw_disk(d, i, level))

        # same rule as the reward: every disk must be on every target rod
        solved = all(d in rods[r] for r in self._targets for d in self._disks)
        num_on_target = len({d for r in self._targets for d in rods[r]})
        title = ('Solved' if solved
                 else f'{num_on_target} / {len(self._disks)} disks on target')
        self.ax.set_title(title, fontsize=13, fontweight='bold',
                          color='#2e7d32' if solved else 'black')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
