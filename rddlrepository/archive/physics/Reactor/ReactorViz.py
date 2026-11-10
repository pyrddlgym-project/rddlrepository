import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

BACKGROUND = '#f5f7fa'
STEEL = '#78909c'
STEEL_DARK = '#455a64'
JACKET = '#b3e5fc'
PIPE = '#90a4ae'
GAUGE_BACK = '#eceff1'
CONC_COLOR = '#6a1b9a'
SETPOINT_COLOR = '#2e7d32'
MERCURY = '#c62828'
TEXT = '#37474f'


class ReactorVisualizer(BaseViz):
    '''Draws the stirred tank in its cooling jacket, with feed and outlet pipes and the
    stirrer. The color of the contents shows their temperature, from blue (cold) to red
    (hot). A gauge shows the concentration of A with the current setpoint, and a
    thermometer shows the temperature.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(7.5, 5.0),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nf = model.ground_vars_with_values(model.non_fluents)
        self._feed_conc = max(float(nf['FEED-CONC']), 1e-6)
        self._time_step = float(nf['TIME-STEP'])
        self._sp_start = float(nf['SETPOINT-START'])
        self._sp_end = float(nf['SETPOINT-END'])
        self._sp_switch = int(nf['SETPOINT-SWITCH'])

        # temperatures from just below the coolant to a little above the feed
        low = min(float(nf['COOLANT-MIN']), float(nf['FEED-TEMP'])) - 10.0
        high = max(float(nf['COOLANT-MAX']), float(nf['FEED-TEMP'])) + 30.0
        self._temp_range = (low, high)
        self._cmap = plt.get_cmap('coolwarm')

        self.fig = plt.figure(figsize=figure_size)
        self.fig.patch.set_facecolor(BACKGROUND)
        self.ax = self.fig.add_axes([0.0, 0.08, 1.0, 0.84])
        self.ax.set_xlim(0, 15)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self._draw_plant()
        self._info = self.fig.text(0.5, 0.025, '', ha='center', va='bottom', fontsize=10, color=TEXT)

    # ===== parts that never change =====

    def _draw_plant(self):
        ax = self.ax

        # feed pipe into the top left, outlet pipe out of the bottom right
        ax.add_patch(patches.Rectangle((0.6, 8.3), 3.6, 0.45, facecolor=PIPE, edgecolor=STEEL_DARK, zorder=1))
        ax.add_patch(patches.Rectangle((3.75, 7.4), 0.45, 1.35, facecolor=PIPE, edgecolor=STEEL_DARK, zorder=1))
        ax.annotate('', xy=(2.9, 8.52), xytext=(1.2, 8.52),
                    arrowprops=dict(arrowstyle='->', color=STEEL_DARK, lw=1.5), zorder=2)
        ax.text(0.6, 8.95, 'feed of A', fontsize=9, color=TEXT)
        ax.add_patch(patches.Rectangle((8.8, 1.6), 2.0, 0.45, facecolor=PIPE, edgecolor=STEEL_DARK, zorder=1))
        ax.annotate('', xy=(10.6, 1.82), xytext=(9.3, 1.82),
                    arrowprops=dict(arrowstyle='->', color=STEEL_DARK, lw=1.5), zorder=2)
        ax.text(9.0, 1.05, 'outlet', fontsize=9, color=TEXT)

        # the jacket around the tank
        ax.add_patch(patches.FancyBboxPatch(
            (2.6, 1.0), 6.6, 6.6, boxstyle='round,pad=0,rounding_size=1.2',
            facecolor=JACKET, edgecolor=STEEL_DARK, linewidth=2, zorder=2))
        ax.text(5.9, 0.45, 'cooling jacket', fontsize=9, color=TEXT, ha='center')

        # the stirrer
        ax.plot([5.9, 5.9], [2.6, 9.0], color=STEEL_DARK, linewidth=3, zorder=6)
        ax.add_patch(patches.Rectangle((5.5, 9.0), 0.8, 0.6, facecolor=STEEL, edgecolor=STEEL_DARK, zorder=6))
        for dx in (-1, 1):
            ax.add_patch(patches.Ellipse((5.9 + dx * 0.65, 2.6), 1.2, 0.35, facecolor=STEEL,
                                         edgecolor=STEEL_DARK, zorder=6))

        # the concentration gauge and the thermometer
        ax.add_patch(patches.Rectangle((11.2, 1.5), 0.8, 6.5, facecolor=GAUGE_BACK, edgecolor=STEEL_DARK, zorder=1))
        for frac in np.linspace(0, 1, 5):
            y = 1.5 + 6.5 * frac
            ax.plot([12.0, 12.15], [y, y], color=STEEL_DARK, linewidth=1)
            ax.text(12.2, y, f'{frac * self._feed_conc:.2g}', fontsize=7, va='center', color=TEXT)
        ax.text(11.6, 8.3, 'A\n(mol/L)', fontsize=8, ha='center', color=TEXT)
        ax.add_patch(patches.Rectangle((13.45, 1.5), 0.5, 6.5, facecolor=GAUGE_BACK, edgecolor=STEEL_DARK, zorder=1))
        ax.add_patch(plt.Circle((13.7, 1.3), 0.45, facecolor=MERCURY, edgecolor=STEEL_DARK, zorder=2))
        low, high = self._temp_range
        for frac in np.linspace(0, 1, 5):
            y = 1.5 + 6.5 * frac
            ax.plot([13.95, 14.1], [y, y], color=STEEL_DARK, linewidth=1)
            ax.text(14.15, y, f'{low + frac * (high - low):.0f}', fontsize=7, va='center', color=TEXT)
        ax.text(13.7, 8.3, 'T\n(K)', fontsize=8, ha='center', color=TEXT)

    # ===== parts that follow the state =====

    def _setpoint(self, step):
        return self._sp_start if step < self._sp_switch else self._sp_end

    def _temp_frac(self, temp):
        low, high = self._temp_range
        return float(np.clip((temp - low) / (high - low), 0.0, 1.0))

    def _draw_state(self, conc, temp, step):
        ax = self.ax
        artists = []

        # contents of the tank, colored by temperature
        artists.append(ax.add_patch(patches.FancyBboxPatch(
            (3.1, 1.5), 5.6, 5.4, boxstyle='round,pad=0,rounding_size=0.9',
            facecolor=self._cmap(self._temp_frac(temp)), edgecolor=STEEL_DARK, linewidth=1.5, zorder=3)))

        # concentration and setpoint
        frac = float(np.clip(conc / self._feed_conc, 0.0, 1.0))
        artists.append(ax.add_patch(patches.Rectangle(
            (11.2, 1.5), 0.8, 6.5 * frac, facecolor=CONC_COLOR, edgecolor='none', alpha=0.8, zorder=2)))
        sp_y = 1.5 + 6.5 * float(np.clip(self._setpoint(step) / self._feed_conc, 0.0, 1.0))
        artists.extend(ax.plot([11.05, 12.15], [sp_y, sp_y], color=SETPOINT_COLOR, linewidth=2.5, zorder=4))
        artists.append(ax.text(11.0, sp_y, 'setpoint', fontsize=7, color=SETPOINT_COLOR,
                               ha='right', va='center', zorder=4))

        # thermometer
        artists.append(ax.add_patch(patches.Rectangle(
            (13.55, 1.3), 0.3, 0.2 + 6.5 * self._temp_frac(temp), facecolor=MERCURY, edgecolor='none', zorder=3)))
        return artists

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        conc, temp, step = float(state['conc']), float(state['temp']), int(state['step'])
        artists = self._draw_state(conc, temp, step)
        setpoint = self._setpoint(step)
        self.ax.set_title(f'Stirred-tank reactor, t = {step * self._time_step:g} min',
                          fontsize=13, fontweight='bold', color=TEXT)
        self._info.set_text(f'A: {conc:.4f} mol/L (setpoint {setpoint:g}, error {conc - setpoint:+.4f})'
                            f'   |   temperature: {temp:.2f} K')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        return img
