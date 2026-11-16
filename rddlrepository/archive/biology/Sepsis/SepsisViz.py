import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

WALL_COLOR = '#eef3f6'
FLOOR_COLOR = '#d9dee2'
FRAME_COLOR = '#cfd8dc'
SKIN_COLOR = '#f6dcc4'
SKIN_EDGE_COLOR = '#c9a27e'
HAIR_COLOR = '#5d4037'
GOWN_COLOR = '#b2dfdb'
GOWN_EDGE_COLOR = '#4db6ac'
BLANKET_COLOR = '#90caf9'
BLANKET_EDGE_COLOR = '#5c9bd6'
SHEET_COLOR = '#f5f5f5'
SHEET_EDGE_COLOR = '#b0bec5'
MATTRESS_COLOR = '#fafafa'
BED_COLOR = '#90a4ae'
BED_DARK_COLOR = '#607d8b'
PILLOW_COLOR = '#ffffff'
DEVICE_COLOR = '#37474f'
CASING_COLOR = '#eceff1'
CASING_EDGE_COLOR = '#90a4ae'
SCREEN_COLOR = '#102027'
POLE_COLOR = '#78909c'
TUBE_COLOR = '#90a4ae'
VENT_TUBE_COLOR = '#b3e5fc'
VENT_TUBE_EDGE_COLOR = '#4fc3f7'
BAG_COLOR = '#f1f8e9'
BAG_FLUID_COLOR = '#fff59d'
PUMP_ON_COLOR = '#ffb300'
DOOR_COLOR = '#d7ccc8'
DOOR_EDGE_COLOR = '#a1887f'
CABINET_COLOR = '#e0e0e0'
BOARD_COLOR = '#ffffff'
MARKER_BLUE = '#1565c0'
MARKER_RED = '#c62828'
MARKER_BLACK = '#263238'
CYANOSIS_COLOR = '#3949ab'
SHIRT_COLOR = '#ffcc80'
SHIRT_EDGE_COLOR = '#ef6c00'
PANTS_COLOR = '#78909c'
SUITCASE_COLOR = '#6d4c41'
TEXT_COLOR = '#37474f'
MUTED_COLOR = '#78909c'

# the channels of the monitor, in the colors of bedside monitors
ECG_COLOR = '#00e676'
ART_COLOR = '#ff5252'
PLETH_COLOR = '#18ffff'
VENT_WAVE_COLOR = '#ffd740'
ALARM_COLOR = '#ffd600'
CRISIS_COLOR = '#ff1744'
STANDBY_COLOR = '#546e7a'

# the simulator has levels, not numbers: the monitor and the meter show a value typical of each
# level (heart rate: bradycardia under 60 and tachycardia over 90 bpm; systolic pressure:
# hypotension under 90 mmHg; oxygen saturation: low under 92%; glucose: the zones of the
# consensus on time in range, as in the Diabetes domain)
HEART_RATE = [52, 82, 124]
PRESSURE = ['84/48', '118/76', '166/98']
PRESSURE_SCALE = [0.55, 0.8, 1.0]
SATURATION = [86, 97]
GLUCOSE = [45, 62, 110, 215, 320]
GLUCOSE_ZONES = [('Very low', '#ff5252'), ('Low', '#ff8a80'), ('In range', '#69f0ae'),
                 ('High', '#ffd180'), ('Very high', '#ff9e80')]
ALARMS = {('hr', 0): 'BRADY', ('hr', 2): 'TACHY', ('sysbp', 0): 'ART LOW', ('sysbp', 2): 'ART HIGH',
          ('oxygen', 0): 'DESAT'}

# sizes in inches: the scene of each patient, the gap between scenes, and the legend
CELL = (4.6, 3.4)
GAP = 0.15
MAX_COLUMNS = 4
LEGEND_HEIGHT = 0.35

# the waveforms span WINDOW seconds of the monitor's sweep
WINDOW = 4.0
SAMPLES = 400

# the backrest of the bed, raised 30 degrees: it rises from the hinge along D, and N is its normal
HINGE = np.array([1.8, 1.15])
D = np.array([-math.cos(math.pi / 6), math.sin(math.pi / 6)])
N = np.array([math.sin(math.pi / 6), math.cos(math.pi / 6)])
BACKREST = 0.8


def _rest(s, h=0.0):
    '''A point at fraction s along the backrest, at height h above it.'''
    return tuple(HINGE + s * BACKREST * D + h * N)


def _rounded(x, y, w, h, r, **style):
    return patches.FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', **style)


def _bump(phase, centre, width):
    return np.exp(-0.5 * ((phase - centre) / width) ** 2)


def _ecg(phase):
    return (0.12 * _bump(phase, 0.15, 0.03) - 0.1 * _bump(phase, 0.27, 0.008)
            + 1.0 * _bump(phase, 0.3, 0.01) - 0.25 * _bump(phase, 0.33, 0.01)
            + 0.25 * _bump(phase, 0.55, 0.05))


def _arterial(phase):
    return 1.0 * _bump(phase, 0.2, 0.07) + 0.35 * _bump(phase, 0.45, 0.06) + 0.15 * (1 - phase)


def _pleth(phase):
    return 1.0 * _bump(phase, 0.3, 0.1) + 0.3 * _bump(phase, 0.58, 0.07)


def _breaths(t):
    phase = (t / 4.0) % 1.0
    return np.where(phase < 0.35, 1.0 - np.exp(-phase * 25), np.exp(-(phase - 0.35) * 12) * 0.95)


class SepsisVisualizer(BaseViz):
    '''Draws each patient in a bay of an intensive care unit: in a bed with its backrest raised,
    under a blanket, with a line in the arm. The bedside monitor sweeps the ECG, arterial pressure
    and pulse oximetry at the patient's heart rate, with the heart rate, blood pressure and oxygen
    saturation beside them, boxed in yellow when abnormal and named in the alarm banner. The
    glucose meter on the cabinet shows the glucose in the colors of its zone. The treatments in
    effect are the equipment in use: the bag of antibiotics on the drip stand and its line, the
    syringe pump of vasopressors on the stand and its line, and the ventilator with its tube to
    the patient's mask. The face shows how many vital signs are abnormal, with blue lips when the
    oxygen is low. The whiteboard has the bed, whether the patient is diabetic and the step of the
    stay. A discharged patient stands at the door with a suitcase, and a patient who died lies
    under a sheet with a flat trace.

    The simulator has levels, not numbers: each level is shown by a value typical of it.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._patients = list(model.type_to_objects['patient'])
        nf = model.ground_vars_with_values(model.non_fluents)
        self._max_steps = int(nf['MAX-STEPS'])
        self._renders = 0

        # a scene for each patient, in rows
        count = max(len(self._patients), 1)
        self._columns = min(count, MAX_COLUMNS)
        rows = math.ceil(count / self._columns)
        width = self._columns * (CELL[0] + GAP) + GAP
        height = rows * (CELL[1] + GAP) + GAP + LEGEND_HEIGHT
        self._figure_size = (width, height)
        self.fig = plt.figure(figsize=self._figure_size)
        self.fig.patch.set_facecolor('white')
        self.ax = self.fig.add_axes([0, LEGEND_HEIGHT / height, 1, 1 - LEGEND_HEIGHT / height])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height - LEGEND_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self._scenes = {}
        for (n, patient) in enumerate(self._patients):
            (row, col) = divmod(n, self._columns)
            x0 = GAP + col * (CELL[0] + GAP)
            y0 = height - LEGEND_HEIGHT - (row + 1) * (CELL[1] + GAP)
            self._scenes[patient] = self._draw_scene(patient, x0, y0)
        self._draw_legend()

    # ---------- drawing a scene, in its own coordinates

    def _add(self, patch):
        patch.set_transform(self._shift)
        return self.ax.add_patch(patch)

    def _line(self, xs, ys, **style):
        return self.ax.plot(xs, ys, transform=self._shift, **style)[0]

    def _text(self, x, y, text, **style):
        style = {'ha': 'center', 'va': 'center', 'color': TEXT_COLOR, 'zorder': 8, **style}
        return self.ax.text(x, y, text, transform=self._shift, **style)

    def _label(self, x, y, text, **style):
        return self._text(x, y, text, **{'fontsize': 5.5, 'color': MUTED_COLOR, 'style': 'italic', **style})

    def _draw_scene(self, patient, x0, y0):
        self._shift = Affine2D().translate(x0, y0) + self.ax.transData
        scene = {'patient': patient}
        self._draw_room()
        self._draw_bed()
        scene.update(self._draw_patient())
        scene.update(self._draw_monitor())
        scene.update(self._draw_stand())
        scene.update(self._draw_ventilator())
        scene.update(self._draw_cabinet())
        scene.update(self._draw_board(patient))
        scene.update(self._draw_discharged())
        return scene

    def _draw_room(self):
        self._add(_rounded(0, 0, *CELL, 0.12, facecolor=WALL_COLOR, edgecolor=FRAME_COLOR,
                           linewidth=0.8, zorder=0))
        self._add(patches.Rectangle((0.02, 0.02), CELL[0] - 0.04, 0.34, facecolor=FLOOR_COLOR,
                                    edgecolor='none', zorder=0.1))

        # the door, with its window and handle
        self._add(patches.Rectangle((4.05, 0.36), 0.47, 2.4, facecolor=DOOR_COLOR,
                                    edgecolor=DOOR_EDGE_COLOR, linewidth=0.9, zorder=0.5))
        self._add(patches.Rectangle((4.15, 1.9), 0.27, 0.55, facecolor='#e3f2fd',
                                    edgecolor=DOOR_EDGE_COLOR, linewidth=0.7, zorder=0.6))
        self._add(plt.Circle((4.13, 1.45), 0.03, facecolor=DOOR_EDGE_COLOR, edgecolor='none',
                             zorder=0.6))

    def _draw_bed(self):
        bed = {'facecolor': BED_COLOR, 'edgecolor': BED_DARK_COLOR, 'linewidth': 0.8}

        # wheels, legs, the base, the head and foot boards and the mattress with its raised back
        for x in (1.1, 3.05):
            self._add(patches.Rectangle((x - 0.03, 0.5), 0.06, 0.3, zorder=1.5, **bed))
            self._add(plt.Circle((x, 0.46), 0.06, facecolor=BED_DARK_COLOR, edgecolor='none',
                                 zorder=1.6))
        self._add(_rounded(0.95, 0.78, 2.4, 0.12, 0.03, zorder=1.7, **bed))
        self._add(_rounded(0.86, 0.78, 0.1, 0.92, 0.04, zorder=1.8, **bed))
        self._add(_rounded(3.27, 0.78, 0.1, 0.6, 0.04, zorder=1.8, **bed))
        mattress = [(3.27, 1.15), tuple(HINGE), _rest(1.0), _rest(1.0, -0.15),
                    (HINGE[0] - 0.05, 0.9), (3.27, 0.9)]
        self._add(patches.Polygon(mattress, closed=True, facecolor=MATTRESS_COLOR,
                                  edgecolor=FRAME_COLOR, linewidth=0.8, joinstyle='round', zorder=1.9))

        # the side rail
        self._line([2.0, 3.0], [1.22, 1.22], color=BED_DARK_COLOR, linewidth=1.6, alpha=0.7,
                   solid_capstyle='round', zorder=4.5)
        for x in (2.1, 2.9):
            self._line([x, x], [1.0, 1.22], color=BED_DARK_COLOR, linewidth=1.0, alpha=0.7, zorder=4.5)

    def _draw_patient(self):
        skin = {'facecolor': SKIN_COLOR, 'edgecolor': SKIN_EDGE_COLOR, 'linewidth': 0.8}
        parts = []

        # the pillow, the shoulders in a gown, the neck and the head with hair and face
        pillow = self._add(patches.Ellipse(_rest(0.93, 0.07), 0.46, 0.15, angle=-30,
                                           facecolor=PILLOW_COLOR, edgecolor=FRAME_COLOR, linewidth=0.7,
                                           zorder=2.0))
        gown = [_rest(0.5), _rest(0.5, 0.22), _rest(0.86, 0.22), _rest(0.86)]
        parts.append(self._add(patches.Polygon(gown, closed=True, facecolor=GOWN_COLOR,
                                               edgecolor=GOWN_EDGE_COLOR, linewidth=0.8, zorder=2.6)))
        neck = _rest(0.86, 0.13)
        parts.append(self._add(patches.Rectangle((neck[0] - 0.05, neck[1] - 0.04), 0.1, 0.12,
                                                 zorder=2.65, **skin)))
        head = np.array(_rest(0.9, 0.3))
        parts.append(self._add(plt.Circle(head, 0.15, zorder=2.7, **skin)))
        parts.append(self._add(patches.Wedge(head + (0, 0.02), 0.16, 20, 160, facecolor=HAIR_COLOR,
                                             edgecolor='none', zorder=2.75)))
        open_eyes = [self._add(plt.Circle(head + (dx, 0.0), 0.022, facecolor=TEXT_COLOR,
                                          edgecolor='none', zorder=2.8)) for dx in (-0.055, 0.055)]
        closed_eyes = [self._add(patches.Arc(head + (dx, 0.0), 0.06, 0.04, theta1=200, theta2=340,
                                             color=TEXT_COLOR, linewidth=0.9, zorder=2.8))
                       for dx in (-0.055, 0.055)]
        mouth = self._add(patches.Arc(head + (0, -0.075), 0.1, 0.06, theta1=200, theta2=340,
                                      color=TEXT_COLOR, linewidth=1.0, zorder=2.8))
        parts += open_eyes + closed_eyes + [mouth]

        # the blanket over the body, from the chest to the feet
        blanket = [(3.27, 1.15), (3.26, 1.3), (3.15, 1.39), (3.0, 1.33), (2.4, 1.32), (1.98, 1.37),
                   _rest(0.3, 0.24), _rest(0.58, 0.25), _rest(0.58), tuple(HINGE)]
        parts.append(self._add(patches.Polygon(blanket, closed=True, facecolor=BLANKET_COLOR,
                                               edgecolor=BLANKET_EDGE_COLOR, linewidth=0.9,
                                               joinstyle='round', zorder=3)))
        parts.append(self._line([_rest(0.58, 0.25)[0], _rest(0.56)[0]],
                                [_rest(0.58, 0.25)[1], _rest(0.56)[1]], color='white', linewidth=2.2,
                                solid_capstyle='round', zorder=3.05))

        # the arm on the blanket, with the line in the forearm
        arm = [_rest(0.78, 0.17), (1.62, 1.5), (2.0, 1.43)]
        parts.append(self._line(*zip(*arm), color=SKIN_EDGE_COLOR, linewidth=6.2, solid_capstyle='round',
                                solid_joinstyle='round', zorder=3.3))
        parts.append(self._line(*zip(*arm), color=SKIN_COLOR, linewidth=5.0, solid_capstyle='round',
                                solid_joinstyle='round', zorder=3.35))
        parts.append(self._add(patches.Rectangle((1.82, 1.43), 0.08, 0.05, angle=-10, facecolor='white',
                                                 edgecolor=SHEET_EDGE_COLOR, linewidth=0.5, zorder=3.4)))

        # what is drawn instead: a made bed, or a sheet over the patient
        made = self._add(patches.Polygon([(3.27, 1.15), (3.27, 1.24), (2.3, 1.24), (2.25, 1.15)],
                                         closed=True, facecolor=BLANKET_COLOR, edgecolor=BLANKET_EDGE_COLOR,
                                         linewidth=0.9, zorder=3, visible=False))
        top = _rest(0.9, 0.48)
        sheet = [(3.3, 1.12), (3.27, 1.33), (3.12, 1.4), (2.4, 1.36), (1.98, 1.4), _rest(0.3, 0.27),
                 _rest(0.75, 0.3), (top[0] + 0.12, top[1] - 0.04), top, (top[0] - 0.1, top[1] - 0.14),
                 _rest(1.0, 0.1), _rest(1.02), tuple(HINGE)]
        shroud = self._add(patches.Polygon(sheet, closed=True, facecolor=SHEET_COLOR, edgecolor=SHEET_EDGE_COLOR,
                                           linewidth=0.9, joinstyle='round', zorder=3.8, visible=False))
        return {'body': parts, 'pillow': pillow, 'open-eyes': open_eyes, 'closed-eyes': closed_eyes,
                'mouth': mouth, 'made': made, 'shroud': shroud, 'cannula': (1.86, 1.455),
                'face': tuple(head + (0, -0.06))}

    def _draw_monitor(self):

        # the monitor on its arm on the wall, with three channels and an alarm banner
        self._add(patches.Rectangle((1.0, 1.9), 0.06, 0.12, facecolor=POLE_COLOR, edgecolor='none',
                                    zorder=1))
        self._add(_rounded(0.12, 1.98, 2.0, 1.3, 0.06, facecolor=DEVICE_COLOR, edgecolor='none', zorder=1.1))
        self._add(_rounded(0.18, 2.04, 1.88, 1.18, 0.03, facecolor=SCREEN_COLOR, edgecolor='none',
                           zorder=1.2))
        banner = self._add(patches.Rectangle((0.18, 3.06), 1.88, 0.16, facecolor=SCREEN_COLOR,
                                             edgecolor='none', zorder=1.3))
        banner_text = self._text(1.12, 3.14, '', fontsize=6.2, fontweight='bold', zorder=1.4)
        channels = {}
        for (name, label, color, y) in (('ecg', 'HR', ECG_COLOR, 2.84), ('art', 'ART', ART_COLOR, 2.55),
                                        ('pleth', 'SpO2', PLETH_COLOR, 2.25)):
            trace = self._line([], [], color=color, linewidth=0.9, zorder=1.4)
            box = self._add(_rounded(1.52, y - 0.12, 0.5, 0.24, 0.02, facecolor=SCREEN_COLOR,
                                     edgecolor='none', zorder=1.3))
            self._text(1.55, y + 0.085, label, ha='left', fontsize=4.6, color=color, zorder=1.4)
            value = self._text(1.99, y - 0.02, '', ha='right', fontsize=10, fontweight='bold', color=color,
                               zorder=1.4)
            channels[name] = {'trace': trace, 'box': box, 'value': value, 'color': color, 'y': y,
                              'label': label}
        standby = self._text(1.12, 2.55, 'STANDBY', fontsize=8, color=STANDBY_COLOR, zorder=1.4, visible=False)
        return {'banner': banner, 'banner-text': banner_text, 'channels': channels, 'standby': standby}

    def _draw_stand(self):

        # the drip stand: a pole on a base, with hooks at the top
        self._line([2.5, 2.5], [0.45, 2.95], color=POLE_COLOR, linewidth=1.6, zorder=1)
        self._line([2.32, 2.68], [2.95, 2.95], color=POLE_COLOR, linewidth=1.3, zorder=1)
        self._line([2.32, 2.68], [0.45, 0.45], color=POLE_COLOR, linewidth=1.6, solid_capstyle='round', zorder=1)
        for x in (2.33, 2.67):
            self._add(plt.Circle((x, 0.41), 0.035, facecolor=BED_DARK_COLOR, edgecolor='none', zorder=1.1))

        # the bag of antibiotics, with its drip chamber, hanging from the left hook
        bag_parts = []
        (x, top, bottom) = (2.36, 2.88, 2.36)
        bag = [(x - 0.1, top), (x + 0.1, top), (x + 0.11, bottom + 0.06), (x + 0.05, bottom),
               (x - 0.05, bottom), (x - 0.11, bottom + 0.06)]
        fluid = [(x - 0.105, top - 0.22), (x + 0.105, top - 0.22), (x + 0.11, bottom + 0.06),
                 (x + 0.05, bottom), (x - 0.05, bottom), (x - 0.11, bottom + 0.06)]
        bag_parts.append(self._add(patches.Polygon(bag, closed=True, facecolor=BAG_COLOR,
                                                   edgecolor=CASING_EDGE_COLOR, linewidth=0.7, zorder=2)))
        bag_parts.append(self._add(patches.Polygon(fluid, closed=True, facecolor=BAG_FLUID_COLOR,
                                                   edgecolor='none', zorder=2.1)))
        bag_parts.append(self._text(x, top - 0.12, 'ABX', fontsize=4.6, fontweight='bold', color=MARKER_BLUE,
                                    zorder=2.2))
        bag_parts.append(self._add(patches.Rectangle((x - 0.025, bottom - 0.1), 0.05, 0.1,
                                                     facecolor='white', edgecolor=CASING_EDGE_COLOR,
                                                     linewidth=0.5, zorder=2)))
        bag_parts.append(self._line([x, x], [top, 2.95], color=POLE_COLOR, linewidth=0.8, zorder=1.9))
        self._label(2.5, 3.04, 'antibiotics')

        # the syringe pump of vasopressors, clamped to the pole
        self._add(_rounded(2.55, 1.66, 0.46, 0.3, 0.03, facecolor=CASING_COLOR, edgecolor=CASING_EDGE_COLOR,
                           linewidth=0.8, zorder=2))
        self._add(patches.Rectangle((2.6, 1.92), 0.36, 0.07, facecolor='white', edgecolor=CASING_EDGE_COLOR,
                                    linewidth=0.6, zorder=2.1))
        plunger = self._add(patches.Rectangle((2.6, 1.935), 0.2, 0.04, facecolor='#ffe0b2',
                                              edgecolor='none', zorder=2.2))
        screen = self._add(patches.Rectangle((2.6, 1.72), 0.3, 0.14, facecolor=SCREEN_COLOR, edgecolor='none',
                                             zorder=2.1))
        display = self._text(2.75, 1.79, '', fontsize=4.4, fontweight='bold', color=PUMP_ON_COLOR, zorder=2.2)
        self._label(2.78, 1.58, 'vasopressor pump')

        # the lines to the arm
        (cx, cy) = (1.86, 1.455)
        abx_line = self._add(patches.FancyArrowPatch((x, bottom - 0.1), (cx, cy), connectionstyle='arc3,rad=0.25',
                                                     arrowstyle='-', color=TUBE_COLOR, linewidth=0.9, zorder=3.6))
        vaso_line = self._add(patches.FancyArrowPatch((2.55, 1.76), (cx + 0.02, cy - 0.01),
                                                      connectionstyle='arc3,rad=-0.2', arrowstyle='-',
                                                      color='#f48fb1', linewidth=0.9, zorder=3.6))
        return {'bag': bag_parts, 'abx-line': abx_line, 'pump-screen': screen, 'pump-display': display,
                'plunger': plunger, 'vaso-line': vaso_line}

    def _draw_ventilator(self):

        # the ventilator on its cart, with a screen on top
        self._line([0.18, 0.62], [0.45, 0.45], color=POLE_COLOR, linewidth=1.6, solid_capstyle='round', zorder=1)
        for x in (0.2, 0.6):
            self._add(plt.Circle((x, 0.41), 0.035, facecolor=BED_DARK_COLOR, edgecolor='none', zorder=1.1))
        self._add(patches.Rectangle((0.36, 0.45), 0.08, 0.5, facecolor=POLE_COLOR, edgecolor='none', zorder=1))
        self._add(_rounded(0.14, 0.95, 0.52, 0.4, 0.04, facecolor=CASING_COLOR, edgecolor=CASING_EDGE_COLOR,
                           linewidth=0.8, zorder=1.2))
        for y in (1.07, 1.19):
            self._add(plt.Circle((0.6, y), 0.025, facecolor=CASING_EDGE_COLOR, edgecolor='none', zorder=1.3))
        self._add(_rounded(0.16, 1.38, 0.48, 0.4, 0.03, facecolor=DEVICE_COLOR, edgecolor='none', zorder=1.2))
        screen = self._add(patches.Rectangle((0.19, 1.41), 0.42, 0.34, facecolor=SCREEN_COLOR, edgecolor='none',
                                             zorder=1.3))
        wave = self._line([], [], color=VENT_WAVE_COLOR, linewidth=0.8, zorder=1.4)
        readout = self._text(0.4, 1.7, '', fontsize=4.2, color=VENT_WAVE_COLOR, fontweight='bold', zorder=1.4)
        self._label(0.4, 1.87, 'ventilator')

        # the tube from the ventilator to a mask over the patient's mouth and nose
        tube = []
        face = np.array(_rest(0.9, 0.3)) + (0, -0.07)
        end = tuple(face + (-0.06, -0.03))
        path = patches.FancyArrowPatch((0.62, 1.15), end, connectionstyle='arc3,rad=-0.45',
                                       arrowstyle='-', color=VENT_TUBE_EDGE_COLOR, linewidth=4.2, zorder=3.7)
        tube.append(self._add(path))
        inner = patches.FancyArrowPatch((0.62, 1.15), end, connectionstyle='arc3,rad=-0.45',
                                        arrowstyle='-', color=VENT_TUBE_COLOR, linewidth=3.0,
                                        linestyle=(0, (1.2, 0.6)), zorder=3.75)
        tube.append(self._add(inner))
        tube.append(self._add(patches.Ellipse(tuple(face), 0.15, 0.11, facecolor='white', alpha=0.65,
                                              edgecolor=VENT_TUBE_EDGE_COLOR, linewidth=0.9, zorder=3.9)))
        return {'vent-screen': screen, 'vent-wave': wave, 'vent-readout': readout, 'tube': tube}

    def _draw_cabinet(self):

        # the bedside cabinet with the glucose meter on it
        self._add(_rounded(3.47, 0.38, 0.48, 0.72, 0.03, facecolor=CABINET_COLOR, edgecolor=CASING_EDGE_COLOR,
                           linewidth=0.8, zorder=1))
        for y in (0.62, 0.86):
            self._line([3.47, 3.95], [y, y], color=CASING_EDGE_COLOR, linewidth=0.6, zorder=1.1)
            self._line([3.66, 3.76], [y + 0.12, y + 0.12], color=BED_DARK_COLOR, linewidth=1.2,
                       solid_capstyle='round', zorder=1.1)
        self._add(_rounded(3.52, 1.1, 0.38, 0.5, 0.05, facecolor=DEVICE_COLOR, edgecolor='none', zorder=1.2))
        self._add(_rounded(3.56, 1.26, 0.3, 0.29, 0.02, facecolor=SCREEN_COLOR, edgecolor='none', zorder=1.3))
        self._add(patches.Rectangle((3.67, 1.03), 0.08, 0.08, facecolor='white', edgecolor='none', zorder=1.25))
        reading = self._text(3.71, 1.44, '', fontsize=9, fontweight='bold', zorder=1.4)
        unit = self._text(3.71, 1.32, 'mg/dL', fontsize=4.2, color=MUTED_COLOR, zorder=1.4)
        zone = self._text(3.71, 1.18, '', fontsize=4.6, zorder=1.4)
        self._label(3.71, 1.68, 'glucose meter')
        return {'glucose': reading, 'glucose-unit': unit, 'glucose-zone': zone}

    def _draw_board(self, patient):

        # the whiteboard of the bay
        self._add(_rounded(2.86, 2.34, 1.08, 0.88, 0.03, facecolor=BOARD_COLOR, edgecolor=FRAME_COLOR,
                           linewidth=1.4, zorder=0.7))
        self._add(patches.Rectangle((2.95, 2.34), 0.9, 0.04, facecolor=FRAME_COLOR, edgecolor='none',
                                    zorder=0.75))
        self._text(3.4, 3.08, f'Bed {patient}', fontsize=7.5, fontweight='bold', color=MARKER_BLUE,
                   zorder=0.8)
        status = self._text(3.4, 2.84, '', fontsize=6.5, color=MARKER_BLACK, zorder=0.8)
        diabetic = self._text(3.4, 2.6, '', fontsize=6.5, fontweight='bold', color=MARKER_RED,
                              zorder=0.8)
        return {'status': status, 'diabetic': diabetic}

    def _draw_discharged(self):

        # the patient going home: standing at the door in clothes, with a suitcase
        parts = []
        skin = {'facecolor': SKIN_COLOR, 'edgecolor': SKIN_EDGE_COLOR, 'linewidth': 0.8}
        for x in (4.18, 4.33):
            parts.append(self._add(_rounded(x - 0.05, 0.4, 0.1, 0.5, 0.03, facecolor=PANTS_COLOR,
                                            edgecolor=BED_DARK_COLOR, linewidth=0.7, zorder=5)))
        parts.append(self._add(_rounded(4.1, 0.86, 0.32, 0.58, 0.08, facecolor=SHIRT_COLOR,
                                        edgecolor=SHIRT_EDGE_COLOR, linewidth=0.8, zorder=5.1)))
        parts.append(self._add(plt.Circle((4.26, 1.6), 0.14, zorder=5.2, **skin)))
        parts.append(self._add(patches.Wedge((4.26, 1.62), 0.15, 20, 160, facecolor=HAIR_COLOR, edgecolor='none',
                                             zorder=5.25)))
        for dx in (-0.05, 0.05):
            parts.append(self._add(plt.Circle((4.26 + dx, 1.6), 0.018, facecolor=TEXT_COLOR, edgecolor='none',
                                              zorder=5.3)))
        parts.append(self._add(patches.Arc((4.26, 1.53), 0.1, 0.06, theta1=200, theta2=340, color=TEXT_COLOR,
                                           linewidth=1.0, zorder=5.3)))
        parts.append(self._line([4.12, 3.98], [1.35, 0.95], color=SKIN_EDGE_COLOR, linewidth=4.4,
                                solid_capstyle='round', zorder=5.05))
        parts.append(self._line([4.12, 3.98], [1.35, 0.95], color=SKIN_COLOR, linewidth=3.4,
                                solid_capstyle='round', zorder=5.06))
        parts.append(self._line([4.4, 4.52], [1.35, 1.72], color=SKIN_EDGE_COLOR, linewidth=4.4,
                                solid_capstyle='round', zorder=5.05))
        parts.append(self._line([4.4, 4.52], [1.35, 1.72], color=SKIN_COLOR, linewidth=3.4,
                                solid_capstyle='round', zorder=5.06))
        parts.append(self._add(_rounded(3.82, 0.42, 0.3, 0.42, 0.04, facecolor=SUITCASE_COLOR,
                                        edgecolor='#3e2723', linewidth=0.8, zorder=5.2)))
        parts.append(self._add(patches.Arc((3.97, 0.9), 0.1, 0.12, theta1=0, theta2=180, color='#3e2723',
                                           linewidth=1.2, zorder=5.2)))
        for part in parts:
            part.set_visible(False)
        return {'home': parts}

    def _draw_legend(self):
        (width, height) = self._figure_size
        ax = self.fig.add_axes([0, 0, 1, LEGEND_HEIGHT / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, LEGEND_HEIGHT)
        ax.axis('off')
        entries = [('HR', 'heart rate', ECG_COLOR), ('ART', 'blood pressure', ART_COLOR),
                   ('SpO2', 'oxygen saturation', '#00b8d4')]
        text = ax.text(width / 2, LEGEND_HEIGHT * 0.62, '   '.join(f'{a}: {b}' for (a, b, _) in entries),
                       ha='center', va='center', fontsize=7, color=TEXT_COLOR)
        ax.text(width / 2, LEGEND_HEIGHT * 0.2, 'The simulator has levels, not numbers: each reading is a value '
                'typical of its level.', ha='center', va='center', fontsize=6, style='italic', color=MUTED_COLOR)
        return text

    # ---------- rendering

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        self._renders += 1
        for (p, scene) in self._scenes.items():
            get = lambda name: state[RDDLPlanningModel.ground_var(name, [p])]
            vitals = {name: int(get(name)) for name in ('hr', 'sysbp', 'oxygen', 'glucose')}
            on = {name: bool(get(name)) for name in ('antibiotics', 'vasopressors', 'ventilation')}
            (admitted, dead, home) = (bool(get('admitted')), bool(get('dead')), bool(get('discharged')))
            steps = int(get('steps'))
            in_bed = admitted and not home
            in_care = in_bed and not dead
            treating = in_care and steps < self._max_steps

            self._show_patient(scene, vitals, in_bed, dead, home)
            self._show_monitor(scene, vitals, in_care, dead, p)
            self._show_treatments(scene, on, treating)
            self._show_glucose(scene, vitals, in_care)

            # the whiteboard
            if not admitted:
                status = 'Admitting...'
            elif dead:
                status = f'Died, step {steps}'
            elif home:
                status = f'Home, step {steps}'
            elif steps >= self._max_steps:
                status = f'Stay ended ({steps} steps)'
            else:
                status = f'Step {steps} of {self._max_steps}'
            scene['status'].set_text(status)
            scene['diabetic'].set_text('Diabetic' if admitted and bool(get('diabetic')) else '')

        self.fig.canvas.draw()
        return self.convert2img(self.fig)

    def _show_patient(self, scene, vitals, in_bed, dead, home):
        for part in scene['body']:
            part.set_visible(in_bed and not dead)
        scene['pillow'].set_visible(not dead)
        scene['made'].set_visible(not in_bed)
        scene['shroud'].set_visible(dead)
        for part in scene['home']:
            part.set_visible(home)
        if not in_bed or dead:
            return

        # the face: eyes open and a smile when all is normal, closed eyes when two or more are not
        abnormal = ((vitals['hr'] != 1) + (vitals['sysbp'] != 1) + (vitals['oxygen'] != 1)
                    + (vitals['glucose'] != 2))
        for eye in scene['open-eyes']:
            eye.set_visible(abnormal < 2)
        for eye in scene['closed-eyes']:
            eye.set_visible(abnormal >= 2)
        mouth = scene['mouth']
        if abnormal == 0:
            (mouth.theta1, mouth.theta2, mouth.height) = (200, 340, 0.06)
        elif abnormal == 1:
            (mouth.theta1, mouth.theta2, mouth.height) = (250, 290, 0.02)
        else:
            (mouth.theta1, mouth.theta2, mouth.height) = (20, 160, 0.06)
        mouth.set_color(CYANOSIS_COLOR if vitals['oxygen'] == 0 else TEXT_COLOR)
        mouth.set_linewidth(1.4 if vitals['oxygen'] == 0 else 1.0)

    def _show_monitor(self, scene, vitals, in_care, dead, p):
        channels = scene['channels']
        scene['standby'].set_visible(not in_care and not dead)
        x0, x1 = 0.22, 1.47
        t = np.linspace(0, WINDOW, SAMPLES)
        xs = x0 + (x1 - x0) * t / WINDOW

        if not in_care and not dead:
            for channel in channels.values():
                channel['trace'].set_data([], [])
                channel['value'].set_text('')
                channel['box'].set_facecolor(SCREEN_COLOR)
            scene['banner'].set_facecolor(SCREEN_COLOR)
            scene['banner-text'].set_text('')
            return
        if dead:
            for channel in channels.values():
                channel['trace'].set_data([x0, x1], [channel['y'] - 0.05] * 2)
                channel['value'].set_text('---')
                channel['value'].set_color(channel['color'])
                channel['box'].set_facecolor(SCREEN_COLOR)
            scene['banner'].set_facecolor(CRISIS_COLOR)
            scene['banner-text'].set_text('*** ASYSTOLE')
            scene['banner-text'].set_color('white')
            return

        # the traces at the heart rate, from a point of the sweep that moves with each render
        rate = HEART_RATE[vitals['hr']]
        offset = (self._renders * 0.37 + sum(map(ord, p)) % 100 / 100.0) % 1.0
        phase = (t * rate / 60.0 + offset) % 1.0
        scale = {'ecg': 0.17, 'art': 0.17 * PRESSURE_SCALE[vitals['sysbp']],
                 'pleth': 0.17 * (0.55 if vitals['oxygen'] == 0 else 0.9)}
        shapes = {'ecg': _ecg(phase), 'art': _arterial(phase), 'pleth': _pleth(phase)}
        values = {'ecg': f'{rate}', 'art': PRESSURE[vitals['sysbp']], 'pleth': f'{SATURATION[vitals["oxygen"]]}'}
        alarming = {'ecg': vitals['hr'] != 1, 'art': vitals['sysbp'] != 1, 'pleth': vitals['oxygen'] != 1}
        for (name, channel) in channels.items():
            base = channel['y'] - 0.08
            channel['trace'].set_data(xs, base + scale[name] * shapes[name])
            channel['value'].set_text(values[name])
            channel['value'].set_fontsize(8 if name == 'art' else 10)
            if alarming[name]:
                channel['box'].set_facecolor(ALARM_COLOR)
                channel['value'].set_color(SCREEN_COLOR)
            else:
                channel['box'].set_facecolor(SCREEN_COLOR)
                channel['value'].set_color(channel['color'])

        # the alarms
        alarms = [text for ((name, level), text) in ALARMS.items() if vitals[name] == level]
        if alarms:
            scene['banner'].set_facecolor(ALARM_COLOR)
            scene['banner-text'].set_text('** ' + '  ** '.join(alarms))
            scene['banner-text'].set_color(SCREEN_COLOR)
        else:
            scene['banner'].set_facecolor(SCREEN_COLOR)
            scene['banner-text'].set_text(f'{scene["patient"]}   ADULT   ICU')
            scene['banner-text'].set_color(STANDBY_COLOR)

    def _show_treatments(self, scene, on, treating):
        abx = treating and on['antibiotics']
        for part in scene['bag']:
            part.set_visible(abx)
        scene['abx-line'].set_visible(abx)

        vaso = treating and on['vasopressors']
        scene['vaso-line'].set_visible(vaso)
        scene['pump-display'].set_text('NOREPI' if vaso else '')
        scene['plunger'].set_width(0.2 if vaso else 0.34)

        vent = treating and on['ventilation']
        for part in scene['tube']:
            part.set_visible(vent)
        if vent:
            t = np.linspace(0, 8.0, 200)
            scene['vent-wave'].set_data(0.22 + 0.36 * t / 8.0, 1.45 + 0.17 * _breaths(t))
            scene['vent-readout'].set_text('RR 15')
        else:
            scene['vent-wave'].set_data([], [])
            scene['vent-readout'].set_text('')

    def _show_glucose(self, scene, vitals, in_care):
        if not in_care:
            for key in ('glucose', 'glucose-zone'):
                scene[key].set_text('')
            scene['glucose-unit'].set_visible(False)
            return
        level = vitals['glucose']
        (name, color) = GLUCOSE_ZONES[level]
        scene['glucose'].set_text(f'{GLUCOSE[level]}')
        scene['glucose'].set_color(color)
        scene['glucose-zone'].set_text(name)
        scene['glucose-zone'].set_color(color)
        scene['glucose-unit'].set_visible(True)
