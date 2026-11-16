import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

SKIN_COLOR = '#f6dcc4'
SKIN_EDGE_COLOR = '#c9a27e'
HAIR_COLOR = '#5d4037'
PANTS_COLOR = '#90a4ae'
PANTS_EDGE_COLOR = '#607d8b'
BELT_COLOR = '#546e7a'
HEART_COLOR = '#ffcdd2'
HEART_EDGE_COLOR = '#c62828'
LIVER_COLOR = '#bc8f80'
LIVER_EDGE_COLOR = '#6d4c41'
STOMACH_COLOR = '#ffd8c8'
STOMACH_EDGE_COLOR = '#d84315'
GUT_COLOR = '#f8bbd0'
GUT_EDGE_COLOR = '#ad1457'
MUSCLE_COLOR = '#ef9a9a'
MUSCLE_EDGE_COLOR = '#b71c1c'
DEPOT_COLOR = '#ede7f6'
DEPOT_EDGE_COLOR = '#7e57c2'
GLUCOSE_COLOR = '#ffb300'
GLUCOSE_EDGE_COLOR = '#e65100'
INSULIN_COLOR = '#5e35b1'
FOOD_COLOR = '#a1887f'
FOOD_EDGE_COLOR = '#5d4037'
DEVICE_COLOR = '#37474f'
SCREEN_COLOR = '#263238'
PUMP_SCREEN_COLOR = '#80cbc4'
TUBE_COLOR = '#90a4ae'
PLATE_COLOR = '#fafafa'
PLATE_EDGE_COLOR = '#b0bec5'
CUTLERY_COLOR = '#78909c'
TEXT_COLOR = '#37474f'
MUTED_COLOR = '#78909c'
SUN_COLOR = '#ffca28'
MOON_COLOR = '#b0bec5'
DAY_SKY_COLOR = '#f1f8fe'
NIGHT_SKY_COLOR = '#e8eaf6'
FRAME_COLOR = '#e0e0e0'

# the glucose zones (mg/dL) of the consensus on time in range: upper bound, name, color
ZONES = [(54, 'Very low', '#ff5252'), (70, 'Low', '#ff8a80'), (180, 'In range', '#69f0ae'),
         (250, 'High', '#ffd180'), (math.inf, 'Very high', '#ff9e80')]

# the particles of each compartment: (name, kind, region, most), with the radius of each kind
PARTICLES = [('food', 'food', 'stomach', 14), ('chyme', 'glucose', 'stomach', 14),
             ('gut', 'glucose', 'gut', 16), ('blood-glucose', 'glucose', 'heart', 14),
             ('blood-insulin', 'insulin', 'heart', 12), ('muscle', 'glucose', 'muscle', 12),
             ('liver', 'insulin', 'liver', 10), ('depot', 'insulin', 'depot', 22),
             ('plate', 'food', 'plate', 15)]
RADIUS = {'glucose': 0.05, 'insulin': 0.032, 'food': 0.055}

# what one particle stands for: mg of glucose in the stomach and the gut, mg/dL of glucose
# in the blood and the muscle, and for insulin, the share of the steady state for one particle
FOOD_MG = 4000.0
GUT_MG = 1500.0
BLOOD_MG_DL = 30.0
MUSCLE_MG_DL = 25.0
STEADY_INSULIN = {'blood-insulin': 5, 'liver': 5, 'depot': 8}
PLATE_G = 4.0

# the day, in minutes after midnight
SUNRISE = 360
SUNSET = 1260

# sizes in inches: the scene of each patient, the gap between scenes, and the legend
CELL = (4.3, 3.75)
GAP = 0.15
MAX_COLUMNS = 4
LEGEND_HEIGHT = 0.4


def _zone(bg):
    return next(zone for zone in ZONES if bg < zone[0])


def _clock(minute):
    minute = int(round(minute)) % 1440
    return f'{minute // 60:02d}:{minute % 60:02d}'


def _rounded(x, y, w, h, r, **style):
    return patches.FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', **style)


def _particle(kind, x, y):
    r = RADIUS[kind]
    if kind == 'glucose':
        return patches.RegularPolygon((x, y), 6, radius=r, facecolor=GLUCOSE_COLOR,
                                      edgecolor=GLUCOSE_EDGE_COLOR, linewidth=0.5, zorder=5)
    if kind == 'insulin':
        return plt.Circle((x, y), r, facecolor=INSULIN_COLOR, edgecolor='white', linewidth=0.4,
                          zorder=5.1)
    return _rounded(x - r, y - 0.75 * r, 2 * r, 1.5 * r, 0.45 * r, facecolor=FOOD_COLOR,
                    edgecolor=FOOD_EDGE_COLOR, linewidth=0.5, zorder=5)


# ---------- the outlines of the organs, in the coordinates of a scene

THETA = np.linspace(0, 2 * np.pi, 80)


def _heart(cx, cy, scale):
    x = 16 * np.sin(THETA) ** 3
    y = 13 * np.cos(THETA) - 5 * np.cos(2 * THETA) - 2 * np.cos(3 * THETA) - np.cos(4 * THETA)
    return np.column_stack([cx + scale * x, cy + scale * y])


def _liver(cx, cy):

    # a wedge, thick on the patient's right (the left of the picture) and thin on its left
    taper = 0.55 + 0.45 * (1 - np.cos(THETA)) / 2
    x = 0.31 * np.cos(THETA)
    y = 0.2 * np.sin(THETA) * taper + 0.01 * np.cos(THETA)
    return np.column_stack([cx + x, cy + y])


def _stomach(cx, cy, angle=-0.5):

    # a bag, wide at the top and narrowing to its outlet, tilted
    x = 0.22 * np.cos(THETA) + 0.06 * np.sin(THETA) ** 2
    y = 0.3 * np.sin(THETA) * (0.75 + 0.25 * np.sin(THETA))
    (c, s) = (math.cos(angle), math.sin(angle))
    return np.column_stack([cx + c * x - s * y, cy + s * x + c * y])


def _coil(rows, left, right, radius):

    # a tube running along rows, turning by a half circle at their ends
    points = []
    for (k, y) in enumerate(rows):
        (start, end) = (left, right) if k % 2 == 0 else (right, left)
        points += [(x, y) for x in np.linspace(start, end, 12)]
        if k + 1 < len(rows):
            side = 1 if k % 2 == 0 else -1
            for a in np.linspace(np.pi / 2, -np.pi / 2, 9)[1:-1]:
                points.append((end + side * radius * math.cos(a),
                               y - radius + radius * math.sin(a)))
    return points


class DiabetesVisualizer(BaseViz):
    '''Draws each patient as an x-ray of the body: the food in the stomach (brown) and the
    glucose (amber hexagons) it becomes in the stomach, the intestine, the blood of the heart and
    the muscle, and the insulin (purple) in the depot under the skin where the pump infuses it,
    in the blood and in the liver, each as many as the compartment holds. Around the patient are
    the sensor on the arm and its receiver with the reading, the zone of the glucose and its
    trend, the pump on the belt with the rate it infuses, the plate with what is left of the meal
    or the next meal of the day, and the time of day.

    The insulin given is not part of the state, so the pump shows the rate recovered from the
    first subcutaneous compartment, of which it is the only input, and the trend of the reading
    is its change since the previous reading: both need the previous state, so they show when
    render is called on consecutive steps.
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._patients = list(model.type_to_objects['patient'])
        self._meals = list(model.type_to_objects['meal'])

        # what the drawing needs of the model
        nf = model.ground_vars_with_values(model.non_fluents)
        init = model.ground_vars_with_values(model.state_fluents)
        value = lambda values, name, *args: float(
            values[RDDLPlanningModel.ground_var(name, list(args))])
        self._dt = value(nf, 'DT')
        self._start = value(nf, 'START-MINUTE')
        self._vg, self._decay, self._gain, self._steady = {}, {}, {}, {}
        for p in self._patients:
            rate = value(nf, 'KA1', p) + value(nf, 'KD', p)
            self._vg[p] = value(nf, 'VG', p)
            self._decay[p] = math.exp(-rate * self._dt)
            self._gain[p] = (1 - self._decay[p]) / rate * 6000 / value(nf, 'BW', p)

            # the insulin is shown relative to the steady state the patients start in
            self._steady[p] = {name: max(value(init, name, p), 1e-9)
                               for name in ('ip', 'il', 'isc1', 'isc2')}
        self._previous = {}

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
            self._scenes[patient] = self._draw_scene(patient, x0, y0, seed=n)
        self._draw_legend()

    # ---------- drawing a scene, in its own coordinates

    def _add(self, patch):
        patch.set_transform(self._shift)
        return self.ax.add_patch(patch)

    def _line(self, xs, ys, **style):
        return self.ax.plot(xs, ys, transform=self._shift, **style)[0]

    def _text(self, x, y, text, **style):
        style = {'ha': 'center', 'va': 'center', 'color': TEXT_COLOR, 'zorder': 6, **style}
        return self.ax.text(x, y, text, transform=self._shift, **style)

    def _label(self, x, y, text):
        return self._text(x, y, text, fontsize=5.5, color=MUTED_COLOR, style='italic')

    def _draw_scene(self, patient, x0, y0, seed):
        self._shift = Affine2D().translate(x0, y0) + self.ax.transData
        scene = {'sky': self._add(_rounded(0, 0, *CELL, 0.12, facecolor=DAY_SKY_COLOR,
                                           edgecolor=FRAME_COLOR, linewidth=0.8, zorder=0))}
        if len(self._patients) > 1:
            self._text(0.12, CELL[1] - 0.14, patient, ha='left', va='top', fontsize=8,
                       fontweight='bold')
        scene['mouth'] = self._draw_body()
        regions = self._draw_organs()
        scene.update(self._draw_devices())
        (scene['meal'], regions['plate']) = self._draw_plate()

        # the particles, all hidden until shown by render
        rng = np.random.default_rng(seed)
        scene['pools'] = {name: self._particles(kind, regions[region], most, rng)
                          for (name, kind, region, most) in PARTICLES}
        return scene

    def _draw_body(self):
        skin = {'facecolor': SKIN_COLOR, 'edgecolor': SKIN_EDGE_COLOR, 'linewidth': 0.9}

        # legs, arms with hands, belt, torso, neck and head with hair, eyes and mouth
        for x in (0.98, 1.58):
            self._add(_rounded(x, 0.12, 0.44, 0.55, 0.08, facecolor=PANTS_COLOR,
                               edgecolor=PANTS_EDGE_COLOR, linewidth=0.8, zorder=1))
        for x in (0.56, 2.2):
            self._add(_rounded(x, 1.05, 0.26, 1.72, 0.12, zorder=1, **skin))
            self._add(plt.Circle((x + 0.13, 1.0), 0.13, zorder=1, **skin))
        self._add(_rounded(0.88, 0.62, 1.24, 0.12, 0.04, facecolor=BELT_COLOR, edgecolor='none',
                           zorder=2.6))
        self._add(_rounded(0.84, 0.7, 1.32, 2.15, 0.28, zorder=2, **skin))
        self._add(patches.Rectangle((1.38, 2.8), 0.24, 0.2, zorder=1.9, **skin))
        self._add(plt.Circle((1.5, 3.3), 0.32, zorder=2, **skin))
        self._add(patches.Wedge((1.5, 3.33), 0.34, 15, 165, facecolor=HAIR_COLOR, edgecolor='none',
                                zorder=2.1))
        for x in (1.39, 1.61):
            self._add(plt.Circle((x, 3.24), 0.03, facecolor=TEXT_COLOR, edgecolor='none',
                                 zorder=2.2))
        return self._add(patches.Arc((1.5, 3.16), 0.18, 0.1, theta1=200, theta2=340,
                                     color=TEXT_COLOR, linewidth=1.0, zorder=2.2))

    def _draw_organs(self):
        organ = lambda points, fill, edge: self._add(patches.Polygon(
            points, closed=True, facecolor=fill, edgecolor=edge, linewidth=1.0, joinstyle='round',
            zorder=3))

        # the heart, which holds the blood, with vessels to the organs, the muscle and the sensor
        heart = organ(_heart(1.62, 2.47, 0.021), HEART_COLOR, HEART_EDGE_COLOR)
        self._label(1.62, 2.79, 'heart (blood)')
        vessels = [((1.45, 2.3), (1.15, 1.95), 0.2), ((1.8, 2.3), (2.3, 2.2), -0.3),
                   ((1.62, 2.2), (1.35, 1.45), 0.25), ((1.75, 2.25), (1.85, 1.95), -0.2),
                   ((1.4, 2.5), (0.69, 2.3), 0.25)]
        for (start, end, bend) in vessels:
            self._add(patches.FancyArrowPatch(start, end, connectionstyle=f'arc3,rad={bend}',
                                              arrowstyle='-', color=HEART_EDGE_COLOR, linewidth=1.6,
                                              alpha=0.45, zorder=2.5))

        # the liver and the stomach under it
        liver = organ(_liver(1.2, 1.98), LIVER_COLOR, LIVER_EDGE_COLOR)
        self._label(1.0, 2.2, 'liver')
        stomach = organ(_stomach(1.83, 1.92), STOMACH_COLOR, STOMACH_EDGE_COLOR)
        self._label(2.0, 2.27, 'stomach')

        # the intestine, drawn as a tube with its wall
        gut = _coil([1.5, 1.35, 1.2, 1.05, 0.9], left=1.0, right=1.66, radius=0.075)
        for (color, width, z) in ((GUT_EDGE_COLOR, 7.5, 3), (GUT_COLOR, 6.2, 3.1)):
            self._line(*zip(*gut), color=color, linewidth=width, solid_capstyle='round',
                       solid_joinstyle='round', zorder=z)
        self._label(1.33, 0.79, 'intestine')

        # the muscle of the arm, and the depot of insulin under the skin of the belly
        muscle = self._add(patches.Ellipse((2.33, 1.95), 0.17, 0.62, facecolor=MUSCLE_COLOR,
                                           edgecolor=MUSCLE_EDGE_COLOR, linewidth=0.8, zorder=3))
        self._label(2.33, 2.36, 'muscle')
        depot = self._add(patches.Ellipse((1.95, 1.02), 0.28, 0.32, facecolor=DEPOT_COLOR,
                                          edgecolor=DEPOT_EDGE_COLOR, linewidth=0.8,
                                          linestyle=(0, (2, 1.5)), zorder=3))
        self._label(1.95, 1.27, 'under skin')
        return {'heart': heart, 'liver': liver, 'stomach': stomach, 'gut': gut, 'muscle': muscle,
                'depot': depot}

    def _draw_devices(self):
        device = {'facecolor': DEVICE_COLOR, 'edgecolor': 'none'}

        # the sensor on the arm
        self._add(plt.Circle((0.69, 2.42), 0.085, facecolor='white', edgecolor=DEVICE_COLOR,
                             linewidth=0.9, zorder=4))
        self._add(plt.Circle((0.69, 2.42), 0.03, zorder=4.1, **device))
        self._label(0.69, 2.6, 'sensor')

        # the pump on the belt, with its tube to the skin
        self._add(_rounded(2.33, 0.42, 0.42, 0.5, 0.06, zorder=4, **device))
        self._add(patches.Rectangle((2.38, 0.67), 0.32, 0.18, facecolor=PUMP_SCREEN_COLOR,
                                    edgecolor='none', zorder=4.1))
        pump = self._text(2.54, 0.76, '', fontsize=5.2, color=SCREEN_COLOR, fontweight='bold')
        self._text(2.54, 0.53, 'U/min', fontsize=4.8, color='white')
        self._add(patches.FancyArrowPatch((2.4, 0.9), (1.95, 0.98),
                                          connectionstyle='arc3,rad=0.5', arrowstyle='-',
                                          color=TUBE_COLOR, linewidth=1.2, zorder=4))
        self._add(plt.Circle((1.95, 0.98), 0.035, facecolor='white', edgecolor=DEVICE_COLOR,
                             linewidth=0.8, zorder=4.3))
        self._label(2.54, 0.3, 'pump')

        # the time of day, with the sun, or the moon drawn as a shadow over it
        sun = self._add(plt.Circle((3.05, 3.45), 0.13, facecolor=SUN_COLOR, edgecolor='none',
                                   zorder=1))
        moon = self._add(plt.Circle((3.11, 3.49), 0.11, facecolor=NIGHT_SKY_COLOR,
                                    edgecolor='none', zorder=1.1, visible=False))
        clock = self._text(3.28, 3.45, '', ha='left', fontsize=11, fontweight='bold')

        # the receiver of the sensor
        self._add(_rounded(2.95, 2.08, 1.18, 1.05, 0.1, zorder=4, **device))
        self._add(_rounded(3.01, 2.16, 1.06, 0.89, 0.06, facecolor=SCREEN_COLOR, edgecolor='none',
                           zorder=4.1))
        reading = self._text(3.5, 2.68, '', fontsize=17, fontweight='bold')
        trend = self._text(3.93, 2.7, '', fontsize=12, fontweight='bold')
        self._text(3.5, 2.43, 'mg/dL', fontsize=6, color=MOON_COLOR)
        zone = self._text(3.54, 2.26, '', fontsize=6.5)
        return {'pump': pump, 'sun': sun, 'moon': moon, 'clock': clock, 'reading': reading,
                'trend': trend, 'zone': zone}

    def _draw_plate(self):
        cutlery = {'color': CUTLERY_COLOR, 'solid_capstyle': 'round', 'zorder': 3}

        # the plate, with a fork on its left and a knife on its right
        self._add(patches.Ellipse((3.54, 1.32), 1.0, 0.46, facecolor=PLATE_COLOR,
                                  edgecolor=PLATE_EDGE_COLOR, linewidth=1.0, zorder=3))
        plate = self._add(patches.Ellipse((3.54, 1.32), 0.72, 0.3, facecolor='white',
                                          edgecolor=FRAME_COLOR, linewidth=0.7, zorder=3.1))
        self._line([2.93, 2.93], [1.1, 1.42], linewidth=1.3, **cutlery)
        for dx in (-0.04, 0.0, 0.04):
            self._line([2.93 + dx, 2.93 + dx], [1.42, 1.55], linewidth=0.8, **cutlery)
        self._line([2.89, 2.97], [1.42, 1.42], linewidth=0.8, **cutlery)
        self._line([4.15, 4.15], [1.1, 1.33], linewidth=1.4, **cutlery)
        self._add(patches.Ellipse((4.165, 1.43), 0.07, 0.24, facecolor=FRAME_COLOR,
                                  edgecolor=CUTLERY_COLOR, linewidth=0.7, zorder=3))
        meal = self._text(3.54, 0.9, '', va='top', fontsize=6.5, linespacing=1.4)
        return meal, plate

    def _particles(self, kind, region, count, rng):
        '''Places count particles of a kind in a region, a patch or a path given by its points,
        apart from each other where there is room, all hidden.'''
        r = RADIUS[kind]
        if isinstance(region, list):
            def sample():
                i = rng.uniform(0, len(region) - 1)
                ((xa, ya), (xb, yb)) = (region[int(i)], region[min(int(i) + 1, len(region) - 1)])
                f = i - int(i)
                return (xa + f * (xb - xa) + rng.normal(0, 0.02),
                        ya + f * (yb - ya) + rng.normal(0, 0.02))
        else:
            outline = region.get_path().transformed(region.get_patch_transform())
            box = outline.get_extents()

            def sample():
                while True:
                    point = (rng.uniform(box.x0 + r, box.x1 - r),
                             rng.uniform(box.y0 + r, box.y1 - r))
                    if outline.contains_point(point, radius=-2 * r):
                        return point
        placed, pool = [], []
        for _ in range(count):
            for _ in range(300):
                point = sample()
                if all(math.dist(point, other) > 1.9 * r for other in placed):
                    break
            placed.append(point)
            particle = self._add(_particle(kind, *point))
            particle.set_visible(False)
            pool.append(particle)
        return pool

    def _draw_legend(self):
        (width, height) = self._figure_size
        ax = self.fig.add_axes([0, 0, 1, LEGEND_HEIGHT / height])
        ax.set_xlim(0, width)
        ax.set_ylim(0, LEGEND_HEIGHT)
        ax.set_aspect('equal')
        ax.axis('off')
        entries = [('food', f'Food ({PLATE_G:g} g)'), ('glucose', 'Glucose'),
                   ('insulin', 'Insulin')]
        widths = [0.65 + 0.07 * len(text) for (_, text) in entries]
        x = (width - sum(widths)) / 2
        y = LEGEND_HEIGHT / 2
        for ((kind, text), w) in zip(entries, widths):
            ax.add_patch(_particle(kind, x + 0.15, y))
            ax.text(x + 0.32, y, text, ha='left', va='center', fontsize=8, color=TEXT_COLOR)
            x += w

    # ---------- rendering

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    @staticmethod
    def _show(pool, count):
        for (k, particle) in enumerate(pool):
            particle.set_visible(k < count)

    def render(self, state):
        value = lambda name, *args: state[RDDLPlanningModel.ground_var(name, list(args))]
        t = float(value('t'))
        minute = (self._start + t) % 1440
        for (p, scene) in self._scenes.items():
            get = lambda name, *args: float(value(name, p, *args))
            self._show_body(scene, p, get)
            self._show_plate(scene, p, get, value, minute)
            self._show_devices(scene, p, get, t)

            # the time of day
            day = SUNRISE <= minute < SUNSET
            scene['clock'].set_text(_clock(minute))
            scene['sky'].set_facecolor(DAY_SKY_COLOR if day else NIGHT_SKY_COLOR)
            scene['sun'].set_facecolor(SUN_COLOR if day else MOON_COLOR)
            scene['moon'].set_visible(not day)

        self.fig.canvas.draw()
        return self.convert2img(self.fig)

    def _show_body(self, scene, p, get):
        pools = scene['pools']
        steady = self._steady[p]
        relative = lambda name, amount, base: round(STEADY_INSULIN[name] * amount / base)
        depot = get('isc1') + get('isc2')
        counts = {'food': round(get('qsto1') / FOOD_MG), 'chyme': round(get('qsto2') / FOOD_MG),
                  'gut': round(get('qgut') / GUT_MG),
                  'blood-glucose': round(get('gp') / self._vg[p] / BLOOD_MG_DL),
                  'muscle': round(get('gt') / self._vg[p] / MUSCLE_MG_DL),
                  'blood-insulin': relative('blood-insulin', get('ip'), steady['ip']),
                  'liver': relative('liver', get('il'), steady['il']),
                  'depot': relative('depot', depot, steady['isc1'] + steady['isc2'])}
        for (name, count) in counts.items():
            self._show(pools[name], count)

        # a frown while the glucose is out of range
        bg = get('gs') / self._vg[p]
        mouth = scene['mouth']
        (mouth.theta1, mouth.theta2) = (20, 160) if bg < 70 or bg > 180 else (200, 340)

    def _show_plate(self, scene, p, get, value, minute):
        left = get('to-eat')
        self._show(scene['pools']['plate'], math.ceil(left / PLATE_G))
        if left > 0 or bool(value('eating', p)):
            scene['meal'].set_text(f'Eating, {left:.0f} g left')
            return

        # the next meal of the day
        upcoming = [(get('meal-time', m), m, get('meal-carbs', m)) for m in self._meals
                    if bool(value('taken', p, m)) and get('meal-time', m) > minute]
        if upcoming:
            (when, meal, carbs) = min(upcoming)
            scene['meal'].set_text(f'Next: {meal.replace("-", " ")}\n'
                                   f'at {_clock(when)}, {carbs:.0f} g')
        else:
            scene['meal'].set_text('No more meals today')

    def _show_devices(self, scene, p, get, t):
        cgm = get('cgm')
        (_, name, color) = _zone(cgm)
        for (text, content) in ((scene['reading'], f'{cgm:.0f}'), (scene['zone'], name)):
            text.set_text(content)
            text.set_color(color)

        # the trend of the reading and the rate of the pump, from the previous state
        previous = self._previous.get(p)
        (arrow, rate) = ('', '')
        if previous is not None and abs(t - previous['t'] - self._dt) < 1e-6:
            slope = (cgm - previous['cgm']) / self._dt
            arrow = ('↓' if slope < -2 else '↘' if slope < -1 else
                     '→' if slope <= 1 else '↗' if slope <= 2 else '↑')
            infused = (get('isc1') - previous['isc1'] * self._decay[p]) / self._gain[p]
            infused = max(infused, 0.0)
            rate = f'{infused:.3f}' if infused < 1 else f'{infused:.2f}'
        scene['trend'].set_text(arrow)
        scene['trend'].set_color(color)
        scene['pump'].set_text(rate)
        self._previous[p] = {'t': t, 'cgm': cgm, 'isc1': get('isc1')}
