import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

SEA_COLOR = '#e3f2fd'
GRID_COLOR = '#bbdefb'
FRAME_COLOR = '#90a4ae'
ROUTE_COLOR = '#78909c'
BOOKED_COLOR = '#ffa000'
HUB_COLOR = '#37474f'
SPOKE_COLOR = '#ffffff'
AIRPORT_EDGE_COLOR = '#37474f'
PLANE_COLOR = '#eceff1'
PLANE_EDGE_COLOR = '#455a64'
CABIN_COLOR = '#1e88e5'
FULL_COLOR = '#e53935'
LOW_FARE_COLOR = '#43a047'
HIGH_FARE_COLOR = '#8e24aa'
TEXT_COLOR = '#263238'
MUTED_COLOR = '#607d8b'

MAP_SIZE = 6.4                  # inches of the map, square
STRIP_HEIGHT = 1.5              # inches of the strip under it
RING = 2.45                     # map units from the centre to the outer airports
BEND = 0.16                     # sideways bend of a route, as a fraction of its length
PLANE_LENGTH = 0.46            # map units, for up to 8 legs, and smaller for more
PLANE_RADIUS = 0.58             # distance of an aircraft from the centre, as a fraction of RING


def _plane():
    '''Top view of an airliner of unit length, nose along +x.'''
    half = [(0.5, 0.0), (0.46, 0.035), (0.38, 0.05), (0.1, 0.05), (-0.08, 0.44), (-0.17, 0.44),
            (-0.07, 0.05), (-0.34, 0.04), (-0.44, 0.17), (-0.5, 0.17), (-0.46, 0.0)]
    return half + [(x, -y) for (x, y) in reversed(half[1:-1])]


def _route(a, b):
    '''Control point of the curve of a route from a to b, bent to its right.'''
    (dx, dy) = (b[0] - a[0], b[1] - a[1])
    return ((a[0] + b[0]) / 2 + BEND * dy, (a[1] + b[1]) / 2 - BEND * dx)


def _curve(a, c, b, n=40):
    s = np.linspace(0, 1, n)[:, None]
    return (1 - s) ** 2 * np.array(a) + 2 * s * (1 - s) * np.array(c) + s ** 2 * np.array(b)


class AirlineRevenueVisualizer(BaseViz):
    '''Draws the network as a route map: the airports, a route for each flight leg with its
    aircraft filling up with the seats sold, and under the map the booking horizon running to
    departure, the mix of fares asked for, and the booking of the period.

    A booking is not part of the state, so the route of the last booking is found from the legs
    that lost a seat since the previous state, and shows when render is called on consecutive
    steps (its fare class cannot be told apart).
    '''

    def __init__(self, model: RDDLPlanningModel, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self._airports = list(model.type_to_objects['airport'])
        self._legs = list(model.type_to_objects['leg'])
        self._itineraries = list(model.type_to_objects['itinerary'])

        # what the drawing needs of the model
        nf = model.ground_vars_with_values(model.non_fluents)
        init = model.ground_vars_with_values(model.state_fluents)
        value = lambda values, name, *args: values[RDDLPlanningModel.ground_var(name, list(args))]
        self._periods = int(value(nf, 'PERIODS'))
        self._ends = {}
        for leg in self._legs:
            ends = [[a for a in self._airports if value(nf, end, leg, a)] for end in ('FROM', 'TO')]
            self._ends[leg] = tuple(e[0] if e else None for e in ends)
        self._capacity = {leg: max(int(value(init, 'seats', leg)), 1) for leg in self._legs}
        self._demand = [(float(value(nf, 'OD-PROB', j)), float(value(nf, 'ONSET', j)),
                         bool(value(nf, 'HIGH', j))) for j in self._itineraries]
        self._previous = None

        width, height = MAP_SIZE, MAP_SIZE + STRIP_HEIGHT
        self.fig = plt.figure(figsize=(width, height))
        self.fig.patch.set_facecolor('white')
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        span = RING + 0.75
        self.ax.set_xlim(-span, span)
        self.ax.set_ylim(-span - 2 * span * STRIP_HEIGHT / MAP_SIZE, span)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self._span = span

        self._place_airports()
        self._draw_map()
        self._routes = {leg: self._draw_route(leg) for leg in self._legs}
        self._draw_airports()
        self._strip = self._draw_strip()

    # ---------- drawing

    def _place_airports(self):
        '''The busiest airport at the centre and the others on a ring around it.'''
        degree = {a: sum(a in ends for ends in self._ends.values()) for a in self._airports}
        hub = max(self._airports, key=lambda a: degree[a]) if self._airports else None
        others = [a for a in self._airports if a != hub]
        self._hub = hub
        self._at = {hub: (0.0, 0.0)}
        for (k, a) in enumerate(others):
            angle = math.pi / 2 - 2 * math.pi * k / len(others)
            self._at[a] = (RING * math.cos(angle), RING * math.sin(angle))

    def _draw_map(self):
        span = self._span
        self.ax.add_patch(patches.FancyBboxPatch(
            (-span + 0.05, -span + 0.05), 2 * span - 0.1, 2 * span - 0.1,
            boxstyle='round,pad=0,rounding_size=0.15', facecolor=SEA_COLOR,
            edgecolor=FRAME_COLOR, linewidth=1.0, zorder=0))

        # a graticule, bowed like the lines of a globe
        s = np.linspace(-1, 1, 60)
        for k in np.linspace(-0.8, 0.8, 7):
            x = span * 0.95 * (k + 0.08 * (1 - s ** 2) * np.sign(k))
            self.ax.plot(x, span * 0.95 * s, color=GRID_COLOR, linewidth=0.6, zorder=1)
            y = span * 0.95 * (k + 0.05 * (1 - s ** 2))
            self.ax.plot(span * 0.95 * s, y, color=GRID_COLOR, linewidth=0.6, zorder=1)

    def _draw_route(self, leg):
        (a, b) = self._ends[leg]
        if a is None or b is None:
            return None
        (pa, pb) = (self._at[a], self._at[b])
        c = _route(pa, pb)
        xy = _curve(pa, c, pb)
        line = self.ax.plot(xy[:, 0], xy[:, 1], color=ROUTE_COLOR, linewidth=1.0,
                            linestyle=(0, (4, 3)), zorder=2)[0]

        # the aircraft out along the route, away from the crowd at the centre
        out = np.hypot(xy[:, 0], xy[:, 1])
        mid = xy[np.argmin(np.abs(out - PLANE_RADIUS * RING))]
        angle = math.degrees(math.atan2(pb[1] - pa[1], pb[0] - pa[0]))
        size = PLANE_LENGTH * min(1.0, math.sqrt(8 / max(len(self._legs), 1)))
        place = (Affine2D().scale(size).rotate_deg(angle).translate(*mid)
                 + self.ax.transData)
        body = patches.Polygon(_plane(), closed=True, facecolor=PLANE_COLOR,
                               edgecolor=PLANE_EDGE_COLOR, linewidth=0.8, zorder=4)
        body.set_transform(place)
        self.ax.add_patch(body)

        # the seats sold fill the cabin from the tail
        cabin = patches.Rectangle((-0.5, -0.5), 0.0, 1.0, facecolor=CABIN_COLOR,
                                  edgecolor='none', zorder=5)
        cabin.set_transform(place)
        self.ax.add_patch(cabin)
        cabin.set_clip_path(body)
        outline = patches.Polygon(_plane(), closed=True, fill=False,
                                  edgecolor=PLANE_EDGE_COLOR, linewidth=0.8, zorder=6)
        outline.set_transform(place)
        self.ax.add_patch(outline)

        # the seats sold, beside the aircraft on the outside of the bend
        away = np.array(c) - (np.array(pa) + np.array(pb)) / 2
        away = away / max(np.linalg.norm(away), 1e-9)
        (x, y) = np.array(mid) + 0.27 * away
        label = self.ax.text(x, y, '', ha='center', va='center', fontsize=6,
                             color=TEXT_COLOR, zorder=7)
        return {'line': line, 'cabin': cabin, 'label': label}

    def _draw_airports(self):
        for (a, (x, y)) in self._at.items():
            hub = a == self._hub and len(self._airports) > 2
            radius = 0.3 if hub else 0.17
            self.ax.add_patch(patches.Circle(
                (x, y), radius, facecolor=HUB_COLOR if hub else SPOKE_COLOR,
                edgecolor=AIRPORT_EDGE_COLOR, linewidth=1.2, zorder=8))
            if hub:

                # a control tower
                self.ax.add_patch(patches.Rectangle((x - 0.035, y - 0.17), 0.07, 0.2,
                                                    facecolor='white', zorder=9))
                self.ax.add_patch(patches.Polygon(
                    [(x - 0.1, y + 0.03), (x + 0.1, y + 0.03), (x + 0.07, y + 0.13),
                     (x - 0.07, y + 0.13)], facecolor='#80deea', edgecolor='white',
                    linewidth=0.8, zorder=9))
            else:
                self.ax.text(x, y, a.upper(), ha='center', va='center', fontsize=6.5,
                             fontweight='bold', color=TEXT_COLOR, zorder=9)

    def _draw_strip(self):
        span = self._span
        top = -span - 0.12
        text = lambda x, y, s, **style: self.ax.text(
            x, y, s, **{'ha': 'left', 'va': 'center', 'fontsize': 8, 'color': TEXT_COLOR,
                        **style})

        # the booking horizon, a runway to departure
        (left, right, y) = (-span + 0.3, span - 0.9, top - 0.35)
        self.ax.add_patch(patches.FancyBboxPatch(
            (left - 0.1, y - 0.09), right - left + 0.2, 0.18,
            boxstyle='round,pad=0,rounding_size=0.05', facecolor='#546e7a',
            edgecolor='none', zorder=2))
        self.ax.plot([left, right], [y, y], color='white', linewidth=0.8,
                     linestyle=(0, (5, 4)), zorder=3)
        text(left - 0.1, y + 0.22, 'Bookings open', fontsize=6.5, color=MUTED_COLOR)
        text(right + 0.15, y, 'Departure', fontsize=7.5, fontweight='bold')
        place = Affine2D().scale(0.32).translate(left, y) + self.ax.transData
        plane = patches.Polygon(_plane(), closed=True, facecolor='white',
                                edgecolor=TEXT_COLOR, linewidth=0.8, zorder=4)
        plane.set_transform(place)
        self.ax.add_patch(plane)
        period = text(left, y - 0.24, '', fontsize=7, color=MUTED_COLOR)

        # the requests of the period, low or high fare
        (y2, bar) = (top - 0.85, (left + 0.95, 1.8))
        text(left - 0.1, y2, 'Requests', fontsize=7.5, color=MUTED_COLOR)
        self.ax.add_patch(patches.Rectangle((bar[0], y2 - 0.08), bar[1], 0.16,
                                            facecolor='#eceff1', zorder=2))
        low = patches.Rectangle((bar[0], y2 - 0.08), 0, 0.16, facecolor=LOW_FARE_COLOR, zorder=3)
        high = patches.Rectangle((bar[0], y2 - 0.08), 0, 0.16, facecolor=HIGH_FARE_COLOR,
                                 zorder=3)
        self.ax.add_patch(low)
        self.ax.add_patch(high)
        mix = text(bar[0] + bar[1] + 0.12, y2, '', fontsize=7)

        # the seats sold and the last booking
        sold = text(left - 0.1, top - 1.24, '', fontsize=8, fontweight='bold')
        booked = text(left + 2.6, top - 1.24, '', fontsize=8, color=BOOKED_COLOR,
                      fontweight='bold')
        return {'plane': plane, 'runway': (left, right, y), 'period': period, 'low': low,
                'high': high, 'bar': bar, 'mix': mix, 'sold': sold, 'booked': booked}

    # ---------- rendering

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return Image.fromarray(data[:, :, :3])

    def render(self, state):
        value = lambda name, *args: state[RDDLPlanningModel.ground_var(name, list(args))]
        seats = {leg: int(value('seats', leg)) for leg in self._legs}
        t = int(value('t'))
        for (leg, route) in self._routes.items():
            if route is None:
                continue
            capacity = self._capacity[leg]
            full = seats[leg] <= 0
            route['cabin'].set_width(min(max(1 - seats[leg] / capacity, 0), 1))
            route['cabin'].set_facecolor(FULL_COLOR if full else CABIN_COLOR)
            route['label'].set_text('FULL' if full else f'{capacity - seats[leg]}/{capacity}')
            route['label'].set_color(FULL_COLOR if full else TEXT_COLOR)
        self._show_strip(t, seats)
        self._previous = seats
        self.fig.canvas.draw()
        return self.convert2img(self.fig)

    def _show_strip(self, t, seats):
        strip = self._strip
        (left, right, y) = strip['runway']
        progress = min(max(t / max(self._periods, 1), 0), 1)
        x = left + progress * (right - left)
        strip['plane'].set_transform(Affine2D().scale(0.32).translate(x, y) + self.ax.transData)
        strip['period'].set_position((x - 0.1, y - 0.24))
        strip['period'].set_text(f'period {t} of {self._periods}')

        # the probabilities of a request in this period, as the domain draws them
        left_frac = max(self._periods - t, 1) / max(self._periods, 1)
        (low, high) = (0.0, 0.0)
        for (prob, onset, is_high) in self._demand:
            ratio = max(1 / left_frac - onset, 0)
            if is_high:
                high += prob * ratio / (1 + ratio)
            else:
                low += prob / (1 + ratio)
        (x0, width) = strip['bar']
        strip['low'].set_width(width * min(low, 1))
        strip['high'].set_x(x0 + width * min(low, 1))
        strip['high'].set_width(width * min(high, 1 - min(low, 1)))
        strip['mix'].set_text(f'{100 * low:.0f}% low fare, {100 * high:.0f}% high fare')

        total = sum(self._capacity.values())
        sold = total - sum(max(s, 0) for s in seats.values())
        strip['sold'].set_text(f'Seats sold {sold} of {total}')

        # the route of the last booking, from the legs that lost a seat
        lost = [leg for leg in self._legs
                if self._previous is not None and seats[leg] < self._previous.get(leg, seats[leg])]
        for (leg, route) in self._routes.items():
            if route is not None:
                on = leg in lost
                route['line'].set_color(BOOKED_COLOR if on else ROUTE_COLOR)
                route['line'].set_linewidth(2.6 if on else 1.0)
                route['line'].set_linestyle('-' if on else (0, (4, 3)))
        strip['booked'].set_text(f'Booked: {self._describe(lost)}' if lost else '')

    def _describe(self, legs):
        '''The airports of a trip over the legs, in order.'''
        ends = [self._ends[leg] for leg in legs if None not in self._ends[leg]]
        starts = [a for (a, _) in ends if all(a != b for (_, b) in ends)]
        stop = starts[0] if starts else (ends[0][0] if ends else '?')
        trip = [stop]
        for _ in range(len(ends)):
            nxt = [b for (a, b) in ends if a == trip[-1]]
            if not nxt:
                break
            trip.append(nxt[0])
        return ' → '.join(a.upper() for a in trip)
