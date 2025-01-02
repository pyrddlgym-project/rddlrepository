import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Optional

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class HVACVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel,
                 grid_size: Optional[int] = [50, 50],
                 resolution: Optional[int] = [500, 500]) -> None:
        self._model = model
        self._states = model.ground_vars_with_values(model.state_fluents)
        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._objects = model.type_to_objects
        self._grid_size = grid_size
        self._resolution = resolution
        self._interval = 15

        self._object_layout = None
        self._canvas_info = None
        self._render_frame = None
        self._data = None
        self._fig = None
        self._ax = None

    def build_object_layout(self) -> dict:
        zone_vol = {o: None for o in self._objects['zone']}
        heater_vol = {o: None for o in self._objects['heater']}
        adj_zone = {o: [] for o in self._objects['zone']}
        adj_heater = {o: [] for o in self._objects['heater']}
        temp_zone = {o: None for o in self._objects['zone']}
        temp_heater = {o: None for o in self._objects['heater']}

        # add none-fluents
        for k, v in self._nonfluents.items():
            var, objects = RDDLPlanningModel.parse_grounded(k)
            if var == 'ZONE_VOL':
                zone_vol[objects[0]] = v
            elif var == 'HEATER-VOL':
                heater_vol[objects[0]] = v
            elif var == 'ADJ_ZONES':
                if v == True:
                    adj_zone[objects[0]].append(objects[1])
            elif var == 'ADJ_HEATER':
                if v == True:
                    adj_heater[objects[0]].append(objects[1])

        # add states
        for k, v in self._states.items():
            var, objects = RDDLPlanningModel.parse_grounded(k)
            if var == 'temp-zone':
                temp_zone[objects[0]] = v
            elif var == 'temp-heater':
                temp_heater[objects[0]] = v

        temp_zone_max = self._nonfluents['TEMP-ZONE-MAX']
        temp_zone_min = self._nonfluents['TEMP-ZONE-MIN']
        temp_out = self._nonfluents['TEMP-OUT']

        object_layout = {'temp-zone-max': temp_zone_max, 
                         'temp-zone-min': temp_zone_min,
                         'temp-out': temp_out,
                         'zone-vol': zone_vol,
                         'heater-vol': heater_vol,
                         'temp-zone': temp_zone,
                         'temp-heater': temp_heater,
                         'adj-zone': adj_zone,
                         'adj-heater': adj_heater}
        return object_layout

    def init_canvas_info(self):
        interval = self._interval

        zone_list = self._objects['zone']
        heater_list = self._objects['heater']

        room_grid_size = (max(len(zone_list), len(heater_list)),
                          max(len(zone_list), len(heater_list)) + 1)

        room_center = (room_grid_size[0] * self._interval / 2,
                       (room_grid_size[1] - 1) * self._interval / 2 + interval)

        start_points_zone = []
        for i in range(room_grid_size[0]):
            for j in range(room_grid_size[1]):
                start_points_zone.append((i * interval, j * interval))

        start_points_heater = []
        for i in range(room_grid_size[0]):
            start_points_heater.append((i * interval, 0))

        # rank start point by distance from center
        start_points_zone.sort(
            key=lambda x: abs(x[0] + interval / 2 - room_center[0]) + \
                          abs(x[1] + interval / 2 - room_center[1]))
        start_points_heater.sort(
            key=lambda x: abs(x[0] + interval / 2 - room_center[0]) + \
                          abs(x[1] + interval / 2 - room_center[1]))

        # rank by number of adj rooms
        zone_list.sort(reverse=True,
                       key=lambda x: len(self._object_layout['adj-zone'][x]))
        heater_list.sort(reverse=True,
                         key=lambda x: len(self._object_layout['adj-heater'][x]))

        zone_init_points = {zone_list[i]: start_points_zone[i]
                            for i in range(len(zone_list))}
        heater_init_points = {heater_list[i]: start_points_heater[i]
                              for i in range(len(heater_list))}

        canvas_info = {
            'canvas_size': (room_grid_size[0] * interval,
                            room_grid_size[1] * interval),
            'zone_init_points': zone_init_points,
            'heater_init_points': heater_init_points
        }
        return canvas_info

    def render_conn(self):
        ax = self._ax
        adj_zone = self._object_layout['adj-zone']
        adj_heater = self._object_layout['adj-heater']
        zone_init_points = self._canvas_info['zone_init_points']
        heater_init_points = self._canvas_info['heater_init_points']

        interval = self._interval * 2 / 3

        for k, v in adj_zone.items():
            from_point = (zone_init_points[k][0] + interval / 2,
                          zone_init_points[k][1] + interval / 2)
            for p in v:
                to_point = (zone_init_points[p][0] + interval / 2,
                            zone_init_points[p][1] + interval / 2)
                line = plt.Line2D((from_point[0], to_point[0]),
                                  (from_point[1], to_point[1]),
                                  ls='-', color='black', lw=2, zorder=1)
                ax.add_line(line)
        for k, v in adj_heater.items():
            from_point = (heater_init_points[k][0] + interval / 2,
                          heater_init_points[k][1] + interval / 2)
            for p in v:
                to_point = (zone_init_points[p][0] + interval / 2,
                            zone_init_points[p][1] + interval / 2)
                line = plt.Line2D((from_point[0], to_point[0]),
                                  (from_point[1], to_point[1]),
                                  ls='-', color='red', lw=2, zorder=1)
                ax.add_line(line)

    def render_zone(self, zone: str) -> tuple:
        ax = self._ax

        curr_z = zone
        vol_ratio = 1

        init_x, init_y = self._canvas_info['zone_init_points'][curr_z]
        interval = self._interval * 2 / 3 * vol_ratio

        temp = self._object_layout['temp-zone'][curr_z]
        upper_bound = self._object_layout['temp-zone-max']
        lower_bound = self._object_layout['temp-zone-min']

        upL = [init_x, upper_bound / 100 * interval + init_y]
        upR = [init_x + interval, upper_bound / 100 * interval + init_y]
        lowL = [init_x, lower_bound / 100 * interval + init_y]
        lowR = [init_x + interval, lower_bound / 100 * interval + init_y]

        background_rect = plt.Rectangle((init_x, init_y),
                                        interval,
                                        interval,
                                        fc='lightgrey', alpha=1, zorder=3)
        temp_rect = plt.Rectangle((init_x, init_y), interval, temp / 100 * interval, 
                                  fc='crimson', zorder=3)
        line_up = plt.Line2D((upL[0], upR[0]), (upL[1], upR[1]), 
                             ls='--', color='orange', lw=0.5, zorder=3)
        line_low = plt.Line2D((lowL[0], lowR[0]), (lowL[1], lowR[1]), 
                              ls='--', color='lime', lw=0.5, zorder=4)

        ax.add_patch(background_rect)
        ax.add_patch(temp_rect)
        ax.add_line(line_up)
        ax.add_line(line_low)

        plt.text(init_x + interval * 0.8, init_y + interval * 0.8, "%s" % curr_z, 
                 color='black', fontsize=5, zorder=5)

    def render_heater(self, heater: str) -> tuple:
        ax = self._ax

        curr_h = heater
        vol_ratio = 1

        init_x, init_y = self._canvas_info['heater_init_points'][curr_h]
        interval = self._interval * 2 / 3 * vol_ratio

        temp = self._object_layout['temp-heater'][curr_h]

        background_rect = plt.Rectangle((init_x, init_y), interval, interval,
                                        fc='lightcoral', alpha=1, zorder=3)
        temp_rect = plt.Rectangle((init_x, init_y), interval, temp / 100 * interval, 
                                  fc='crimson', zorder=3)

        ax.add_patch(background_rect)
        ax.add_patch(temp_rect)

        plt.text(init_x + interval * 0.8, init_y + interval * 0.8, "%s" % curr_h, 
                 color='black', fontsize=5, zorder=5)

    def fig2npa(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return data

    def render(self, state) -> np.ndarray:
        self._object_layout = self.build_object_layout()
        self._states = state
        self._canvas_info = self.init_canvas_info()

        canvas_size = self._canvas_info['canvas_size']
        self._fig = plt.figure(figsize=(canvas_size[0] / 10, canvas_size[1] / 10),
                               dpi=200)
        self._ax = plt.gca()

        plt.xlim([-canvas_size[0] / 5, canvas_size[0]])
        plt.ylim([-canvas_size[1] / 5, canvas_size[1]])
        plt.axis('scaled')
        plt.axis('off')

        for z in self._objects['zone']:
            self.render_zone(z)
        for h in self._objects['heater']:
            self.render_heater(h)

        self.render_conn()

        img = self.convert2img(self._fig, self._ax)

        self._ax.cla()
        plt.cla()
        plt.close()

        return img

    def convert2img(self, fig, ax):
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        img = Image.fromarray(data)
        self._data = data
        self._img = img
        return img
