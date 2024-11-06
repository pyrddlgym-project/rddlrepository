import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle
from matplotlib import collections  as mc
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class IntrudersVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(4, 4),
                 vector_len=0.15,
                 lb=-2, ub=2,
                 wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._vector_len = vector_len
        self._wait_time = wait_time
        
        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = plt.gca()
        
        # draw zones
        FLUENT_SEP = RDDLPlanningModel.FLUENT_SEP
        obj = model.type_to_objects['zone']
        lines = []
        for o in obj:
            l = self._nonfluents['DANGER-L' + FLUENT_SEP + o]
            r = self._nonfluents['DANGER-R' + FLUENT_SEP + o]
            t = self._nonfluents['DANGER-T' + FLUENT_SEP + o]
            b = self._nonfluents['DANGER-B' + FLUENT_SEP + o]
            zone = Rectangle((l, b), r - l, t - b, color='orange', alpha=0.4)
            self.ax.add_patch(zone)
            
        #     lines.append([(l, t), (r, t)])
        #     lines.append([(l, t), (l, b)])
        #     lines.append([(r, t), (r, b)])
        #     lines.append([(l, b), (r, b)])
        # lc = mc.LineCollection(lines, linewidths=2)
        # self.ax.add_collection(lc)
        
    def convert2img(self, canvas):
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))
        return Image.fromarray(data)

    def render(self, state):
        FLUENT_SEP = RDDLPlanningModel.FLUENT_SEP
        
        camera_circs, camera_arrows = [], []
        for camera in self._model.type_to_objects['camera']: 
            x = state['camera-x' + FLUENT_SEP + camera]
            y = state['camera-y' + FLUENT_SEP + camera]
            vx = state['camera-vx' + FLUENT_SEP + camera]
            vy = state['camera-vy' + FLUENT_SEP + camera]            
            circ = plt.Circle((x, y), 
                              self._nonfluents['CAMERA-RADIUS' + FLUENT_SEP + camera],
                              color='green', alpha=0.4)
            camera_circs.append(self.ax.add_patch(circ))
            arrow = Arrow(x, y, vx, vy, width=0.1, color='green')
            camera_arrows.append(self.ax.add_patch(arrow))
            
        intr_circs, intr_arrows = [], []
        for intr in self._model.type_to_objects['intruder']: 
            x = state['intruder-x' + FLUENT_SEP + intr]
            y = state['intruder-y' + FLUENT_SEP + intr]
            vx = state['intruder-vx' + FLUENT_SEP + intr]
            vy = state['intruder-vy' + FLUENT_SEP + intr]            
            circ = plt.Circle((x, y), 0.01, color='red')
            intr_circs.append(self.ax.add_patch(circ))
            arrow = Arrow(x, y, vx, vy, width=0.05, color='red')
            intr_arrows.append(self.ax.add_patch(arrow))
                
        self.fig.canvas.draw()
        img = self.convert2img(self.fig.canvas)   
        
        for obj in camera_circs + camera_arrows + intr_circs + intr_arrows:
            obj.remove()
        del camera_circs, camera_arrows, intr_circs, intr_arrows
        
        return img
