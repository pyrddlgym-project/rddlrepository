import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class PongVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(4, 4),
                 ball_radius=0.02,
                 wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._ball_radius = ball_radius
        self._wait_time = wait_time
        
        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
        self._balls = model.type_to_objects['ball']
        
        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = plt.gca()
        
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        
    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        
        # draw the balls
        balls = []
        for b in self._balls:
            ball = plt.Circle((state[f'ball-x___{b}'], 
                               state[f'ball-y___{b}']), self._ball_radius, 
                              color='red')
            balls.append(ball)
            self.ax.add_patch(ball)
        
        # draw the paddle
        path = Path([(0.99, state['paddle-y']),
                     (0.99, state['paddle-y'] + self._nonfluents['PADDLE-H'])], 
                    [Path.MOVETO, Path.LINETO])
        paddle = patches.PathPatch(path, color='black', lw=3.0)
        self.ax.add_patch(paddle)
        
        self.fig.canvas.draw()        
        img = self.convert2img(self.fig)
        for ball in balls:
            ball.remove()
        del balls
        paddle.remove()
        del paddle
        
        return img

