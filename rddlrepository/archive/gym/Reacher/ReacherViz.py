import numpy as np
from PIL import Image
import pygame
from pygame import gfxdraw

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class ReacherVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel, screen_dim=500, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self.screen_dim = screen_dim
        self._nonfluents = model.ground_vars_with_values(model.non_fluents)
    
    def init_canvas(self, figure_size):
        screen = pygame.Surface(figure_size)
        surf = pygame.Surface(figure_size)
        return screen, surf
        
    def convert2img(self, screen):
        data = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), 
                            axes=(1, 0, 2))
        img = Image.fromarray(data)
        return img
    
    def render(self, state):
        segments = self._model.type_to_objects['segment']
        xpos = [state['tip-x___' + s] for s in segments]
        ypos = [state['tip-y___' + s] for s in segments]
        lengths = [self._nonfluents['LENGTH___' + s] for s in segments]
        bound = sum(lengths)
        
        screen, surf = self.init_canvas((self.screen_dim, self.screen_dim))
        surf.fill((255, 255, 255))
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2
        
        # draw the segments
        x, y = 0, 0
        for x2, y2 in zip(xpos, ypos):
            angle = np.atan2(y2 - y, x2 - x)
            l = np.sqrt((x2 - x) ** 2 + (y2 - y) ** 2)
            rod_length = l * scale
            rod_width = 0.05 * scale
            l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for c in coords:
                c = pygame.math.Vector2(c).rotate_rad(angle)
                c = (c[0] + offset + x * scale, c[1] + offset + y * scale)
                transformed_coords.append(c)
            gfxdraw.aapolygon(surf, transformed_coords, (204, 77, 77))
            gfxdraw.filled_polygon(surf, transformed_coords, (204, 77, 77))
            px = int(offset + x * scale)
            py = int(offset + y * scale)
            gfxdraw.aacircle(surf, px, py, int(rod_width / 2), (77, 77, 204))
            gfxdraw.filled_circle(surf, px, py, int(rod_width / 2), (77, 77, 204))
            x, y = x2, y2
        px = int(offset + x * scale)
        py = int(offset + y * scale)
        gfxdraw.aacircle(surf, px, py, int(rod_width / 2), (77, 77, 204))
        gfxdraw.filled_circle(surf, px, py, int(rod_width / 2), (77, 77, 204))
        
        # draw the target
        tx = self._model.non_fluents['TARGET-X']
        ty = self._model.non_fluents['TARGET-Y']
        px = int(offset + tx * scale)
        py = int(offset + ty * scale)

        gfxdraw.aacircle(surf, px, py, int(0.05 * scale / 2), (77, 204, 77))
        gfxdraw.filled_circle(surf, px, py, int(0.05 * scale / 2), (77, 204, 77))
        
        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))
        pygame.time.wait(self._wait_time)
        img = self.convert2img(screen)
        del screen, surf        
        
        return img
    
