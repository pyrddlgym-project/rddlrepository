import numpy as np
from PIL import Image
import pygame
from pygame import gfxdraw

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


# code comes from openai gym
class AcrobotVisualizer(BaseViz):

    def __init__(self, model: RDDLPlanningModel, screen_dim=500, wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time
        self.screen_dim = screen_dim
    
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
        screen, surf = self.init_canvas((self.screen_dim, self.screen_dim))
        surf.fill((255, 255, 255))
        
        bound = 1 + 1 + 0.2 
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2
        
        # draw the arm
        p1 = [
            -1 * np.cos(state['theta1']) * scale,
            1 * np.sin(state['theta1']) * scale,
        ]

        p2 = [
            p1[0] - 1 * np.cos(state['theta1'] + state['theta2']) * scale,
            p1[1] + 1 * np.sin(state['theta1'] + state['theta2']) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [state['theta1'] - np.pi / 2, state['theta1'] + state['theta2'] - np.pi / 2]
        link_lengths = [1 * scale, 1 * scale]

        pygame.draw.line(
            surf,
            start_pos=(-2.2 * scale + offset, 1 * scale + offset),
            end_pos=(2.2 * scale + offset, 1 * scale + offset),
            color=(0, 0, 0),
        )
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))

            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
        
        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))
        pygame.time.wait(self._wait_time)
        img = self.convert2img(screen)
        del screen, surf        
        
        return img
    
