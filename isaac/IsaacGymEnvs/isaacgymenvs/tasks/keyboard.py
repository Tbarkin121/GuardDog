import pygame
import numpy as np
import torch

class Keyboard():
    def __init__(self, num_actions=1):
        pygame.quit()
        # Initialize Pygame
        pygame.init()
        # Set up the screen
        screen = pygame.display.set_mode((400, 300))
        self.num_actions = num_actions

    def refresh(self):
        pygame.event.pump()

    def get_keys(self):
        pygame.event.pump()

        # Check for arrow key presses
        a = torch.zeros(self.num_actions)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            a[0] = 1.0
        if keys[pygame.K_DOWN]:
            a[0] = -1.0
        if keys[pygame.K_LEFT]:
            a[1] = 1.0
        if keys[pygame.K_RIGHT]:
            a[1] = -1.0
        if keys[pygame.K_a]:
            a[2] = 1.0
        if keys[pygame.K_d]:
            a[2] = -1.0

        return a

                
