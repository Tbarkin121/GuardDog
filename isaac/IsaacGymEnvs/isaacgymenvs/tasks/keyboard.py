import pygame
import numpy as np

class Keyboard():
    def __init__(self):
        pygame.quit()
        # Initialize Pygame
        pygame.init()
        # Set up the screen
        screen = pygame.display.set_mode((400, 300))

    def refresh(self):
        pygame.event.pump()

    def get_keys(self):
        pygame.event.pump()

        # Check for arrow key presses
        a = [0.0]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            a[0] = 1.0
        if keys[pygame.K_RIGHT]:
            a[0] = -1.0

        return a

                
