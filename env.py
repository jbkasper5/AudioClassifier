import pygame
import time
from DL.models import Transformer, CNN
from maze import Maze

# gray is (30, 30, 30)
# brown is (120, 70, 0)
# green is (0, 150, 0)

class Environment:
    def __init__(self, screen, xdim = 5, ydim = 5):
        self.pcoords = (275, 275, 50, 50)
        self.player = pygame.Rect(self.pcoords)
        self.screen = screen
        self.maze = Maze(xdim = xdim, ydim = ydim)
        self.player_index = self.maze.start

    def updateEnvironment(self, direction = None):
        self.connections = self.maze.get_connections(self.player_index)
        directory = ''
        if 'u' in self.connections:
            directory += 'u'
        if 'd' in self.connections:
            directory += 'd'
        if 'l' in self.connections:
            directory += 'l'
        if 'r' in self.connections:
            directory += 'r'
        if direction != None:
            image = pygame.image.load(f'/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/Assets/{directory}/{directory}_{direction}.png')
            self.screen.blit(image, (0, 0))
            pygame.display.update()
            time.sleep(1)
        else:
            image = pygame.image.load(f'/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/Assets/{directory}/{directory}.png')
            pygame.display.update()
            self.screen.blit(image, (0, 0))

    def valid_direction(self, direction) -> bool:
        return direction in self.connections

    def switch_screen(self, direction):
        self.player_index = self.maze.go(index = self.player_index, direction = direction)
        if(self.player_index == -1):
            self.screen.fill((0, 255, 0))
            pygame.display.update()
            time.sleep(5)
            exit(0)
        self.updateEnvironment()

    def updatePlayer(self, delX, delY):
        self.pcoords = (self.pcoords[0] + delX, self.pcoords[1] + delY, self.pcoords[2], self.pcoords[3])
        self.player = pygame.Rect(self.pcoords)