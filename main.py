import pygame
import sys
from env import Environment
from processor import Processor

pygame.init()

xdim = int(input("Length of the maze: "))
ydim = int(input("Height of the maze: "))

screen = pygame.display.set_mode([600, 600])

processor = Processor()

env = Environment(screen, xdim = xdim, ydim = ydim)
env.updateEnvironment()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit(0)
        
        if event.type == pygame.KEYDOWN:
            # if event.key == pygame.K_UP:
            #     if env.valid_direction(direction = 'u'):
            #         env.updateEnvironment(direction = 'u')
            #         env.switch_screen(direction = 'u')
            # elif event.key == pygame.K_DOWN:
                # if env.valid_direction(direction = 'd'):
                #     env.updateEnvironment(direction = 'd')
                #     env.switch_screen(direction = 'd')
            # elif event.key == pygame.K_LEFT:
                # if env.valid_direction(direction = 'l'):
                #     env.updateEnvironment(direction = 'l')
                #     env.switch_screen(direction = 'l')
            # elif event.key == pygame.K_RIGHT:
                # if env.valid_direction(direction = 'r'):
                #     env.updateEnvironment(direction = 'r')
                #     env.switch_screen(direction = 'r')
            if event.key == pygame.K_SPACE:
                choice = processor.predict()
                if choice == 0:
                    if env.valid_direction(direction = 'u'):
                        env.updateEnvironment(direction = 'u')
                        env.switch_screen(direction = 'u')
                elif choice == 1:
                    if env.valid_direction(direction = 'd'):
                        env.updateEnvironment(direction = 'd')
                        env.switch_screen(direction = 'd')
                elif choice == 2:
                    if env.valid_direction(direction = 'l'):
                        env.updateEnvironment(direction = 'l')
                        env.switch_screen(direction = 'l')
                else:
                    if env.valid_direction(direction = 'r'):
                        env.updateEnvironment(direction = 'r')
                        env.switch_screen(direction = 'r')

        
		
    pygame.display.update()
	
	# clock.tick(60) means that for every second at most
	# 60 frames should be passed.
	# clock.tick(60)
