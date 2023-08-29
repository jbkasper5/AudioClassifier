# import sys module
import pygame
import sys
import env


# pygame.init() will initialize all
# imported module
pygame.init()

clock = pygame.time.Clock()

# it will display on screen
screen = pygame.display.set_mode([600, 600])

environment = env.Environment(screen = screen)

while True:
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if pygame.key.get_pressed()[pygame.K_SPACE]:
                environment.update_environment((255, 0, 0))
                pygame.display.update()
                environment.validate()
                environment.update_environment((0, 0, 0))

    environment.update_environment()

    pygame.display.update()
	
	# clock.tick(60) means that for every second at most
	# 60 frames should be passed.
	# clock.tick(60)
