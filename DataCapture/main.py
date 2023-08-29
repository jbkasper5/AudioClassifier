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

color = (0, 255, 0)
hover = False
active = False

while True:
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        hover = False
        for btn_name in environment.page.buttons:
            if environment.page.buttons[btn_name].collidepoint(pygame.mouse.get_pos()):
                btn_active = btn_name
                hover = True
                environment.page.active_button = btn_active
            else:
                pass

        if not hover:
            environment.page.active_button = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            for btn_name in environment.page.buttons:
                if environment.page.buttons[btn_name].collidepoint(event.pos):
                    active = True
                    btn_active = btn_name
                    environment.page.active_button = btn_active
                    break
                else:
                    active = False

        if event.type == pygame.MOUSEBUTTONUP:
            if active:
                environment.handle_event(button = btn_active)
            else:
                pass
            active = False
            btn_active_idx = None

        if event.type == pygame.KEYDOWN and environment.page.readyToRecord:
            if pygame.key.get_pressed()[pygame.K_SPACE]:
                color = (255, 0, 0)
                environment.update_environment(color)
                pygame.display.update()
                environment.page.recorder.record()

    environment.update_environment()

    pygame.display.update()
	
	# clock.tick(60) means that for every second at most
	# 60 frames should be passed.
	# clock.tick(60)
