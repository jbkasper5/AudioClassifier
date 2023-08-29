from Pages.home import HomePage
import pygame

color_active = pygame.Color(75, 75, 75)
color_passive = pygame.Color(150, 150, 150)

fonts = pygame.font.get_fonts()
font = fonts[55]

class Environment:
    def __init__(self, screen):
        self.screen = screen
        self.page = HomePage()
        self.base_font = pygame.font.SysFont(fonts[55], 42)

    def update_environment(self, color = (0, 0, 0)):
        self.render_page(color)

    def render_page(self, color):
        self.screen.fill(color)
        for btn_name in self.page.buttons:
            if btn_name == self.page.active_button:
                pygame.draw.rect(self.screen, color_active, self.page.buttons[btn_name], border_radius = 5)
            else:
                pygame.draw.rect(self.screen, color_passive, self.page.buttons[btn_name], border_radius = 5)
        self.render_text()

    def render_text(self):
        for btn_name in self.page.buttons:
            surface = self.base_font.render(btn_name, True, (255, 255, 255))
            y = self.page.buttons[btn_name].y + ((self.page.buttons[btn_name].h - surface.get_height()) / 2) + 2
            x = self.page.buttons[btn_name].x + ((self.page.buttons[btn_name].w - surface.get_width()) / 2)
            self.screen.blit(surface, (x, y))

        if self.page.textboxes:
            for text in self.page.textboxes:
                surface = self.base_font.render(text, True, (255, 255, 255))
                x = (600 - surface.get_width()) / 2
                self.screen.blit(surface, (x, self.page.textboxes[text][1]))

    def handle_event(self, button):
        self.page.handle_event(button = button)
        self.page = self.page.getNextPage()