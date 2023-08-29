import pygame
import torch
from data import Recorder, Spectrogramer

color_active = pygame.Color(75, 75, 75)
color_passive = pygame.Color(150, 150, 150)

fonts = pygame.font.get_fonts()
font = fonts[55]

class Environment:
    def __init__(self, screen):
        self.screen = screen
        self.base_font = pygame.font.SysFont(fonts[55], 42)
        self.recorder = Recorder()
        self.spectrogramer = Spectrogramer()
        self.textboxes = {
            "[space] To Record": (100, 275)
        }

    def update_environment(self, color = (0, 0, 0)):
        self.render_page(color)

    def render_page(self, color):
        self.screen.fill(color)
        self.render_text()

    def render_text(self):
        if self.textboxes:
            for text in self.textboxes:
                surface = self.base_font.render(text, True, (255, 255, 255))
                x = (600 - surface.get_width()) / 2
                self.screen.blit(surface, (x, self.textboxes[text][1]))

    def validate(self):
        self.recorder.record()
        model = torch.load("/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/DL/Models/model")
        spectrogram = self.spectrogramer.create_spectrogram()
        out = model(spectrogram)
        print(f"Raw output: {out.tolist()}")
        pred = torch.argmax(out)
        print(pred.item())
        outs = ["up", "down", "left", "right"]
        print(f"The prediction is: {outs[pred]}")


