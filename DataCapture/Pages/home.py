import pygame
import Pages.gender

class HomePage:
    def __init__(self):
        self.buttons = {
            'Up': pygame.Rect(90, 160, 200, 125),
            'Down': pygame.Rect(310, 160, 200, 125),
            'Left': pygame.Rect(90, 305, 200, 125),
            'Right': pygame.Rect(310, 305, 200, 125)
        }
        self.textboxes = {
            'Choose Direction:': (115, 100) # h: 42, w: 370
        }
        self.active_button = None
        self.readyToRecord = False
    
    def handle_event(self, button):
        self.direction = button

    def getNextPage(self):
        newPage = Pages.gender.GenderPage(direction = self.direction)
        return newPage