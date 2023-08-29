import pygame
import Pages.home
import Pages.record as record

class GenderPage:
    def __init__(self, direction):
        self.buttons = {
            'Back': pygame.Rect(20, 20, 100, 42),
            'Male': pygame.Rect(125, 240, 350, 75),
            'Female': pygame.Rect(125, 340, 350, 75),
        }
        self.textboxes = {
            'Voice Gender': (115, 180) # h: 42, w: 370
        }
        self.active_button = None
        self.readyToRecord = False
        self.direction = direction
    
    def handle_event(self, button):
        self.gender = button

    def getNextPage(self):
        if self.gender.lower() == 'back':
            newPage = Pages.home.HomePage()
        else:
            newPage = record.RecordPage(record_info = (self.direction, self.gender))
        return newPage