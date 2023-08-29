import pygame
import Pages.gender
import Pages.recorder

class RecordPage:
    def __init__(self, record_info):
        self.buttons = {
            'Back': pygame.Rect(20, 20, 100, 42),
        }
        self.textboxes = {
            "[space] To Record": (100, 100),
            f"Say \"{record_info[0]}\"": (100, 300)
        }
        self.active_button = None
        self.readyToRecord = True
        self.record_info = record_info
        self.recorder = Pages.recorder.Recorder(record_info = record_info)
    
    def handle_event(self, button):
        pass

    def getNextPage(self):
        newPage = Pages.gender.GenderPage(self.record_info[0])
        return newPage