import torch
import torch.nn as nn
from trainer import Trainer

# images are (307, 284)

trainer = Trainer(dataset = "transformer_audio", batch_size = 16)
trainer.train(epochs = 8, save_path = "/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/DL/Models/aug_model")
# trainer = Trainer(dataset = "audio", batch_size = 16, load_path = "/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/DL/Models/model")
# trainer.train(epochs = 1, save_path = "/Users/jakekasper/Pitt/Pitt2022-23/Fall/DigitalMedia/MediaProject/DL/Models/model")
