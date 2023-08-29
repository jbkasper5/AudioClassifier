import torch
import torch.nn as nn
import models
import torch.optim as optim
import time
import dataloaders
import math

class Trainer:
    def __init__(self, dataset, batch_size, load_path = None, **kwargs):
        # self.model = models.Transformer(n_encoders = 6, n_decoders = 6, dmodel = 512, vocab_size = 256)
        if load_path == None:
            if dataset == "cnn_audio":
                self.model = models.CNN(kwargs)
            elif dataset == "transformer_audio":
                # self.model = models.Transformer(**kwargs)
                self.model = models.Transformer(**kwargs)
            else:
                self.model = models.CIFAR_CNN()
        else:
            self.model = torch.load(load_path)
            self.trainloader = self.model.trainloader
            self.testloader = self.model.testloader

    def train(self, epochs, save_path = None):
        criterion = nn.NLLLoss()
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr = 0.001, momentum = 0.9)
        print_size = math.floor(len(self.model.trainloader) / 10)
        print(f"Training on {len(self.model.trainloader)} batches for {epochs} epochs, reports every {print_size} batches...")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            start = time.time()
            epoch_loss = 0
            total = 0
            correct = 0
            for i, data in enumerate(self.model.trainloader):
                inputs, labels = data
                optimizer.zero_grad()
                # outputs = self.model(inputs)
                outputs = self.model(inputs, labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                total += labels.size(0)
                correct += torch.sum(predictions == labels).item()
                if i % print_size == 0 and i != 0:
                    print(f'{i}/{len(self.trainloader)} | loss: {epoch_loss / total:.3f} accuracy: {100.*(correct / total):.3f}%')

            print(f"Epoch {epoch + 1} time: {time.time() - start:.3f} seconds | loss: {epoch_loss / total:.3f} | accuracy: {100.*correct/total:.3f}%\n")
            epoch_loss = 0
        if save_path != None:
            torch.save(self.model, save_path)
        print("Training complete.")


    def test(self):
        pass

    def load(self, path):
        pass