
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, data_loader, epochs, device,
                 valid_data_loader=None, class_weights=None):

        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.optimizer = optimizer
        self.class_weights = torch.tensor(class_weights)
        self.device = device




    def train_epoch(self):
        """
        Training logic for an epoch
        """
        batch_id = 0
        losses = []

        for (features, target) in tqdm(self.trainloader):
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            loss = self.criterion(predictions, target, self.class_weights, self.device)

            losses.append(float(loss))

            loss.backward()
            self.optimizer.step()
            

        if self.do_validation:
            with torch.no_grad():
                for (features, target) in (self.valid_data_loader):
                    predictions = self.model(features)
                    predicted_classes = torch.argmax(predictions, dim=1)
                    accuracy = int(torch.sum(predicted_classes==target))/len(predicted_classes)
                    loss = float(self.criterion(predictions, target, self.class_weights, self.device))
                    
                    print('average train loss: ', np.mean(losses))
                    print('validation loss: ', loss)
                    print('validation accuracy: ', accuracy)

        return(loss, accuracy)