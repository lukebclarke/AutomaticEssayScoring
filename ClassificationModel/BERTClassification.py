import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ray import tune

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator

import torch
from torch import cuda
from torchmetrics.classification import MulticlassCohenKappa

from .EarlyStopping import EarlyStopping
from .BERTModel import CustomBERTModel

EARLY_STOP_PATIENCE = 3
EARLY_STOP_DELTA = 0.01

NUM_TRAITS = 4
NUM_CLASSES_PER_TRAIT = 4
TRAITS = ['Ideas', 'Organization', 'Style', 'Conventions']

class BertClassifier():
    """Defines the BERT model that will be used for training/predicting """

    def __init__(self, weights=None, epochs=15, train_batch_size=16, val_batch_size=8, learning_rate=5e-05, max_len=512, modelPath=None, train_size=0.8, dropout=0.3, weight_decay=0.01):
        """Initialses the BERT model

        Args:
            weights ([float], optional): A list of weights that are used to calculate the loss. Defaults to None.
            epochs (int, optional): The number of epochs during training. Defaults to EPOCHS.
            train_batch_size (int, optional): The size of each batch during training. Defaults to TRAIN_BATCH_SIZE.
            val_batch_size (int, optional): The size of each batch during validation. Defaults to VALID_BATCH_SIZE.
            learning_rate (int, optional): The learning rate of the model. Defaults to LEARNING_RATE.
            max_len (int, optional): The maximum length of an input. Defaults to MAX_LEN.
            modelPath (str, optional): The path of a pre-trained model to be loaded. Defaults to None.
            train_size (float, optional): The percentage of data (as a decimal) used for training the data. Defaults to TRAINING_SPLIT.
        """
        
        #Initialses variables
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.learning_rate = learning_rate
        self.max_len = max_len
        self.train_size = train_size
        self.weights = weights

        main_directory = os.path.dirname((os.path.abspath(__file__))) 

        self.model = CustomBERTModel(num_traits=NUM_TRAITS, num_classes=NUM_CLASSES_PER_TRAIT, dropout=dropout)

        #Setting up the device for GPU usage
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        #Load file if a previous model has been saved
        if modelPath != None: 
            state_dict = torch.load(modelPath, map_location=torch.device(self.device))
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Loaded model.")
        else: #If training, model folder needs to be created
            model_folder = os.path.join(main_directory, "Models")
            directory_name = time.strftime("%Y%m%d-%H%M%S")

            self.model_folder = os.path.join(model_folder, directory_name)
            

            if not os.path.exists(self.model_folder):
                os.makedirs(self.model_folder)

        self.model.to(self.device) #Send to GPU

        #Sets up optimizer
        self.optimizer = torch.optim.AdamW(params = self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        #Defines early stopping parameters
        self.early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, delta=EARLY_STOP_DELTA)

        #Tracks the current best validaton loss so checkpoints can be made
        self.best_val_loss = 100000000 

        #Used for producing graphs at the end
        self.avg_training_loss = []
        self.totalLosses = []
        self.accuracies = []
        self.f1_score_macros = []
        self.f1_score_micros = []
        self.val_losses = []
        self.trait_losses = [[] for _ in range(4)]
        self.qwk_scores = [[] for _ in range(5)]
        self.fileOutput = []

    def fit(self, training_loader, validation_loader):
        """Used to train the model

        Args:
            training_loader (DataLoader): The DataLoader with the training samples
            validation_loader (DataLoader): The DataLoader with the validation samples
        """
        for epoch in range(self.epochs):
            self.model.train() #Set to training mode

            losses = [] #Used for graphing

            #Data is processed in batches
            for _, data in enumerate(training_loader, 0): 

                #Moves the tensors to the appropiate device and ensures correct format
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.long)

                #Makes predictions for batch
                outputs = self.model(ids, mask)

                #Computes loss  
                loss = self.loss_fn(outputs, targets)
                loss.backward()

                #Checks progress every 10% of batches
                interval = len(training_loader) // 10
                if _%interval == 0: 
                    lossInt = loss.item()
                    print(f'Epoch: {epoch}, Loss:  {lossInt}')
                    losses.append(lossInt)

                self.optimizer.step() 

                #Clears gradients ready for the next batch
                self.optimizer.zero_grad()

            #Finds the average loss during training across the epoch
            try:
                lossMean = sum(losses) / len(losses)
            except ZeroDivisionError:
                print("No samples trained")
        
            self.avg_training_loss.append(lossMean)

            #Validate the epoch
            val_loss = self.validateEpoch(validation_loader)

            #Make a checkpoint if model peforms better than previous best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_name = "best_checkpoint.pth"

                #Saves the checkpoint
                try:
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss
                    }, file_name)
                    print(f"Checkpoint successfully saved at {file_name}")
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")

            #Stops early if the model maintains a consistent loss
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.stop:
                print("Early stopping")
                break

    def predict(self, testing_loader):
        """Predicts the scores for the samples in a given DataLoader. Used when no targets are available.

        Args:
            testing_loader (DataLoader): A DataLoader containing the samples to predict

        Returns:
            Tensor: A tensor containing all the final score predictions for each sample
        """

        self.model.eval() #Set to evaluation mode

        fin_predictions=[]

        with torch.no_grad(): #Model parameters are not updated within evaluation process
            for _, data in enumerate(testing_loader, 0): #Data is processed in batches within validation set

                #Moves the tensors to the appropiate device and ensures correct format
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                                
                #Makes predictions for batch
                outputs = self.model(ids, mask)

                #Finds the actual model predictions (the most probable class)
                predictions = torch.argmax(outputs, dim=2)
                fin_predictions.extend(predictions.cpu().detach().numpy().tolist())
            
        return fin_predictions

    def predictWithTargets(self, validation_loader, loss=True):
        """Predicts scores and calculates the loss for each sample within a DataLoader

        Args:
            validation_loader (DataLoader): The DataLoader containing the samples to be validated

        Raises:
            ZeroDivisionError: Occurs when no samples are present to predict

        Returns:
            [int]: A list of all the predicted scores
            [int]: A list of all the actual scores
            int: The validation loss across the samples
        """

        self.model.eval() #Set to evaluation mode

        targets_list=[]
        predictions_list=[]

        val_losses = []
        trait_losses = [[], [], [], []]
        avg_trait_losses = [0, 0, 0, 0]

        avg_val_loss = 0

        with torch.no_grad(): #Model parameters are not updated within evaluation process
            for _, data in enumerate(validation_loader, 0): #Data is processed in batches within validation set

                #Moves the tensors to the appropiate device and ensures correct format
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.long)

                #Makes predictions for batch
                outputs = self.model(ids, mask)

                predictions = torch.argmax(outputs, dim=2)

                #Store predictions and targets
                targets_list.extend(targets.cpu().detach().numpy().tolist())
                predictions_list.extend(predictions.cpu().detach().numpy().tolist())

                #Finds losses
                if (loss == True):
                    trait_loss, val_loss = self.calculate_loss(outputs, targets)

                    val_losses.append(val_loss)
                    for i in range(len(trait_loss)):
                        trait_losses[i].append(trait_loss[i])
    
        #Flatten the lists for validation  
        X_val_flat = [item for sublist in predictions_list for item in sublist] 
        Y_val_flat = [item for sublist in targets_list for item in sublist]

        #Finds the validation loss across the DataLoader
        if (loss == True):
            try:
                avg_val_loss = sum(val_losses) / len(val_losses)
            except ZeroDivisionError:
                print("No samples validated")
                raise ZeroDivisionError
            
            #Finds the validation trait losses across the DataLoader
            try:
                for j in range(len(trait_losses)):
                    avg_trait_losses[j] = sum(trait_losses[j]) / len(trait_losses[j])
            except ZeroDivisionError:
                print("No samples validated")
                raise ZeroDivisionError
            
            return X_val_flat, Y_val_flat, avg_val_loss, avg_trait_losses
        else:
            return X_val_flat, Y_val_flat

    def validateEpoch(self, validation_loader):
        """Evaluates the performance of an epoch

        Args:
            validation_loader (DataLoader): The DataLoader containing the samples to be validated

        Returns:
            int: The validation loss across the samples
        """
        X, Y, loss, traitLosses = self.predictWithTargets(validation_loader)

        #Finds evaluation metrics 
        accuracy = metrics.accuracy_score(Y, X)
        f1_score_micro = metrics.f1_score(Y, X, average='micro')
        f1_score_macro = metrics.f1_score(Y, X, average='macro')
        
        qwks = self.calculate_qwk(X, Y)
        avg_qwk = sum(qwks) / len(qwks)

        #Used for plotting graphs
        self.accuracies.append(accuracy)
        self.f1_score_micros.append(f1_score_micro)
        self.f1_score_macros.append(f1_score_macro)
        self.val_losses.append(loss)

        #Add QWK scores
        for i in range(len(qwks)):
            self.qwk_scores[i].append(qwks[i])

        #Add Trait Losses
        for i in range(len(traitLosses)):
            self.trait_losses[i].append(traitLosses[i])

        #Add average QWK score
        self.qwk_scores[NUM_TRAITS].append(avg_qwk)

        #Evaluation metrics will be written to file
        print("Epoch:" + str((len(self.accuracies)-1)))
        self.fileOutput.append(f"Epoch: {(len(self.accuracies)-1)}")
        print(f"Accuracy Score = {accuracy}")
        self.fileOutput.append(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        self.fileOutput.append(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        self.fileOutput.append(f"F1 Score (Macro) = {f1_score_macro}")

        for i in range(len(qwks)):
            print(f"Qudratic Weighted Kappa for {TRAITS[i]} Trait = {qwks[i]}")
            self.fileOutput.append(f"Qudratic Weighted Kappa for {TRAITS[i]} Trait = {qwks[i]}")

        print(f"Average Qudratic Weighted Kappa = {avg_qwk}")
        self.fileOutput.append(f"Average Qudratic Weighted Kappa = {avg_qwk}")

        print(f"Training Loss = {self.avg_training_loss[-1]}")
        self.fileOutput.append(f"Training Loss = {self.avg_training_loss[-1]}")
        print(f"Validation Loss = {loss}")
        self.fileOutput.append(f"Validation Loss = {loss}")
        print(f"Trait Losses:")
        self.fileOutput.append(f"Trait Losses:")
        for t in range(len(traitLosses)):
            print(f"{TRAITS[t]} Loss = {traitLosses[t]}")
            self.fileOutput.append(f"{TRAITS[t]} Loss = {traitLosses[t]}")
        print("")
        self.fileOutput.append("")

        return loss
    
    def score(self, validation_loader):
        X, Y, loss, trait_losses = self.predictWithTargets(validation_loader)

        Xarray = np.array(X)
        Yarray = np.array(Y)

        num_samples = len(X) // NUM_TRAITS
        Xarray = Xarray.reshape(num_samples, NUM_TRAITS)
        Yarray = Yarray.reshape(num_samples, NUM_TRAITS)

        #Generates a classification report for each trait
        for i in range(NUM_TRAITS):
            scores = [0, 1, 2, 3]
            self.fileOutput.append("\nTrait " + str(i) + " Classification Report:")
            self.fileOutput.append(classification_report(Yarray[:,i], Xarray[:,i], labels=scores, zero_division=0))

        qwks = self.calculate_qwk(X, Y)
        #Ensures it is a float rather than tensor
        avg_qwk = (sum(qwks) / len(qwks)).item()

        #The QWK is the most relevant metric for hyperparameter tuning here
        return avg_qwk
    
    def scoreMultipleMetrics(self, validation_loader):
        X, Y = self.predictWithTargets(validation_loader, loss=False)

        Xarray = np.array(X)
        Yarray = np.array(Y)

        for i in range(len(Yarray)):
            print(f"X: {Xarray[i]}, Y: {Yarray[i]}")

        num_samples = len(X) // NUM_TRAITS
        Xarray = Xarray.reshape(num_samples, NUM_TRAITS)
        Yarray = Yarray.reshape(num_samples, NUM_TRAITS)

        #Generates a classification report for each trait
        for i in range(NUM_TRAITS):
            scores = [0, 1, 2, 3]
            print("\nTrait " + str(i) + " Classification Report:")
            print(classification_report(Yarray[:,i], Xarray[:,i], labels=scores, zero_division=0))

        #Finds evaluation metrics 
        accuracy = metrics.accuracy_score(Y, X)
        print(f"\nAccuracy: {accuracy}")
        f1_score_micro = metrics.f1_score(Y, X, average='micro')
        print(f"F1-Score Micro: {f1_score_micro}")
        f1_score_macro = metrics.f1_score(Y, X, average='macro')
        print(f"F1-Score Macro: {f1_score_macro}")

        qwks = self.calculate_qwk(X, Y)
        avg_qwk = sum(qwks) / len(qwks)
        print(f"Average QWK: {avg_qwk}")

        #The QWK is the most relevant metric for hyperparameter tuning here
        return accuracy, f1_score_micro, f1_score_macro, avg_qwk

    def calculate_qwk(self, flat_outputs, flat_targets):
        #Ensures targets and outputs are a tensor
        if isinstance(flat_targets, list):
            flat_targets = torch.tensor(flat_targets)
        if isinstance(flat_outputs, list):
            flat_outputs = torch.tensor(flat_outputs)
            
        qwks = []

        #Finds QWK for each trait category
        for i in range(NUM_TRAITS):
            trait_X = flat_outputs[::NUM_TRAITS+i]
            trait_Y = flat_targets[::NUM_TRAITS+i]

            qwk = MulticlassCohenKappa(num_classes=4, weights='quadratic')(trait_X, trait_Y)
            qwks.append(qwk)

        return qwks 
    
    def loss_fn(self, outputs, targets):
        """A weighted loss function that only returns total loss (used for training)

        Args:
            outputs (torch.tensor): The model's outputs for a batch
            targets (torch.tensor): The correct labels for the batch

        Returns:
            torch.tensor: The final loss for the batch
        """
        trait_losses, total_loss = self.calculate_loss(outputs, targets)

        return total_loss
    
    def calculate_loss(self, outputs, targets):
        """A weighted loss function that utilises Cross-Entropy loss

        Args:
            outputs (torch.tensor): The model's outputs for a batch
            targets (torch.tensor): The correct labels for the batch

        Returns:
            [torch.tensor]: The final losses for each trait
            torch.tensor: The final average loss
        """
        num_samples, num_traits = targets.shape
        total_loss = 0
        trait_losses = [0, 0, 0, 0]

        #Converts weights grid to tensor so it can interact with loss function
        weights = torch.tensor(self.weights, dtype=torch.float).to(outputs.device) 

        #Finds loss for each head
        for traitIndex in range(num_traits): 
            #Finds the weight for classes in the head
            traitWeights = weights[traitIndex]

            #Defines loss function and sets weight
            criterion = torch.nn.CrossEntropyLoss(weight=traitWeights) 

            head_outputs = outputs[:, traitIndex, :]
            head_targets = targets[:, traitIndex]

            loss = criterion(head_outputs, head_targets)

            total_loss += loss
            trait_losses[traitIndex] = loss

        #Find average loss over the heads
        total_loss = total_loss / NUM_TRAITS
        
        return trait_losses, total_loss
    
    def report(self):
        """Writes the metrics tracked during training to a results file, and sets up graphs that can be plotted later"""
        #Writing the training losses and evaluation metrics file
        fileName = f"{self.epochs}epochs{self.learning_rate:.2f}learningRate_Results.txt"
        path = os.path.join(self.model_folder, fileName)

        with open(path, "w") as f:
            for line in self.fileOutput:
                f.write(line + "\n")

    def plot_graphs(self):
        plt.plot(self.avg_training_loss, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        title = f"Loss per Epoch"
        filepath = os.path.join(self.model_folder, f"{title}.png")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(bottom=0)
        plt.legend()
        plt.savefig(filepath)
        plt.close()
        plt.clf()

        plt.plot(self.accuracies)
        title = f"Accuracy per Epoch"
        filepath = os.path.join(self.model_folder, f"{title}.png")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(bottom=0, top=1)
        plt.savefig(filepath)
        plt.close()
        plt.clf()

        plt.plot(self.f1_score_macros)
        title = "Macro F1-Score per Epoch"
        filepath = os.path.join(self.model_folder, f"{title}.png")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1-Score")
        plt.ylim(bottom=0, top=1)
        plt.savefig(filepath)
        plt.close()
        plt.clf()

        for i in range(len(TRAITS)):
            plt.plot(self.qwk_scores[i])
            title = f"QWK per Epoch ({TRAITS[i]} Trait)"
            filepath = os.path.join(self.model_folder, f"{title}.png")
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel("QWK")
            plt.ylim(bottom=0, top=1)
            plt.savefig(filepath)
            plt.close()
            plt.clf()

        for i in range(len(TRAITS)):
            plt.plot(self.trait_losses[i])
            title = f"Loss per Epoch ({TRAITS[i]} Trait)"
            filepath = os.path.join(self.model_folder, f"{title}.png")
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.ylim(bottom=0, top=1)
            plt.savefig(filepath)
            plt.close()
            plt.clf()

        plt.plot(self.qwk_scores[NUM_TRAITS])
        title = f"Average QWK per Epoch"
        filepath = os.path.join(self.model_folder, f"{title}.png")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("QWK")
        plt.ylim(bottom=0, top=1)
        plt.savefig(filepath)
        plt.close()
        plt.clf()

    def saveBestModel(self):
        """Saves the model that has the best performance"""

        #Load checkpoint
        main_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(main_dir, "best_checkpoint.pth")
        torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)  

        modelName = "model.pth" 

        path = os.path.join(self.model_folder, modelName)
        torch.save(self.model.state_dict(), path)

def train_model_tune(config, training_loader, validation_loader, testing_loader, weights):
    """Trains a model with a specified configuration (used with RayTune for hyperparameter tuning)

    Args:
        config (dict): Dictionary containing the settings for the model
        training_loader (DataLoader): The DataLoader with the training samples
        validation_loader (DataLoader): The DataLoader with the validation samples
        testing_loader (DataLoader): The DataLoader with the testing samples
        weights ([float]): A list of weights that are used to resolve the imbalance in the dataset
    """

    #Moves to main directory so results can be easily accessed
    main_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(main_dir)

    #Creates the model from the specified configuration
    model = BertClassifier(weights=weights, epochs=config['epochs'], learning_rate=config['learning_rate'], train_batch_size=config['batch_size'], val_batch_size=config['batch_size'])

    #Train and evaluate the model
    model.fit(training_loader, validation_loader)
    qwk = model.score(validation_loader)

    #Report results to RayTune
    tune.report({"qwk": qwk})

    #Saves the hyperparameters and the model to its folder
    with open(os.path.join(model.model_folder, "hyperparameters.txt"), "w") as file:
        for key, value in config.items():
            file.write(f"{key}: {value}\n")

    model.report()
    
    model.saveBestModel()

def train_model(epochs, learning_rate, dropout, weight_decay, batch_size, training_loader, testing_loader, weights):
    """Trains a model on specified parameters

    Args:
        epochs (int): The number of epochs to train for
        learning_rate (float): The learning rate of the model
        dropout (float): The dropout of the model
        weight_decay (float): The decay rate of weights
        batch_size (int): The batch size of the model
        training_loader (DataLoader): The DataLoader with the training samples
        validation_loader (DataLoader): The DataLoader with the validation samples
        testing_loader (DataLoader): The DataLoader with the testing samples
        weights ([float]): A list of weights that are used to resolve the imbalance in the dataset
    """
    #Creates model
    model = BertClassifier(weights=weights, epochs=epochs, learning_rate=learning_rate, train_batch_size=batch_size, val_batch_size=batch_size, dropout=dropout, weight_decay=weight_decay)

    #Saves the hyperparameters and the model to its folder
    with open(os.path.join(model.model_folder, "hyperparameters.txt"), "w") as file:
        file.write(f"Epochs: {epochs}")
        file.write(f"Learning Rate: {learning_rate}")
        file.write(f"Batch size: {batch_size}")
        file.write(f"Dropout: {dropout}")
        file.write(f"Weight Decay: {weight_decay}")

    #Trains + reports on model
    model.fit(training_loader, testing_loader)
    model.report()
    model.plot_graphs()

    #Saves model
    model.saveBestModel()

