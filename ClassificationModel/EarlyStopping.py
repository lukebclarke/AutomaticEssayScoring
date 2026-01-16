class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """Early Stopping class stops the model from training any further epochs if the (validation) loss sees no improvement

        Args:
            patience (int, optional): If there is no improvement, this is how many epochs the model waits until stopping. Defaults to 5.
            delta (int, optional): The minimum change needed in the loss function to qualify as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.delta = delta
        self.bestLoss = None
        self.stop = False
        self.similarEpochs = 0 #Used to track how many epochs in a row have not shown improvement
        self.bestModelState = None #Best model (can be loaded later)

    def __call__(self, loss, model):
        """Checks whether the current epoch should cause the model to stop

        Args:
            loss (float): The validation loss for the epoch
            model (dict): The state dictionary mapping each layer of the current model to its parameter tensor
        """
        #The first epoch sets the best score to beat
        if self.bestLoss is None:
            self.bestLoss = loss
            self.bestModelState = model.state_dict() 
        #If the score is not a notable improvement, 
        elif loss > (self.bestLoss - self.delta): 
            self.similarEpochs += 1
            if self.similarEpochs >= self.patience: #And there has not been a notable improvement in a while, time to early stop
                self.stop = True
        #If the score is continuing to improve
        else: 
            self.bestLoss = loss 
            self.bestModelState = model.state_dict() #This (best) model is then saved so can be loaded later
            self.similarEpochs = 0 

    def load_best_model(self, model):
        """Loads the model that has resulted in the best (smallest) loss

        Args:
            model (dict): The state dictionary mapping each layer of the best model to its parameter tensor
        """
        model.load_state_dict(self.best_model_state)