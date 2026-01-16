import os

from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch

from ClassificationModel.Dataset import getBaseDataframe, findWeights, getDataloaders
from ClassificationModel.BERTClassification import train_model_tune

#Sets up the data
df = getBaseDataframe()
weights = findWeights(df)
training_loader, validation_loader, testing_loader = getDataloaders(df, weights)

#Configures hyperparameters for grid search
config = {
    'epochs': 10,  
    'learning_rate': tune.uniform(1e-5, 5e-5),
    'batch_size': 16,
    'dropout': tune.uniform(0.0, 0.5),
    'weight_decay': tune.uniform(1e-4, 1e-1)
}

if __name__ == "__main__":
    bayes_search = BayesOptSearch(metric="qwk", mode="max")  #Optimise for highest qwk

    #Performs hyperparameter tuning
    tuner = tune.Tuner(
        tune.with_parameters(train_model_tune, 
                            training_loader=training_loader, 
                            validation_loader=validation_loader, 
                            testing_loader=testing_loader,
                            weights=weights),
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=15, 
            metric="qwk", 
            mode="max", 
            search_alg=bayes_search,
            max_concurrent_trials=1  #Only run one trial at a time due to computational limitations
        )
    )
    tuner.fit()
