import ClassificationModel.BERTClassification as BERT
import os 
import ClassificationModel.Dataset as ASAP
from sklearn.model_selection import KFold

EPOCHS = 10
LEARNING_RATE = 3e-05
BATCH_SIZE = 16
DROPOUT = 0.5
WEIGHT_DECAY = 0.001

def trainModel_KFold(df):
    X = df['essay']
    Y = df['traitAverages']

    kf = KFold(n_splits=5, shuffle=True, random_state=47)
    kf.get_n_splits(X, Y)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        #Saves testing essays to file (so they can be used for demo)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        with open(f'x_test_fold_{i}.txt', 'w') as f:
            for essay in X_test:
                f.write(essay.strip() + '\n\n')
                
        training_loader, testing_loader = ASAP.getDataloadersWeightedKFold(df, train_index, test_index, weights)

        BERT.train_model(EPOCHS, LEARNING_RATE, DROPOUT, WEIGHT_DECAY, BATCH_SIZE, training_loader, testing_loader, weights)

if __name__ == "__main__":
    #Sets up the data
    df = ASAP.getBaseDataframe()
    weights = ASAP.findWeights(df)

    trainModel_KFold(df)