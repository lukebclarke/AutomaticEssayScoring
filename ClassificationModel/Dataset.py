import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizerFast
import numpy as np

MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TRAINING_SPLIT = 0.8
VALIDATION_SPLIT = 0.6

NUM_TRAITS = 4
NUM_CLASSES_PER_TRAIT = 4

class CustomDataset(Dataset):
    """
    Preprocesses the dataset to be used with BERT
    """

    def __init__(self, texts, tokenizer, max_len, labels=None):
        self.tokenizer = tokenizer
        self.essay = texts #Essay column of dataset
        self.targets = labels #Trait averages column of dataset
        self.max_len = max_len
        #self.doc_stride=doc_stride #Recommended to be 25%-50% of max length

    def __len__(self):
        """
        Returns total number of samples in the dataset
        """
        return len(self.essay)

    def __getitem__(self, index):
        """
        Retrieves a tokenized sample from the dataset at the specified index
        
        Args:
        index (int): The index of the sample

        Returns:
        dictionary: Contains the tokenized input, attention mask and the target values for the sample
        """

        essay = str(self.essay.iloc[index]) #Converts essay to string

        #Tokenize the text
        inputs = self.tokenizer( 
            essay,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
        )

        ids = inputs['input_ids'] #Numerical representations of tokens building the sequences that make up the input of the model
        mask = inputs['attention_mask'] #Specifies which tokens should be attended to (padded tokens are not relevant)

        #Check if targets are provided
        if self.targets is not None:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'targets': torch.tensor(self.targets.iloc[index], dtype=torch.float)
            }
        else:  #If no targets, return only the input data and attention mask
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long)
            }

def findWeights(df):
    """Finds the weights associated with each of the classes per trait, used to resolve the issue of the imbalanced dataset

    Args:
        df (pd.DataFrame): The dataset

    Returns:
        list[list[float]]: A 2D list that describes the weight attributed to each score, for each trait
    """
    #Initialses 6x6 matrix (with all counts set very low to start) to store counts for each trait and value
    position_counts = [[1e-6 for _ in range(NUM_TRAITS)] for _ in range(NUM_CLASSES_PER_TRAIT)] #Uses a very small value to prevent division by 0

    numTotalSamples = 0

    #Iterate over each row and each index in the list and find how many occurences there are of each score within each trait
    for idx, row in df.iterrows():
        for trait, score in enumerate(row['traitAverages']):
            position_counts[trait][int(score)] += 1
        numTotalSamples += 1

    #Finds the weights
    weights = INS(position_counts)

    #Normalise weights (so they sum to 1)   
    for trait in range(NUM_TRAITS):
        allWeights = sum(weights[trait])
        for score in range(NUM_CLASSES_PER_TRAIT):
            weights[trait][score] = weights[trait][score] / allWeights

    return weights

def INS(position_counts):
    """Finds the weights based on number of class instances using Inverse of Number Samples (INS)

    Args:
        position_counts ([int]): A 2D list that describes the number of occurences of each score, for each trait

    Returns:
        list[list[float]]: A 2D list that describes the non-normalised weight attributed to each score, for each trait
    """
    for position in range(NUM_TRAITS):
        for score in range(NUM_CLASSES_PER_TRAIT):
            #Using Inverse of Number Samples (INS) method to find weights
            instances = position_counts[position][score]
            weight = 1 / instances #Scores with fewer occurences are weighted more heavily

            position_counts[position][score] = weight 

    return position_counts

def ISNS(position_counts):
    """Finds the weights based on number of class instances using Inverse of Square Root of Number Samples (ISNS)

    Args:
        position_counts ([int]): A 2D list that describes the number of occurences of each score, for each trait

    Returns:
        list[list[float]]: A 2D list that describes the non-normalised weight attributed to each score, for each trait
    """
    for position in range(NUM_TRAITS):
        for score in range(NUM_CLASSES_PER_TRAIT):
            #Using Inverse of Number Samples (INS) method to find weights
            instances = position_counts[position][score]
            weight = 1 / math.sqrt(instances) #Scores with fewer occurences are weighted more heavily

            position_counts[position][score] = weight 

    return position_counts

def getSampleWeights(df, class_weights):
    """Generates a list of weights associated with samples, with samples containing minority classes having a greater weight

    Args:
        df (pd.DataFrame): A DataFrame containing the samples to be weighted
        class_weights ([float]): A list containing the weights associated with each class in each trait

    Returns:
        [float]: A list of weights corresponding to the samples
    """
    sample_weights_list = []

    for index, row in df.iterrows():
        labels = row['traitAverages']

        sample_label_weights = [0, 0, 0, 0]

        #Finds the weight for each trait in the sample
        for trait in range(len(labels)):
            label_weights = class_weights[trait][labels[trait]]
            sample_label_weights[trait] = label_weights

        #Finds mean across all traits for sample - samples with more minority labels have a higher weight
        sample_weights = sum(sample_label_weights) / len(sample_label_weights) 
        sample_weights_list.append(sample_weights)

    return sample_weights_list

def getSampleWeightsKFold(Y_train, class_weights):
    """Generates a list of weights associated with samples, with samples containing minority classes having a greater weight

    Args:
        df (pd.DataFrame): A DataFrame containing the samples to be weighted
        class_weights ([float]): A list containing the weights associated with each class in each trait

    Returns:
        [float]: A list of weights corresponding to the samples
    """
    sample_weights_list = []

    for labels in Y_train:
        sample_label_weights = [0, 0, 0, 0]

        #Finds the weight for each trait in the sample
        for trait in range(len(labels)):
            label_weights = class_weights[trait][labels[trait]]
            sample_label_weights[trait] = label_weights

        #Finds mean across all traits for sample - samples with more minority labels have a higher weight
        sample_weights = sum(sample_label_weights) / len(sample_label_weights) 
        sample_weights_list.append(sample_weights)

    return sample_weights_list

def getBaseDataframe():
    """Gets a dataframe from an excel spreadsheet with essays and associated trait scores

    Returns:
        pd.DataFrame: The relevant dataframe
    """

    #Importing data
    df = pd.read_excel("Datasets/training_set_rel3.xlsx", index_col=0)
    df = df[df.essay_set == 7] #Only look at dataset 7

    #Find the average score for each trait
    averageHeaders = []
    for trait in range(1, NUM_TRAITS+1): #Loop through each of the traits
        headers = []

        #Finds all the relevant column headings
        for rater in range(1,2):
            header = "rater" + str(rater) + "_trait" + str(trait) 
            headers.append(header)

        #Finds the average of ratings for each trait
        newHeader = "average_trait_" + str(trait)   
        averageHeaders.append(newHeader)
        
        #To prevent bias towards central values, we add some noise to the rounding
        rounding_noise = np.random.uniform(-0.5, 0.5, size=len(df))

        #Score is rounded to give a distinct integer (class) from 0-3
        df[newHeader] = ((df[headers].mean(axis=1) + rounding_noise).round()).astype(int)

    df['traitAverages'] = df[averageHeaders].values.tolist()

    cleanDf = df[['essay', 'traitAverages']].copy() #Only takes relevant columns of the original dataset

    cleanDf = cleanDf.reset_index(drop=True) #Ensures index starts from correct value

    return cleanDf

def adaptTraitAveragesForASAP8(trait_list):
    #Converts from 6 trait scores -> 4
    avg_score = round(sum(trait_list[2:5]) / 3)
    new_trait_list = trait_list[:2] + [avg_score] + trait_list[5:]

    #Converts scores from range 1-6 to range 0-3
    for i in range(len(new_trait_list)):
        new_score = round(((new_trait_list[i] - 1) / 5) * 3)

        new_trait_list[i] = new_score

    return new_trait_list

def getASAP8Dataframe():
    num_traits = 6

    #Importing data
    df = pd.read_excel("Datasets/training_set_rel3.xlsx", index_col=0)
    df = df[df.essay_set == 8] #Only look at dataset 8

    #Find the average score for each trait
    averageHeaders = []
    for trait in range(1, num_traits+1): #Loop through each of the traits
        headers = []

        #Finds all the relevant column headings
        for rater in range(1,2):
            header = "rater" + str(rater) + "_trait" + str(trait) 
            headers.append(header)

        #Finds the average of ratings for each trait
        newHeader = "average_trait_" + str(trait)   
        averageHeaders.append(newHeader)
        
        #To prevent bias towards central values, we add some noise to the rounding
        rounding_noise = np.random.uniform(-0.5, 0.5, size=len(df))

        #Score is rounded to give a distinct integer (class) from 0-3
        df[newHeader] = ((df[headers].mean(axis=1) + rounding_noise).round()).astype(int)

    df['traitAverages'] = df[averageHeaders].values.tolist()

    cleanDf = df[['essay', 'traitAverages']].copy() #Only takes relevant columns of the original dataset

    cleanDf = cleanDf.reset_index(drop=True) #Ensures index starts from correct value

    #Change the scores so they are represented the same as in ASAP7
    cleanDf["traitAverages"] = cleanDf["traitAverages"].apply(adaptTraitAveragesForASAP8)

    return cleanDf

def getDataloaders(df, weights):
    """Generates the dataloaders for the dataframe

    Args:
        df (pd.DataFrame): The dataframe to be loaded
        weights ([float]): The class weights

    Returns:
        DataLoader: The training DataLoader
        DataLoader: The validation DataLoader
        DataLoader: The testing DataLoader
    """
    #Defines which essays are in which set
    train_dataset = df.sample(frac=TRAINING_SPLIT, random_state=200)
    temp_test_dataset = df.drop(train_dataset.index).reset_index(drop=True) #Remaining data is used for testing
    train_dataset = train_dataset.reset_index(drop=True) #Resets the indices
    
    validation_dataset = temp_test_dataset.sample(frac=VALIDATION_SPLIT, random_state=200)
    test_dataset = temp_test_dataset.drop(validation_dataset.index).reset_index(drop=True)
    validation_dataset = validation_dataset.reset_index(drop=True) 

    #Uses batches with minority classes more frequently than those with majority classes 
    sample_weights = getSampleWeights(train_dataset, weights)
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

    #Splits the data
    X_train = train_dataset['essay']
    Y_train = train_dataset['traitAverages']

    X_validation = validation_dataset['essay']
    Y_validation = validation_dataset['traitAverages']

    X_test = test_dataset['essay']
    Y_test = test_dataset['traitAverages']

    tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True,
            pad_token="[PAD]",
            )

    training_set = CustomDataset(X_train, tokenizer, MAX_LEN, Y_train)
    validation_set = CustomDataset(X_validation, tokenizer, MAX_LEN, Y_validation)
    testing_set = CustomDataset(X_test, tokenizer, MAX_LEN)

    #Parameters
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'sampler': sampler,
                'num_workers': 0 
                }
    
    validation_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0 
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0 
                    }

    #Splits into batches and shuffles the data
    training_loader = DataLoader(training_set, **train_params, sampler=sampler)
    validation_loader = DataLoader(validation_set, **test_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, validation_loader, testing_loader

def getDataloadersWeightedKFold(df, train_index, test_index, weights):
    """Generates the dataloaders for the dataframe

    Args:
        df (pd.DataFrame): The dataframe to be loaded
        train_index ([int]): The samples to be used for training
        test_index ([int]): The samples to be used for testing

    Returns:
        DataLoader: The training DataLoader
        DataLoader: The testing DataLoader
    """

    X = df['essay']
    Y = df['traitAverages']

    #Splits data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True,
            pad_token="[PAD]",
            )
    
    training_set = CustomDataset(X_train, tokenizer, MAX_LEN, Y_train)
    testing_set = CustomDataset(X_test, tokenizer, MAX_LEN, Y_test)

    sample_weights = getSampleWeightsKFold(Y_train, weights)
    sampler = WeightedRandomSampler(sample_weights, len(training_set), replacement=True)

    #Parameters
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'sampler': sampler,
                'num_workers': 0 
                }
    
    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0 
                    }

    #Splits into batches and shuffles the data
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader

def getDataloadersKFold(df, train_index, test_index):
    """Generates the dataloaders for the dataframe

    Args:
        df (pd.DataFrame): The dataframe to be loaded
        train_index ([int]): The samples to be used for training
        test_index ([int]): The samples to be used for testing

    Returns:
        DataLoader: The training DataLoader
        DataLoader: The testing DataLoader
    """

    X = df['essay']
    Y = df['traitAverages']

    #Splits data
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True,
            pad_token="[PAD]",
            )
    
    training_set = CustomDataset(X_train, tokenizer, MAX_LEN, Y_train)
    testing_set = CustomDataset(X_test, tokenizer, MAX_LEN, Y_test)

    #Parameters
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0 
                }
    
    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0 
                    }

    #Splits into batches and shuffles the data
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader