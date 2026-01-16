import ClassificationModel.BERTClassification as BERT
import ClassificationModel.Dataset as Dataset
from ClassificationModel.Dataset import CustomDataset
import random
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import os
import pandas as pd 
import numpy as np
from transformers import BertTokenizerFast, BertModel, BertConfig, BertForSequenceClassification

NUM_TRAITS = 4
NUM_RATERS = 2
MAX_LEN = 512

TRAITS = ['Ideas', 'Organization', 'Style', 'Conventions']

def loadClassificationModel(filepath):
    parent_dir = os.path.dirname(os.path.abspath(__file__)) #Points to Project folder
    PATH = os.path.join(parent_dir, filepath)

    #Initialses base model
    model = BERT.BertClassifier(modelPath=PATH)

    return model

def get_original_scores(noisy_file, original_file):
    #Load dataset
    noisy_df = pd.read_excel(noisy_file)
    ids = noisy_df['essay_id'].tolist()

    original_df = pd.read_excel(original_file)

    original_scores = []

    for id in ids:
        row = original_df.loc[original_df['essay_id'] == id]
        row_scores = []

        #Finds averages of trait scores 
        for trait in range(1, NUM_TRAITS+1):
            trait_score = 0
            for rater in range(1,NUM_RATERS+1):
                    header = "rater" + str(rater) + "_trait" + str(trait) 
                    trait_score += row[header].values[0]

            rounding_noise = random.uniform(-0.5, 0.5)
            trait_score = round((trait_score / NUM_RATERS) + rounding_noise)
            row_scores.append(trait_score)
        
        original_scores.append(row_scores)

    return original_scores

def get_new_scores(noisy_file, column_header, model_path):
    #Get essays
    noisy_df = pd.read_excel(noisy_file)

    #Load classification model
    classification_model = loadClassificationModel(model_path)

    tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True,
            pad_token="[PAD]",
            )

    essaySet = CustomDataset(pd.DataFrame(noisy_df, columns=[column_header]), tokenizer, MAX_LEN, labels=None)

    loader_params = {'batch_size': 8,
                    'shuffle': False,
                    'num_workers': 0 
                    }

    #Splits into batches and shuffles the data
    testing_loader = DataLoader(essaySet, **loader_params)

    predictions = classification_model.predict(testing_loader)

    return predictions

def compare_accuracy_decrease(original_file, noisy_file, headers, trait, model_path):
    trait_index = TRAITS.index(trait)

    original_scores = get_new_scores(noisy_file, 'essay', model_path)
    
    for essay_header in headers:
        new_scores = get_new_scores(noisy_file, essay_header, model_path)
        num_decreased = 0
        num_checked = 0 

        for i in range(len(original_scores)):
            original_trait_score = original_scores[i][trait_index]
            new_trait_score = new_scores[i][trait_index]

            #Trait scores of 0 cannot decrease
            if original_trait_score >= 1:
                num_checked += 1

                #If the score has decreased
                if new_trait_score <= original_trait_score:
                    num_decreased += 1

        accuracy = num_decreased/num_checked
    
        print(f"Accuracy for {essay_header}: {accuracy}")

              
model_path = "ClassificationModel/Models/Multi-headDropoutSampler/10epochs1e-05learningRate_Model.pth"
compare_accuracy_decrease("Datasets/training_set_rel3.xlsx", "Datasets/organisation.xlsx", ['essay25', 'essay50', 'essay75','essay100'], "Organization", model_path)
