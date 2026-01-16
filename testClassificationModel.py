import os
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

import ClassificationModel.BERTClassification as BERT
from ClassificationModel.Dataset import CustomDataset
from ClassificationModel.Dataset import getASAP8Dataframe

def loadClassificationModel(filepath):
    parent_dir = os.path.dirname(os.path.abspath(__file__)) #Points to Project folder
    PATH = os.path.join(parent_dir, filepath)

    #Initialses base model
    model = BERT.BertClassifier(modelPath=PATH)

    return model

def testASAP8(model_filepath):
    df = getASAP8Dataframe()

    tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True,
            pad_token="[PAD]",
            )

    validation = CustomDataset(df["essay"], tokenizer, 512, labels=df["traitAverages"])

    loader_params = {'batch_size': 8,
                    'shuffle': False,
                    'num_workers': 0 
                    }

    #Splits into batches and shuffles the data
    validation_loader = DataLoader(validation, **loader_params)

    #Loads classification model and scores 
    model = loadClassificationModel(model_filepath)
    model.scoreMultipleMetrics(validation_loader)

model_filepath = "ClassificationModel/Models/Multi-headDropoutSampler/10epochs1e-05learningRate_Model.pth"
testASAP8(model_filepath)


