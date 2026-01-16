###IMPORTS###
import pandas as pd
import os
import random
from unidecode import unidecode
import numpy as np

from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader

from openai import OpenAI
from huggingface_hub import InferenceClient
from together import Together

import ClassificationModel.BERTClassification as BERT
import ClassificationModel.Dataset as Dataset
import GenerativeModel.generate_feedback as Feedback
import GenerativeModel.evaluate_feedback as Evaluate

categories = ['Ideas', 'Organization', 'Style', 'Conventions']
MAX_LEN = 512

def loadClassificationModel(filepath):
    parent_dir = os.path.dirname(os.path.abspath(__file__)) #Points to Project folder
    PATH = os.path.join(parent_dir, filepath)

    #Initialses base model
    model = BERT.BertClassifier(modelPath=PATH)

    return model

def getNoisyEssays(filename):
    df = pd.read_excel(filename)

    column_headings = df.columns.tolist()[1:] #Finds headings for each version of essay

    essay_sets = []

    for _, row in df.iterrows(): #For each essay
        valid_essay = True
        essay_versions = []
        
        #Find each noisy version of the ssay
        for column in column_headings:
            #Ensures essay is in ASCII format
            essay = unidecode(str(row[column]))

            essay_versions.append(essay)
            #Only evaluates essay if noisy versions exist
            if essay == 'nan':
                valid_essay = False
                break

        if valid_essay:
            essay_sets.append(essay_versions)

    return column_headings, essay_sets

def getStandardEssays(filename):
    df = pd.read_excel(filename)
    df = df[df.essay_set == 7] #Only look at dataset 7
    essays = df['essay'].apply(unidecode).tolist() #Ensures there are in ASCII format

    return essays

def getSamples(essays, scores, samples_per_score):
    scores_options = [0, 1, 2, 3]
    final_essays = []
    final_scores = []
    
    #Finds a good distribution of samples (for each score in each category)
    for category_num in reversed(range(len(categories))):
        for score in scores_options:
            matches = 0
            for i in reversed(range(len(essays))):
                if scores[i][category_num] == scores_options[score]:
                    #Will use essays
                    final_essays.append(essays[i])
                    final_scores.append(scores[i])

                    #Ensures same essay cannot be selected twice
                    essays.pop(i)
                    scores.pop(i)

                    matches += 1

                    #Only continues until x amount of samples are found
                    if matches >= samples_per_score:
                        break

    return final_essays, final_scores

def getEssayClassificationScores(essays, model):

    tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True,
            pad_token="[PAD]",
            )

    essaySet = Dataset.CustomDataset(pd.DataFrame(essays, columns=["essay"]), tokenizer, MAX_LEN, labels=None)

    loader_params = {'batch_size': 8,
                    'shuffle': False,
                    'num_workers': 0 
                    }

    #Splits into batches and shuffles the data
    testing_loader = DataLoader(essaySet, **loader_params)

    predictions = model.predict(testing_loader)

    return predictions


def generateFeedbackForSet(essays, scores, model, client):
    feedbacks = []
    for i in range(len(essays)):
        feedback = Feedback.generateFeedback(essays[i], scores[i], client, model)

        feedbacks.append(feedback)

    return feedbacks

def evaluateUsingFeedbackExamples():

    #Load classification model
    classification_model = loadClassificationModel("ClassificationModel/Models/Multi-headDropoutSampler/10epochs1e-05learningRate_Model.pth")
    
    #Load dataset
    df = pd.read_excel("Datasets/feedbackExamples.xlsx")
    original_essays = df['Essay'].tolist()
    original_feedbacks = df['Feedback'].tolist()
    feedback_categories = df['Category'].tolist()

    #Generate feedbacks
    scores = getEssayClassificationScores(original_essays, classification_model)
    generated_feedbacks = generateFeedbackForSet(original_essays, scores)

    relevant_feedbacks = []
    #Only look at the part of the feedback relevant to the category
    for i in range(len(original_feedbacks)):
        #Find which category is being investigated
        category_index = categories.index(feedback_categories[i])

        #Generate feedback and add to list
        relevant_feedback = generated_feedbacks[i][category_index]
        relevant_feedbacks.append(relevant_feedback)

    #Finds BERTScore
    avg_precision, avg_recall, avg_f1 = Evaluate.evaluateFeedbackUsingBERTScore(original_feedbacks, relevant_feedbacks)
    print(f"BERTScore Results:\nAverage Precision: {avg_precision}\nAverage Recall: {avg_recall}\nAverage F1-Score: {avg_f1}")

    #Finds BLEU Score
    BLEU_Score = Evaluate.evaluateFeedbackUsingBLEUScore(relevant_feedbacks, original_feedbacks)
    print(f"\nAverage BLEU Score: {BLEU_Score}")

    #Finds ROUGE Scores
    ROUGE_Metrics, ROUGE_Scores = Evaluate.evaluateFeedbacksUsingROUGEScore(relevant_feedbacks, original_feedbacks)
    for m in range(len(ROUGE_Metrics)):
        print(f"\n{ROUGE_Metrics[m]} Score:\nAverage Precision: {ROUGE_Scores[m][0]}\nAverage Recall: {ROUGE_Scores[m][1]}\nAverage F-Score: {ROUGE_Scores[m][2]}")
            
#Compares BLEU, BERTScore, and ROUGE difference on noisy feedbacks 
def evaluateNoisyFeedback(classification_model_file, category, feedback_model, feedback_client):
    category_index = categories.index(category)

    classification_model = loadClassificationModel(classification_model_file)

    noisy_filename = f"Datasets/{category}.xlsx"
    headings, essays = getNoisyEssays(noisy_filename)

    #Only look at first 10 essay sets
    essays = essays[:10]
    
    #Results foreach headings (essay, essay25, essay50, essay75, etc.) - for altered category only
    BERTScore_precision_cat = [[] for _ in headings]
    BERTScore_recall_cat = [[] for _ in headings]
    BERTScore_f1_cat = [[] for _ in headings]
    BLEU_results_cat = [[] for _ in headings]
    ROUGE_precision_cat = [[] for _ in headings]
    ROUGE_recall_cat = [[] for _ in headings]
    ROUGE_f1_cat = [[] for _ in headings]

    #Scores for non-altered categories
    BERTScore_precision_nocat = [[] for _ in headings]
    BERTScore_recall_nocat = [[] for _ in headings]
    BERTScore_f1_nocat = [[] for _ in headings]
    BLEU_results_nocat = [[] for _ in headings]
    ROUGE_precision_nocat = [[] for _ in headings]
    ROUGE_recall_nocat = [[] for _ in headings]
    ROUGE_f1_nocat = [[] for _ in headings]

    #For each essay set
    for essay_set in essays:
        #Generate feedbacks
        scores = getEssayClassificationScores(essay_set, classification_model)
        feedbacks = generateFeedbackForSet(essay_set, scores, feedback_model, feedback_client)

        original_feedback = feedbacks[0]

        #For each essay in set
        for i in range(1,len(essay_set)):      
            #For each category
            for c in range(len(original_feedback)):
                original_category_feedback = original_feedback[c]
                altered_category_feedback = feedbacks[i][c]

                bert_precision, bert_recall, bert_f1 = Evaluate.evaluateFeedbackUsingBERTScore([original_category_feedback], [altered_category_feedback])
                print(bert_precision)
                print(type(bert_precision))
                print(bert_recall)
                print(type(bert_recall))

                #Finds BLEU Score
                BLEU_Score = Evaluate.evaluateFeedbackUsingBLEUScore([altered_category_feedback], [original_category_feedback])
                print(BLEU_Score)
                print(type(BLEU_Score))

                #Finds ROUGE Scores
                ROUGE_Scores = Evaluate.evaluateFeedbacksUsingROUGEScore([altered_category_feedback], [original_category_feedback])
                rouge_precision = ROUGE_Scores[0][0]
                rouge_recall = ROUGE_Scores[0][1]
                rouge_f1 = ROUGE_Scores[0][2]
                print(rouge_precision)
                print(type(rouge_precision))
                print('---------------')

                #If comparing relevant category (we should expect change):
                if c == category_index:
                    BERTScore_precision_cat[i].append(bert_precision)
                    BERTScore_recall_cat[i].append(bert_recall)
                    BERTScore_f1_cat[i].append(bert_f1)
                    BLEU_results_cat[i].append(BLEU_Score)
                    ROUGE_precision_cat[i].append(rouge_precision)
                    ROUGE_recall_cat[i].append(rouge_recall)
                    ROUGE_f1_cat[i].append(rouge_f1)
                #If not comparing relevant category (we should expect no change)
                else: 
                    BERTScore_precision_nocat[i].append(bert_precision)
                    BERTScore_recall_nocat[i].append(bert_recall)
                    BERTScore_f1_nocat[i].append(bert_f1)
                    BLEU_results_nocat[i].append(BLEU_Score)
                    ROUGE_precision_nocat[i].append(rouge_precision)
                    ROUGE_recall_nocat[i].append(rouge_recall)
                    ROUGE_f1_nocat[i].append(rouge_f1)

    print("CATEGORY ALTERED RESULTS")
    for i in range(1,len(headings)):
        print(headings[i])
        print(f"BERTScore Precision: {np.sum(BERTScore_precision_cat[i])/len(BERTScore_precision_cat[i])}")
        print(f"BERTScore Recall: {np.sum(BERTScore_recall_cat[i])/len(BERTScore_recall_cat[i])}")
        print(f"BERTScore F1: {np.sum(BERTScore_f1_cat[i])/len(BERTScore_f1_cat[i])}")
        print(f"BLEU: {sum(BLEU_results_cat[i])/len(BLEU_results_cat[i])}")
        print(f"ROUGE Precision: {sum(BERTScore_precision_cat[i])/len(BERTScore_precision_cat[i])}")
        print(f"ROUGE Recall: {sum(BERTScore_recall_cat[i])/len(BERTScore_recall_cat[i])}")
        print(f"ROUGE F1: {sum(BERTScore_f1_cat[i])/len(BERTScore_f1_cat[i])}")

    print("CATEGORY NOT ALTERED RESULTS")
    for i in range(1,len(headings)):
        print(headings[i])
        print(f"BERTScore Precision: {np.sum(BERTScore_precision_nocat[i])/len(BERTScore_precision_nocat[i])}")
        print(f"BERTScore Recall: {np.sum(BERTScore_recall_nocat[i])/len(BERTScore_recall_cat[i])}")
        print(f"BERTScore F1: {np.sum(BERTScore_f1_nocat[i])/len(BERTScore_f1_nocat[i])}")
        print(f"BLEU: {sum(BLEU_results_nocat[i])/len(BLEU_results_nocat[i])}")
        print(f"ROUGE Precision: {sum(BERTScore_precision_nocat[i])/len(BERTScore_precision_nocat[i])}")
        print(f"ROUGE Recall: {sum(BERTScore_recall_nocat[i])/len(BERTScore_recall_nocat[i])}")
        print(f"ROUGE F1: {sum(BERTScore_f1_nocat[i])/len(BERTScore_f1_nocat[i])}")

###LLM MATCHING METHODS###
def LLMMatching_Minimal_Human(classification_model_path, feedback_client, feedback_model, evaluator_client, category):
    classification_model = loadClassificationModel(classification_model_path)
    column_headings, essay_sets = getNoisyEssays(f"Datasets/{category}.xlsx")

    total_matched_essays_llm = []
    total_matched_essays_human = []

    #Chooses random essays
    sets_to_samples = random.sample(essay_sets, 15)

    for essays in sets_to_samples:
        minimal_set = [essays[0], essays[-1]]
        minimal_scores = getEssayClassificationScores(minimal_set, classification_model)
        minimal_feedbacks = generateFeedbackForSet(minimal_set, minimal_scores, feedback_model, feedback_client)

        correctly_matched_set_llm = Evaluate.evaluateFeedbackSetUsingLLM_Comparison(minimal_set, minimal_feedbacks, minimal_scores, evaluator_client, category)
        total_matched_essays_llm.extend(correctly_matched_set_llm)

        correctly_matched_set_human = Evaluate.evaluateFeedbackSetUsingHuman_Comparison(minimal_set, minimal_feedbacks, minimal_scores, evaluator_client, category)
        total_matched_essays_human.extend(correctly_matched_set_human)

    acc_llm = Evaluate.getModelAccuracy(total_matched_essays_llm)
    acc_human = Evaluate.getModelAccuracy(total_matched_essays_human)
   
    return acc_llm, acc_human

def LLMMatching_All_Human(classification_model_path, feedback_client, feedback_model, evaluator_client, category):
    classification_model = loadClassificationModel(classification_model_path)
    column_headings, essay_sets = getNoisyEssays(f"Datasets/{category}.xlsx")

    total_matched_essays_llm = []
    total_matched_essays_human = []

    #Chooses random essays
    sets_to_samples = random.sample(essay_sets, 15)

    for essays in sets_to_samples:
        scores = getEssayClassificationScores(essays, classification_model)
        feedbacks = generateFeedbackForSet(essays, scores, feedback_model, feedback_client)

        correctly_matched_set_llm = Evaluate.evaluateFeedbackSetUsingLLM_Comparison(essays, feedbacks, scores, evaluator_client, category)
        total_matched_essays_llm.extend(correctly_matched_set_llm)

        correctly_matched_set_human = Evaluate.evaluateFeedbackSetUsingHuman_Comparison(essays, feedbacks, scores, evaluator_client, category)
        total_matched_essays_human.extend(correctly_matched_set_human)

    acc_llm = Evaluate.getModelAccuracy(total_matched_essays_llm)
    acc_human = Evaluate.getModelAccuracy(total_matched_essays_human)

    return acc_llm, acc_human

def LLMMatching_Minimal(classification_model_path, feedback_client, feedback_model, evaluator_client, category):
    classification_model = loadClassificationModel(classification_model_path)
    column_headings, essay_sets = getNoisyEssays(f"Datasets/{category}.xlsx")

    total_matched_essays_llm = []

    #Chooses random essays
    sets_to_samples = random.sample(essay_sets, 5)

    for essays in sets_to_samples:
        minimal_set = [essays[0], essays[-1]]
        minimal_scores = getEssayClassificationScores(minimal_set, classification_model)
        minimal_feedbacks = generateFeedbackForSet(minimal_set, minimal_scores, feedback_model, feedback_client)

        correctly_matched_set_llm = Evaluate.evaluateFeedbackSetUsingLLM_Comparison(minimal_set, minimal_feedbacks, evaluator_client, category)
        total_matched_essays_llm.extend(correctly_matched_set_llm)

    acc_llm = Evaluate.getModelAccuracy(total_matched_essays_llm)
   
    return acc_llm

def LLMMatching_All(classification_model_path, feedback_client, feedback_model, evaluator_client, category):
    classification_model = loadClassificationModel(classification_model_path)
    column_headings, essay_sets = getNoisyEssays(f"Datasets/{category}.xlsx")

    total_matched_essays_llm = []

    #Chooses random essays
    sets_to_samples = random.sample(essay_sets, 15)

    for essays in sets_to_samples:
        scores = getEssayClassificationScores(essays, classification_model)
        feedbacks = generateFeedbackForSet(essays, scores, feedback_model, feedback_client)

        correctly_matched_set_llm = Evaluate.evaluateFeedbackSetUsingLLM_Comparison(essays, feedbacks, evaluator_client, category)
        total_matched_essays_llm.extend(correctly_matched_set_llm)

    acc_llm = Evaluate.getModelAccuracy(total_matched_essays_llm)

    return acc_llm

###LLM RATING METHOD###

def evaluateUsingLLM_Rating(classification_model, feedback_model, feedback_client, evaluator_client):
    classification_model = loadClassificationModel(classification_model)
    essays = getStandardEssays("Datasets/training_set_rel3.xlsx")
    scores = getEssayClassificationScores(essays, classification_model)

    essays_to_test, scores_to_test = getSamples(essays, scores, 15)

    feedbacks = generateFeedbackForSet(essays_to_test, scores_to_test, feedback_model, feedback_client)
        
    Evaluate.evaluateFeedbackSetUsingLLM_Rating(essays_to_test, feedbacks, evaluator_client)


###MAIN###
if __name__=="__main__":
    feedback_model = "gpt-3.5-turbo"

    #OpenAI Client
    openai_client = OpenAI()

    #Together Client
    together_client = Together()

    ##LLM MATCHING###
    #GPT
    feedback_model = "gpt-3.5-turbo"
    gpt_ideas_all_llm, gpt_ideas_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", openai_client, feedback_model, openai_client, "Ideas")
    gpt_organization_all_llm, gpt_organization_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", openai_client, feedback_model, openai_client, "Organization")
    gpt_style_all_llm, gpt_style_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", openai_client, feedback_model, openai_client, "Style")
    gpt_conventions_all_llm, gpt_conventions_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", openai_client, feedback_model, openai_client, "Conventions")

    gpt_ideas_minimal_llm, gpt_ideas_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Ideas")
    gpt_organization_minimal_llm, gpt_organization_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Organization")
    gpt_style_minimal_llm, gpt_style_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Style")
    gpt_conventions_minimal_llm, gpt_conventions_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Conventions")

    #Llama
    feedback_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    llama_ideas_all_llm, llama_ideas_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Ideas")
    llama_organization_all_llm, llama_organization_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Organization")
    llama_style_all_llm, llama_style_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Style")
    llama_conventions_all_llm, llama_conventions_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Conventions")

    llama_ideas_minimal_llm, llama_ideas_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Ideas")
    llama_organization_minimal_llm, llama_organization_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Organization")
    llama_style_minimal_llm, llama_style_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Style")
    llama_conventions_minimal_llm, llama_conventions_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Conventions")

    #Mistral
    feedback_model = "mistralai/Mistral-7B-Instruct-v0.2"
    mistral_ideas_all_llm, mistral_ideas_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Ideas")
    mistral_organization_all_llm, mistral_organization_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Organization")
    mistral_style_all_llm, mistral_style_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Style")
    mistral_conventions_all_llm, mistral_conventions_all_human = LLMMatching_All_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Conventions")

    mistral_ideas_minimal_llm, mistral_ideas_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Ideas")
    mistral_organization_minimal_llm, mistral_organization_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Organization")
    mistral_style_minimal_llm, mistral_style_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Style")
    mistral_conventions_minimal_llm, mistral_conventions_minimal_human = LLMMatching_Minimal_Human("ClassificationModel/Models/Final/model.pth", together_client, feedback_model, openai_client, "Conventions")

    print("Final Results")
    print("GPT")
    print(f"All Ideas LLM: {gpt_ideas_all_llm} - Human: {gpt_ideas_all_human}")
    print(f"All Organization LLM: {gpt_organization_all_llm} - Human: {gpt_organization_all_human}")
    print(f"All Style LLM: {gpt_style_all_llm} - Human: {gpt_style_all_human}")
    print(f"All Conventions LLM: {gpt_conventions_all_llm} - Human: {gpt_conventions_all_human}")
    print("\nLlama")
    print(f"All Ideas LLM: {llama_ideas_all_llm} - Human: {llama_ideas_all_human}")
    print(f"All Organization LLM: {llama_organization_all_llm} - Human: {llama_organization_all_human}")
    print(f"All Style LLM: {llama_style_all_llm} - Human: {llama_style_all_human}")
    print(f"All Conventions LLM: {llama_conventions_all_llm} - Human: {llama_conventions_all_human}")
    print("\nMistral")
    print(f"All Ideas LLM: {mistral_ideas_all_llm} - Human: {mistral_ideas_all_human}")
    print(f"All Organization LLM: {mistral_organization_all_llm} - Human: {mistral_organization_all_human}")
    print(f"All Style LLM: {mistral_style_all_llm} - Human: {mistral_style_all_human}")
    print(f"All Conventions LLM: {mistral_conventions_all_llm} - Human: {mistral_conventions_all_human}")

    ###LLM RATING###
    print("GPT")
    evaluateUsingLLM_Rating("ClassificationModel/Models/Final/model.pth", "gpt-3.5-turbo", openai_client, openai_client)
    print("Llama")
    evaluateUsingLLM_Rating("ClassificationModel/Models/Final/model.pth", "meta-llama/Llama-3.3-70B-Instruct-Turbo", together_client, openai_client)
    print("Mistral")
    evaluateUsingLLM_Rating("ClassificationModel/Models/Final/model.pth", "mistralai/Mistral-7B-Instruct-v0.2", together_client, openai_client)

###MODELS###
#gpt-3.5-turbo
#meta-llama/Llama-3.3-70B-Instruct-Turbo
#mistralai/Mistral-7B-Instruct-v0.2