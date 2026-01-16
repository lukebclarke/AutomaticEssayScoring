import numpy as np
import math
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer
import re
import random

evaluation_model = "gpt-4o-mini"
grade = "7th"
prompt = "Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining. Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience."

categories = ['Ideas', 'Organization', 'Style', 'Conventions']
criteria_set = ['level of detail', 'accuracy', 'relevance', 'helpfulness']

def evaluateFeedbackSetUsingLLM_Comparison(essay_set, feedback_set, client, category):
    correctly_matched_feedbacks = []
    category_num = categories.index(category.title())

    for i in range(len(feedback_set)): #For each feedback generated
        feedback = feedback_set[i]
        category_feedback = feedback[category_num]

        system_prompt = (f"You will be given a piece of feedback (regarding the {category} of an essay) and a selection of essays from {grade} grade students.\n\n\
                         Your task is to compare the essays to each other, and determine the essay that the feedback is most appropiate for\n\
                         You must:\n\
                         - Carefully read each essay and thoroughly compare them to the feedback\n\
                         - Select the best match - the essay that the feedback is most relevant to\n\
                         - Ensure that every point made in the feedback is relevant to the chosen essay\n\
                         - Ensure that no other essay is a better match for the feedback\n\
                         - Ensure you do not have a bias towards order the essays are presented in - carefully consider each essay\n\
                         - Do not favor the first essay shown\n\
                         - Only make your judgement based on the feedback - do not focus on errors in essays that aren't mentioned in the feedback\n")
        content_prompt = f"Feedback:\n{category_feedback}\n\n"

        for j in range(len(essay_set)): #For each essay, check if matches feedback
            content_prompt += f"Essay {j}:\n{essay_set[j]}\n\n"

        valid_attempts = 0
        while valid_attempts < 10 and valid_attempts != -1: #Only attempts 10 times before passing
            response = client.chat.completions.create(
                model=evaluation_model, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_prompt}
                ],
                temperature=0.1, #Makes model predictable
                max_tokens=1000,
            )

            response_text = response.choices[0].message.content

            #Extracts single number from response
            choice = re.findall(r'\b\d+\b', response_text)[0]

            try: 
                best_match_index = int(choice) #Correctly converts to int
                valid_attempts = -1
            except:
                valid_attempts += 1 #Keeps going until can convert to int

        if valid_attempts != -1: #No best match found
            print("Failed to identify a match")
            continue

        #Find most likely essay based on the probability of "YES" being the first token
        if best_match_index == i:
            correctly_matched_feedbacks.append(1)
        else:
            correctly_matched_feedbacks.append(0)

    return correctly_matched_feedbacks

def evaluateFeedbackSetUsingHuman_Comparison(essay_set, feedback_set, client, category):
    correctly_matched_feedbacks = []
    category_num = categories.index(category.title())

    #Shuffle essays and feedbacks
    combined = list(zip(essay_set, feedback_set))
    random.shuffle(combined)
    essay_set, feedback_set = zip(*combined)

    for i in range(len(feedback_set)): #For each feedback generated
        feedback = feedback_set[i]
        category_feedback = feedback[category_num]

        #Shuffle essay display order so we can't predict what order they will appear in
        indices = list(range(len(essay_set)))
        random.shuffle(indices)

        content_prompt = f"Feedback:\n{category_feedback}\n\n"

        for j in range(len(essay_set)): #For each essay, check if matches feedback
            content_prompt += f"Essay {j}:\n{essay_set[indices[j]]}\n\n"

        print(content_prompt)

        valid = 0
        while valid == 0:
            try:
                choice = int(input("Enter essay number: "))
                choice = indices[choice]
                valid = 1
            except:
                continue

        print("Actual match:")
        print(i)

        #Find most likely essay based on the probability of "YES" being the first token
        if choice == i:
            correctly_matched_feedbacks.append(1)
        else:
            correctly_matched_feedbacks.append(0)

    return correctly_matched_feedbacks

def getModelAccuracy(correctly_matched_feedbacks):
    accuracy = correctly_matched_feedbacks.count(1) / len(correctly_matched_feedbacks)

    return accuracy

def evaluateFeedbackSetUsingLLM_Rating(essay_set, feedback_set, client):
    num_criteria = len(criteria_set)
    scores = [[] for _ in range(num_criteria)]

    for i in range(len(feedback_set)):
        essay = essay_set[i]
        feedback = feedback_set[i]

        for j in range(num_criteria):

            for c in range(len(categories)):
                category = categories[c]
                criteria = criteria_set[j]
                category_feedback = feedback[c]

                system_prompt = (f"An American {grade} Grade student has been tasked with writing an essay in response to the following prompt: {prompt}\n Now read the following essay and its associated feedback. The feedback is regarding the {category} of the essay.\n\nYour task is to rate the {criteria} of the feedback on a 7-point Likert level, with a score of 7 representing the highest {criteria}, and a score of 1 representing the lowest {criteria}. Only rate based on this criteria. Just say the score without providing any additional comment, note or explanation.")
                content_prompt = (f"Feedback: {category_feedback}\n\nEssay: {essay}")

                response = client.responses.create(
                    model=evaluation_model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content_prompt}
                        ],
                    temperature=1,
                    max_output_tokens=2048,
                )

                #If cannot convert to int, simply skip the score for that essay
                try:
                    score = int(response.output[0].content[0].text)
                    #Add score to list
                    scores[j].append(score)
                except:
                    pass

    for i in range(len(scores)):
        avg_score = sum(scores[i])/len(scores[i])

        print(f"{criteria_set[i]} average score: {avg_score}")

def evaluateFeedbackUsingBERTScore(generated_feedbacks, target_feedbacks):
    bertscore = load("bertscore")

    similarity = bertscore.compute(predictions=generated_feedbacks, references=target_feedbacks, lang="en", rescale_with_baseline=True)

    avg_precision = np.mean(similarity['precision'])
    avg_recall = np.mean(similarity['recall'])
    avg_f1 = np.mean(similarity['f1'])

    return avg_precision, avg_recall, avg_f1

def evaluateFeedbackUsingBLEUScore(generated_feedbacks, target_feedbacks):
    score = corpus_bleu(target_feedbacks, generated_feedbacks)

    return score

def evaluateFeedbacksUsingROUGEScore(generated_feedbacks, target_feedbacks):
    
    metric = 'rougeL'
    avg_per_metric = []

    scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)

    precisions = []
    recalls = []
    f_measures = []

    #Finds ROUGE score for each feedback pair
    for i in range(len(generated_feedbacks)):
        scores = scorer.score(target_feedbacks[i], generated_feedbacks[i])

        precisions.append(scores[metric].precision)
        recalls.append(scores[metric].recall)
        f_measures.append(scores[metric].fmeasure)

    avg_precision = sum(precisions)/len(precisions)
    avg_recall = sum(recalls)/len(recalls)
    avg_fmeasures = sum(f_measures)/len(f_measures)

    avg_per_metric.append([avg_precision, avg_recall, avg_fmeasures])

    return avg_per_metric
