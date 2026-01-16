import random
import string
import pandas as pd
import re

def remove_sentences_from_text(text, percent=0.5):
    #Checks that the text is a string
    if not isinstance(text, str):
        return text
    
    new_text = []

    #Splits text into sentences
    sentences = re.split(r'(?<!\?\?)(?<=[.!?])\s+', text.strip())
    num_sentences = len(sentences)

    if num_sentences <= 3:
        return "NA"

    indicies = list(range(num_sentences))

    num_sentences_to_keep = max(1, num_sentences - round(num_sentences * percent)) #Ensures at least 1 sentence remains

    indicies_to_keep = random.sample(indicies, num_sentences_to_keep)
    indicies_to_keep.sort()

    for i in indicies_to_keep:
        new_text.append(sentences[i])

    essay = " ".join(new_text)

    return essay

def process_spreadsheet(input_file, output_file):
    df = pd.read_excel(input_file, index_col=0)
    df = df[df.changed_category == 'Ideas'] #Only apply noise to relevant rows

    #Creates new columns
    df.loc[df['changed_category'] == 'Ideas', 'essay25'] = df['essay'].apply(lambda x: remove_sentences_from_text(x, percent=0.25))
    df.loc[df['changed_category'] == 'Ideas', 'essay50'] = df['essay'].apply(lambda x: remove_sentences_from_text(x, percent=0.50))
    df.loc[df['changed_category'] == 'Ideas', 'essay75'] = df['essay'].apply(lambda x: remove_sentences_from_text(x, percent=0.75))

    df.to_excel(output_file, sheet_name='Ideas', columns=['essay', 'essay25', 'essay50', 'essay75'], index=True)
    print(f"Sentences removed. Modified spreadsheet saved to {output_file}")

process_spreadsheet("Datasets/NoisyDataset.xlsx", "Datasets/ideas.xlsx")
