import random
import string
import pandas as pd
import re

def shuffle_sentences(text, percent=0.5):
    #Checks that the text is a string
    if not isinstance(text, str):
        return text
    
    #Splits text into sentences
    sentences = re.split(r'(?<!\?\?)(?<=[.!?])\s+', text.strip())
    num_sentences = len(sentences)

    new_text = sentences.copy()

    indices = list(range(num_sentences))

    num_sentences_to_move = max(2, round(num_sentences * percent)) #Must move at least 2 sentences

    sentences_to_move = random.sample(indices, min(num_sentences_to_move, len(indices)))

    available_positions = sentences_to_move.copy()

    #Shuffles sentences
    for i in sentences_to_move:
        filtered_list = [item for item in available_positions if item != i]
        
        if filtered_list:
            newPos = random.choice(filtered_list) #Ensures that index doesn't remain same
            new_text[newPos] = sentences[i]

            available_positions.remove(newPos)
        else:
            new_text[i] = sentences[i]

    order = []
    for i in new_text:
        order.append(sentences.index(i))

    essay = " ".join(new_text)

    return essay

def process_spreadsheet(input_file, output_file):
    df = pd.read_excel(input_file, index_col=0)
    df = df[df.changed_category == 'Organisation'] #Only apply noise to relevant rows

    #Creates new columns
    df.loc[df['changed_category'] == 'Organisation', 'essay25'] = df['essay'].apply(lambda x: shuffle_sentences(x, percent=0.25))
    df.loc[df['changed_category'] == 'Organisation', 'essay50'] = df['essay'].apply(lambda x: shuffle_sentences(x, percent=0.50))
    df.loc[df['changed_category'] == 'Organisation', 'essay75'] = df['essay'].apply(lambda x: shuffle_sentences(x, percent=0.75))

    columns_to_save = ['essayID', 'essay', 'essay25', 'essay50', 'essay75', 'essay100']

    df.to_excel(output_file, sheet_name='Organization', columns=['essay', 'essay25', 'essay50', 'essay75'], index=True)
    print(f"Sentences shuffled. Modified spreadsheet saved to {output_file}")

process_spreadsheet('Datasets/NoisyDataset.xlsx', 'Datasets/Organization.xlsx')
