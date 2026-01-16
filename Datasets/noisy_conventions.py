import random
import pandas as pd
import re
from openai import OpenAI
from unidecode import unidecode

client = OpenAI()

def correct_errors_LLM(text, percent=0.5):
    text = unidecode(text) #Ensures ascii characters
    sentences = re.split(r'(?<!\?\?)(?<=[.!?])\s+', text.strip())
    num_sentences = len(sentences)
    num_sentences_to_correct = round(num_sentences * percent)

    indices = list(range(num_sentences))
    sentences_to_correct_indices = random.sample(indices, min(num_sentences_to_correct, len(indices)))

    for i in sentences_to_correct_indices:
        #Find what sentence is being altered
        sentence = sentences[i]

        #Alter sentence
        response = client.responses.create(
            model="gpt-3.5-turbo",
            input=[
                {"role": "system", "content": f"You will be provided with a sentence from a Grade 7 esssay, and your task is to convert them to standard English. Only alter the spelling and grammar in the sentence - do not alter the essay in any other way. Only return the sentence - no additional text."},
                {"role": "user", "content": sentence}
                ],
            temperature=0.1,
            max_output_tokens=2048,
        )

        #Replace sentence
        sentences[i] = response.output[0].content[0].text

    new_essay = " ".join(sentences)
    print(new_essay)

    return new_essay

def process_spreadsheet(input_file, output_file):
    df = pd.read_excel(input_file, index_col=0)
    df = df[df.changed_category == 'Conventions'] #Only apply noise to relevant rows

    #Creates new columns
    df.loc[df['changed_category'] == 'Conventions', 'essay25'] = df['essay'].apply(lambda x: correct_errors_LLM(x, percent=0.25))
    df.loc[df['changed_category'] == 'Conventions', 'essay50'] = df['essay'].apply(lambda x: correct_errors_LLM(x, percent=0.50))
    df.loc[df['changed_category'] == 'Conventions', 'essay75'] = df['essay'].apply(lambda x: correct_errors_LLM(x, percent=0.75))

    print(df)
    df.to_excel(output_file, sheet_name='Organisation', columns=['essay', 'essay25', 'essay50', 'essay75'], index=True)
    print(f"Sentences corrected. Modified spreadsheet saved to {output_file}")

process_spreadsheet("Datasets/NoisyDataset.xlsx", "Datasets/conventions.xlsx")
