from openai import OpenAI
import json
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import evaluate
import numpy as np

def finetuneGPT():
    client = OpenAI()

    client.fine_tuning.jobs.create(
        training_file="file-L3P4JWU7hwig3KNNVAHjHb", #File uploaded to servers using API key
        model="gpt-3.5-turbo-0125"
    )



