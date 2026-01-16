import gradio as gr
import os
import pandas as pd
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from openai import OpenAI
from huggingface_hub import InferenceClient
from together import Together

import ClassificationModel.BERTClassification as BERT
from ClassificationModel.Dataset import CustomDataset
import GenerativeModel.generate_feedback as Feedback

categories = ["Ideas", "Organization", "Style", "Conventions"]
model_path = "ClassificationModel/Models/Multi-headV2/15epochs3e-05learningRate_Model.pth"
openai_client = OpenAI()

results_by_category = {}

def getClassifierScores(essay):
    parent_dir = os.path.dirname(os.path.abspath(__file__)) #Points to Project folder
    PATH = os.path.join(parent_dir, model_path)

    #Initialses base model
    model = BERT.BertClassifier(modelPath=PATH)

    tokenizer = BertTokenizerFast.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True,
            pad_token="[PAD]",
            )
    
    #Converts essay to form CustomDataset handles (e.g. Panda DataFrame)
    essay_df = pd.DataFrame([essay], columns=["essay"]) 
    dataset = CustomDataset(essay_df, tokenizer, 512)

    #Creates dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    #Finds scores
    scores = model.predict(loader)[0]

    #Generates feedback
    feedback = Feedback.generateFeedback(essay, scores, openai_client, "gpt-3.5-turbo")

    formatted_results = []
    for i in range(len(categories)):
        results_by_category[categories[i]]= (f"### {categories[i]}\n\n**Score**: {scores[i]}\n\n**Feedback**:\n\n{feedback[i]}")
    
    return "Results generated."

def show_section(category):
    if category in results_by_category:
        return results_by_category[category]
    else:
        return "Please process an essay first."

with gr.Blocks() as demo:
    essay_input = gr.Textbox(label="Enter your essay", lines=10)
    status = gr.Markdown("")
    output = gr.Markdown()

    with gr.Row():
        process_button = gr.Button("Process Essay")
    
    #Can click each button to view feedback for that section
    with gr.Row():
        ideas_btn = gr.Button("Ideas")
        org_btn = gr.Button("Organization")
        style_btn = gr.Button("Style")
        conv_btn = gr.Button("Conventions")

    process_button.click(getClassifierScores, inputs=[essay_input], outputs=[status])
    ideas_btn.click(show_section, inputs=gr.State("Ideas"), outputs=output)
    org_btn.click(show_section, inputs=gr.State("Organization"), outputs=output)
    style_btn.click(show_section, inputs=gr.State("Style"), outputs=output)
    conv_btn.click(show_section, inputs=gr.State("Conventions"), outputs=output)

demo.launch()