import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline

import openai
from huggingface_hub import InferenceClient
from together import Together

categories = ['Ideas', 'Organization', 'Style', 'Conventions']
grade = "7th"
essay_prompt = "Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining. Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience."

def generateFeedback(essay, scores, client, model):
    responses = []

    #Generate feedback for each category
    for c in range(4):
        category = categories[c]
        category_score = scores[c]

        system_prompt = f"""You are a teacher of a {grade} class.

            Your students have been asked to write a Persuasive/Narrative/Expository essay about the following prompt: "{essay_prompt}"

            You are tasked with generating a paragraph of feedback related to **one** of four categories:
            1. Ideas
            2. Organization
            3. Style
            4. Conventions

            Each category has a 0-3 score, with 3 being the highest.

            Generate feedback **only** for the category '{category}' with a score of {category_score}.

            The feedback must:
            - Describe how the student performed
            - Mention a maximum of 2-3 specific mistakes, phrased as questions
            - Include an explanation of what the student struggled with
            - Only focus on the selected category
            - Be in paragraph form
            - Use second person point of view
            - Ignore placeholders like @NAME1 in the essay

            Respond with the paragraph of feedback.
        """

        if type(client) == openai.OpenAI or type(client) == Together: 
            prompt = f"{system_prompt}\n\nEssay:\n\"{essay}\"\n\n{category} Score: {category_score}"

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=2048
            )

            responses.append(f"{category} Score: {category_score}/3\n{response.choices[0].message.content}")
        else:
            raise Exception("Invalid client")
        
    return responses
