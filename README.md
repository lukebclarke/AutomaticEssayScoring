# AutomaticEssayScoring
A first-class dissertation project that automatically scores student essays and generates written feedback based on the score. 

There are two main components to the project:
- Automatic Essay Scoring in 4 different categories (ideas, conventions, organisation, style)
  - This was achieved by fine-tuning the BERT language model
  - Hyperparameter Exploration and K-Fold Cross Validation training was employed to improve model performance
- LLM Feedback Generation
  - A selection of LLMs are given essays and their corresponding category scores, and are tasked with generating feedback based on each category
  - Various prompting techniques were explored to improve performance
 
All data for training and evaluation was ASAP Dataset provided by Kaggle (https://kaggle.com/competitions/asap-aes)
