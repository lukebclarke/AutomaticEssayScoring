from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch

class CustomBERTModel(nn.Module):
    def __init__(self, num_traits, num_classes, dropout):
        super(CustomBERTModel, self).__init__()
        self.config = AutoConfig.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained('bert-base-uncased', config=self.config)
        self.dropout = nn.Dropout(dropout)

        #Define separate classifier heads for each trait
        self.classifier_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, num_classes)
            ) for _ in range(num_traits)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        #Gets the hidden states of the [CLS] token
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        #Passes them through the classifier heads
        logits = [head(pooled_output) for head in self.classifier_heads]

        return torch.stack(logits, dim=1)