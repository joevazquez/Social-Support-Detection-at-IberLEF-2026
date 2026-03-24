# Código del paper para Social Support Detection at IberLEF 2026

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
import os

# CONFIGURACIÓN (Volvemos a XLM-RoBERTa porque en Colab la GPU lo aguanta)
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16 

# Mapeos
ST1_MAP = {'Support': 1, 'Supportive': 1, 'Non Support': 0, 'Non-Supportive': 0}
ST2_MAP = {'Group': 0, 'Individual': 1, 'No': 0}
ST3_MAP = {'Black Community': 0, 'LGBTQ': 1, 'Nation': 2, 'Other': 3, 'Religion': 4, 'Women': 5, 'No': 3}

# Arquitectura Multitarea
class MultiTaskSocialModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.head_st1 = nn.Linear(hidden_size, 2)
        self.head_st2 = nn.Linear(hidden_size, 2)
        self.head_st3 = nn.Linear(hidden_size, 6)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        logits_st1 = self.head_st1(pooled_output)
        logits_st2 = self.head_st2(pooled_output)
        logits_st3 = self.head_st3(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits_st1, labels[:, 0]) + loss_fct(logits_st2, labels[:, 1]) + loss_fct(logits_st3, labels[:, 2])
        return {"loss": loss, "logits": (logits_st1, logits_st2, logits_st3)} if loss is not None else (logits_st1, logits_st2, logits_st3)

class MultiTaskDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# Cargar datos (Asegúrate de haber subido los archivos a la carpeta de Colab)
df_es = pd.read_csv('train-spanish.csv').rename(columns={'comment': 'text'})
df_en = pd.read_csv('train-english.csv')
df = pd.concat([df_es, df_en], ignore_index=True)

labels = np.stack([
    df['task1'].map(ST1_MAP).fillna(0).values,
    df['task2'].map(ST2_MAP).fillna(0).values,
    df['task3'].map(ST3_MAP).fillna(3).values
], axis=1).astype(int)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encodings = tokenizer(df['text'].astype(str).tolist(), truncation=True, padding=True, max_length=MAX_LEN)

train_idx, val_idx = train_test_split(range(len(df)), test_size=0.1, random_state=42)
train_dataset = MultiTaskDataset({k: [v[i] for i in train_idx] for k, v in encodings.items()}, labels[train_idx])

model = MultiTaskSocialModel(MODEL_NAME)

training_args = TrainingArguments(
    output_dir='./res',
    num_train_epochs=3, 
    per_device_train_batch_size=BATCH_SIZE,
    save_strategy="no",
    learning_rate=2e-5,
    fp16=True, # ESTO ACTIVA LA VELOCIDAD DE LA GPU
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()

# Predicción final
def predict_and_save(test_path, text_col, lang):
    test_df = pd.read_csv(test_path)
    test_enc = tokenizer(test_df[text_col].astype(str).tolist(), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    model.eval()
    device = "cuda"
    model.to(device)
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in test_enc.items()}
        l1, l2, l3 = model(inputs['input_ids'], inputs['attention_mask'])
        p1, p2, p3 = torch.argmax(l1, -1), torch.argmax(l2, -1), torch.argmax(l3, -1)
    pd.DataFrame({'id': test_df['id'], f'{lang}_support_pred': p1.cpu(), f'{lang}_individual_pred': p2.cpu(), f'{lang}_multiclass_pred': p3.cpu()}).to_csv(f'submission_{lang}.csv', index=False)
    print(f"✅ Archivo {lang} generado.")

predict_and_save('test_phase_spanish.csv', 'comment', 'es')
predict_and_save('test_phase_english.csv', 'text', 'en')