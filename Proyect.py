import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import os

# ==========================================
# 1. ARQUITECTURA Y DATASET
# ==========================================
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
        l1, l2, l3 = self.head_st1(pooled_output), self.head_st2(pooled_output), self.head_st3(pooled_output)
        
        loss = None
        if labels is not None:
            fn = nn.CrossEntropyLoss()
            loss = fn(l1, labels[:, 0]) + fn(l2, labels[:, 1]) + fn(l3, labels[:, 2])
        return {"loss": loss, "logits": (l1, l2, l3)} if loss is not None else (l1, l2, l3)

class MultiTaskDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

# ==========================================
# 2. FUNCIÓN PRINCIPAL (PROTEGE EL PROCESO)
# ==========================================
def main():
    MODEL_NAME = "xlm-roberta-base"
    MAX_LEN = 128 
    BATCH_SIZE = 16 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16_enabled = True if device == "cuda" else False
    
    print(f"✅ Hardware detectado: {device.upper()}")

    ST1_MAP = {'Support': 1, 'Supportive': 1, 'Non Support': 0, 'Non-Supportive': 0}
    ST2_MAP = {'Group': 0, 'Individual': 1, 'No': 0}
    ST3_MAP = {'Black Community': 0, 'LGBTQ': 1, 'Nation': 2, 'Other': 3, 'Religion': 4, 'Women': 5, 'No': 3}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MultiTaskSocialModel(MODEL_NAME).to(device)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    configs = [
        {'lang': 'en', 'train': 'train-english.csv', 'test': 'test_phase_english.csv', 'text': 'text'},
        {'lang': 'es', 'train': 'train-spanish.csv', 'test': 'test_phase_spanish.csv', 'text': 'comment'}
    ]

    final_losses = {}

    for item in configs:
        if not os.path.exists(item['train']):
            print(f"⚠️ Archivo {item['train']} no encontrado. Saltando...")
            continue

        print(f"\n🔥 Entrenando Idioma: {item['lang'].upper()}")
        df = pd.read_csv(item['train'])
        
        labels = np.stack([
            df['task1'].map(ST1_MAP).fillna(0), 
            df['task2'].map(ST2_MAP).fillna(0), 
            df['task3'].map(ST3_MAP).fillna(3)
        ], axis=1).astype(int)
        
        encodings = tokenizer(df[item['text']].astype(str).tolist(), truncation=True, padding=True, max_length=MAX_LEN)
        train_ds = MultiTaskDataset(encodings, labels)

        args = TrainingArguments(
            output_dir=f'./tmp_{item["lang"]}',
            num_train_epochs=1,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=3e-5,
            fp16=fp16_enabled,
            save_strategy="no",
            dataloader_num_workers=0, # 0 evita errores de multiprocessing en Windows
            report_to="none"
        )

        trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=data_collator)
        train_result = trainer.train()
        final_losses[item['lang']] = train_result.training_loss

        print(f"📝 Generando submission_{item['lang']}.csv...")
        test_df = pd.read_csv(item['test'])
        model.eval()
        
        all_p1, all_p2, all_p3 = [], [], []
        with torch.no_grad():
            for i in range(0, len(test_df), BATCH_SIZE):
                batch_texts = test_df[item['text']].iloc[i:i+BATCH_SIZE].astype(str).tolist()
                inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(device)
                l1, l2, l3 = model(inputs['input_ids'], inputs['attention_mask'])
                all_p1.extend(torch.argmax(l1, -1).cpu().numpy())
                all_p2.extend(torch.argmax(l2, -1).cpu().numpy())
                all_p3.extend(torch.argmax(l3, -1).cpu().numpy())

        pd.DataFrame({
            'id': test_df['id'],
            f"{item['lang']}_support_pred": all_p1,
            f"{item['lang']}_individual_pred": all_p2,
            f"{item['lang']}_multiclass_pred": all_p3
        }).to_csv(f"submission_{item['lang']}.csv", index=False)

    print("\n" + "="*30)
    print("📊 RESULTADOS PARA EL PAPER")
    print("="*30)
    for lang, loss in final_losses.items():
        print(f"Final Training Loss ({lang.upper()}): {loss:.4f}")
    print("="*30)

# ESTA LÍNEA ES LA QUE CORRIGE TU ERROR:
if __name__ == '__main__':
    main()