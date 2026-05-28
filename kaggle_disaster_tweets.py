import os
import gc
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. AYARLAR VE DONANIM
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan donanım: {device}")

# EFSANE GERİ DÖNDÜ: Görev adamı BERT
model_name = "bert-base-uncased"
n_splits = 5 
max_length = 64
text_column = 'final_text' 

# ==========================================
# 2. YENİ VE AKILLI TEMİZLENMİŞ VERİLER
# ==========================================
train_path = "./DATA/train_bert_ready.csv"
test_path = "./DATA/test_bert_ready.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
test_ids = df_test['id'].values

# Güvenlik ağı
df_train[text_column] = df_train[text_column].fillna("")
df_test[text_column] = df_test[text_column].fillna("")

print(f"Eğitim verisi boyutu: {df_train.shape}")

# ==========================================
# 3. K-FOLD VE DATASET SINIFI
# ==========================================
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class DisasterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx]).long()
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'f1': f1_score(labels, predictions), 'accuracy': accuracy_score(labels, predictions)}

# ==========================================
# 4. EĞİTİM DÖNGÜSÜ (5 FOLD BERT)
# ==========================================
all_test_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train[text_column], df_train['target'])):
    print(f"\n--- FOLD {fold+1} BAŞLIYOR ---")
    
    train_texts = df_train[text_column].iloc[train_idx].astype(str).tolist()
    val_texts = df_train[text_column].iloc[val_idx].astype(str).tolist()
    train_labels = df_train['target'].iloc[train_idx].values
    val_labels = df_train['target'].iloc[val_idx].values
    
    train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    test_encodings = tokenizer(df_test[text_column].astype(str).tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    train_ds = DisasterDataset(train_encodings, train_labels)
    val_ds = DisasterDataset(val_encodings, val_labels)
    test_ds = DisasterDataset(test_encodings)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    
    
    training_args = TrainingArguments(
        output_dir=f'./results_fold_{fold}',
        num_train_epochs=2,
        per_device_train_batch_size=32,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    fold_preds = trainer.predict(test_ds).predictions
    all_test_preds.append(fold_preds)
    
    del model, trainer, train_ds, val_ds
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# 5. ENSEMBLE VE SUBMISSION
# ==========================================
print("\n🔥 Tüm foldlar tamamlandı, birleştiriliyor...")
avg_preds = np.mean(all_test_preds, axis=0)
final_labels = np.argmax(avg_preds, axis=-1)

submission_df = pd.DataFrame({'id': test_ids, 'target': final_labels})
submission_df.to_csv("submission_BERT_V2_Top50.csv", index=False) 


