import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from tqdm import tqdm
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_data(path):
    df = pd.read_csv(path)
    df = df[['Utterance', 'Emotion']].dropna()
    return df

train_df = load_data("train_sent_emo.csv")
dev_df = load_data("dev_sent_emo.csv")
test_df = load_data("test_sent_emo.csv")

label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['Emotion'])
dev_df['label'] = label_encoder.transform(dev_df['Emotion'])
test_df['label'] = label_encoder.transform(test_df['Emotion'])

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class MELDDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=64)
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MELDDataset(train_df['Utterance'], train_df['label'])
dev_dataset = MELDDataset(dev_df['Utterance'], dev_df['label'])
test_dataset = MELDDataset(test_df['Utterance'], test_df['label'])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

num_labels = len(label_encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

def train_model(epochs=3):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        total_loss = 0
        for batch in loop:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} - Avg loss: {total_loss / len(train_loader)}")

def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    acc = accuracy_score(labels, preds)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()
    print("Validation:")
    evaluate(dev_loader)
    print("Test:")
    evaluate(test_loader)

    # Save model and label encoder
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
    joblib.dump(label_encoder, "./model/label_encoder.pkl")
