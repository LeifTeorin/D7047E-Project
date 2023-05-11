import pandas as pd
#from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv('preprocessed_hasoc_dataset_task2.csv', sep= ',')
df_test = pd.read_csv('preprocessed_hasoc_test_dataset_task2.csv', sep= ',')

label_encoder = LabelEncoder()

df["task_2"] = label_encoder.fit_transform(df["task_2"])
df_test["task_2"] = label_encoder.fit_transform(df_test["task_2"])

# Split the data into training and validation sets
train_texts, train_labels   = (df['tweet'], df['task_2'])
val_texts, val_labels = (df_test['tweet'], df_test['task_2'])
# Convert the labels to PyTorch tensors
# train_labels = torch.tensor(train_labels)
# val_labels = torch.tensor(val_labels)

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = torch.load('Bert-task_2.pt')

# Tokenize the texts and encode the labels
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)
train_labels_enc = torch.tensor(train_labels.tolist()).to(device)
val_labels_enc = torch.tensor(val_labels.tolist()).to(device)

# Create the PyTorch datasets
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              train_labels_enc)
val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']),
                            torch.tensor(val_encodings['attention_mask']),
                            val_labels_enc)

# Create the PyTorch data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader = list(train_loader)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_loader = list(val_loader)


# Evaluate the model on the validation set after each epoch
model.eval()
val_loss = 0
val_correct = 0
with torch.no_grad():
    for batch in val_loader:
        inputs = {'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device),
                    'labels': batch[2].to(device)}
        outputs = model(**inputs)
        #loss = outputs[0]
        #val_loss += loss.item()
        preds = torch.argmax(outputs[1], dim=1)
        val_correct += torch.sum(preds == batch[2])
    
#val_loss /= len(val_loader)
val_acc = val_correct / len(val_dataset)
    
#print(f'Test Loss: {val_loss:.4f}')
print(f'Test Accuracy: {val_acc:.4f}')
