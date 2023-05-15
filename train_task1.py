import pandas as pd
#from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from BERTModel import BertClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv('preprocessed_hasoc_dataset_task1.csv', sep= ',')

label_encoder = LabelEncoder()
df["task_1"] = label_encoder.fit_transform(df["task_1"])

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['tweet'], df['task_1'], test_size=0.1)

# Convert the labels to PyTorch tensors
# train_labels = torch.tensor(train_labels)
# val_labels = torch.tensor(val_labels)

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)


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




### Train the model ###
epochs = 2
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay = 0.03)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_f1 = 0
    for batch in train_loader:
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs[0]
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Compute f1_score
        preds = torch.argmax(outputs[1], dim=1)
        train_f1 += f1_score(batch[2].cpu(), preds.cpu(), average='macro')
    
    # Evaluate the model on the validation set after each epoch
    model.eval()
    val_loss = 0
    val_f1 = 0
    val_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[2].to(device)}
            outputs = model(**inputs)
            loss = outputs[0]
            val_loss += loss.item()
            preds = torch.argmax(outputs[1], dim=1)
            val_correct += torch.sum(preds == batch[2])

            # Compute f1_score
            val_f1 += f1_score(batch[2].cpu(), preds.cpu(), average='macro')
    
    train_loss /= len(train_loader)
    train_f1 /= len(train_loader)
    val_loss /= len(val_loader)
    val_f1 /= len(val_loader)
    val_acc = val_correct / len(val_dataset)
    
    print(f'Epoch {epoch + 1}:')
    print(f'Training Loss {train_loss:.4f}, Training F1: {train_f1:.4f}')
    print(f'Validation Loss {val_loss:.4f}, Validation F1: {val_f1:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')

torch.save(model, 'Bert-task_1.pt')