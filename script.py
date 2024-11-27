import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd


s3_path = "s3://tutorial-project-multiclass-text-classification-bucket/training-data/newsCorpora.csv"
df = pd.read_csv(s3_path, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

df = df[['TITLE', 'CATEGORY']]

my_dict = {

    'e': 'Entertainment',
    'b': 'Business',
    't': 'Science',
    'm': 'Health'

}


df['CATEGORY'] = df['CATEGORY'].apply(lambda x: my_dict[x])
df = df.reset_index(drop=True)
print(df)

# For testing the model training
# df = df.sample(frac=0.05, random_state=1)
# df = df.reset_index(drop=True)

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))
df = df.reset_index(drop=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, index):
        title = str(self.data.iloc[index, 0])
        title = " ".join(title.split())
        
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = 'max_length',
            return_token_type_ids = True,
            truncations = True
        )
        ids = inputs['input_ids']
        mask['attention_mask']
        
        return {
            'ids':torch.tensor(ids, dtype=torch.long),
            'mask':torch.tensor(mask, dtype=torch.long),
            'targets':torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        }
    
    def __len__(self):
        return self.len
    
    
    
train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
train_dataset.reset_index(drop=True)


print('Full dataset: {}'.format(df.shape))
print('Train dataset: {}'.format(train_dataset.shape))
print('Test dataset: {}'.format(test_dataset.shape))



MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2

training_set = NewsDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = NewsDataset(test_dataset, tokenizer, MAX_LEN)

train_parameters = {
                    'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0,
                    }

test_parameters = {
                    'batch_size': TEST_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0,
                    }

training_loader = DataLoader(training_set, **train_parameters)
testing_loader = DataLoader(testing_set, **test_parameters)


class DistilBERTClf(torch.nn.Module):
    
    def __init__(self):
        super(DistilBERTClf, self).__init__()
        
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.activation_fn = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(768, 4)
        
    
    def forward(self, input_ids, attention_mask):
        
        out = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_state = out[0]
        
        pooler = hidden_state[:,0]
        
        pooler = self.pre_classifier(pooler)
        
        pooler = self.activation_fn(pooler)
        
        pooler = self.dropout(pooler)
        
        output = self.classifier(pooler)
        
        return output
    
    

def calculate_acc(pred, gold):
    n_correct = (pred==gold).sum().item()
    return n_correct


def train(epoch, model, device, training_loader, optimizer, loss_function):
    epoch_loss = 0
    n_correct = 0
    nb_epoch_steps = 0
    nb_epoch_examples = 0
    model.train()
    
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        gold = data['targets'].to(device, dtype=torch.long)
        
        outputs = model(ids, mask)
        
        loss = loss_function(outputs, gold)
        
        epoch_loss += loss.item()
        pred_val, pred_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_acc(pred_idx, gold)
        
        nb_epoch_steps += 1
        nb_epoch_examples += gold.size(0)
        
        if _ % 5000 == 0:
            loss_step = epoch_loss/nb_epoch_steps
            acc_step = (n_correct*100)/nb_epoch_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {acc_step}")
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()

    epoch_loss = epoch_loss / nb_epoch_steps
    epoch_acc = (n_correct*100) / nb_epoch_examples
    
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_acc}")
    
    return
    

def validation(epoch, model, testing_loader, device, loss_function):
    
    model.eval()
    
    n_correct = 0
    epoch_loss = 0
    nb_epoch_steps = 0
    nb_epoch_examples = 0
    
    with torch.no_grad():
        
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            gold = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask).squeeze()

            loss = loss_function(outputs, gold)

            epoch_loss += loss.item()
            pred_val, pred_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_acc(pred_idx, gold)

            nb_epoch_steps += 1
            nb_epoch_examples += gold.size(0)

            if _ % 1000 == 0:
                loss_step = epoch_loss/nb_epoch_steps
                acc_step = (n_correct*100)/nb_epoch_examples
                print(f"Validation Loss per 1000 steps: {loss_step}")
                print(f"Validation Accuracy per 1000 steps: {acc_step}")


        epoch_loss = epoch_loss / nb_epoch_steps
        epoch_acc = (n_correct*100) / nb_epoch_examples

        print(f"Validation Loss Epoch: {epoch_loss}")
        print(f"Validation Accuracy Epoch: {epoch_acc}")
        
        return


def main():
    
    print("start")
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--epochs", type=int, default=10)
    # args = parser.parse_arg()
    # args.epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    model = DistilBERTClf()
    model.to(device)
    
    LEARNING_RATE = 1e-05
    optimizer = torch.optim.Adam(params=model.parameters, lr=LEARNING_RATE)
    
    loss_function = torch.nn.CrossEntropy()
    
    EPOCHS = 4
    
    for epoch in range(EPOCHS):
        print(f"starting epoch: {epoch}")
        
        train(epoch, model, device, training_loader, optimizer, loss_function)
        
        validation(epoch, model, testing_loader, device, loss_function)
    
    output_dir = os.environ['SM_MODEL_DIR']
    
    output_model_file = os.path.join(output_dir, 'pytorch_distilbert_news.bin')
    
    output_vocab_file = os.path.join(output_dir, 'vocab_distilbert_news.bin')
    
    torch.save(model.state_dict(), output_model_file)
    
    tokenizer.save_vocabulary(output_vocab_file)

    
if __name__ == '__main__':
    
    main()