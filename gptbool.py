"""
BoolQ using GPT2
Source Code: https://huggingface.co/docs/transformers/model_doc/gpt2

"""
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)

tokenizer.pad_token = tokenizer.eos_token

from transformers import GPT2ForSequenceClassification, AdamW, GPT2Config
model = GPT2ForSequenceClassification.from_pretrained(
    "gpt2", 
    num_labels = 2,  
    output_attentions = False, 
    output_hidden_states = False  
)
model.config.pad_token_id = model.config.eos_token_id

model.cuda()

def load_data(tokenizer, questions, passages, max_length):
    #Encode the question/passage pairs into features than can be fed to the model
    input_ids = []
    attention_masks = []

    for question, passage in zip(questions, passages):
        encoded_data = tokenizer.encode_plus(question, passage, truncation=True, max_length=max_length, pad_to_max_length=True, truncation_strategy="longest_first")
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)
    return np.array(input_ids), np.array(attention_masks)


train_data_df = pd.read_csv("/home/csgrads/rao00134/SuperGlue-tasks-using-BERT/dataset/BoolQ/train.csv", header=None, names=('question','passage','label','idx'))
dev_data_df = pd.read_csv("/home/csgrads/rao00134/SuperGlue-tasks-using-BERT/dataset/BoolQ/val.csv", header=None,names=('question','passage','label','idx'))


passages_train = train_data_df.passage.values
questions_train = train_data_df.question.values
answers_train = train_data_df.label.values.astype(int)
print(answers_train)

passages_dev = dev_data_df.passage.values
questions_dev = dev_data_df.question.values
answers_dev = dev_data_df.label.values.astype(int)

# Encoding data
max_seq_length = 256
input_ids_train, attention_masks_train = load_data(tokenizer, questions_train, passages_train, max_seq_length)
input_ids_dev, attention_masks_dev = load_data(tokenizer, questions_dev, passages_dev, max_seq_length)

train_features = (input_ids_train, attention_masks_train, answers_train)
dev_features = (input_ids_dev, attention_masks_dev, answers_dev)

# Building Dataloaders
batch_size = 32

train_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in train_features]
dev_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in dev_features]

train_dataset = TensorDataset(*train_features_tensors)
dev_dataset = TensorDataset(*dev_features_tensors)

train_sampler = RandomSampler(train_dataset)
dev_sampler = SequentialSampler(dev_dataset)

train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=batch_size)

import time
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

# Number of EPOCHS
EPOCHS = 4
total_steps = len(train_dataloader) * EPOCHS
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)


def accuracy(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc


#for _ in tqdm(range(epochs), desc="Epoch"):
for epoch in range(EPOCHS):

  # Training
  model.train()

  # Reset the total loss and accuracy for this epoch.
  total_train_loss = 0
  total_train_acc = 0
  
  # Measure how long the training epoch takes.
  start = time.time()

  for step, batch in enumerate(train_dataloader):
      # Unpack this training batch from our dataloader.
      input_ids = batch[0].to(device)
      attention_masks = batch[0].to(device)
      labels = batch[1].to(device)  

      #clear any previously calculated gradients before performing a backward pass
      optimizer.zero_grad()

      loss, prediction = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels).values()
      acc = accuracy(prediction, labels)

      # Accumulate the training loss and accuracy over all of the batches so that we can
      # calculate the average loss at the end
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

      # Perform a backward pass to calculate the gradients.
      loss.backward()

      # Clip the norm of the gradients to 1.0.
      # This is to help prevent the "exploding gradients" problem.
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      # Update parameters and take a step using the computed gradient.
      optimizer.step()

      # Update the learning rate.
      scheduler.step()

  # Calculate the average accuracy and loss over all of the batches.
  train_acc  = total_train_acc/len(train_dataloader)
  train_loss = total_train_loss/len(train_dataloader)

  # Put the model in evaluation mode
  model.eval()

  total_val_acc  = 0
  total_val_loss = 0

  with torch.no_grad():
    for batch in dev_dataloader:

      #clear any previously calculated gradients before performing a backward pass
      optimizer.zero_grad()

      # Unpack this validation batch from our dataloader.
      input_ids = batch[0].to(device)
      attention_masks = batch[0].to(device)
      labels = batch[1].to(device) 

      #Get the loss and prediction
      loss, prediction = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels).values()

      # Calculate the accuracy for this batch
      acc = accuracy(prediction, labels)

      # Accumulate the validation loss and Accuracy
      total_val_loss += loss.item()
      total_val_acc  += acc.item()

  # Calculate the average accuracy and loss over all of the batches.
  val_acc  = total_val_acc/len(dev_dataloader)
  val_loss = total_val_loss/len(dev_dataloader)

  end = time.time()
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)

  print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
  print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


#Load Test data
test_data_df = pd.read_csv("/home/csgrads/rao00134/SuperGlue-tasks-using-BERT/dataset/BoolQ/test.csv", header=None, names=('passage','question','idx'))

passage_list = test_data_df['passage'].to_list()
question_list = test_data_df['question'].to_list()

# Prediction
def predict(question, passage):
  sequence = tokenizer.encode_plus(question, passage, return_tensors="pt")['input_ids'].to(device)
  logits = model(sequence)[0]
  probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
  proba_yes = round(probabilities[1], 3)
  proba_no = round(probabilities[0], 3)
  print(f"Question: {question}, True: {proba_yes}, False: {proba_no}")


for i in range(10):
  question = question_list[i]
  passage = passage_list[i]
  predict(question,passage)