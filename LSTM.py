# LSTM.py
# pytorch lstm for evaluating question 2a Tweet Categorisation task.
# By Bryce Dowie 3281924, 2023

import torch
import torch.nn as nn
import datasets
import torchtext
import numpy as np

# load onto gpu if possible

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is available")
    
else:
    device = torch.device("cpu")
    

## Training parameters
global_hidden_size = 64
epochs = 20
    
# Loads training and test datasets
training_dataset = datasets.load_dataset("cardiffnlp/tweet_topic_single", split = "train_coling2022_random")

test_dataset = datasets.load_dataset("cardiffnlp/tweet_topic_single", split = "test_coling2022_random")

### Preprocess targets to array format
# I guess this should be batched or something? whatever

output_length = 6
 
# already implemented preprocess apparently
 
    #def preprocess(target, length):    
    #    output = np.zeros(length)
        
    #    output[target] = 1
        
    #    return output
    
#def batch_preprocess(targets, length):    
#    output = np.zeros((targets.size, length))
    
#    for i in range(targets.size):
#        output[i][targets[i]] = 1
    
#    return output

## Tokenise Data


tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

def tokenize(text, tokenizer):
    tokens = tokenizer(text["text"])
    return {"tokens": tokens}

# for each dataset...

# already implemented

    #for entries in training_dataset:
    #    entries["label"] = preprocess(entries["label"], output_length)
        
    #for entries in test_dataset:
    #    entries["label"] = preprocess(entries["label"], output_length)
    
training_dataset = training_dataset.map(tokenize, fn_kwargs = {"tokenizer": tokenizer})

test_dataset = test_dataset.map(tokenize, fn_kwargs = {"tokenizer": tokenizer})

# build vocab from training data
vocab = torchtext.vocab.build_vocab_from_iterator(training_dataset["tokens"], min_freq = 5, specials = ["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])
pad_index = vocab["<pad>"]

## Numericalise Datasets

# "grabbed from lstm word embedding"

    #def numericalize(dataset, vocab):
    #    ids = [vocab[token] for token in dataset["text"]]
    #    return {"ids": ids}

    #training_dataset = training_dataset.map(numericalize, fn_kwargs = {"vocab": vocab})

### Batching the datasets or something

bloop = torchtext.vocab.GloVe()

pad_length = 99
  
def pad_and_glove(tokens):
    token_list = tokens
    
    # pad
    if len(token_list) < pad_length:
        token_list += [""] * (pad_length - len(token_list))
        
    token_list = bloop.get_vecs_by_tokens(token_list)
        
    return token_list

# lengths holds the length of each tweet before padding

training_lengths = [len(tokens) for tokens in training_dataset["tokens"]]
test_lengths = [len(tokens) for tokens in test_dataset["tokens"]]

# data holds all the vectorised tokens

training_data = [pad_and_glove(tokens) for tokens in training_dataset["tokens"]]

test_data = [pad_and_glove(tokens) for tokens in test_dataset["tokens"]]

#for i in range (len(data)):
#    data[i] = training_dataset["tokens"][i]

# targets holds the values for all the labels, for comparing against the predictions

training_targets = np.zeros((len(training_dataset), output_length))
    
for i in range (len(training_targets)):
    training_targets[i][training_dataset["label"][i]] = 1

test_targets = np.zeros((len(test_dataset), output_length))

for i in range (len(test_targets)):
    test_targets[i][test_dataset["label"][i]] = 1

## convert it all to tensors



training_lengths = torch.from_numpy(np.asarray(training_lengths))
training_data = torch.stack([entries for entries in training_data]).to(device)
training_targets = torch.from_numpy(training_targets).type(torch.FloatTensor).to(device)

test_lengths = torch.from_numpy(np.asarray(test_lengths))
test_data = torch.stack([entries for entries in test_data]).to(device)
test_targets = torch.from_numpy(test_targets).type(torch.FloatTensor).to(device)

# print(training_data)
    
print("data preprocessed")

## LSTM Implementation

class LSTM(nn.Module):
    def __init__(self, input):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input, hidden_size = global_hidden_size, num_layers = 1, bidirectional = False)
        self.linear = nn.Linear(in_features = global_hidden_size, out_features = 6)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, text, length):
        ## pack the data
        text = nn.utils.rnn.pack_padded_sequence(text, length, batch_first = True, enforce_sorted = False)
        
        ## do the lstm part
        _, (text, _) = self.lstm(text)
        
        text = text[-1]
        
        text = torch.tanh(text)
        
        ## linear layer to convert to a useable outptu
        text = self.linear(text)
        
        text = self.softmax(text)
        
        return text

network = LSTM(input = 300)

# from the comp3330 thingo

optimizer = torch.optim.Adam(network.parameters(), lr = 0.01)

#def loss_fn(predictions, targets):
#    return np.mean(np.power(np.subtract(targets, predictions), 2))

# loss function used and adapted from Ethan Brown, casual academic at UNSW

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self._mse = nn.MSELoss()
    
    def forward(self, output, target):
        return self._mse(output, target).float()
    
lossFunc = loss()

# training function for each epoch

def train(network, training_data, training_lengths):
    network.train()
    
    losses = []
    
    optimizer.zero_grad()
    predictions = network(training_data, training_lengths)
    
    loss = lossFunc(predictions, training_targets)
    losses.append(loss.item())
    
    # Accuracy used from Lab LSTMExample.py
    
    accuracy = torch.sum(torch.argmax(predictions, dim = -1) == torch.argmax(training_targets, dim = -1)) / training_data.shape[0]
    
    loss.backward()
    optimizer.step()
    return losses, accuracy

# evaluation function for the end of each epoch

def evaluate(network, test_data, test_lengths):
    network.eval()
    
    losses = []
    
    with torch.no_grad():
        predictions = network(test_data, test_lengths)
        loss = lossFunc(predictions, test_targets)
        losses.append(loss.item())
        
        # Accuracy used from Week 8 Lab LSTMExample.py
    
        accuracy = torch.sum(torch.argmax(predictions, dim = -1) == torch.argmax(test_targets, dim = -1)) / test_data.shape[0]
        
        
    return losses, accuracy


# results stored from each epoch

training_losses = []
validation_losses = []

training_accuracies = []
validation_accuracies = []

# runs the lstm across the specified number of epochs

for epoch in range(epochs):
    training_loss, training_accuracy = train(network, training_data, training_lengths)
    
    print("||epoch ", epoch, "||:")
    print("training loss:   ", training_loss)
    print("training accuracy:   ", training_accuracy)
    
    training_losses.append(training_loss)
    training_accuracies.append(training_accuracy)
    
test_loss, test_accuracy = evaluate(network, test_data, test_lengths)
    
validation_losses.append(test_loss)
validation_accuracies.append(test_accuracy)
    
# plot the results, adapted from Week 8 Lab LSTMExample.py

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, figsize = (12, 8), sharex = True)

ax1.plot(training_losses, label = "training")
ax1.plot(validation_losses, label = "validation")
ax1.set_ylabel("Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(training_accuracies, label = "training")
ax2.plot(validation_accuracies, label = "validation")
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Epoch")
ax2.legend()

fig.savefig("figures")