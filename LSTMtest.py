#pytorchlstmtest.py

import torch
import torch.nn as nn
import datasets
import torchtext
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is available")
    
else:
    device = torch.device("cpu")
    
# Loads training and test datasets
training_dataset = datasets.load_dataset("cardiffnlp/tweet_topic_single", split = "train_coling2022_random")

test_dataset = datasets.load_dataset("cardiffnlp/tweet_topic_single", split = "test_coling2022_random")

### Preprocess targets to array format
# I guess this should be batched or something? whatever

output_length = 6
    
def preprocess(target, length):    
    output = np.zeros(length)
    
    output[target] = 1
    
    return output
    
def batch_preprocess(targets, length):    
    output = np.zeros((targets.size, length))
    
    for i in range(targets.size):
        output[i][targets[i]] = 1
    
    return output

example_text = "Hello hello my name is bryce this is a string of words, what shall I do with these words? I can't tell you that because I am not sure so I will just make sure to write a bunch of them"

## Tokenise Data

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

def tokenize(text):
    tokens = tokenizer(text)
    return tokens

#*!* do this shit with batches or something idfk
#*!* it'd also be nice if it mapped, but I don't wanna put something in that I don't actually understand yet

for entries in training_dataset:
    entries["text"] = tokenize(entries["text"])
    entries["label"] = preprocess(entries["label"], output_length)
    
example_text = tokenize(example_text)

print(example_text)

print()
print()

## Build Vocabulary

#vocab = torchtext.vocab.build_vocab_from_iterator(iterator = training_dataset["text"], min_freq = 5)

vocab = torchtext.vocab.build_vocab_from_iterator(example_text, min_freq = 1)

print(vocab.get_itos())

#print(vocab.get_itos())

## Numericalise Datesets

# "grabbed from lstm word embedding"

def numericalize(dataset, vocab):
    ids = [vocab[token] for token in dataset["text"]]
    return ids
    
training_dataset["text"] = numericalize(training_dataset["text"], vocab)

## Convert Datasets to Torch Type

## Pad Words

## Wrapping

## LSTM Implementation

class LSTM():
    def __init__(inputs):
        rnn = nn.LSTM(input_size = inputs, hidden_size = 32, num_layers = 1, bidirectional = False)


#lstm = LSTM(inputs = 64).to(device)

## Define the optimiser and tell it what parameters to update, as well as the loss function

# takes the difference between the certainty of the correct answer for each result

def loss_fn(predictions, targets):
    return np.mean(np.power(np.subtract(targets, predictions), 2))

## Train function

## Evaluation Function

## Run training (Try just 5 epochs)

## Plot Data