import torch
import evaluate
import transformers as tf
from datasets import load_dataset
import numpy as np


# Models:
# - roberta-base
# - bert-base-uncased
pretrained_model = "bert-base-uncased"
epoch = 5

# Load tweet topic dataset
dataset = load_dataset("cardiffnlp/tweet_topic_single")


# Define labels
LABEL2ID = {
    "arts_&_culture": 0,
    "business_&_entrepreneurs": 1,
    "pop_culture": 2,
    "daily_life": 3,
    "sports_&_gaming": 4,
    "science_&_technology": 5
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# Check which device is available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.has_mps:
    device = 'mps'
    # apple gpu acceleration
print('Using {}'.format(device))


# Define pretrained model and apply acceleration
model = tf.AutoModelForSequenceClassification.from_pretrained(
    pretrained_model, 
    num_labels=6,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
).to(torch.device(device))


# Prepare data
tokenizer = tf.AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)


# Define metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Define hyperparameters
training_args = tf.TrainingArguments(
    num_train_epochs=epoch,
    output_dir="{} model Epoch: {}".format(pretrained_model, epoch), 
    evaluation_strategy="epoch",
    save_strategy="epoch",
    use_mps_device=True if device=='mps' else False
)


# Pass parameters (ensuring to use specified datasets for training and testing)
trainer = tf.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train_coling2022"],#.shuffle(seed=42).select(range(10)),
    eval_dataset=tokenized_dataset["test_coling2022"],#.shuffle(seed=42).select(range(10)),
    compute_metrics=compute_metrics,
)

# Train the thing
trainer.train()

# Evaluate (using test set to report results)
trainer.evaluate(eval_dataset=tokenized_dataset["test_coling2022"])
