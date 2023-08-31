# Fine-tuning a model on a multiple choice task WITHOUT CONTEXT

import os
import torch
import transformers
from tqdm.auto import tqdm
import wandb

wandb.init(project="project_name")

print('Fine tuning the model on MedMCQA dataset....')

### Model Naming

model_name = "BioOntoBERT"

print("#########################")
print(f'Using Model - {model_name}')

finetuned_dir_path = "models/finetuned"
finetuned_model_name = f"{finetuned_dir_path}/bert_test"
  
if not os.path.exists(finetuned_model_name):
    os.makedirs(finetuned_model_name)

"""## PreProcessing

### Loading Dataset
"""

from datasets import load_dataset, load_metric

"""`load_dataset` will cache the dataset to avoid downloading it again the next time you run this cell."""

print('loading medmcqa dataset from hugging face dataset....')
datasets = load_dataset("medmcqa")
print('dataset loaded !')


# Renaming as the BERT will know only the one's with name 'label' or 'labels'
datasets = datasets.rename_column("cop", "label")

print(datasets)

### Show one Example

def show_one(example):
    print(f"  A - {example['question']} {example['opa']}")
    print(f"  B - {example['question']} {example['opb']}")
    print(f"  C - {example['question']} {example['opc']}")
    print(f"  D - {example['question']} {example['opd']}")
    print(f"\nGround truth: option {['A', 'B', 'C', 'D'][example['label']]}")
    print(f"Explanation : {example['exp']}")

print('Showing some examples from training set')
show_one(datasets["train"][0])

print('Showing some examples from validation set')
show_one(datasets['validation'][0])

print('Showing some examples from testing set')
show_one(datasets["train"][5])

"""## Tokenizer

### Loading Tokenizer
"""

from transformers import BertTokenizer,AutoTokenizer


print("#########################")
print(f'loading tokenizer for {model_name} ...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('tokenizer loaded !')
print("#########################")

### Preprocess Function

ending_names = ["opa", "opb", "opc", "opd"]

def preprocess_function(examples):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first = [[str(qs)] * 4 for qs in examples["question"]]
    first = sum(first, [])
    second =[]
    siz = len(examples['opa'])
    for i in range(siz):
      temp=[]
      for end in ending_names:
        temp.append(examples[end][i])
      second.append(temp)
    second = sum(second, [])
    tokenized_examples = tokenizer(first, second ,padding=True, truncation=True,max_length=50, add_special_tokens = True)
    return {k: [v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

"""### Encoding Dataset"""

print("#########################")
print('Tokenizing the dataset.....')
encoded_datasets = datasets.map(preprocess_function, batched = True)
print('tokenzing dataset completed !')
print("#########################")
"""### Saving the Tokenized Dataset

Saving in model's dir only
"""
encoded_datasets.save_to_disk(finetuned_model_name)
print("#########################")
print(f'tokenized dataset saved to {finetuned_model_name}')
print("#########################")

"""
## Fine-tuning the model

### Setup Model

"""

from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

print("#########################")
print(f'loading {model_name} model ....')
model = AutoModelForMultipleChoice.from_pretrained(model_name)
print('model loaded !')
print("#########################")

OUTPUT_DIR = f'{finetuned_dir_path}/pubmed-finetuned-wocontext-OUTPUTDIR-v1-b32-e100'

learning_rate = 5e-5
epochs = 1
weight_decay=0.01
batch_size = 8


print("#########################")
print('training args set')
print(f'Batch size = {batch_size}')
print(f'Epochs = {epochs}')
print(f'Learning rate = {learning_rate}')
print(f'Weight Decay = {weight_decay}')
print("#########################")

wandb.config = {
  "learning_rate": learning_rate,
  "epochs": epochs,
  "batch_size": batch_size
}

# model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    evaluation_strategy = "epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=weight_decay,
    report_to='wandb'
)

### DataCollator

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


accepted_keys = ["input_ids", "attention_mask", "label"]
features = [{k: v for k, v in encoded_datasets["train"][i].items() if k in accepted_keys} for i in range(10)]
batch = DataCollatorForMultipleChoice(tokenizer)(features)

print("#########################")
print('Data collator created !!!')
print("#########################")

"""
### Compute Metrics setup
"""

import numpy as np

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

### Trainer

print("#########################")
print('Trainer being loaded...')
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics
)

print('Trainer loaded !!! ')
print("#########################")

"""We can now finetune our model by just calling the `train` method:"""

"""### Training cycle"""
print("#########################")
print('Training cycle start .....')
trainer.train()
print('Training completed !!!! ')
print("#########################")
### Saving Model

model.save_pretrained(finetuned_model_name)

tokenizer.save_pretrained(finetuned_model_name)

print("#########################")
print(f'Model & Tokenizer saved in {finetuned_model_name} ')

print('Fine tuning completed !')

### Evaluating model

from datasets import load_from_disk
# Here will get tokenized dataset
data_dir = finetuned_model_name 

from transformers import AutoModelForMultipleChoice,AutoTokenizer

new_model = AutoModelForMultipleChoice.from_pretrained(finetuned_model_name)
new_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)

new_encoded_datasets = load_from_disk(data_dir)

from transformers import Trainer

reloaded_trainer = Trainer(
                    model = new_model,
                    tokenizer = new_tokenizer
                    )


predictions = reloaded_trainer.predict(new_encoded_datasets['validation'])

print(predictions.predictions.shape, predictions.label_ids.shape)

import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)

import evaluate

metric = evaluate.load("accuracy")

result = metric.compute(predictions=preds, references=predictions.label_ids)
print(result)

with open("output.txt", "a") as f:
  f.write(str(finetuned_model_name) + 'epochs - ' + str(epochs) + 'batch size - ' + str(batch_size) + '     ' +str(result))

print('Results Stored !!!')
