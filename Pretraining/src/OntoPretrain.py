"""PreTraining BERT using Onto2Sen Generated Biomedical ontologies corpus (MLM task)

"""

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForMaskedLM
import wandb

wandb.init(project="project-name")

"""## Model Naming"""

model_name = "bert-base-uncased"

print(f'Pre Training on {model_name}')

corpus_path = "path-to-onto2sen-corpus"

"""## Tokenizer"""

tokenizer = AutoTokenizer.from_pretrained(model_name)
print('######### Tokenizing Created successfully !!! ')

"""## Model Setup"""

model = AutoModelForMaskedLM.from_pretrained(model_name)
print('######### Model Created successfully !!! ')

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path= corpus_path,
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

print('######### Tokenizing done successfully !!! ')

"""## Training"""

epochs = 3
batch_size = 8

OUTPUT_DIR = f"model/bert-pretrained-outputdir-e{epochs}-b{batch_size}"

wandb.config = {
  "epochs": epochs,
  "batch_size": batch_size
}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    save_steps=10000,
    save_total_limit=2,
    seed=1,
    report_to='wandb'
)


print('######### Arguments set successfully !!! ')

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

print('######### Trainer created successfully !!! ')

"""## Training Cycle"""

trainer.train()


print('######### Training done successfully !!! ')

"""## Saving Model"""

model_save_path = f"model/pubmed-pretrained-e{epochs}-b{batch_size}"

model.save_pretrained(model_save_path)

tokenizer.save_pretrained(model_save_path)

print('######### Model saved successfully !!! ')
print(f'Pretraining completed, model saved at {model_save_path}')
