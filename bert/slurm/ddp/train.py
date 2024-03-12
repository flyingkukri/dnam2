from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForMaskedLM, BertConfig, BertModel, BertForPreTraining, BertForMaskedLM, AutoModel, PretrainedConfig, AutoConfig

import sys 
import os
sys.path.append(os.path.abspath("../.."))

from collate import DataCollatorForLanguageModelingSpan


dataset = load_from_disk("/s/project/semi_supervised_multispecies/all_fungi_reference/fungi/Annotation/Sequences/AAA_Concatenated/pretokenized/speciesdownstream300")
dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.remove_columns(["species_name", "__index_level_0__"])


tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")

# This way we don't load weights
# https://stackoverflow.com/questions/65072694/make-sure-bert-model-does-not-load-pretrained-weights
# TODO AutConfig or AutoModel? i guess it doesn't matter
config = PretrainedConfig.from_pretrained("togethercomputer/m2-bert-80M-2k")
config.vocab_size = tokenizer.vocab_size
model = BertForMaskedLM(config)

os.environ["WANDB_PROJECT"] = "singlesamplednam2"

data_collator = DataCollatorForLanguageModelingSpan(tokenizer, mlm=True, mlm_probability = 0.025, span_length = 6)

training_args = TrainingArguments(
    output_dir="./results/mp",
    
    max_steps=100000,
    
    seed=17,
   # per_device_train_batch_size=64,
   # per_device_eval_batch_size=64,
   # gradient_accumulation_steps=16,
    
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=32,
    
    learning_rate=4.0e-4,
    adam_epsilon=1.0e-6,
    adam_beta1=0.9,
    adam_beta2=0.98,
    
    warmup_steps=10000,
    
    logging_strategy="steps",
    logging_steps=1,
    
    evaluation_strategy="steps",
    eval_steps=2000,
    
    #dataloader_num_workers=4,
    #dataloader_prefetch_factor=2,
    run_name="fullset_bs1024_6gpu_monarchhf",
    report_to="wandb",
    
    fp16=True,
    
  #  hub_model_id="fullds",
    
  #  push_to_hub=True,
  #  hub_token="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"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator
)

trainer.train()

# data_collator = DataCollatorForLanguageModelingSpan(tokenizer, mlm=True, mlm_probability = 0.033333, span_length = 6)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator
)

trainer.train()


model.save_pretrained("model")
