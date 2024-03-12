from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForMaskedLM, BertConfig, BertModel, BertForPreTraining, BertForMaskedLM, AutoModel, PretrainedConfig, AutoConfig

import sys 
import os
sys.path.append(os.path.abspath(".."))

from collate import DataCollatorForLanguageModelingSpan


dataset = load_from_disk("../300")
dataset = dataset.remove_columns(["species_name", "__index_level_0__"])


tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")

# This way we don't load weights
# https://stackoverflow.com/questions/65072694/make-sure-bert-model-does-not-load-pretrained-weights
# TODO AutConfig or AutoModel? i guess it doesn't matter
config = PretrainedConfig.from_pretrained("togethercomputer/m2-bert-80M-2k")
model = BertForMaskedLM(config)

os.environ["WANDB_PROJECT"] = "singlesamplednam2"

data_collator = DataCollatorForLanguageModelingSpan(tokenizer, mlm=True, mlm_probability = 0.02, span_length = 6)

training_args = TrainingArguments(
    output_dir="./results",
    
    max_steps=100000,
    
    seed=17,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=32,
    
    logging_strategy="steps",
    logging_steps=1,
    
    evaluation_strategy="no",
    
    #dataloader_num_workers=4,
    #dataloader_prefetch_factor=2,
    run_name="300set_bs1024_monarchhf",
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator
)

trainer.train()