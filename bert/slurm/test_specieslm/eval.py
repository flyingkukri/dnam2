from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoModelForMaskedLM, BertConfig, BertModel, BertForPreTraining, BertForMaskedLM, AutoModel, PretrainedConfig, AutoConfig
import pickle

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
specieslm = AutoModelForMaskedLM.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")

specieslm.eval()
data_collator = DataCollatorForLanguageModelingSpan(tokenizer, mlm=True, mlm_probability = 0.02, span_length = 6)

os.environ["WANDB_PROJECT"] = "singlesamplednam2"

trainingArguments = TrainingArguments(   
    output_dir="eval_res",
    run_name="2batch64_monarchhf",
    report_to="wandb",
    eval_accumulation_steps=16,
)

trainer = Trainer(
    args=trainingArguments,
    model=specieslm,
    tokenizer=tokenizer,
    data_collator=data_collator
)

predictions = trainer.predict(dataset["train"])

import pickle 
with open("eval.pickle", "wb") as outfile:
    pickle.dump(predictions, outfile)