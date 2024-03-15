from omegaconf import OmegaConf as om
from omegaconf import DictConfig
from typing import cast
import sys 
import os
sys.path.append(os.path.abspath("../.."))

from main import build_model
import src.create_bert as bert_module
import src.create_model as model_module


yaml_path = "../../yamls/pretrain/micro_dna_noglu_monarch-mixer-pretrain-786dim-80m-parameters.yaml"

with open(yaml_path) as f:
    cfg = om.load(f)
cfg = cast(DictConfig, cfg)
print(cfg.max_duration)
model = model_module.create_model(cfg.model.get("model_config"))
#model = bert_module.create_bert_mlm(model_config = cfg.model.get("model_config", None))

from datasets import load_from_disk
from collate import DataCollatorForLanguageModelingSpan
from transformers import AutoTokenizer, Trainer, TrainingArguments

#Load model directly
os.environ["WANDB_PROJECT"] = "singlesamplednam2"

tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")


dataset = load_from_disk("../../micro")
dataset = dataset.remove_columns(["species_name", "__index_level_0__"])

data_collator = DataCollatorForLanguageModelingSpan(tokenizer, mlm=True, mlm_probability = 0.02, span_length = 6)

training_args = TrainingArguments(
    output_dir="./results",
    
    max_steps=20000,
    
    seed=17,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    
    logging_strategy="steps",
    logging_steps=1,
    
    remove_unused_columns=False,
    evaluation_strategy="no",
    
    #dataloader_num_workers=0,
    run_name="repo_model_1sample_noglu",
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
