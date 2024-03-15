from omegaconf import OmegaConf as om
from omegaconf import DictConfig
from typing import cast
import sys 
import os
import wandb
sys.path.append(os.path.abspath("../.."))

from main import build_model
import src.create_bert as bert_module
import src.create_model as model_module

from datasets import load_from_disk
from collate import DataCollatorForLanguageModelingSpan
from transformers import AutoTokenizer, Trainer, TrainingArguments


def train_with_set(name: str, samples: int):
    run_name = f'train_full_noglu'
    run = wandb.init(reinit=True,
                    project = "singlesamplednam2",
                    name=run_name)
    batch_size = samples
    gradient_accumulation_steps = 1
    if batch_size > 64:
        batch_size = 64
        gradient_accumulation_steps = samples // batch_size
        
    
    tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")

    yaml_path = "../../yamls/pretrain/micro_dna_noglu_monarch-mixer-pretrain-786dim-80m-parameters.yaml"

    with open(yaml_path) as f:
        cfg = om.load(f)
    cfg = cast(DictConfig, cfg)
    print(cfg.max_duration)
    model = model_module.create_model(cfg.model.get("model_config"), tokenizer)
    #model = bert_module.create_bert_mlm(model_config = cfg.model.get("model_config", None))



    #Load model directly
    os.environ["WANDB_PROJECT"] = "singlesamplednam2"



    dataset = load_from_disk("/s/project/semi_supervised_multispecies/all_fungi_reference/fungi/Annotation/Sequences/AAA_Concatenated/pretokenized/speciesdownstream300")
    dataset = dataset.train_test_split(test_size=0.1)
    dataset = dataset.remove_columns(["species_name", "__index_level_0__"])

    data_collator = DataCollatorForLanguageModelingSpan(tokenizer, mlm=True, mlm_probability = 0.02, span_length = 6)

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(run_name)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    training_args = TrainingArguments(
        output_dir=f'./results/{samples}',

        max_steps=100000,

        seed=17,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=32,
        per_device_eval_batch_size=64,

        logging_strategy="steps",
        logging_steps=1,

        remove_unused_columns=False,
        
        learning_rate=4.0e-4,
        adam_epsilon=1.0e-6,
        adam_beta1=0.9,
        adam_beta2=0.98,
    
        warmup_steps=10000,
        
        evaluation_strategy="steps",
        eval_steps=2000,
                
        #fp16=True,

        dataloader_num_workers=0,
        #run_name= run_name,
        report_to="wandb",
        use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator
    )


    trainer.train()
    wandb.finish()
    
train_with_set("microset", 2048)