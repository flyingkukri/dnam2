print('started script')

from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
import pandas as pd

#dataset_path = "/s/project/semi_supervised_multispecies/all_fungi_reference/fungi/Annotation/Sequences/AAA_Concatenated/pretokenized/speciesdownstream300"
dataset_path = "../../batch"
print('started loading dataset')
dataset = load_from_disk(dataset_path)
dataset = dataset["train"]

#dataset = dataset[:10]

print('loaded dataset')

def delete_2nd_fun (example):
    del example["input_ids"][1]
    return example

print('deleting the second (species) token')
updated_dataset = dataset.map(delete_2nd_fun)
print('deleted the second token')


tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision="downstream_species_lm")
start_species_ids = tokenizer.convert_tokens_to_ids(["GGGGGG"])[0]
species_col = dataset["species_name"]
print('extracted species column')
species_ids = tokenizer.convert_tokens_to_ids(species_col)
print('converted species tokens to ids')
species_ids = [species_id - start_species_ids for species_id in species_ids]
print('subtracted_starting species id')
# delete the species token in the input_ids

updated_dataset = updated_dataset.add_column("species_ids", species_ids)
print('added new species embedding column to the dataset')

dataset.save_to_disk("../../batch_embed")
print("saved to disk")
