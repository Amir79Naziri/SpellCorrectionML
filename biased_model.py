from datasets import load_dataset
from datasets import Dataset

import transformers
from transformers import AutoTokenizer
from transformers import BertForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import torch_default_data_collator
from transformers import EarlyStoppingCallback, IntervalStrategy
from transformers import Trainer, TrainingArguments

import torch
import numpy as np
import gc
import collections
from tqdm import tqdm


# check cuda
if not torch.cuda.is_available():
    print("No GPU available, using the CPU instead.")

print(transformers.__version__)


# data_path = '/mnt/disk1/users/naziri/train test datasets'
data_path = input("train test datasets: ")
original_model = input("input model: ")
biased_model = input("output model: ")
tokenizer_path = input("tokenizer_path: ")

# load dataset
print("load dataset ...")

correct_dataset = load_dataset(
    "text",
    data_files={"train": data_path + "/train" + "/correct_sentences.txt"},
    split="train",
)
dataset = load_dataset(
    "text",
    data_files={"train": data_path + "/train" + "/corrupted_sentences.txt"},
    split="train",
)
dataset = dataset.add_column("labels", correct_dataset["text"].copy())

del correct_dataset
torch.cuda.empty_cache()
gc.collect()
print(dataset)


# load tokenizer
print("load tokenizer ...")
model_checkpoint = "HooshvareLab/bert-base-parsbert-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# tokenizer.save_pretrained(tokenizer_path)


### batched must be False
def tokenize_function(tokenizer, dataset):
    final_text = []
    final_label = []
    for idx in tqdm(range(dataset.num_rows)):
        temp_text = []
        temp_label = []

        for text_word, label_word in zip(
            dataset[idx]["text"].split(), dataset[idx]["labels"].split()
        ):
            temp_result = tokenizer([text_word, label_word], padding="longest")[
                "input_ids"
            ]

            for i in range(2):
                temp_result[i].remove(2)
                temp_result[i].remove(4)

            temp_text.extend(temp_result[0])
            temp_label.extend(temp_result[1])

        temp_text.insert(0, 2)
        temp_text.append(4)
        temp_label.insert(0, 2)
        temp_label.append(4)

        final_text.append(temp_text)
        final_label.append(temp_label)
    return Dataset.from_dict({"input_ids": final_text, "labels": final_label})


tokenized_dataset = tokenize_function(tokenizer, dataset)

del dataset
torch.cuda.empty_cache()
gc.collect()
# print(tokenized_dataset)
# print(tokenized_dataset[0])

# grouping
print("group text ...")


def group_texts(data):
    block_size = 64
    concatenated_data = {key: sum(data[key], []) for key in data.keys()}

    total_length = len(concatenated_data[list(data.keys())[0]])

    new_total_length = (total_length // block_size) * block_size

    result = {
        key: [
            val[idx : idx + block_size]
            for idx in range(0, new_total_length, block_size)
        ]
        for key, val in concatenated_data.items()
    }

    return result


final_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=64,
    num_proc=4,
)


del tokenized_dataset
torch.cuda.empty_cache()
gc.collect()


# data collector
def whole_word_masking_data_collator_V2(features):
    wwm_probability = 0.15

    for feature in features:
        word_ids = feature["input_ids"]

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)

        indices_replaced = torch.bernoulli(torch.full((len(mapping),), 0.8)).bool()
        indices_random = (
            torch.bernoulli(torch.full((len(mapping),), 0.5)).bool() & ~indices_replaced
        )
        random_words = torch.randint(len(tokenizer), (len(mapping),), dtype=torch.long)

        # ERRORS (NOT MASK)
        for idx, (inp_ids, label) in enumerate(zip(input_ids, labels)):
            if inp_ids != label:
                new_labels[idx] = label

        # 15% RANDOM SELECTED
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()

            # 80% MASK
            if indices_replaced[word_id].item():
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id

            # 10% RANDOM
            elif indices_random[word_id].item():
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = random_words[word_id]

            # 10% NOT CHANGE
            else:
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]

        feature["labels"] = new_labels

    return torch_default_data_collator(features)


# split train, evaluation dataset
print("split train evaluation dataset ...")
final_dataset = final_dataset.train_test_split(test_size=0.2)
print(final_dataset)
# print(final_dataset["train"])
# print(final_dataset["train"][0])


# load model
print("load model ...")
model_checkpoint = "HooshvareLab/bert-base-parsbert-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = BertForMaskedLM.from_pretrained(original_model)
# model.save_pretrained(original_model)

# define trainer and args
training_args = TrainingArguments(
    biased_model,
    overwrite_output_dir=True,
    evaluation_strategy=IntervalStrategy.STEPS,  # "steps",
    save_steps=250,
    logging_steps=250,
    eval_steps=250,  # Evaluation and Save happens every 250 steps
    save_total_limit=2,  # Only 2 models are saved. best and last.
    report_to="all",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-7,
    weight_decay=0.01,
    load_best_model_at_end=True
    # push_to_hub=True,
    # hub_model_id="Amir79Naziri/bert-base-parsbert-uncased-finetuned",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["test"],
    data_collator=whole_word_masking_data_collator_V2,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("start training ...")
trainer.train()

trainer.save_model(biased_model)

print(trainer.state.best_model_checkpoint)

print("done. :)")
