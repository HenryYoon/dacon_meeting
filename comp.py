# -*- coding: utf-8 -*-

import argparse

from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer)

from datasets import load_metric, Dataset
from dacon_submit_api import dacon_submit_api

import numpy as np
import pandas as pd
import json
import os

from konlpy.tag import Kkma
import warnings
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--pretrain', type=str, default="ainize/kobart-news")
args = parser.parse_args()


pretrained = args.pretrain
batch_size = args.batch
epochs = args.epoch

encoder_max_length = 500
decoder_max_length = 50
learning_rate = 1e-4

tokenizer = AutoTokenizer.from_pretrained(pretrained)
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
metric = load_metric("rouge")

DIR = "./data"
TRAIN_SOURCE = os.path.join(DIR, "train.json")
TEST_SOURCE = os.path.join(DIR, "test.json")

with open(TRAIN_SOURCE) as f:
    TRAIN_DATA = json.loads(f.read())

with open(TEST_SOURCE) as f:
    TEST_DATA = json.loads(f.read())

train = pd.DataFrame(columns=['uid', 'title', 'region', 'context', 'summary'])
uid = 1000
for data in TRAIN_DATA:
    for agenda in data['context'].keys():
        context = ''
        for line in data['context'][agenda]:
            context += data['context'][agenda][line]
            context += ' '
        train.loc[uid, 'uid'] = uid
        train.loc[uid, 'title'] = data['title']
        train.loc[uid, 'region'] = data['region']
        train.loc[uid, 'context'] = context[:-1]
        train.loc[uid, 'summary'] = data['label'][agenda]['summary']
        uid += 1

test = pd.DataFrame(columns=['uid', 'title', 'region', 'context'])
uid = 2000
for data in TEST_DATA:
    for agenda in data['context'].keys():
        context = ''
        for line in data['context'][agenda]:
            context += data['context'][agenda][line]
            context += ' '
        test.loc[uid, 'uid'] = uid
        test.loc[uid, 'title'] = data['title']
        test.loc[uid, 'region'] = data['region']
        test.loc[uid, 'context'] = context[:-1]
        uid += 1

train['total'] = train.title + ' ' + train.region + ' ' + train.context
test['total'] = test.title + ' ' + test.region + ' ' + test.context

df_train = train.iloc[:-200]
df_val = train.iloc[-200:]


def preprocess_function(batch):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in batch["total"]]
    inputs = tokenizer(inputs, padding="max_length",
                       truncation=True,
                       max_length=encoder_max_length)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        outputs = tokenizer(batch["summary"], padding="max_length",
                            truncation=True,
                            max_length=encoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id
            else token for token in labels]
        for labels in batch["labels"]]
    return batch


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     kkma = Kkma()

#     decoded_preds = tokenizer(predictions, skip_special_tokens=True)
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer(labels, skip_special_tokens=True)

#     # Rouge expects a newline after each sentence
#     decoded_preds = ["\n".join(kkma.sentences(pred.strip())) for pred in decoded_preds]
#     decoded_labels = ["\n".join(kkma.sentences(label.strip())) for label in decoded_labels]

#     result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     # Extract a few results
#     result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

#     # Add mean generated length
#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#     result["gen_len"] = np.mean(prediction_lens)

#     return {k: round(v, 4) for k, v in result.items()}

train_data = Dataset.from_pandas(df_train)
val_data = Dataset.from_pandas(df_val)
test_data = Dataset.from_pandas(test)

train_data = train_data.map(
    preprocess_function,
    batched=True,
    batch_size=batch_size,
    remove_columns=[
        'uid', 'title', 'region',
        'context', 'summary', 'total']
)
train_data.set_format(
    type="torch",
    columns=[
        "input_ids", "attention_mask", "decoder_input_ids",
        "decoder_attention_mask", "labels"],
)


val_data = val_data.map(
    preprocess_function,
    batched=True,
    batch_size=batch_size,
    remove_columns=[
        'uid', 'title', 'region',
        'context', 'summary', 'total']
)
val_data.set_format(
    type="torch",
    columns=[
        "input_ids", "attention_mask", "decoder_input_ids",
        "decoder_attention_mask", "labels"],
)


training_args = Seq2SeqTrainingArguments(
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs=epochs,
    logging_strategy="epoch",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir='./log',
    load_best_model_at_end=True,
    learning_rate=learning_rate,
    disable_tqdm=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    optimizers='AdamW',
    # compute_metrics=compute_metrics,
)

trainer.train()

model.to("cuda")


def generate_summary(batch):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in batch["total"]]
    inputs = tokenizer(inputs,
                       padding="max_length",
                       truncation=True,
                       max_length=encoder_max_length,
                       return_tensors="pt")

    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=decoder_max_length,)

    output_str = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True)

    batch["summary"] = output_str

    return batch


results = test_data.map(
    generate_summary,
    batched=True,
    batch_size=batch_size)

pred_str = results["total"]
label_str = results["summary"]

sample_submission = pd.read_csv("./data/sample_submission.csv")
sample_submission['summary'] = label_str
sample_submission.to_csv('sub1.csv', index=False)

result = dacon_submit_api.post_submission_file(
    'sub1.csv',
    '63af198a705c8b8bcb5c3f0bcd84b1c8e8435c9da10875843871a770c269d0e6',
    '235813',
    '영락태왕',
    'submission'
)
