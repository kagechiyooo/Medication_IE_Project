import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


LABEL_LIST = [
    "O",
    "B-DRUG",
    "I-DRUG",
    "B-STRENGTH",
    "I-STRENGTH",
    "B-DOSAGE",
    "I-DOSAGE",
    "B-FREQUENCY",
    "I-FREQUENCY",
    "B-ROUTE",
    "I-ROUTE",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    all_labels = []

    for i, ner_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_name = ner_tags[word_idx]
                label_ids.append(LABEL2ID[label_name])

            else:
                # subword ต่อจาก token เดิม ไม่เอามาคิด loss ซ้ำ
                label_ids.append(-100)

            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):

        pred_seq = []
        label_seq = []

        for p, l in zip(prediction, label):

            if l == -100:
                continue

            pred_seq.append(ID2LABEL[int(p)])
            label_seq.append(ID2LABEL[int(l)])

        true_predictions.append(pred_seq)
        true_labels.append(label_seq)

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="path ของ config JSON"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    model_name = config["model_name"]
    output_dir = config["output_dir"]
    result_dir = config["result_dir"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    seed = config.get("seed", 42)
    set_seed(seed)

    print("===== CONFIG =====")
    print("Model:", model_name)
    print("Train:", config["train_file"])
    print("Validation:", config["validation_file"])
    print("Test:", config["test_file"])
    print("Output:", output_dir)
    print("Result:", result_dir)

    data_files = {
        "train": config["train_file"],
        "validation": config["validation_file"],
        "test": config["test_file"],
    }

    dataset = load_dataset(
        "json",
        data_files=data_files
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples,
            tokenizer
        ),
        batched=True,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=output_dir,

        learning_rate=config.get(
            "learning_rate",
            2e-5
        ),

        per_device_train_batch_size=config.get(
            "batch_size",
            8
        ),

        per_device_eval_batch_size=config.get(
            "batch_size",
            8
        ),

        num_train_epochs=config.get(
            "epochs",
            5
        ),

        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="f1",
        greater_is_better=True,

        save_total_limit=2,

        logging_dir=os.path.join(
            result_dir,
            "logs"
        ),

        logging_strategy="steps",
        logging_steps=20,

        report_to="none",

        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n===== START TRAINING =====")

    trainer.train()

    print("\n===== VALIDATION =====")

    val_metrics = trainer.evaluate(
        tokenized_dataset["validation"]
    )

    print(val_metrics)

    print("\n===== TEST =====")

    test_output = trainer.predict(
        tokenized_dataset["test"]
    )

    test_metrics = test_output.metrics

    print(test_metrics)

    # save best model
    final_model_dir = os.path.join(
        output_dir,
        "best_model"
    )

    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # save summary metrics
    metrics = {
        "model": model_name,
        "validation": val_metrics,
        "test": test_metrics,
    }

    metrics_path = os.path.join(
        result_dir,
        "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False
        )

    # detailed test classification report
    predictions = np.argmax(
        test_output.predictions,
        axis=2
    )

    labels = test_output.label_ids

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predictions,
        labels
    ):

        pred_seq = []
        label_seq = []

        for p, l in zip(
            prediction,
            label
        ):

            if l == -100:
                continue

            pred_seq.append(
                ID2LABEL[int(p)]
            )

            label_seq.append(
                ID2LABEL[int(l)]
            )

        true_predictions.append(
            pred_seq
        )

        true_labels.append(
            label_seq
        )

    report = classification_report(
        true_labels,
        true_predictions,
        digits=4
    )

    report_path = os.path.join(
        result_dir,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    print("\n===== FINISHED =====")
    print("Best model:", final_model_dir)
    print("Metrics:", metrics_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
