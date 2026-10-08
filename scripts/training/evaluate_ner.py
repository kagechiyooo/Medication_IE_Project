import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
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

    labels = []

    for i, ner_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(
                    LABEL2ID[ner_tags[word_idx]]
                )

            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="path ของ config JSON"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    model_dir = os.path.join(
        config["output_dir"],
        "best_model"
    )

    result_dir = config["result_dir"]

    os.makedirs(result_dir, exist_ok=True)

    print("===== EVALUATION =====")
    print("Model:", model_dir)
    print("Test:", config["test_file"])

    dataset = load_dataset(
        "json",
        data_files={
            "test": config["test_file"]
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_dir
    )

    tokenized_dataset = dataset.map(
        lambda examples:
            tokenize_and_align_labels(
                examples,
                tokenizer
            ),
        batched=True,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(
            result_dir,
            "eval_temp"
        ),
        per_device_eval_batch_size=config.get(
            "batch_size",
            8
        ),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    output = trainer.predict(
        tokenized_dataset["test"]
    )

    predictions = np.argmax(
        output.predictions,
        axis=2
    )

    labels = output.label_ids

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

    precision = precision_score(
        true_labels,
        true_predictions
    )

    recall = recall_score(
        true_labels,
        true_predictions
    )

    f1 = f1_score(
        true_labels,
        true_predictions
    )

    report = classification_report(
        true_labels,
        true_predictions,
        digits=4
    )

    metrics = {
        "model": config["model_name"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    with open(
        os.path.join(
            result_dir,
            "test_metrics.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False
        )

    with open(
        os.path.join(
            result_dir,
            "test_classification_report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    print("\n===== TEST RESULT =====")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    print("\n")
    print(report)

    print(
        "\nSaved to:",
        result_dir
    )


if __name__ == "__main__":
    main()
