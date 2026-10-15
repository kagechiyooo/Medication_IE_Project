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



from ner_common import (additional_metrics, seed_everything, save_json, read_json,
                        signature_for, completed_run, write_grid_results, decode_labels, compatible_signatures)


def evaluate_config(config):
    best = read_json(os.path.join(config['result_dir'], 'best_config.json'))
    if not best.get('grid_complete') or best['seed'] != 42 or best['signature'] not in compatible_signatures(config):
        raise ValueError('Require complete matching seed-42 grid before final testing')
    model_dir = best['best_model_path']

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
        per_device_eval_batch_size=best['batch_size'],
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

    extra_metrics = additional_metrics(true_labels, true_predictions)
    report += (
        f"\nToken-level BIO Accuracy: {extra_metrics['accuracy']:.4f} "
        f"({extra_metrics['accuracy']:.2%})\n"
    )
    metrics = {
        "model": config["model_name"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **extra_metrics,
        "learning_rate":best['learning_rate'],
        "batch_size":best['batch_size'],
        "configured_epochs":best['configured_epochs'],
        "best_checkpoint_epoch":best['best_checkpoint_epoch'],
        "best_model_path":model_dir,
        "seed":42,
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

    import csv
    with open(os.path.join(result_dir,'test_metrics.csv'),'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics));writer.writeheader();writer.writerow(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', help='Evaluate one model after all four grids are complete')
    group.add_argument('--configs', nargs=4, help='Evaluate all four models')
    args = parser.parse_args()
    paths = args.configs or [args.config]
    configs = [load_config(path) for path in paths]
    # All four grid results must exist before any final test, even with --config.
    config_folder = os.path.dirname(os.path.abspath(paths[0]))
    all_configs = [load_config(os.path.join(config_folder,key+'.json')) for key in
                   ['biobert','bioclinicalbert','clinicalbert','pubmedbert']]
    for item in all_configs:
        best=read_json(os.path.join(item['result_dir'],'best_config.json'))
        if not best.get('grid_complete') or best['seed'] != 42 or best['signature'] not in compatible_signatures(item):
            raise ValueError('Complete all four matching grids before final testing')
    if args.configs and {c['model_key'] for c in configs} != {c['model_key'] for c in all_configs}:
        raise ValueError('Provide each of the four model configs exactly once')
    if len({os.path.realpath(c['test_file']) for c in all_configs}) != 1:
        raise ValueError('All four models must use the same test file')
    results=[]
    for config in configs:
        import torch
        try:
            results.append(evaluate_config(config))
        except torch.cuda.OutOfMemoryError as error:
            save_json(os.path.join(config['result_dir'],'final_test_status.json'),{'status':'OOM','error':str(error)})
            raise
        save_json(os.path.join(config['result_dir'],'final_test_status.json'),{'status':'Completed'})
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if args.configs:
        import csv
        os.makedirs('results/final_comparison',exist_ok=True)
        save_json('results/final_comparison/comparison.json',results)
        with open('results/final_comparison/comparison.csv','w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)


if __name__ == '__main__':
    main()
