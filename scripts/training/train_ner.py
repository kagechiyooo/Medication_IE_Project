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

    extra_metrics = additional_metrics(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **extra_metrics,
    }



from ner_common import (additional_metrics, seed_everything, save_json, read_json,
                        signature_for, completed_run, write_grid_results, decode_labels, maintain_storage, load_storage_best, compatible_signatures)

import gc
import itertools
import shutil
import statistics
import time
from pathlib import Path


def train_configuration(config, tokenized_dataset, tokenizer, seed, experiment_dir):
    model_name = config['model_name']
    output_dir = str(experiment_dir / 'checkpoints')
    result_dir = str(experiment_dir)
    seed_everything(seed)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
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

        metric_for_best_model="macro_f1",
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
        data_seed=seed,
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

    # Re-seed before training as requested; model is newly loaded above.
    seed_everything(seed)
    trainer.train()
    val_metrics = trainer.evaluate(tokenized_dataset['validation'])
    val_output = trainer.predict(tokenized_dataset['validation'])
    true_labels, true_predictions = decode_labels(val_output.predictions, val_output.label_ids, ID2LABEL)
    report = classification_report(true_labels, true_predictions, digits=4)
    report_path = experiment_dir / 'validation_classification_report.txt'
    report_path.write_text(report, encoding='utf-8')
    final_model_dir = experiment_dir / 'best_model'
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    best_state = read_json(Path(trainer.state.best_model_checkpoint) / 'trainer_state.json')
    save_json(experiment_dir / 'trainer_history.json', trainer.state.log_history)
    result = {
        'validation': {key:float(val_metrics['eval_'+key]) for key in ['precision','recall','f1','macro_f1','accuracy']},
        'best_checkpoint_epoch': best_state['epoch'],
        'best_model_path': str(final_model_dir.resolve()),
        'classification_report_path': str(report_path.resolve()),
        'total_training_steps':trainer.state.global_step,
        'scheduler_type':str(training_args.lr_scheduler_type),
        'warmup_steps':training_args.get_warmup_steps(trainer.state.max_steps),
        'model_commit':getattr(model.config,'_commit_hash',None),
    }
    # Optimizer checkpoints are not used by restart-from-pretrained resume.
    # Exported best_model and epoch history remain available.
    del trainer, model, val_output
    gc.collect()
    shutil.rmtree(experiment_dir / 'checkpoints')
    return result


def run_grid(config, repeat_best=False):
    if config.get('seed',42) != 42 or config.get('repeat_seeds',[42,123,2026]) != [42,123,2026]:
        raise ValueError('Use the agreed seed 42 and repeat seeds [42,123,2026]')
    if int(os.environ.get('WORLD_SIZE','1')) != 1 or torch.cuda.device_count() > 1:
        raise ValueError('Use one process and one visible GPU to keep batch sizes comparable')
    model_name = config['model_name']
    result_dir = Path(config['result_dir'])
    result_dir.mkdir(parents=True, exist_ok=True)
    signature = signature_for(config)
    manifest_path = result_dir / 'grid_manifest.json'
    if manifest_path.exists():
        existing_signature = read_json(manifest_path)['signature']
        if existing_signature not in compatible_signatures(config):
            raise ValueError('Config, code, dependencies or data changed. Use new output_dir and result_dir to avoid mixing runs.')
        signature = existing_signature
    save_json(manifest_path, {'signature':signature,'model':model_name,'seed':42,
                             'metric_mode':'seqeval default; fixed five-entity macro F1'})
    # Original JSON loader; Test intentionally excluded during tuning.
    data_files = {'train':config['train_file'], 'validation':config['validation_file']}
    dataset = load_dataset('json', data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenized_dataset = dataset.map(
        lambda examples:tokenize_and_align_labels(examples,tokenizer), batched=True)
    if repeat_best:
        best = read_json(result_dir/'best_config.json')
        if not best.get('grid_complete') or best['signature'] != signature:
            raise ValueError('Complete the matching seed-42 grid first')
        seeds = config.get('repeat_seeds',[42,123,2026])
        experiments = [(best['learning_rate'],best['batch_size'],best['configured_epochs'],s) for s in seeds]
    else:
        space = config['grid']
        experiments = [(lr,batch,epochs,42) for lr,batch,epochs in itertools.product(
            space['learning_rate'],space['batch_size'],space['epochs'])]
    load_storage_best(config, signature)  # Verify retained weights before trusting pruned results.
    records = []
    for index,(lr,batch,epochs,seed) in enumerate(experiments,1):
        experiment_id = f'lr_{lr:g}_batch_{batch}_epochs_{epochs}_seed_{seed}'
        experiment_dir = Path(config['output_dir']) / 'grid' / experiment_id
        record_path = experiment_dir/'result.json'
        record = {}
        if record_path.exists():
            try: record = read_json(record_path)
            except (ValueError,OSError): pass
        print(f'\nModel: {model_name} | Experiment: {index}/{len(experiments)} | {experiment_id}',flush=True)
        if completed_run(record, signature):
            print('Completed: reuse existing complete result',flush=True)
        else:
            if experiment_dir.exists(): shutil.rmtree(experiment_dir)
            experiment_dir.mkdir(parents=True)
            record = {'experiment_id':experiment_id,'model':model_name,'learning_rate':lr,
                      'batch_size':batch,'configured_epochs':epochs,'seed':seed,
                      'signature':signature,'status':'Running'}
            save_json(record_path,record)
            run_config = dict(config,learning_rate=lr,batch_size=batch,epochs=epochs)
            started = time.perf_counter()
            try:
                record.update(train_configuration(run_config,tokenized_dataset,tokenizer,seed,experiment_dir))
                record['status'] = 'Completed'
            except torch.cuda.OutOfMemoryError as error:
                record.update(status='OOM',error=str(error))
            except Exception as error:
                record.update(status='Failed',error=f'{type(error).__name__}: {error}')
            finally:
                record['training_seconds'] = time.perf_counter()-started
                save_json(record_path,record)
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        if record['status'] == 'Completed':
            maintain_storage(config, signature, record_path)
            record = read_json(record_path)
            # Earlier winners may have just been superseded and pruned.
            records = [read_json(Path(config['output_dir'])/'grid'/r['experiment_id']/'result.json') for r in records]
        records.append(record)
        print(record['status'],record.get('validation',record.get('error')),flush=True)
        if not repeat_best: write_grid_results(result_dir,records)
        else: save_json(result_dir/'multi_seed_runs.json',records)
        if record['status'] != 'Completed':
            if not repeat_best: (result_dir/'best_config.json').unlink(missing_ok=True)
            raise RuntimeError('Run stopped and status recorded. No batch/LR/epochs changed. Resolve before rerunning the same command.')
    if repeat_best:
        metrics = {key:{'mean':statistics.mean(r['validation'][key] for r in records),
                        'sd':statistics.stdev(r['validation'][key] for r in records)}
                   for key in ['precision','recall','f1','macro_f1','accuracy']}
        save_json(result_dir/'multi_seed_summary.json',{'seeds':[r['seed'] for r in records],
                  'sd_definition':'sample standard deviation, ddof=1','validation':metrics,'runs':records})
    else:
        # Ties: first configuration in declared grid order.
        best = max(records,key=lambda row:row['validation']['macro_f1'])
        save_json(result_dir/'best_config.json',dict(best,grid_complete=True,configurations=len(records)))
        save_json(result_dir/'metrics.json',{'model':model_name,'validation':best['validation']})
        shutil.copyfile(best['classification_report_path'],result_dir/'validation_classification_report.txt')


def run_with_lock(config, repeat_best=False):
    result_dir=Path(config['result_dir']); result_dir.mkdir(parents=True,exist_ok=True)
    lock=result_dir/'RUNNING.lock'
    try:
        fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'{lock} exists. Check no training process is running before removing a stale lock.')
    try:
        with os.fdopen(fd,'w') as stream:stream.write(str(os.getpid()))
        run_grid(config,repeat_best)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path ของ config JSON')
    args = parser.parse_args()
    config = load_config(args.config)
    run_with_lock(config)


if __name__ == '__main__':
    main()
