"""Dataset-size experiment: Stage 1 training, Stage 2 selection, Stage 3 test."""
import argparse, csv, gc, hashlib, importlib.metadata, json, os, shutil, time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
    DataCollatorForTokenClassification, EarlyStoppingCallback, Trainer,
    TrainerCallback, TrainingArguments)
from ner_common import (ENTITY_TYPES, ID2LABEL, LABEL2ID, LABEL_LIST,
    MISSED_ENTITY, SPURIOUS_ENTITY, additional_metrics, decode_labels,
    entity_confusion, entity_counts, format_classification_report,
    model_files_exist, normalize_matrix_label, read_json, save_json,
    seed_everything)

SIZES = ['dataset_50', 'dataset_75', 'dataset_100']
MODELS = {'biobert', 'clinicalbert', 'pubmedbert'}
HISTORY_FIELDS = ['epoch','training_step','train_loss','validation_loss',
    'validation_precision','validation_recall','validation_f1',
    'validation_macro_f1','validation_accuracy','learning_rate']
BEST_FIELDS = ['dataset_size','maximum_epochs','actual_final_epoch',
    'best_checkpoint_epoch','validation_loss','validation_precision',
    'validation_recall','validation_f1','validation_macro_f1',
    'validation_accuracy','seed','status','best_model_path']
SELECTED_FIELDS = ['model','dataset_size','best_checkpoint_epoch','validation_loss',
    'validation_precision','validation_recall','validation_f1',
    'validation_macro_f1','validation_accuracy','seed','best_model_path',
    'signature','run_summary_path']
TEST_FIELDS = ['Model','Dataset Size','Best Epoch','Test Loss','Precision',
    'Recall','Micro-F1','Macro-F1','BIO Accuracy']
EXPECTED_TEST_ENTITIES = {'DRUG':445,'STRENGTH':95,'DOSAGE':73,'FREQUENCY':329,'ROUTE':259}

def load_config(path):
    with open(path, encoding='utf-8') as stream: return json.load(stream)

def write_csv(path, fields, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temp,path)

def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as stream: return list(csv.DictReader(stream))

def validate_config(config):
    if config.get('model_key') not in MODELS: raise ValueError('Only biobert, clinicalbert, and pubmedbert are in this experiment')
    expected={'learning_rate':2e-5,'train_batch_size':16,'maximum_epochs':100,
        'seed':42,'gradient_accumulation_steps':1,'early_stopping_patience':10,
        'early_stopping_threshold':0.0,'lr_scheduler_type':'linear','warmup_steps':0}
    for key,value in expected.items():
        if config.get('training',{}).get(key)!=value: raise ValueError(f'training.{key} must be {value!r}')
    if list(config.get('datasets',{}))!=SIZES: raise ValueError(f'Dataset order must be {SIZES}')

def tokenize(examples,tokenizer):
    result=tokenizer(examples['tokens'],truncation=True,is_split_into_words=True)
    labels=[]
    for index,tags in enumerate(examples['ner_tags']):
        previous=None; aligned=[]
        for word in result.word_ids(batch_index=index):
            aligned.append(-100 if word is None or word==previous else LABEL2ID[tags[word]])
            previous=word
        labels.append(aligned)
    result['labels']=labels; return result

def metrics(output):
    truth,predicted=decode_labels(output.predictions,output.label_ids,ID2LABEL)
    return {'precision':precision_score(truth,predicted),'recall':recall_score(truth,predicted),
        'f1':f1_score(truth,predicted),**additional_metrics(truth,predicted)}

def history_rows(logs):
    train=[row for row in logs if 'loss' in row and 'eval_loss' not in row]; rows=[]
    for item in (row for row in logs if 'eval_macro_f1' in row):
        epoch=float(item['epoch']); candidates=[x for x in train if float(x.get('epoch',-1))<=epoch+1e-9]
        logged=candidates[-1] if candidates else {}
        rows.append({'epoch':epoch,'training_step':int(item.get('step',0)),
            'train_loss':logged.get('loss'),'validation_loss':item['eval_loss'],
            'validation_precision':item['eval_precision'],'validation_recall':item['eval_recall'],
            'validation_f1':item['eval_f1'],'validation_macro_f1':item['eval_macro_f1'],
            'validation_accuracy':item['eval_accuracy'],'learning_rate':logged.get('learning_rate')})
    return list({row['epoch']:row for row in rows}.values())

def plot_curve(rows,path,title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    figure,axis=plt.subplots(figsize=(8,5))
    if rows:
        epochs=[float(x['epoch']) for x in rows]; scores=[float(x['validation_macro_f1']) for x in rows]
        best=max(range(len(rows)),key=lambda i:scores[i]); axis.plot(epochs,scores,marker='o')
        axis.scatter(epochs[best],scores[best],color='red',zorder=3)
        axis.annotate(f'Best: epoch {epochs[best]:g}, {scores[best]:.4f}',(epochs[best],scores[best]),xytext=(8,8),textcoords='offset points')
    axis.set(title=title,xlabel='Epoch',ylabel='Validation Entity-level Macro-F1'); axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(path,dpi=160); plt.close(figure)

class ArtifactCallback(TrainerCallback):
    def __init__(self,directory,title): self.directory=Path(directory); self.title=title
    def on_evaluate(self,args,state,control,**kwargs):
        rows=history_rows(state.log_history)
        write_csv(self.directory/'epoch_history.csv',HISTORY_FIELDS,rows)
        plot_curve(rows,self.directory/'validation_curve.png',self.title)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def signature(config,size):
    files=config['datasets'][size]
    payload={'model':config['model_name'],'size':size,'training':config['training'],
        'data':{key:{'path':value,'sha256':sha(value)} for key,value in files.items()},
        'code':{name:sha(Path(__file__).with_name(name)) for name in ['dataset_size_experiment.py','ner_common.py']},
        'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','datasets','accelerate','seqeval','numpy']}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload

def source_labels(path):
    answer=[]
    with open(path,encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            row=json.loads(line)
            if len(row['tokens'])!=len(row['ner_tags']): raise ValueError(f'Unaligned row {path}:{number}')
            unknown=set(row['ner_tags'])-set(LABEL_LIST)
            if unknown: raise ValueError(f'Unknown labels {sorted(unknown)} at {path}:{number}')
            answer.append(row['ner_tags'])
    return answer

def completed(summary,run_signature):
    return summary.get('status')=='Completed' and summary.get('signature')==run_signature \
        and Path(summary.get('epoch_history_path','')).is_file() \
        and Path(summary.get('validation_report_path','')).is_file() \
        and model_files_exist(summary.get('best_model_path',''))

def update_stage1(config):
    root=Path(config['result_root']); table=[]; histories={}
    for size in SIZES:
        path=root/size/'run_summary.json'; summary=read_json(path) if path.is_file() else {'status':'Not Started'}
        values=summary.get('best_validation_metrics',{})
        table.append({'dataset_size':size,'maximum_epochs':summary.get('maximum_epochs',100),
            'actual_final_epoch':summary.get('actual_final_epoch'),'best_checkpoint_epoch':summary.get('best_checkpoint_epoch'),
            **{'validation_'+key:values.get(key) for key in ['loss','precision','recall','f1','macro_f1','accuracy']},
            'seed':summary.get('seed',42),'status':summary.get('status'),'best_model_path':summary.get('best_model_path')})
        history=root/size/'epoch_history.csv'
        if history.is_file(): histories[size]=read_csv(history)
    write_csv(root/'best_checkpoints.csv',BEST_FIELDS,table)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    colors={'dataset_50':'#1f77b4','dataset_75':'#ff7f0e','dataset_100':'#2ca02c'}
    figure,axis=plt.subplots(figsize=(9,5.5))
    for size,rows in histories.items():
        x=[float(r['epoch']) for r in rows]; y=[float(r['validation_macro_f1']) for r in rows]
        best=max(range(len(rows)),key=lambda i:y[i]); axis.plot(x,y,marker='o',markersize=3,color=colors[size],label=size.replace('dataset_','Dataset ')+'%')
        axis.scatter(x[best],y[best],color=colors[size],edgecolor='black',zorder=3)
    suffix=' (partial results)' if len(histories)<3 else ''
    axis.set(title=f"{config['model_key']}: Dataset-size comparison{suffix}",xlabel='Epoch',ylabel='Validation Entity-level Macro-F1')
    if histories: axis.legend()
    axis.grid(alpha=.25); figure.tight_layout(); figure.savefig(root/'dataset_comparison.png',dpi=160); plt.close(figure)

def train_one(config,size):
    settings=config['training']; files=config['datasets'][size]
    result=Path(config['result_root'])/size; weights=Path(config['model_root'])/size
    run_signature,payload=signature(config,size); summary_path=result/'run_summary.json'
    if summary_path.is_file() and completed(read_json(summary_path),run_signature):
        print(config['model_key'],size,'already Completed; skipping'); update_stage1(config); return
    Path(config['result_root']).mkdir(parents=True,exist_ok=True)
    lock=Path(config['result_root'])/f'.{size}.RUNNING.lock'
    try: descriptor=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as error: raise RuntimeError(f'{lock} exists; verify no duplicate run is active before removing a stale lock') from error
    with os.fdopen(descriptor,'w') as stream: stream.write(str(os.getpid()))
    if result.exists(): shutil.rmtree(result)
    if weights.exists(): shutil.rmtree(weights)
    result.mkdir(parents=True); weights.mkdir(parents=True)
    summary={'status':'Running','model':config['model_key'],'model_id':config['model_name'],'dataset_size':size,
        'dataset_paths':files,
        'seed':42,'training_settings':settings,'maximum_epochs':100,'signature':run_signature,'signature_payload':payload,
        'train_loss_definition':'Trainer loss logged during each epoch; not a loss recomputed at the best checkpoint.',
        'learning_rate_note':'2e-5 is the initial LR; the linear scheduler reduces LR during training.'}
    save_json(summary_path,summary); started=time.perf_counter(); trainer=model=output=None
    try:
        train_labels=source_labels(files['train_file']); validation_labels=source_labels(files['validation_file'])
        summary.update(train_examples=len(train_labels),validation_examples=len(validation_labels)); save_json(summary_path,summary)
        if int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()>1: raise ValueError('Use one process and one visible GPU')
        split_files={'train':files['train_file'],'validation':files['validation_file']}
        seed_everything(42); dataset=load_dataset('json',data_files=split_files)
        tokenizer=AutoTokenizer.from_pretrained(config['model_name'],use_fast=True)
        tokenized=dataset.map(lambda batch:tokenize(batch,tokenizer),batched=True)
        seed_everything(42)
        model=AutoModelForTokenClassification.from_pretrained(config['model_name'],num_labels=len(LABEL_LIST),id2label=ID2LABEL,label2id=LABEL2ID)
        arguments=TrainingArguments(output_dir=str(weights/'checkpoints'),learning_rate=2e-5,
            per_device_train_batch_size=16,per_device_eval_batch_size=settings.get('eval_batch_size',16),
            gradient_accumulation_steps=1,num_train_epochs=100,weight_decay=settings.get('weight_decay',.01),
            lr_scheduler_type='linear',warmup_steps=0,eval_strategy='epoch',save_strategy='epoch',logging_strategy='epoch',
            load_best_model_at_end=True,metric_for_best_model='macro_f1',greater_is_better=True,
            save_total_limit=2,report_to='none',seed=42,data_seed=42)
        callback=ArtifactCallback(result,f"{config['model_key']} {size}: validation Macro-F1")
        trainer=Trainer(model=model,args=arguments,train_dataset=tokenized['train'],eval_dataset=tokenized['validation'],
            processing_class=tokenizer,data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),compute_metrics=metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10,early_stopping_threshold=0.0),callback])
        seed_everything(42); trainer.train(resume_from_checkpoint=False)
        history=history_rows(trainer.state.log_history); write_csv(result/'epoch_history.csv',HISTORY_FIELDS,history)
        plot_curve(history,result/'validation_curve.png',callback.title)
        if not history: raise RuntimeError('No validation history was produced')
        best=max(history,key=lambda row:float(row['validation_macro_f1'])) # first maximum wins
        trainer.save_model(str(weights/'best_model')); tokenizer.save_pretrained(str(weights/'best_model'))
        output=trainer.predict(tokenized['validation'],metric_key_prefix='validation_best')
        truth,predicted=decode_labels(output.predictions,output.label_ids,ID2LABEL)
        before=entity_counts(validation_labels); after=entity_counts(truth); truncated=before!=after
        report=format_classification_report(truth,predicted,float(best['validation_loss']),'Validation')
        report_path=result/'validation_classification_report.txt'; report_path.write_text(report,encoding='utf-8')
        actual=float(history[-1]['epoch']); best_metrics={key:float(best['validation_'+key]) for key in ['loss','precision','recall','f1','macro_f1','accuracy']}
        summary.update(status='Completed',actual_final_epoch=actual,best_checkpoint_epoch=float(best['epoch']),
            stopping_reason='early_stopping_patience_10' if actual<100 else 'maximum_epochs_reached',
            best_validation_metrics=best_metrics,best_model_path=str((weights/'best_model').resolve()),
            epoch_history_path=str((result/'epoch_history.csv').resolve()),validation_report_path=str(report_path.resolve()),
            scheduler='linear',warmup_steps=0,initial_learning_rate=2e-5,
            validation_entity_counts_before_tokenization=before,validation_entity_counts_after_tokenization=after,
            validation_truncation_detected=truncated,training_seconds=time.perf_counter()-started)
        if truncated: summary['warning']='Validation entities were truncated; metrics cover retained entities only.'
        save_json(summary_path,summary); shutil.rmtree(weights/'checkpoints',ignore_errors=True); update_stage1(config)
    except torch.cuda.OutOfMemoryError as error:
        summary.update(status='OOM',error=str(error),training_seconds=time.perf_counter()-started); save_json(summary_path,summary); update_stage1(config)
        raise RuntimeError('OOM recorded; settings were not changed') from error
    except Exception as error:
        summary.update(status='Failed',error=f'{type(error).__name__}: {error}',training_seconds=time.perf_counter()-started); save_json(summary_path,summary); update_stage1(config); raise
    finally:
        lock.unlink(missing_ok=True); del trainer,model,output; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def select(config_paths,output):
    configs=[load_config(path) for path in config_paths]
    if {x['model_key'] for x in configs}!=MODELS: raise ValueError('Provide exactly the three selected model configs')
    selected=[]
    for config in configs:
        candidates=[]
        for size in SIZES:
            path=Path(config['result_root'])/size/'run_summary.json'
            if not path.is_file(): raise RuntimeError(f'Missing complete run: {path}')
            summary=read_json(path); run_signature,_=signature(config,size)
            if not completed(summary,run_signature): raise RuntimeError(f'Incomplete or mismatched run: {path}')
            candidates.append(summary)
        winner=max(candidates,key=lambda row:float(row['best_validation_metrics']['macro_f1']))
        values=winner['best_validation_metrics']; selected.append({'model':config['model_key'],'dataset_size':winner['dataset_size'],
            'best_checkpoint_epoch':winner['best_checkpoint_epoch'],**{'validation_'+key:values[key] for key in ['loss','precision','recall','f1','macro_f1','accuracy']},
            'seed':winner['seed'],'best_model_path':winner['best_model_path'],'signature':winner['signature'],
            'run_summary_path':str((Path(config['result_root'])/winner['dataset_size']/'run_summary.json').resolve())})
    write_csv(output,SELECTED_FIELDS,selected)

def read_matrix_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        records=list(csv.reader(stream))
    if not records:
        raise ValueError(f'Empty entity confusion matrix CSV: {path}')
    columns=[normalize_matrix_label(label) for label in records[0][1:]]
    rows=[]; values=[]
    for record in records[1:]:
        if len(record)!=len(columns)+1:
            raise ValueError(f'Invalid entity confusion matrix row in {path}: {record}')
        rows.append(normalize_matrix_label(record[0]))
        values.append([np.nan if value=='N/A' else float(value) for value in record[1:]])
    expected_rows=ENTITY_TYPES+[SPURIOUS_ENTITY]
    expected_columns=ENTITY_TYPES+[MISSED_ENTITY]
    if rows!=expected_rows or columns!=expected_columns:
        raise ValueError(f'Unexpected entity confusion matrix labels in {path}')
    if not np.isnan(values[-1][-1]):
        raise ValueError(f'Bottom-right entity confusion matrix cell must be N/A: {path}')
    return rows,columns,np.asarray(values,dtype=float)

def normalize_matrix_csv_labels(path):
    path=Path(path)
    with path.open(encoding='utf-8-sig',newline='') as stream:
        records=list(csv.reader(stream))
    if not records:
        raise ValueError(f'Empty entity confusion matrix CSV: {path}')
    records[0][1:]=[normalize_matrix_label(label) for label in records[0][1:]]
    for record in records[1:]:
        record[0]=normalize_matrix_label(record[0])
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as stream:
        csv.writer(stream).writerows(records)
    os.replace(temp,path)

def render_matrix_from_csv(csv_path,png_path,title):
    rows,columns,values=read_matrix_csv(csv_path)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    figure,axis=plt.subplots(figsize=(10,8)); image=axis.imshow(values,cmap='Blues'); figure.colorbar(image,ax=axis)
    axis.set_xticks(range(len(columns)),columns,rotation=35,ha='right')
    axis.set_yticks(range(len(rows)),rows)
    axis.set(xlabel='Prediction',ylabel='Ground Truth',title=title)
    for i in range(len(rows)):
        for j in range(len(columns)):
            axis.text(j,i,'N/A' if np.isnan(values[i,j]) else str(int(values[i,j])),ha='center',va='center')
    figure.tight_layout(); figure.savefig(png_path,dpi=160,bbox_inches='tight'); plt.close(figure)

def write_matrix(rows,columns,matrix,csv_path,png_path,title):
    with Path(csv_path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['Ground Truth / Prediction']+columns)
        for row in rows:
            writer.writerow([row]+[('N/A' if row==SPURIOUS_ENTITY and col==MISSED_ENTITY else matrix[row][col]) for col in columns])
    render_matrix_from_csv(csv_path,png_path,title)

def test_one(config,selected_path,comparison_path):
    selected=[row for row in read_csv(selected_path) if row['model']==config['model_key']]
    if len(selected)!=1: raise ValueError(f'Stage 2 selection missing for {config["model_key"]}')
    choice=selected[0]; summary=read_json(choice['run_summary_path'])
    if summary['signature']!=choice['signature'] or not model_files_exist(choice['best_model_path']): raise RuntimeError('Selected Stage 1 weights are missing or mismatched')
    test_path=config['test_file']; source=source_labels(test_path); outdir=Path(config['stage3_root'])/config['model_key']; outdir.mkdir(parents=True,exist_ok=True)
    dataset=load_dataset('json',data_files={'test':test_path}); tokenizer=AutoTokenizer.from_pretrained(choice['best_model_path'],use_fast=True)
    model=AutoModelForTokenClassification.from_pretrained(choice['best_model_path']); tokenized=dataset.map(lambda batch:tokenize(batch,tokenizer),batched=True)
    arguments=TrainingArguments(output_dir=str(outdir/'eval_temp'),per_device_eval_batch_size=16,report_to='none')
    trainer=Trainer(model=model,args=arguments,processing_class=tokenizer,data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer))
    output=trainer.predict(tokenized['test']); truth,predicted=decode_labels(output.predictions,output.label_ids,ID2LABEL)
    extra=additional_metrics(truth,predicted); loss=float(output.metrics['test_loss']); before=entity_counts(source); after=entity_counts(truth); truncated=before!=after
    values={'model':config['model_key'],'dataset_size':choice['dataset_size'],'best_checkpoint_epoch':float(choice['best_checkpoint_epoch']),
        'seed':42,'best_model_path':choice['best_model_path'],'test_loss':loss,'precision':precision_score(truth,predicted),
        'recall':recall_score(truth,predicted),'f1':f1_score(truth,predicted),**extra,'evaluated_examples':len(truth),
        'entity_counts_before_tokenization':before,'entity_counts_after_tokenization':after,
        'expected_source_entity_counts':EXPECTED_TEST_ENTITIES,'source_entity_counts_match_expected':before==EXPECTED_TEST_ENTITIES,
        'truncation_detected':truncated}
    warnings=[]
    if before!=EXPECTED_TEST_ENTITIES: warnings.append('Calculated source entity counts differ from the expected Test-400 counts.')
    if truncated: warnings.append('Test entities were truncated; metrics do not cover every source entity.')
    if warnings: values['warnings']=warnings; print('WARNING:', ' '.join(warnings),flush=True)
    save_json(outdir/'test_metrics.json',values); (outdir/'test_classification_report.txt').write_text(format_classification_report(truth,predicted,loss,'Test'),encoding='utf-8')
    rows,columns,matrix=entity_confusion(truth,predicted); write_matrix(rows,columns,matrix,outdir/'entity_confusion_matrix.csv',outdir/'entity_confusion_matrix.png',f"{config['model_key']}: Entity confusion matrix")
    existing=read_csv(comparison_path) if Path(comparison_path).is_file() else []; existing=[row for row in existing if row['Model']!=config['model_key']]
    existing.append({'Model':config['model_key'],'Dataset Size':choice['dataset_size'],'Best Epoch':choice['best_checkpoint_epoch'],
        'Test Loss':loss,'Precision':values['precision'],'Recall':values['recall'],'Micro-F1':values['f1'],'Macro-F1':values['macro_f1'],'BIO Accuracy':values['accuracy']})
    existing.sort(key=lambda row:['biobert','clinicalbert','pubmedbert'].index(row['Model'])); write_csv(comparison_path,TEST_FIELDS,existing)
    shutil.rmtree(outdir/'eval_temp',ignore_errors=True)

def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='stage',required=True)
    one=sub.add_parser('stage1'); one.add_argument('--config',required=True); one.add_argument('--dataset-size',choices=SIZES); one.add_argument('--all-dataset-sizes',action='store_true'); one.add_argument('--plots-only',action='store_true')
    two=sub.add_parser('stage2'); two.add_argument('--configs',nargs=3,required=True); two.add_argument('--output',default='results/dataset_size_experiment/stage2/selected_models.csv')
    three=sub.add_parser('stage3'); three.add_argument('--config'); three.add_argument('--configs',nargs=3); three.add_argument('--selected',default='results/dataset_size_experiment/stage2/selected_models.csv'); three.add_argument('--comparison',default='results/dataset_size_experiment/stage3/test_comparison.csv')
    args=parser.parse_args()
    if args.stage=='stage1':
        config=load_config(args.config); validate_config(config)
        if args.plots_only: update_stage1(config)
        else:
            if not args.dataset_size and not args.all_dataset_sizes: parser.error('stage1 requires --dataset-size or --all-dataset-sizes')
            for size in (SIZES if args.all_dataset_sizes else [args.dataset_size]): train_one(config,size)
    elif args.stage=='stage2': select(args.configs,args.output)
    else:
        paths=args.configs or ([args.config] if args.config else [])
        if not paths: parser.error('stage3 requires --config or --configs')
        if args.configs and {load_config(path)['model_key'] for path in paths}!=MODELS: raise ValueError('Provide exactly three selected configs')
        for path in paths:
            config=load_config(path); validate_config(config); test_one(config,args.selected,args.comparison)

if __name__=='__main__': main()
