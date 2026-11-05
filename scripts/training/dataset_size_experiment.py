"""Dataset-size experiment: Stage 1 training, Stage 2 selection, Stage 3 test."""
# ==============================================================================
# โครงการวิจัย: Medication Information Extraction - Dataset Size Experiment
# สคริปต์หลัก: dataset_size_experiment.py
#
# ภาพรวมการทดลองแบบ 3 Stage:
# - Stage 1 (Training & Validation):
#     ฝึกโมเดล Transformer การแพทย์ 3 ตัว (BioBERT, ClinicalBERT, PubMedBERT)
#     กับชุดข้อมูล 3 ขนาด (50%, 75%, 100%) ที่แบ่งเตรียมไว้ล่วงหน้า (รวม 9 การทดลอง)
#     ประเมินผลบน Validation Set ในทุก Epoch และคัดเลือก Best Checkpoint ของแต่ละการทดลอง
# - Stage 2 (Model Selection):
#     เปรียบเทียบผล Validation Macro-F1 ของทั้ง 3 ขนาดข้อมูลในแต่ละโมเดล
#     เพื่อคัดเลือก "ขนาดข้อมูลที่ดีที่สุด" (Best Dataset Size) เพียง 1 ขนาดต่อ 1 โมเดล
# - Stage 3 (Final Evaluation on Shared Test Set):
#     นำโมเดลตัวแทนที่ดีที่สุดของทั้ง 3 สถาปัตยกรรมที่คัดเลือกจาก Stage 2
#     มาประเมินผลขั้นสุดท้ายบน Shared Test Set ชุดเดียวกัน เพื่อสรุปและเปรียบเทียบประสิทธิภาพ
# ==============================================================================

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

# ==============================================================================
# ค่าคงที่หลักสำหรับการทดลอง Dataset Size Experiment
# ==============================================================================

# รายชื่อ key ขนาดชุดข้อมูลที่แบ่งเตรียมไว้แล้วล่วงหน้า (50%, 75%, 100%)
# จุดสำคัญ: 'size' เป็นตัวแปรสตริงที่ใช้เลือก key ใน config['datasets']
# ข้อมูลทั้ง 3 ขนาดถูกแบ่งและเตรียมไว้ในโฟลเดอร์ data/dataset_size/ เรียบร้อยแล้ว โปรแกรมไม่ได้สร้าง split ใหม่
SIZES = ['dataset_50', 'dataset_75', 'dataset_100']

# สถาปัตยกรรมโมเดล Transformer ทางการแพทย์ 3 ตัวที่ใช้ในการทดลองนี้
# (ไม่รวม BioClinicalBERT และการทดลองเดิม เพื่อควบคุมตัวแปรและตอบโจทย์วิจัยเฉพาะกลุ่มนี้)
MODELS = {'biobert', 'clinicalbert', 'pubmedbert'}

# คอลัมน์ที่บันทึกลง epoch_history.csv ในแต่ละ epoch ระหว่างการฝึก Stage 1
HISTORY_FIELDS = ['epoch','training_step','train_loss','validation_loss',
    'validation_precision','validation_recall','validation_f1',
    'validation_macro_f1','validation_accuracy','learning_rate']

# คอลัมน์สำหรับสรุปผล Best Checkpoint ของแต่ละขนาดข้อมูลใน Stage 1 (best_checkpoints.csv)
BEST_FIELDS = ['dataset_size','maximum_epochs','actual_final_epoch',
    'best_checkpoint_epoch','validation_loss','validation_precision',
    'validation_recall','validation_f1','validation_macro_f1',
    'validation_accuracy','seed','status','best_model_path']

# คอลัมน์สำหรับบันทึกโมเดลและขนาดข้อมูลที่ดีที่สุดที่ได้รับคัดเลือกใน Stage 2 (selected_models.csv)
SELECTED_FIELDS = ['model','dataset_size','best_checkpoint_epoch','validation_loss',
    'validation_precision','validation_recall','validation_f1',
    'validation_macro_f1','validation_accuracy','seed','best_model_path',
    'signature','run_summary_path']

# คอลัมน์สำหรับตารางเปรียบเทียบผลการทดสอบขั้นสุดท้ายใน Stage 3 (test_comparison.csv)
TEST_FIELDS = ['Model','Dataset Size','Best Epoch','Test Loss','Precision',
    'Recall','Micro-F1','Macro-F1','BIO Accuracy']

# จำนวน Entity อ้างอิงตามจริงในชุดข้อมูล Test (Shared Test 400 ฉบับ)
# ใช้ตรวจสอบความครบถ้วนของข้อมูล Ground Truth และตรวจจับ Truncation ก่อน/หลัง Tokenization
EXPECTED_TEST_ENTITIES = {'DRUG':445,'STRENGTH':95,'DOSAGE':73,'FREQUENCY':329,'ROUTE':259}

# ------------------------------------------------------------------------------
# ฟังก์ชัน: load_config
# หน้าที่: โหลดไฟล์การตั้งค่า (JSON config) ของโมเดล
# รับค่า: path (str หรือ Path) - ที่อยู่ไฟล์คอนฟิก เช่น scripts/training/configs/dataset_size/biobert.json
# คืนค่า: dict ของการตั้งค่าทั้งหมด (model_key, model_name, paths, training hyperparameters)
# ถูกเรียกจาก: main() ใน Stage 1/Stage 3 และ select() ใน Stage 2
# ------------------------------------------------------------------------------
def load_config(path):
    with open(path, encoding='utf-8') as stream: return json.load(stream)

# ------------------------------------------------------------------------------
# ฟังก์ชัน: write_csv
# หน้าที่: เขียนข้อมูลแถวตารางลงไฟล์ CSV แบบ Atomic Write เพื่อป้องกันไฟล์เสียหายกรณีระบบหยุดกะทันหัน
#         (เขียนลงไฟล์ .tmp ก่อน แล้วใช้วิธี atomic replace แทนที่ไฟล์จริง) และใช้การเข้ารหัส UTF-8 with BOM (utf-8-sig)
# รับค่า: path (str หรือ Path), fields (list ของชื่อคอลัมน์), rows (list ของ dict ข้อมูลแต่ละแถว)
# คืนค่า: ไม่มี (เขียน/แทนที่ไฟล์ CSV บนดิสก์)
# ถูกเรียกจาก: ArtifactCallback (Stage 1), update_stage1 (Stage 1), select (Stage 2), test_one (Stage 3)
# ------------------------------------------------------------------------------
def write_csv(path, fields, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temp,path)

# ------------------------------------------------------------------------------
# ฟังก์ชัน: read_csv
# หน้าที่: อ่านไฟล์ CSV และแปลงเป็น list ของ dict เพื่อนำข้อมูลประวัติหรือผลลัพธ์มาประมวลผลต่อ
# รับค่า: path (str หรือ Path) - ที่อยู่ไฟล์ CSV ที่ต้องการอ่าน
# คืนค่า: list ของ dict (แต่ละ dict แทนข้อมูล 1 แถวในตาราง)
# ถูกเรียกจาก: update_stage1 (Stage 1) และ test_one (Stage 3)
# ------------------------------------------------------------------------------
def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as stream: return list(csv.DictReader(stream))

# ------------------------------------------------------------------------------
# ฟังก์ชัน: validate_config
# หน้าที่: ตรวจสอบความถูกต้องของค่า Configuration เพื่อให้มั่นใจว่าการทดลองเป็นไปตามระเบียบวิธีวิจัยที่กำหนดไว้
#         และป้องกันการแก้ไขพารามิเตอร์โดยไม่ตั้งใจ (Hyperparameter Integrity Guard)
# รับค่า: config (dict) - ค่าคอนฟิกที่โหลดมาจากไฟล์ JSON
# คืนค่า: ไม่มี (จะ raise ValueError หากพบว่าค่าใดไม่ตรงตามข้อกำหนด)
# ถูกเรียกจาก: main() ก่อนเริ่มทำงานใน Stage 1 และ Stage 3
# ------------------------------------------------------------------------------
def validate_config(config):
    # ตรวจสอบว่า model_key อยู่ในชุดโมเดลที่กำหนด (biobert, clinicalbert, pubmedbert)
    if config.get('model_key') not in MODELS: raise ValueError('Only biobert, clinicalbert, and pubmedbert are in this experiment')
    # ตรวจสอบค่า Hyperparameters สำหรับการฝึกที่กำหนดคงที่ในระเบียบวิธีวิจัย:
    # - learning_rate: 2e-5 (initial learning rate)
    # - train_batch_size: 16 (ขนาด batch ในการฝึก)
    # - maximum_epochs: 100 (จำนวน epoch สูงสุด)
    # - seed: 42 (สุ่มแบบกำหนดค่าคงที่เพื่อความสามารถในการทำซ้ำได้)
    # - gradient_accumulation_steps: 1
    # - early_stopping_patience: 10 (หยุดเมื่อ validation macro-f1 ไม่ดีขึ้นต่อเนื่อง 10 epoch)
    # - early_stopping_threshold: 0.0 (ต้องการการปรับปรุงค่า > 0.0)
    # - lr_scheduler_type: 'linear' (ลด learning rate แบบเส้นตรงตลอดการฝึก)
    # - warmup_steps: 0
    expected={'learning_rate':2e-5,'train_batch_size':16,'maximum_epochs':100,
        'seed':42,'gradient_accumulation_steps':1,'early_stopping_patience':10,
        'early_stopping_threshold':0.0,'lr_scheduler_type':'linear','warmup_steps':0}
    for key,value in expected.items():
        if config.get('training',{}).get(key)!=value: raise ValueError(f'training.{key} must be {value!r}')
    # ตรวจสอบว่า datasets มีครบและเรียงลำดับ ['dataset_50', 'dataset_75', 'dataset_100']
    if list(config.get('datasets',{}))!=SIZES: raise ValueError(f'Dataset order must be {SIZES}')

# ------------------------------------------------------------------------------
# ฟังก์ชัน: tokenize
# หน้าที่: แปลงคำ (Tokens) เป็น Subwords ตาม Tokenizer ของแต่ละโมเดล พร้อมจัดแนว BIO Labels ให้ตรงกับ Subwords
#         โดยให้ Label เฉพาะ Subword แรกของคำ และกำหนด -100 ให้กับ Subword ที่เหลือและ Special Tokens
# รับค่า: examples (dict ของ batch ข้อมูลที่มี 'tokens' และ 'ner_tags')
#        tokenizer (PreTrainedTokenizerFast จาก Hugging Face)
# คืนค่า: result (dict ที่มี 'input_ids', 'attention_mask' และ 'labels' ที่จัดแนวแล้ว)
# ถูกเรียกจาก: train_one() (Stage 1) และ test_one() (Stage 3) ผ่าน dataset.map(batched=True)
# ------------------------------------------------------------------------------
def tokenize(examples,tokenizer):
    # ทำ Tokenization จากข้อความที่ถูกตัดเป็นคำไว้แล้ว (is_split_into_words=True)
    # และตัดทอนประโยคที่ยาวเกินขีดจำกัดโมเดล (truncation=True)
    result=tokenizer(examples['tokens'],truncation=True,is_split_into_words=True)
    labels=[]
    # วนลูปจัดแนวป้ายกำกับ (Label Alignment) สำหรับแต่ละตัวอย่างใน batch
    for index,tags in enumerate(examples['ner_tags']):
        previous=None; aligned=[]
        # ใช้ word_ids() เพื่อระบุว่าแต่ละ subword ตรงกับคำดั้งเดิมลำดับที่เท่าใด
        for word in result.word_ids(batch_index=index):
            # กฎการกำหนด Label ตามมาตรฐาน Hugging Face Token Classification:
            # 1. ถ้าเป็น Special Token (word is None) หรือเป็น Subword ตัวถัดมาของคำเดิม (word == previous)
            #    ให้กำหนดเป็น -100 ซึ่ง PyTorch CrossEntropyLoss จะละเว้น ไม่นำมาคิด Loss และ seqeval จะตัดทิ้ง
            # 2. ถ้าเป็น Subword แรกของคำ (word != previous) ให้แปลง BIO Tag ของคำนั้นเป็น ID ผ่าน LABEL2ID
            aligned.append(-100 if word is None or word==previous else LABEL2ID[tags[word]])
            previous=word
        labels.append(aligned)
    result['labels']=labels; return result

# ------------------------------------------------------------------------------
# ฟังก์ชัน: metrics
# หน้าที่: คำนวณ Evaluation Metrics สำหรับ Hugging Face Trainer ในระดับ Entity-level และ Token-level
# รับค่า: output (EvalPrediction object ที่มี predictions และ label_ids จากโมเดล)
# คืนค่า: dict ของตัววัดประสิทธิภาพ ได้แก่:
#         - precision, recall, f1 (Entity-level แบบ Micro-F1 ตามมาตรฐาน seqeval)
#         - macro_f1 (Macro-F1 เฉลี่ย 5 Entity Types หลักอย่างเท่าเทียม ไม่คิดคลาส O)
#         - accuracy (Token-level BIO Accuracy วัดความถูกต้องระดับ Subword แรก)
# ถูกเรียกจาก: Trainer ใน compute_metrics เมื่อประเมิน Validation ในแต่ละ epoch (Stage 1)
# ------------------------------------------------------------------------------
def metrics(output):
    # ถอดรหัส predictions (argmax) และกรองตำแหน่ง -100 ออกให้เหลือเฉพาะป้ายกำกับของคำจริง
    truth,predicted=decode_labels(output.predictions,output.label_ids,ID2LABEL)
    # คำนวณ Entity-level Precision, Recall, Micro-F1 จาก seqeval ร่วมกับ Macro-F1 และ Accuracy จาก additional_metrics
    return {'precision':precision_score(truth,predicted),'recall':recall_score(truth,predicted),
        'f1':f1_score(truth,predicted),**additional_metrics(truth,predicted)}

# ------------------------------------------------------------------------------
# ฟังก์ชัน: history_rows
# หน้าที่: สกัดและจับคู่ข้อมูลประวัติการฝึกจาก Trainer logs ให้เป็นตารางราย Epoch
# รับค่า: logs (list ของ dict จาก trainer.state.log_history)
# คืนค่า: list ของ dict ที่แต่ละแถวประกอบด้วย epoch, training_step, train_loss, validation metrics และ learning_rate
# หมายเหตุสำคัญ: train_loss ที่บันทึกมาจาก log ของ Trainer ที่เฉลี่ยระหว่าง epoch นั้นๆ
#                ไม่ใช่ loss ที่คำนวณใหม่ ณ best checkpoint
# ถูกเรียกจาก: ArtifactCallback.on_evaluate() และ train_one() (Stage 1)
# ------------------------------------------------------------------------------
def history_rows(logs):
    # แยก log ของฝั่ง training (ที่มี loss แต่ไม่มี eval_loss)
    train=[row for row in logs if 'loss' in row and 'eval_loss' not in row]; rows=[]
    # วนลูปจับคู่กับ log ของฝั่ง evaluation (ที่มี eval_macro_f1)
    for item in (row for row in logs if 'eval_macro_f1' in row):
        epoch=float(item['epoch']); candidates=[x for x in train if float(x.get('epoch',-1))<=epoch+1e-9]
        logged=candidates[-1] if candidates else {}
        rows.append({'epoch':epoch,'training_step':int(item.get('step',0)),
            'train_loss':logged.get('loss'),'validation_loss':item['eval_loss'],
            'validation_precision':item['eval_precision'],'validation_recall':item['eval_recall'],
            'validation_f1':item['eval_f1'],'validation_macro_f1':item['eval_macro_f1'],
            'validation_accuracy':item['eval_accuracy'],'learning_rate':logged.get('learning_rate')})
    # กำจัดแถว epoch ซ้ำซ้อน (ถ้ามี) ให้เหลือค่าล่าสุดของแต่ละ epoch
    return list({row['epoch']:row for row in rows}.values())

# ------------------------------------------------------------------------------
# ฟังก์ชัน: plot_curve
# หน้าที่: วาดและบันทึกภาพกราฟเส้น (Learning Curve) แสดง Validation Entity-level Macro-F1 ในแต่ละ Epoch
#         พร้อมระบุจุดสีแดงและข้อความกำกับ Best Checkpoint Epoch
# รับค่า: rows (list ของ dict ประวัติการฝึกจาก history_rows), path (Path สำหรับบันทึกไฟล์ภาพ PNG), title (str ชื่อหัวกราฟ)
# คืนค่า: ไม่มี (บันทึกไฟล์ภาพลงดิสก์)
# ถูกเรียกจาก: ArtifactCallback.on_evaluate() และ train_one() (Stage 1)
# ------------------------------------------------------------------------------
def plot_curve(rows,path,title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    figure,axis=plt.subplots(figsize=(8,5))
    if rows:
        epochs=[float(x['epoch']) for x in rows]; scores=[float(x['validation_macro_f1']) for x in rows]
        # หาตำแหน่ง Best Checkpoint ที่ให้ค่า Validation Macro-F1 สูงสุด (first maximum wins)
        best=max(range(len(rows)),key=lambda i:scores[i]); axis.plot(epochs,scores,marker='o')
        axis.scatter(epochs[best],scores[best],color='red',zorder=3)
        axis.annotate(f'Best: epoch {epochs[best]:g}, {scores[best]:.4f}',(epochs[best],scores[best]),xytext=(8,8),textcoords='offset points')
    axis.set(title=title,xlabel='Epoch',ylabel='Validation Entity-level Macro-F1'); axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(path,dpi=160); plt.close(figure)

# ------------------------------------------------------------------------------
# คลาส: ArtifactCallback (สืบทอดจาก TrainerCallback)
# หน้าที่: Callback สำหรับสร้างและอัปเดตไฟล์ผลลัพธ์ (Artifacts) แบบ Real-time ทันทีที่จบการประเมินในแต่ละ Epoch
#         ทำให้สามารถติดตามผลและตรวจสอบกราฟได้ทันทีแม้การฝึกยังไม่สิ้นสุด
# รับค่าตอน __init__: directory (Path โฟลเดอร์ผลลัพธ์), title (str ชื่อกราฟ)
# เมธอด on_evaluate: ถูกเรียกโดย Hugging Face Trainer ทุกครั้งที่ประเมิน Validation เสร็จสิ้น
# ถูกเรียกจาก: Trainer ใน Stage 1 train_one()
# ------------------------------------------------------------------------------
class ArtifactCallback(TrainerCallback):
    def __init__(self,directory,title): self.directory=Path(directory); self.title=title
    def on_evaluate(self,args,state,control,**kwargs):
        # บันทึกประวัติ epoch_history.csv และวาดกราฟ validation_curve.png ทันทีที่มีผล validation ใหม่
        rows=history_rows(state.log_history)
        write_csv(self.directory/'epoch_history.csv',HISTORY_FIELDS,rows)
        plot_curve(rows,self.directory/'validation_curve.png',self.title)

# ------------------------------------------------------------------------------
# ฟังก์ชัน: sha
# หน้าที่: คำนวณรหัสแฮช SHA-256 จากเนื้อหาไบนารีของไฟล์
# รับค่า: path (str หรือ Path)
# คืนค่า: str (รหัสแฮชฐานสิบหก 64 ตัวอักษร)
# ถูกเรียกจาก: signature()
# ------------------------------------------------------------------------------
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# ------------------------------------------------------------------------------
# ฟังก์ชัน: signature
# หน้าที่: สร้าง Run Signature (SHA-256) สำหรับตรวจสอบความถูกต้องและการทำซ้ำได้ (Reproducibility & Integrity)
#         คำนวณจาก: ข้อมูลโมเดล, ขนาดชุดข้อมูล, ค่า Hyperparameters, แฮชไฟล์ข้อมูล Train/Validation,
#         แฮชไฟล์ซอร์สโค้ด (dataset_size_experiment.py, ner_common.py) และเวอร์ชันของไลบรารีที่เกี่ยวข้อง
# รับค่า: config (dict), size (str เช่น 'dataset_50')
# คืนค่า: tuple ของ (run_signature_hash, payload_dict)
# ข้อควรระวัง: การแก้ไขไฟล์ Python ซอร์สโค้ด (แม้เพียง comment) จะเปลี่ยนค่าแฮชของไฟล์ ทำให้ได้ signature ใหม่
# ถูกเรียกจาก: train_one() (Stage 1) และ select() (Stage 2)
# ------------------------------------------------------------------------------
def signature(config,size):
    files=config['datasets'][size]
    payload={'model':config['model_name'],'size':size,'training':config['training'],
        'data':{key:{'path':value,'sha256':sha(value)} for key,value in files.items()},
        'code':{name:sha(Path(__file__).with_name(name)) for name in ['dataset_size_experiment.py','ner_common.py']},
        'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','datasets','accelerate','seqeval','numpy']}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload

# ------------------------------------------------------------------------------
# ฟังก์ชัน: source_labels
# หน้าที่: อ่านและตรวจสอบความถูกต้องของ Tokens และ BIO Labels จากไฟล์ข้อมูล JSONL ดั้งเดิมก่อนการ Tokenize
# รับค่า: path (str หรือ Path ของไฟล์ JSONL)
# คืนค่า: list ของ list ของ BIO labels ในแต่ละประโยค
# ตรวจสอบ: 1) ความยาว tokens ต้องเท่ากับ ner_tags ในทุกประโยค 2) ต้องไม่มี Tag นอกเหนือจาก LABEL_LIST
# ถูกเรียกจาก: train_one() (Stage 1 สำหรับ train/val) และ test_one() (Stage 3 สำหรับ test)
# ------------------------------------------------------------------------------
def source_labels(path):
    answer=[]
    with open(path,encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            row=json.loads(line)
            # ตรวจสอบการจับคู่ 1:1 ระหว่าง token และ tag ในข้อมูลตั้งต้น
            if len(row['tokens'])!=len(row['ner_tags']): raise ValueError(f'Unaligned row {path}:{number}')
            # ตรวจสอบว่าไม่มี Label แปลกปลอมนอกเหนือจาก BIO List ที่กำหนด
            unknown=set(row['ner_tags'])-set(LABEL_LIST)
            if unknown: raise ValueError(f'Unknown labels {sorted(unknown)} at {path}:{number}')
            answer.append(row['ner_tags'])
    return answer

# ------------------------------------------------------------------------------
# ฟังก์ชัน: completed
# หน้าที่: ตรวจสอบว่าผลการทดลองเดิมเสร็จสมบูรณ์และตรงตาม Signature ปัจจุบันทุกประการหรือไม่
#         เพื่อตัดสินใจว่าสามารถข้ามการรัน (Skip) ได้อย่างปลอดภัยโดยไม่ต้องฝึกซ้ำ
# รับค่า: summary (dict จาก run_summary.json), run_signature (str แฮชที่คำนวณใหม่)
# คืนค่า: bool (True ถ้าผลลัพธ์สมบูรณ์และตรงกับ signature, False ถ้าต้องรันใหม่)
# เงื่อนไข: status == 'Completed' และ signature ตรงกัน และมี epoch_history.csv และมี validation_report และมีไฟล์โมเดนครบ
# ถูกเรียกจาก: train_one() (Stage 1) และ select() (Stage 2)
# ------------------------------------------------------------------------------
def completed(summary,run_signature):
    return summary.get('status')=='Completed' and summary.get('signature')==run_signature \
        and Path(summary.get('epoch_history_path','')).is_file() \
        and Path(summary.get('validation_report_path','')).is_file() \
        and model_files_exist(summary.get('best_model_path',''))

# ------------------------------------------------------------------------------
# ฟังก์ชัน: update_stage1
# หน้าที่: รวบรวมผลลัพธ์ของโมเดลตัวนั้นจากทั้ง 3 ขนาดข้อมูล (50%, 75%, 100%) ใน Stage 1
#         สร้างตารางสรุป best_checkpoints.csv และวาดภาพกราฟเปรียบเทียบขนาดข้อมูล dataset_comparison.png
# รับค่า: config (dict ของโมเดล)
# คืนค่า: ไม่มี (บันทึก best_checkpoints.csv และ dataset_comparison.png ในโฟลเดอร์ result_root ของโมเดล)
# ถูกเรียกจาก: train_one() (เมื่อจบการฝึกแต่ละขนาด) และ main() (เมื่อสั่งรันด้วย flag --plots-only)
# อยู่ใน: Stage 1
# ------------------------------------------------------------------------------
def update_stage1(config):
    root=Path(config['result_root']); table=[]; histories={}
    # วนลูปอ่านผล run_summary.json และประวัติ epoch_history.csv ของทั้ง 3 ขนาด
    for size in SIZES:
        path=root/size/'run_summary.json'; summary=read_json(path) if path.is_file() else {'status':'Not Started'}
        values=summary.get('best_validation_metrics',{})
        table.append({'dataset_size':size,'maximum_epochs':summary.get('maximum_epochs',100),
            'actual_final_epoch':summary.get('actual_final_epoch'),'best_checkpoint_epoch':summary.get('best_checkpoint_epoch'),
            **{'validation_'+key:values.get(key) for key in ['loss','precision','recall','f1','macro_f1','accuracy']},
            'seed':summary.get('seed',42),'status':summary.get('status'),'best_model_path':summary.get('best_model_path')})
        history=root/size/'epoch_history.csv'
        if history.is_file(): histories[size]=read_csv(history)
    # บันทึกตารางสรุปเปรียบเทียบ Best Checkpoint ของแต่ละขนาด
    write_csv(root/'best_checkpoints.csv',BEST_FIELDS,table)
    # วาดกราฟเปรียบเทียบเส้นทางการเรียนรู้ (Validation Macro-F1) ของทั้ง 3 ขนาดข้อมูล
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

# ------------------------------------------------------------------------------
# ฟังก์ชัน: train_one
# หน้าที่: ฟังก์ชันหลักของ Stage 1 ดำเนินการฝึกโมเดล 1 สถาปัตยกรรม กับ 1 ขนาดข้อมูล (50%, 75% หรือ 100%)
#         เริ่มจาก Pretrained weights ใหม่เสมอ (seed=42), ฝึกด้วย Trainer สูงสุด 100 epochs พร้อม Early Stopping
#         ประเมิน Validation ทุก Epoch, คัดเลือก Best Checkpoint ตาม Validation Macro-F1
#         และบันทึกโมเดล ประวัติ กราฟ และรายงานผลการประเมิน
# รับค่า: config (dict ของโมเดล), size (str เช่น 'dataset_50')
# คืนค่า: ไม่มี (บันทึกโมเดลและไฟล์ผลลัพธ์ทั้งหมดลงโฟลเดอร์ model_root และ result_root)
# ถูกเรียกจาก: main() ใน Stage 1
# อยู่ใน: Stage 1
# ------------------------------------------------------------------------------
def train_one(config,size):
    # [1] อ่านและตรวจ config สำหรับขนาดข้อมูลนี้
    settings=config['training']; files=config['datasets'][size]
    result=Path(config['result_root'])/size; weights=Path(config['model_root'])/size
    run_signature,payload=signature(config,size); summary_path=result/'run_summary.json'

    # [2] ตรวจสถานะการทดลองเดิมและป้องกันการรันซ้ำ (ถ้าสถานะ Completed และ Signature ตรงกันครบถ้วน ให้ข้ามได้)
    if summary_path.is_file() and completed(read_json(summary_path),run_signature):
        print(config['model_key'],size,'already Completed; skipping'); update_stage1(config); return

    # สร้างโฟลเดอร์ผลลัพธ์ และสร้างไฟล์ Lock ป้องกันการรันซ้ำซ้อนพร้อมกันใน process อื่น
    Path(config['result_root']).mkdir(parents=True,exist_ok=True)
    lock=Path(config['result_root'])/f'.{size}.RUNNING.lock'
    try: descriptor=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as error: raise RuntimeError(f'{lock} exists; verify no duplicate run is active before removing a stale lock') from error
    with os.fdopen(descriptor,'w') as stream: stream.write(str(os.getpid()))

    # หากผลเดิมไม่สมบูรณ์หรือไม่ตรง signature ให้เคลียร์โฟลเดอร์เดิมและเริ่มใหม่แบบ clean slate
    if result.exists(): shutil.rmtree(result)
    if weights.exists(): shutil.rmtree(weights)
    result.mkdir(parents=True); weights.mkdir(parents=True)

    # เริ่มต้นบันทึกสถานะ 'Running' และข้อมูล metadata ของการทดลอง
    summary={'status':'Running','model':config['model_key'],'model_id':config['model_name'],'dataset_size':size,
        'dataset_paths':files,
        'seed':42,'training_settings':settings,'maximum_epochs':100,'signature':run_signature,'signature_payload':payload,
        'train_loss_definition':'Trainer loss logged during each epoch; not a loss recomputed at the best checkpoint.',
        'learning_rate_note':'2e-5 is the initial LR; the linear scheduler reduces LR during training.'}
    save_json(summary_path,summary); started=time.perf_counter(); trainer=model=output=None

    try:
        # [3] ตรวจ tokens และ BIO labels ในไฟล์ Train และ Validation ต้นฉบับ
        train_labels=source_labels(files['train_file']); validation_labels=source_labels(files['validation_file'])
        summary.update(train_examples=len(train_labels),validation_examples=len(validation_labels)); save_json(summary_path,summary)
        # ตรวจสอบสภาพแวดล้อมฮาร์ดแวร์ (ควบคุมตัวแปรให้รัน 1 Process และมองเห็น 1 GPU เท่านั้น)
        if int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()>1: raise ValueError('Use one process and one visible GPU')

        # [4] โหลด Train/Validation จาก JSONL ด้วย Hugging Face datasets
        split_files={'train':files['train_file'],'validation':files['validation_file']}
        seed_everything(42); dataset=load_dataset('json',data_files=split_files)

        # [5] โหลด tokenizer และจัด labels ตาม subword (ให้ label เฉพาะ subword แรก, ส่วนอื่นใช้ -100)
        tokenizer=AutoTokenizer.from_pretrained(config['model_name'],use_fast=True)
        tokenized=dataset.map(lambda batch:tokenize(batch,tokenizer),batched=True)

        # [6] โหลด pretrained model สำหรับ Token Classification (เริ่มใหม่จาก Hugging Face Hub ไม่ใช้ checkpoint เก่า)
        seed_everything(42)
        model=AutoModelForTokenClassification.from_pretrained(config['model_name'],num_labels=len(LABEL_LIST),id2label=ID2LABEL,label2id=LABEL2ID)

        # [7] กำหนด TrainingArguments, Trainer และ callbacks
        # - LR: 2e-5, Batch: 16, Max Epochs: 100, Linear Scheduler, Warmup: 0
        # - ประเมินและบันทึกทุก Epoch (eval_strategy='epoch', save_strategy='epoch')
        # - คัดเลือก Checkpoint ที่ดีที่สุดด้วย Validation Macro-F1 (metric_for_best_model='macro_f1')
        # - Early Stopping patience=10 (หยุดเมื่อไม่ดีขึ้น 10 epoch ติดต่อกัน)
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

        # [8] ฝึก ประเมิน Validation และเลือก best checkpoint ใน Stage 1
        seed_everything(42); trainer.train(resume_from_checkpoint=False)
        history=history_rows(trainer.state.log_history); write_csv(result/'epoch_history.csv',HISTORY_FIELDS,history)
        plot_curve(history,result/'validation_curve.png',callback.title)
        if not history: raise RuntimeError('No validation history was produced')
        best=max(history,key=lambda row:float(row['validation_macro_f1'])) # first maximum wins

        # [9] บันทึกโมเดล รายงาน ประวัติ และกราฟ
        trainer.save_model(str(weights/'best_model')); tokenizer.save_pretrained(str(weights/'best_model'))
        # ประเมิน Validation อีกครั้งด้วย Best Model เพื่อสร้าง Classification Report และตรวจจับ Truncation
        output=trainer.predict(tokenized['validation'],metric_key_prefix='validation_best')
        truth,predicted=decode_labels(output.predictions,output.label_ids,ID2LABEL)
        before=entity_counts(validation_labels); after=entity_counts(truth); truncated=before!=after
        report=format_classification_report(truth,predicted,float(best['validation_loss']),'Validation')
        report_path=result/'validation_classification_report.txt'; report_path.write_text(report,encoding='utf-8')

        # สรุปและบันทึกสถานะการทดลองเป็น 'Completed'
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
        # ปลด Lock file และคืนหน่วยความจำ GPU
        lock.unlink(missing_ok=True); del trainer,model,output; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# ------------------------------------------------------------------------------
# ฟังก์ชัน: select
# หน้าที่: ฟังก์ชันหลักของ Stage 2 คัดเลือก "ขนาดข้อมูลที่ดีที่สุด" (Best Dataset Size) สำหรับแต่ละโมเดล
#         โดยอ่านผลลัพธ์ของทั้ง 3 โมเดล x 3 ขนาด (รวม 9 การทดลอง) จาก Stage 1
#         เปรียบเทียบค่า Validation Macro-F1 ของขนาด 50%, 75%, 100% ในแต่ละโมเดล และเลือกขนาดที่ได้คะแนนสูงสุด
#         (ความแตกต่าง: Stage 1 เลือก Best Checkpoint แต่ละ run ส่วน Stage 2 เลือก Best Dataset Size ของแต่ละโมเดล)
# รับค่า: config_paths (list ของ 3 config files), output (path ของ selected_models.csv)
# คืนค่า: ไม่มี (บันทึกตาราง selected_models.csv)
# ถูกเรียกจาก: main() ใน Stage 2
# อยู่ใน: Stage 2
# ------------------------------------------------------------------------------
def select(config_paths,output):
    # [1] โหลดและตรวจสอบคอนฟิกของทั้ง 3 โมเดล
    configs=[load_config(path) for path in config_paths]
    if {x['model_key'] for x in configs}!=MODELS: raise ValueError('Provide exactly the three selected model configs')
    selected=[]
    # [2] วนลูปตรวจสอบผล Stage 1 ของแต่ละโมเดลในทุกขนาดข้อมูล (50%, 75%, 100%)
    for config in configs:
        candidates=[]
        for size in SIZES:
            path=Path(config['result_root'])/size/'run_summary.json'
            if not path.is_file(): raise RuntimeError(f'Missing complete run: {path}')
            summary=read_json(path); run_signature,_=signature(config,size)
            # ตรวจสอบว่าการทดลองเสร็จสมบูรณ์และ signature ตรงกัน
            if not completed(summary,run_signature): raise RuntimeError(f'Incomplete or mismatched run: {path}')
            candidates.append(summary)
        # [3] เลือกขนาดข้อมูลใน Stage 2: เปรียบเทียบ Validation Macro-F1 และเลือกตัวที่ได้คะแนนสูงสุด
        winner=max(candidates,key=lambda row:float(row['best_validation_metrics']['macro_f1']))
        values=winner['best_validation_metrics']; selected.append({'model':config['model_key'],'dataset_size':winner['dataset_size'],
            'best_checkpoint_epoch':winner['best_checkpoint_epoch'],**{'validation_'+key:values[key] for key in ['loss','precision','recall','f1','macro_f1','accuracy']},
            'seed':winner['seed'],'best_model_path':winner['best_model_path'],'signature':winner['signature'],
            'run_summary_path':str((Path(config['result_root'])/winner['dataset_size']/'run_summary.json').resolve())})
    # [4] บันทึกโมเดลและขนาดข้อมูลที่ชนะลง selected_models.csv
    write_csv(output,SELECTED_FIELDS,selected)

# ------------------------------------------------------------------------------
# ฟังก์ชัน: read_matrix_csv
# หน้าที่: อ่าน Entity Confusion Matrix จากไฟล์ CSV เพื่อเตรียมนำไปพล็อต Heatmap
# รับค่า: path (str หรือ Path ไปยังไฟล์ CSV)
# คืนค่า: tuple ของ (rows, columns, values_numpy_array)
# ตรวจสอบ: แถวและคอลัมน์ต้องตรงตามรูปแบบ ENTITY_TYPES + Spurious/Missed และช่องขวาล่างสุดต้องเป็น N/A
# ถูกเรียกจาก: render_matrix_from_csv()
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# ฟังก์ชัน: normalize_matrix_csv_labels
# หน้าที่: ปรับแก้หัวตารางของไฟล์ CSV Confusion Matrix เดิมให้เป็นชื่อมาตรฐานภาษาอังกฤษ
# รับค่า: path (str หรือ Path)
# คืนค่า: ไม่มี (เขียนทับไฟล์ CSV เดิมแบบ atomic replace)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# ฟังก์ชัน: render_matrix_from_csv
# หน้าที่: เรนเดอร์ภาพ Heatmap ของ Entity Confusion Matrix จากไฟล์ CSV ออกมาเป็นไฟล์ PNG
# รับค่า: csv_path, png_path, title
# คืนค่า: ไม่มี (บันทึกภาพ PNG ลงดิสก์)
# ถูกเรียกจาก: write_matrix()
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# ฟังก์ชัน: write_matrix
# หน้าที่: บันทึกข้อมูล Entity Confusion Matrix ลงไฟล์ CSV และเรียกเรนเดอร์เป็นภาพ PNG
# รับค่า: rows, columns, matrix (dict), csv_path, png_path, title
# คืนค่า: ไม่มี (บันทึก CSV และ PNG)
# ถูกเรียกจาก: test_one() ใน Stage 3
# ------------------------------------------------------------------------------
def write_matrix(rows,columns,matrix,csv_path,png_path,title):
    with Path(csv_path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['Ground Truth / Prediction']+columns)
        for row in rows:
            writer.writerow([row]+[('N/A' if row==SPURIOUS_ENTITY and col==MISSED_ENTITY else matrix[row][col]) for col in columns])
    render_matrix_from_csv(csv_path,png_path,title)

# ------------------------------------------------------------------------------
# ฟังก์ชัน: test_one
# หน้าที่: ฟังก์ชันหลักของ Stage 3 ดำเนินการประเมินผลขั้นสุดท้ายบน Shared Test Set สำหรับโมเดลที่ชนะใน Stage 2
#         โหลด Best Checkpoint Weights ที่ได้รับคัดเลือก, ประเมินบน test_15.jsonl ด้วย Trainer.predict()
#         คำนวณ Metrics ทั้งหมด (Macro-F1, Micro-F1, Precision, Recall, BIO Accuracy)
#         สร้าง Entity Confusion Matrix และอัปเดตผลลง test_comparison.csv
# รับค่า: config (dict ของโมเดล), selected_path (path ไปยัง selected_models.csv), comparison_path (path ไปยัง test_comparison.csv)
# คืนค่า: ไม่มี (บันทึกผลการทดสอบทั้งหมดลง stage3_root/model_key)
# ถูกเรียกจาก: main() ใน Stage 3
# อยู่ใน: Stage 3
# ------------------------------------------------------------------------------
def test_one(config,selected_path,comparison_path):
    # [1] โหลดข้อมูลโมเดลที่ได้รับการคัดเลือกมาจาก Stage 2 และตรวจสอบความสมบูรณ์ของ signature กับ weights
    selected=[row for row in read_csv(selected_path) if row['model']==config['model_key']]
    if len(selected)!=1: raise ValueError(f'Stage 2 selection missing for {config["model_key"]}')
    choice=selected[0]; summary=read_json(choice['run_summary_path'])
    if summary['signature']!=choice['signature'] or not model_files_exist(choice['best_model_path']): raise RuntimeError('Selected Stage 1 weights are missing or mismatched')

    # [2] โหลด Shared Test Set (test_15.jsonl) และนับจำนวน Entity ต้นฉบับก่อนการ Tokenize
    test_path=config['test_file']; source=source_labels(test_path); outdir=Path(config['stage3_root'])/config['model_key']; outdir.mkdir(parents=True,exist_ok=True)
    dataset=load_dataset('json',data_files={'test':test_path}); tokenizer=AutoTokenizer.from_pretrained(choice['best_model_path'],use_fast=True)

    # [3] โหลดโมเดลและ Tokenizer ที่ชนะมาจาก Stage 2 และจัดแนว labels ตาม Subwords
    model=AutoModelForTokenClassification.from_pretrained(choice['best_model_path']); tokenized=dataset.map(lambda batch:tokenize(batch,tokenizer),batched=True)
    arguments=TrainingArguments(output_dir=str(outdir/'eval_temp'),per_device_eval_batch_size=16,report_to='none')
    trainer=Trainer(model=model,args=arguments,processing_class=tokenizer,data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer))

    # [4] ประเมิน Test ใน Stage 3 ด้วย Trainer.predict()
    output=trainer.predict(tokenized['test']); truth,predicted=decode_labels(output.predictions,output.label_ids,ID2LABEL)
    extra=additional_metrics(truth,predicted); loss=float(output.metrics['test_loss'])

    # [5] ตรวจสอบ Entity Counts และตรวจจับ Truncation เทียบกับชุดข้อมูล Test อ้างอิง (400 ฉบับ)
    before=entity_counts(source); after=entity_counts(truth); truncated=before!=after
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

    # [6] บันทึกผลลัพธ์: test_metrics.json, รายงานการจำแนกประเภท และ Entity Confusion Matrix
    save_json(outdir/'test_metrics.json',values); (outdir/'test_classification_report.txt').write_text(format_classification_report(truth,predicted,loss,'Test'),encoding='utf-8')
    rows,columns,matrix=entity_confusion(truth,predicted); write_matrix(rows,columns,matrix,outdir/'entity_confusion_matrix.csv',outdir/'entity_confusion_matrix.png',f"{config['model_key']}: Entity confusion matrix")

    # [7] อัปเดตผลลัพธ์ลงตารางเปรียบเทียบ test_comparison.csv รวมทุกโมเดล
    existing=read_csv(comparison_path) if Path(comparison_path).is_file() else []; existing=[row for row in existing if row['Model']!=config['model_key']]
    existing.append({'Model':config['model_key'],'Dataset Size':choice['dataset_size'],'Best Epoch':choice['best_checkpoint_epoch'],
        'Test Loss':loss,'Precision':values['precision'],'Recall':values['recall'],'Micro-F1':values['f1'],'Macro-F1':values['macro_f1'],'BIO Accuracy':values['accuracy']})
    existing.sort(key=lambda row:['biobert','clinicalbert','pubmedbert'].index(row['Model'])); write_csv(comparison_path,TEST_FIELDS,existing)

    # ลบไดเรกทอรี eval_temp ชั่วคราว
    shutil.rmtree(outdir/'eval_temp',ignore_errors=True)

# ------------------------------------------------------------------------------
# ฟังก์ชัน: main
# หน้าที่: จุดเริ่มต้นการทำงานของโปรแกรม (CLI Entrypoint) ควบคุมการทำงานของทั้ง 3 Stage
#         แยก Subcommand ออกเป็น 'stage1', 'stage2', และ 'stage3' ด้วย argparse
# ------------------------------------------------------------------------------
def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='stage',required=True)
    # Subcommand สำหรับ Stage 1: ฝึกและประเมิน Validation
    one=sub.add_parser('stage1'); one.add_argument('--config',required=True); one.add_argument('--dataset-size',choices=SIZES); one.add_argument('--all-dataset-sizes',action='store_true'); one.add_argument('--plots-only',action='store_true')
    # Subcommand สำหรับ Stage 2: คัดเลือก Best Dataset Size ของแต่ละโมเดล
    two=sub.add_parser('stage2'); two.add_argument('--configs',nargs=3,required=True); two.add_argument('--output',default='results/dataset_size_experiment/stage2/selected_models.csv')
    # Subcommand สำหรับ Stage 3: ประเมินผลโมเดลที่ได้รับคัดเลือกบน Shared Test Set
    three=sub.add_parser('stage3'); three.add_argument('--config'); three.add_argument('--configs',nargs=3); three.add_argument('--selected',default='results/dataset_size_experiment/stage2/selected_models.csv'); three.add_argument('--comparison',default='results/dataset_size_experiment/stage3/test_comparison.csv')
    args=parser.parse_args()

    # ประมวลผลตาม Stage ที่ผู้ใช้เลือกสั่งรัน
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
