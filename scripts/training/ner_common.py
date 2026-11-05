"""Small shared additions; all entity scoring remains seqeval default mode."""
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

# ==============================================================================
# ค่าคงที่และโครงสร้าง Label สำหรับโมเดล Token Classification (Named Entity Recognition)
# ==============================================================================

# ประเภท Entity ทั้ง 5 ชนิดที่สนใจสกัดในงาน Medication Information Extraction:
# - DRUG: ชื่อยา
# - STRENGTH: ความแรงยา (เช่น 500 mg)
# - DOSAGE: ขนาดยาหรือจำนวนที่รับประทาน (เช่น 1 เม็ด)
# - FREQUENCY: ความถี่ในการรับประทาน (เช่น วันละ 2 ครั้ง)
# - ROUTE: ช่องทางการให้ยา (เช่น oral, IV)
ENTITY_TYPES = ['DRUG', 'STRENGTH', 'DOSAGE', 'FREQUENCY', 'ROUTE']

# รายชื่อตัววัดประสิทธิภาพ 5 ค่าหลักที่ใช้ในการประเมินผลการเรียนรู้
METRIC_NAMES = ['precision', 'recall', 'f1', 'macro_f1', 'accuracy']

# รายการป้ายกำกับ BIO Scheme ทั้งหมด 11 คลาส (O + B-/I- ของทั้ง 5 Entity)
# รูปแบบ: 'O', 'B-DRUG', 'I-DRUG', 'B-STRENGTH', 'I-STRENGTH', ...
LABEL_LIST = ['O'] + [prefix + entity for entity in ENTITY_TYPES for prefix in ('B-', 'I-')]

# พจนานุกรมสำหรับแปลงป้ายกำกับสตริงเป็นตัวเลข ID เพื่อป้อนให้โมเดลประมวลผล
LABEL2ID = {label: index for index, label in enumerate(LABEL_LIST)}

# พจนานุกรมสำหรับแปลงตัวเลข ID ที่โมเดลทำนายกลับเป็นป้ายกำกับสตริง
ID2LABEL = {index: label for index, label in enumerate(LABEL_LIST)}

# ป้ายกำกับพิเศษสำหรับ Entity Confusion Matrix:
# SPURIOUS_ENTITY: False Positive (โมเดลทำนายมี Entity แต่ใน Ground Truth ไม่มี)
# MISSED_ENTITY: False Negative (ใน Ground Truth มี Entity แต่โมเดลตรวจไม่พบหรือขอบเขตผิด)
SPURIOUS_ENTITY = 'Spurious Entity'
MISSED_ENTITY = 'Missed Entity'

# ตารางจับคู่สำหรับแปลงชื่อป้ายกำกับเดิมภาษาไทยให้เป็นชื่อมาตรฐานภาษาอังกฤษ
LEGACY_MATRIX_LABELS = {
    'ไม่มี Entity จริง': SPURIOUS_ENTITY,
    'ไม่พบ Entity': MISSED_ENTITY,
}

# ------------------------------------------------------------------------------
# ฟังก์ชัน: normalize_matrix_label
# หน้าที่: แปลงชื่อป้ายกำกับของ Entity Confusion Matrix ให้เป็นชื่อภาษาอังกฤษมาตรฐาน
# รับค่า: label (str) - ชื่อป้ายกำกับเดิม
# คืนค่า: str - ชื่อป้ายกำกับมาตรฐาน
# ถูกเรียกจาก: read_matrix_csv, normalize_matrix_csv_labels ใน dataset_size_experiment.py
# ------------------------------------------------------------------------------
def normalize_matrix_label(label):
    """Return the canonical English label for old or current matrix files."""
    return LEGACY_MATRIX_LABELS.get(label, label)

# ------------------------------------------------------------------------------
# ฟังก์ชัน: additional_metrics
# หน้าที่: คำนวณ Fixed Five-Entity Macro-F1 (เฉลี่ย F1 ของทั้ง 5 Entity โดยไม่นำคลาส 'O' มาคิด)
#         และ Token-level BIO Accuracy (วัดความแม่นยำของทุกตำแหน่ง Token/Subword แรกที่ไม่ใช่ -100)
# รับค่า: true_labels (list ของ list ของ label strings จาก Ground Truth)
#        true_predictions (list ของ list ของ label strings จาก Prediction ของโมเดล)
# คืนค่า: dict ที่มี 'macro_f1' และ 'accuracy'
# ถูกเรียกจาก: metrics() ใน Stage 1 และ test_one() ใน Stage 3
# ------------------------------------------------------------------------------
def additional_metrics(true_labels, true_predictions):
    from seqeval.metrics import classification_report, accuracy_score
    # Default seqeval mode, exactly as the existing P/R/F1 evaluation.
    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
    return {
        'macro_f1': sum(report.get(entity, {}).get('f1-score', 0.0) for entity in ENTITY_TYPES) / len(ENTITY_TYPES),
        'accuracy': accuracy_score(true_labels, true_predictions),
    }

# ------------------------------------------------------------------------------
# ฟังก์ชัน: seed_everything
# หน้าที่: กำหนดค่า Random Seed (seed=42) ให้กับทุกระบบสุ่ม เพื่อความสามารถในการทำซ้ำได้ 100% (Reproducibility)
#         ครอบคลุม Python random, numpy, PyTorch CPU, PyTorch CUDA (GPU), Transformers set_seed
#         และตั้งค่า cuDNN ให้เป็น Deterministic
# รับค่า: seed (int) - ค่า seed คงที่ (กำหนดเป็น 42)
# คืนค่า: ไม่มี
# ถูกเรียกจาก: train_one() ใน Stage 1 (ก่อนโหลด dataset, ก่อนโหลด model, และก่อน trainer.train)
# ------------------------------------------------------------------------------
def seed_everything(seed):
    import numpy as np
    import torch
    from transformers import set_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------------------
# ฟังก์ชัน: decode_labels
# หน้าที่: ถอดรหัสผลลัพธ์ดิบ (Logits) จากโมเดลให้กลายเป็นชื่อป้ายกำกับ BIO แบบข้อความ
#         โดยใช้ argmax หา class ID ที่มีความน่าจะเป็นสูงสุดในแต่ละตำแหน่ง
#         และคัดกรองตำแหน่งที่เป็น -100 ออก (Subword ตัวถัดไปและ Special Tokens)
# รับค่า: predictions (numpy array ของ logits [batch, seq_len, num_labels])
#        labels (numpy array ของ ground truth IDs [batch, seq_len])
#        id2label (dict แมป ID เป็นชื่อ label)
# คืนค่า: tuple ของ (true_labels, true_predictions) ที่เป็น list ของ list ของข้อความ label
# ถูกเรียกจาก: metrics(), train_one() (Stage 1) และ test_one() (Stage 3)
# ------------------------------------------------------------------------------
def decode_labels(predictions, labels, id2label):
    # Same argmax / -100 filtering loop as the original script.
    import numpy as np
    predictions = np.argmax(predictions, axis=2)
    true_predictions, true_labels = [], []
    for prediction, label in zip(predictions, labels):
        pred_seq, label_seq = [], []
        for p, l in zip(prediction, label):
            if l == -100:
                continue
            pred_seq.append(id2label[int(p)])
            label_seq.append(id2label[int(l)])
        true_predictions.append(pred_seq)
        true_labels.append(label_seq)
    return true_labels, true_predictions

# ------------------------------------------------------------------------------
# ฟังก์ชัน: entity_counts
# หน้าที่: นับจำนวน Entity แต่ละประเภทจากลำดับป้ายกำกับ BIO ตามมาตรฐานของ seqeval get_entities
#         เพื่อใช้วัดจำนวน Entity รวม และใช้เปรียบเทียบก่อน/หลัง Tokenization เพื่อตรวจจับ Truncation
# รับค่า: label_sequences (list ของ list ของ BIO label strings)
# คืนค่า: dict ของจำนวน Entity แต่ละประเภท เช่น {'DRUG': 445, 'STRENGTH': 95, ...}
# ถูกเรียกจาก: train_one() (Stage 1) และ test_one() (Stage 3)
# ------------------------------------------------------------------------------
def entity_counts(label_sequences):
    """Count entities with seqeval's existing default BIO interpretation."""
    from seqeval.metrics.sequence_labeling import get_entities
    counts = Counter()
    for labels in label_sequences:
        counts.update(entity for entity, _start, _end in get_entities(labels))
    return {entity: counts.get(entity, 0) for entity in ENTITY_TYPES}

# ------------------------------------------------------------------------------
# ฟังก์ชัน: entity_confusion
# หน้าที่: สร้าง Entity Confusion Matrix ระดับ Exact-span (ขอบเขตคำต้องตรงกันพอดี) แบบ 1 ต่อ 1
#         เปรียบเทียบระหว่าง Ground Truth Entity กับ Predicted Entity
#         โดยจำแนกข้อผิดพลาดออกเป็น Misclassification (ทายผิดประเภท), Missed Entity (หลุดการตรวจจับ)
#         และ Spurious Entity (ทำนายเกินขึ้นมา)
# รับค่า: true_labels, predicted_labels (list ของ list ของ BIO tags)
# คืนค่า: tuple ของ (rows, columns, matrix_dict)
# ถูกเรียกจาก: test_one() ใน Stage 3
# ------------------------------------------------------------------------------
def entity_confusion(true_labels, predicted_labels):
    """Build an exact-span, one-to-one entity confusion matrix per document."""
    from seqeval.metrics.sequence_labeling import get_entities
    rows = ENTITY_TYPES + [SPURIOUS_ENTITY]
    columns = ENTITY_TYPES + [MISSED_ENTITY]
    matrix = {row: {column: 0 for column in columns} for row in rows}
    for truth, prediction in zip(true_labels, predicted_labels):
        true_spans = {(start, end): entity for entity, start, end in get_entities(truth)}
        predicted_spans = {(start, end): entity for entity, start, end in get_entities(prediction)}
        for boundary in sorted(set(true_spans) | set(predicted_spans)):
            true_entity = true_spans.get(boundary)
            predicted_entity = predicted_spans.get(boundary)
            if true_entity is not None and predicted_entity is not None:
                matrix[true_entity][predicted_entity] += 1
            elif true_entity is not None:
                matrix[true_entity][MISSED_ENTITY] += 1
            else:
                matrix[SPURIOUS_ENTITY][predicted_entity] += 1
    return rows, columns, matrix

# ------------------------------------------------------------------------------
# ฟังก์ชัน: format_classification_report
# หน้าที่: จัดรูปแบบรายงานสรุปผลการจำแนกประเภท (Classification Report)
#         แสดง Precision, Recall, F1-score และ Support ราย Entity
#         พร้อมระบุ Token-level BIO Accuracy และ Loss ของชุดข้อมูลนั้น
# รับค่า: true_labels, predicted_labels, loss (float), split_name (str เช่น 'Validation' หรือ 'Test')
# คืนค่า: str - ข้อความรายงานพร้อมจัดหน้าสวยงาม
# ถูกเรียกจาก: train_one() (Stage 1) และ test_one() (Stage 3)
# ------------------------------------------------------------------------------
def format_classification_report(true_labels, predicted_labels, loss, split_name):
    from seqeval.metrics import accuracy_score, classification_report
    report = classification_report(
        true_labels, predicted_labels, digits=4, zero_division=0
    ).replace('support', 'จำนวน Entity', 1)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return (
        report
        + f'\nToken-level BIO Accuracy: {accuracy:.4f} ({accuracy:.2%})\n'
        + f'{split_name} Loss: {loss:.6f}\n'
    )

# ------------------------------------------------------------------------------
# ฟังก์ชัน: save_json
# หน้าที่: บันทึกข้อมูลแบบ JSON ลงไฟล์แบบ Atomic Write เพื่อความปลอดภัยของข้อมูล
#         (เขียนลงไฟล์ .tmp แล้วแทนที่) พร้อมจัด Indent 2 เคาะและบันทึกเป็น UTF-8
# รับค่า: path (str หรือ Path), data (dict/list)
# คืนค่า: ไม่มี
# ถูกเรียกจาก: train_one() (Stage 1) และ test_one() (Stage 3)
# ------------------------------------------------------------------------------
def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)

# ------------------------------------------------------------------------------
# ฟังก์ชัน: read_json
# หน้าที่: โหลดและแปลงไฟล์ JSON เป็น Python dict ด้วยการเข้ารหัส UTF-8
# รับค่า: path (str หรือ Path)
# คืนค่า: ข้อมูลที่แปลงแล้ว (dict/list)
# ถูกเรียกจาก: update_stage1, train_one, select, test_one
# ------------------------------------------------------------------------------
def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)

def signature_for(config):
    # No access to test data during tuning; only hash train/validation.
    payload = {'config':config, 'train_validation_sha256':{}}
    for key in ['train_file','validation_file']:
        payload['train_validation_sha256'][key] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    payload['code'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['train_ner.py','ner_common.py']}
    import importlib.metadata
    payload['versions'] = {name:importlib.metadata.version(name) for name in ['torch','transformers','datasets','accelerate','seqeval','numpy']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def completed_run(record, signature):
    if record.get('status') != 'Completed' or record.get('signature') != signature:
        return False
    metadata_complete = (
        all(isinstance(record.get('validation',{}).get(name),(int,float))
            and math.isfinite(record['validation'][name]) for name in METRIC_NAMES)
        and Path(record.get('classification_report_path','')).is_file())
    if not metadata_complete:
        return False
    # Intentional pruning is distinct from a missing/corrupt retained checkpoint.
    if record.get('weights_state') == 'pruned':
        return (record.get('pruned_by') == 'storage_policy_v1'
                and Path(record.get('history_path','')).is_file())
    return model_files_exist(record.get('best_model_path',''))


# ------------------------------------------------------------------------------
# ฟังก์ชัน: model_files_exist
# หน้าที่: ตรวจสอบความสมบูรณ์ของโฟลเดอร์โมเดลที่บันทึกไว้ ว่ามีไฟล์สำคัญตามมาตรฐานครบถ้วนหรือไม่
#         ประกอบด้วย: config.json, tokenizer_config.json และไฟล์น้ำหนักโมเดล (model.safetensors หรือ pytorch_model.bin)
# รับค่า: path (str หรือ Path ไปยังโฟลเดอร์ best_model)
# คืนค่า: bool (True ถ้าไฟล์ครบถ้วน, False ถ้าขาดหาย)
# ถูกเรียกจาก: completed() ใน Stage 1/Stage 2 และ test_one() ใน Stage 3
# ------------------------------------------------------------------------------
def model_files_exist(path):
    model = Path(path)
    return ((model/'config.json').is_file()
            and (model/'tokenizer_config.json').is_file()
            and any((model/name).is_file() for name in ['model.safetensors','pytorch_model.bin']))


def load_storage_best(config, signature):
    pointer = Path(config['result_dir'])/'storage_best.json'
    if not pointer.exists():
        # No recovery is possible from pruned metadata alone: never pretend
        # missing global-best weights can be reconstructed from metrics.
        for path in (Path(config['output_dir'])/'grid').glob('*/result.json'):
            try:
                record = read_json(path)
            except (ValueError,OSError):
                continue
            if record.get('signature') == signature and record.get('weights_state') == 'pruned':
                raise RuntimeError('storage_best.json is missing but pruned runs exist. Restore the storage pointer and retained weights before resuming.')
        return None
    best = read_json(pointer)
    if best.get('signature') != signature or not model_files_exist(best.get('best_model_path','')):
        raise RuntimeError('Retained seed-42 best model is missing or mismatched. Restore it before Resume/Final Test; no weights were deleted.')
    return best


def maintain_storage(config, signature, record_path):
    """Keep the running seed-42 winner; journal metadata before deleting losers.

    Called only AFTER result.json is durably marked Completed. No model loading,
    training, metrics or selection rules change here.
    """
    import shutil
    record_path = Path(record_path)
    record = read_json(record_path)
    if record.get('status') != 'Completed' or record.get('signature') != signature:
        raise ValueError('Storage cleanup requires a completed matching run')
    legacy_report = Path(record.get('classification_report_path',''))
    if legacy_report.name == 'classification_report.txt' and legacy_report.is_file():
        validation_report = record_path.parent/'validation_classification_report.txt'
        if not validation_report.exists(): shutil.copyfile(legacy_report, validation_report)
        record['classification_report_path'] = str(validation_report.resolve())
        save_json(record_path, record)
    pointer = Path(config['result_dir'])/'storage_best.json'
    best = load_storage_best(config, signature)
    # Grid order remains the tiebreaker, including when resuming out of order.
    space = config['grid']
    def priority(row):
        return (row['validation']['macro_f1'],
                -space['learning_rate'].index(row['learning_rate']),
                -space['batch_size'].index(row['batch_size']),
                -space['epochs'].index(row['configured_epochs']))
    previous = None
    if record['seed'] == 42 and (best is None or priority(record) > priority(best)):
        if not model_files_exist(record['best_model_path']):
            raise RuntimeError('Candidate best weights are missing; cannot promote or prune')
        previous = best
        best = dict(record, result_path=str(record_path.resolve()))
        # Commit the new valid winner before touching the old winner.
        save_json(pointer, best)
    if best is None:
        raise RuntimeError('Keep a valid seed-42 grid winner before pruning multi-seed weights')
    protected = Path(best['best_model_path']).resolve()

    def prune(path):
        row = read_json(path)
        model = Path(row['best_model_path'])
        if model.resolve() == protected:
            if row.get('weights_state') != 'retained':
                row['weights_state'] = 'retained'
                save_json(path,row)
            return
        # Delete only this run's generated best_model subdirectory, never a
        # user-supplied arbitrary model path or the retained final-test model.
        expected = path.parent/'best_model'
        if model.resolve() != expected.resolve():
            raise RuntimeError('Unexpected model path; cleanup refused')
        history = path.parent/'trainer_history.json'
        if not history.is_file() or not Path(row['classification_report_path']).is_file():
            raise RuntimeError('Missing history/report; keep weights and stop cleanup')
        row.update(weights_state='pruned',pruned_by='storage_policy_v1',
                   history_path=str(history.resolve()))
        # Journal first: a crash during rmtree must not trigger re-training.
        save_json(path,row)
        if model.exists(): shutil.rmtree(model)

    prune(record_path)
    if previous is not None:
        prune(Path(previous['result_path']))

def write_grid_results(result_dir, records):
    result_dir = Path(result_dir)
    fields = ['experiment_id','model','learning_rate','batch_size','configured_epochs','seed','status',
              'best_checkpoint_epoch','best_model_path','weights_state','training_seconds'] + ['val_'+k for k in METRIC_NAMES] + ['error']
    path = result_dir/'grid_results.csv'
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key:record.get(key) for key in fields}
            row.update({'val_'+key:record.get('validation',{}).get(key) for key in METRIC_NAMES})
            writer.writerow(row)
    os.replace(temp,path)


# Compatibility only with the exact Minimal Changes source preceding this
# storage/report-only patch. Data, config and installed versions still must match.
def pre_storage_signature_for(config):
    # No access to test data during tuning; only hash train/validation.
    payload = {'config':config, 'train_validation_sha256':{}}
    for key in ['train_file','validation_file']:
        payload['train_validation_sha256'][key] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    payload['code'] = {'train_ner.py': '8b570c62542c8f23885fd32188c5249246b4e81701c689341fb4127f11556c26', 'ner_common.py': '467de3959efd50ddb729c3451faeae5d53b111eddff671aa278dc7f5be3a1ad1'}
    import importlib.metadata
    payload['versions'] = {name:importlib.metadata.version(name) for name in ['torch','transformers','datasets','accelerate','seqeval','numpy']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def compatible_signatures(config):
    return {signature_for(config), pre_storage_signature_for(config)}
