# ==============================================================================
# Unit Tests สำหรับตรรกะหลักของ Dataset Size Experiment
# ==============================================================================
# วัตถุประสงค์:
# ตรวจสอบความถูกต้องของตรรกะการคำนวณสถิติ (Entity Confusion Matrix) และความถูกต้องเชิงโครงสร้าง (Syntax & Configs)
#
# ข้อจำกัดสำคัญ:
# ชุดทดสอบนี้ใช้ข้อมูลจำลอง (Mock/Synthetic Data) และการวิเคราะห์โค้ดแบบคงที่ (Static AST Parsing)
# เพื่อให้รันได้อย่างรวดเร็วโดยไม่ต้องพึ่งพาโมเดลจริงหรือ GPU
# **ไม่ได้เป็นการยืนยันหรือรับประกันการฝึกโมเดลจริงครบทั้ง Pipeline**

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'training'))
from ner_common import (
    ENTITY_TYPES,
    MISSED_ENTITY,
    SPURIOUS_ENTITY,
    entity_confusion,
    entity_counts,
    normalize_matrix_label,
)


class DatasetSizeLogicTests(unittest.TestCase):
    # --------------------------------------------------------------------------
    # ฟังก์ชันทดสอบ: test_entity_confusion_exact_boundaries
    # หน้าที่: ทดสอบตรรกะการสร้าง Entity Confusion Matrix แบบ Exact Boundaries
    # ข้อมูลจำลองที่ใช้:
    #   - Ground Truth: มี 3 entity spans คือ DRUG (2 tokens), ROUTE (1 token), DOSAGE (1 token)
    #   - Prediction: โมเดลจำลองทำนาย STRENGTH (2 tokens), ROUTE (1 token ตำแหน่งใหม่), DOSAGE (1 token)
    # สิ่งที่ยืนยันได้:
    #   1. DRUG -> STRENGTH (Classification Error ข้ามประเภท ที่ span ตรงกัน): matrix['DRUG']['STRENGTH'] == 1
    #   2. DOSAGE -> DOSAGE (ทำนายประเภทและ span ถูกต้องสมบูรณ์): matrix['DOSAGE']['DOSAGE'] == 1
    #   3. ROUTE -> Missed Entity (Ground Truth มี แต่โมเดลทายไม่ตรงตำแหน่งเดิม): matrix['ROUTE'][MISSED_ENTITY] == 1
    #   4. Spurious Entity -> ROUTE (โมเดลทาย ROUTE ในตำแหน่งที่ไม่มีใน Ground Truth): matrix[SPURIOUS_ENTITY]['ROUTE'] == 1
    #   5. ผลรวมของแถว Entity แต่ละประเภท ต้องเท่ากับจำนวน entity จริงที่นับได้จาก entity_counts()
    #   6. แถวสุดท้ายคือ Spurious Entity และคอลัมน์สุดท้ายคือ Missed Entity
    #   7. ฟังก์ชัน normalize_matrix_label สามารถแปลงป้ายชื่อภาษาไทยเดิมเป็นชื่อมาตรฐานภาษาอังกฤษได้ถูกต้อง
    # ข้อจำกัด: ตรวจสอบเฉพาะตรรกะการจับคู่ Entity เท่านั้น ไม่ได้ทดสอบ inference จากโมเดลจริง
    # --------------------------------------------------------------------------
    def test_entity_confusion_exact_boundaries(self):
        truth = [['B-DRUG','I-DRUG','O','B-ROUTE','O','B-DOSAGE']]
        predicted = [['B-STRENGTH','I-STRENGTH','O','O','B-ROUTE','B-DOSAGE']]
        rows, columns, matrix = entity_confusion(truth, predicted)
        self.assertEqual(matrix['DRUG']['STRENGTH'], 1)
        self.assertEqual(matrix['DOSAGE']['DOSAGE'], 1)
        self.assertEqual(matrix['ROUTE'][MISSED_ENTITY], 1)
        self.assertEqual(matrix[SPURIOUS_ENTITY]['ROUTE'], 1)
        counts = entity_counts(truth)
        for entity in ENTITY_TYPES:
            self.assertEqual(sum(matrix[entity].values()), counts[entity])
        self.assertEqual(rows[-1], SPURIOUS_ENTITY)
        self.assertEqual(columns[-1], MISSED_ENTITY)
        self.assertEqual(normalize_matrix_label('ไม่มี Entity จริง'), SPURIOUS_ENTITY)
        self.assertEqual(normalize_matrix_label('ไม่พบ Entity'), MISSED_ENTITY)

    # --------------------------------------------------------------------------
    # ฟังก์ชันทดสอบ: test_configs_and_source_parse_without_runtime_import
    # หน้าที่: ทดสอบไวยากรณ์ (Syntax) ของสคริปต์หลัก และความถูกต้องของไฟล์คอนฟิกโดยไม่ต้องโหลดโมเดลจริง
    # สิ่งที่ยืนยันได้:
    #   1. scripts/training/dataset_size_experiment.py สามารถแปลงเป็น AST (ast.parse) ได้โดยไม่มี SyntaxError
    #   2. ไฟล์คอนฟิก JSON ใน configs/dataset_size/ มีครบทั้ง 3 โมเดลเป้าหมาย (biobert, clinicalbert, pubmedbert)
    #   3. ค่า Hyperparameters ในทุกคอนฟิกตรงตามเกณฑ์: learning_rate == 2e-5, maximum_epochs == 100
    #   4. รายการชุดข้อมูลในทุกคอนฟิกมีครบและเรียงลำดับถูกต้อง: ['dataset_50', 'dataset_75', 'dataset_100']
    # ข้อจำกัด: เป็นการตรวจสอบเชิงโครงสร้าง (Static Verification) เท่านั้น ไม่ได้ยืนยันความถูกต้องระหว่างการฝึกจริง
    # --------------------------------------------------------------------------
    def test_configs_and_source_parse_without_runtime_import(self):
        ast.parse((ROOT/'scripts/training/dataset_size_experiment.py').read_text(encoding='utf-8'))
        expected = {'biobert','clinicalbert','pubmedbert'}
        found = set()
        for path in (ROOT/'scripts/training/configs/dataset_size').glob('*.json'):
            config=json.loads(path.read_text(encoding='utf-8')); found.add(config['model_key'])
            self.assertEqual(config['training']['learning_rate'], 2e-5)
            self.assertEqual(config['training']['maximum_epochs'], 100)
            self.assertEqual(list(config['datasets']), ['dataset_50','dataset_75','dataset_100'])
        self.assertEqual(found, expected)


if __name__ == '__main__':
    unittest.main()
