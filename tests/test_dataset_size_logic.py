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
