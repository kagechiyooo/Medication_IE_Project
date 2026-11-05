# Medication_IE_Project

Medication Information Extraction จาก Clinical Notes ด้วย BIO labels: DRUG, STRENGTH, DOSAGE, FREQUENCY และ ROUTE

งานปัจจุบันเปรียบเทียบ Dataset 50%, 75%, 100% ด้วย BioBERT, ClinicalBERT และ PubMedBERT

- [โครงสร้างโปรเจกต์](docs/PROJECT_STRUCTURE.md)
- [ขั้นตอนเตรียมและตรวจ Dataset](docs/DATASET_WORKFLOW.md)
- [การทดลองและผลลัพธ์](docs/EXPERIMENTS_AND_RESULTS.md)
- [คู่มือเดิมและ Annotation Guideline](archive/legacy_readme.md) — พาธและคำสั่งในคู่มือเดิมเป็นข้อมูลย้อนหลัง ให้ใช้เอกสารปัจจุบันสำหรับพาธใหม่

รันคำสั่งทั้งหมดจาก project root และติดตั้ง dependency สำหรับ Training ด้วย `python -m pip install -r requirements-training.txt`

ตรวจ Dataset ปัจจุบัน:

```bash
python scripts/validation/validate_dataset_size_splits.py
python -m unittest discover -s tests
```
