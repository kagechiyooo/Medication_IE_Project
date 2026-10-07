# Medication_IE_Project
# Medication_IE_Project

โปรเจกต์นี้ใช้สำหรับงาน Medication Information Extraction จาก Clinical Notes

งานหลักตอนนี้คือการกำกับข้อมูลจาก MTSamples ด้วย Doccano เพื่อสร้าง Ground Truth สำหรับงาน Named Entity Recognition (NER)

Labels ที่ใช้มี 5 ประเภท

- DRUG
- STRENGTH
- DOSAGE
- FREQUENCY
- ROUTE

---

## Project Structure

```text
Medication_IE_Project/
├── data/
│   ├── glaucoma/
│   └── mtsamples/
│       ├── annotation_parts/
│       └── annotated_parts/
│
├── scripts/
├── models/
├── results/
├── README.md
└── requirements-doccano.txt
```

### ความหมายของโฟลเดอร์

`annotation_parts/`

คือไฟล์ที่แบ่งไว้ให้แต่ละคนเอาไปกำกับข้อมูล

ตัวอย่าง

```text
mtsamples_001_200.jsonl
mtsamples_201_400.jsonl
```

`annotated_parts/`

คือไฟล์ที่แต่ละคน Export ออกจาก Doccano หลังจากกำกับข้อมูลเสร็จแล้ว

ตัวอย่าง

```text
annotated_001_200.jsonl
annotated_201_400.jsonl
```

---

# การใช้งาน GitHub

## 1. Clone โปรเจกต์ครั้งแรก

สำหรับคนที่ยังไม่มีโปรเจกต์ในเครื่อง

```bash
git clone https://github.com/kagechiyooo/Medication_IE_Project.git
```

จากนั้นเข้าโฟลเดอร์

```bash
cd Medication_IE_Project
```

ตรวจสอบว่าอยู่ใน Git Repository แล้ว

```bash
git status
```

---

## 2. ก่อนเริ่มทำงานทุกครั้ง

ให้อัปเดต `main` ก่อน

```bash
git checkout main
git pull origin main
```

เพื่อให้ไฟล์ในเครื่องเป็นเวอร์ชันล่าสุด

---

## 3. สร้าง Branch ของตัวเอง

ไม่ควรทำงานตรงบน `main`

ตัวอย่าง คนที่ได้รับช่วง 1-200

```bash
git checkout -b annotation-001-200
```

คนที่ได้รับช่วง 201-400

```bash
git checkout -b annotation-201-400
```

ตรวจสอบว่าอยู่ branch ไหน

```bash
git branch --show-current
```

ตัวอย่าง

```text
annotation-201-400
```

---

# การ Annotation

## 4. ไฟล์ที่ต้องใช้

ไฟล์สำหรับนำไปกำกับข้อมูลอยู่ที่

```text
data/mtsamples/annotation_parts/
```

ตัวอย่าง

```text
data/mtsamples/annotation_parts/mtsamples_001_200.jsonl
```

หรือ

```text
data/mtsamples/annotation_parts/mtsamples_201_400.jsonl
```

แต่ละคนทำเฉพาะช่วงที่ตัวเองได้รับมอบหมาย

---

## 5. สร้าง Project ใน Doccano

Project Type ต้องเลือก

```text
Sequence Labeling
```

ไม่ใช่ Text Classification

จากนั้นสร้าง Labels ให้ครบ 5 ตัว

```text
DRUG
STRENGTH
DOSAGE
FREQUENCY
ROUTE
```

ทุกคนต้องใช้ชื่อ Label เหมือนกัน

---

## 6. Import Dataset เข้า Doccano

เข้าโปรเจกต์ใน Doccano

ไปที่

```text
Dataset
→ Actions
→ Import Dataset
```

ตั้งค่า

```text
File format: JSONL
Column Data: text
Column Label: label
Encoding: utf_8
```

จากนั้นเลือกไฟล์ที่ตัวเองได้รับ

ตัวอย่าง

```text
mtsamples_201_400.jsonl
```

แล้วกด Import

---

# Annotation Guideline

ทุกคนต้องใช้กฎเดียวกัน

## DRUG

ชื่อยา

ตัวอย่าง

```text
Keflex
Lidocaine
Omeprazole
Heparin
```

ตัวอย่างประโยค

```text
Keflex 500 mg
```

กำกับ

```text
Keflex → DRUG
```

---

## STRENGTH

ความแรงหรือความเข้มข้นของยา

ตัวอย่าง

```text
500 mg
20 mg
1%
0.5%
40 mg/cc
1:100,000
```

ตัวอย่าง

```text
Keflex 500 mg
```

กำกับ

```text
Keflex → DRUG
500 mg → STRENGTH
```

---

## DOSAGE

ปริมาณยาที่ให้หรือรับประทานต่อครั้ง

ตัวอย่าง

```text
one tablet
three tablets
20 mL
10 cc
500 units
```

ตัวอย่าง

```text
Keflex one tablet q.i.d.
```

กำกับ

```text
Keflex → DRUG
one tablet → DOSAGE
q.i.d. → FREQUENCY
```

---

## FREQUENCY

ความถี่ในการใช้ยา

ตัวอย่าง

```text
daily
twice daily
three times a day
BID
b.i.d.
q.i.d.
```

ตัวอย่าง

```text
Omeprazole 20 mg daily
```

กำกับ

```text
Omeprazole → DRUG
20 mg → STRENGTH
daily → FREQUENCY
```

---

## ROUTE

ช่องทางหรือวิธีการให้ยา

ตัวอย่าง

```text
IV
IM
p.o.
oral
orally
both eyes
```

ตัวอย่าง

```text
heparin IV
```

กำกับ

```text
heparin → DRUG
IV → ROUTE
```

---

# สิ่งที่ไม่ต้องกำกับ

ตอนนี้ใช้เพียง 5 Labels เท่านั้น

ดังนั้นไม่ต้องกำกับ

```text
FORM
DURATION
ACTION
START
STOP
```

ตัวอย่าง

```text
Keflex 500 mg tablets
```

กำกับ

```text
Keflex → DRUG
500 mg → STRENGTH
```

ไม่ต้องกำกับคำว่า

```text
tablets
```

---

# กฎสำคัญ

กำกับเฉพาะข้อมูลที่มีอยู่จริงในข้อความ

ห้ามเดาข้อมูลเพิ่มเอง

ตัวอย่าง

```text
Keflex 500 mg
```

ข้อความนี้ไม่มีความถี่

ดังนั้นไม่ต้องกำกับ FREQUENCY

---

# STRENGTH กับ DOSAGE ต่างกันอย่างไร

ตัวอย่าง

```text
Keflex 500 mg tablet
```

`500 mg` คือความแรงของยา

```text
500 mg → STRENGTH
```

แต่ถ้าประโยคบอกชัดว่าเป็นปริมาณที่ให้จริง เช่น

```text
500 units heparin IV
```

กำกับ

```text
500 units → DOSAGE
heparin → DRUG
IV → ROUTE
```

ถ้าไม่แน่ใจ ห้ามเดาเอง ให้ถามในกลุ่มก่อน เพื่อให้ทุกคนใช้กฎเดียวกัน

---

# ตัวอย่าง Annotation

## Example 1

```text
Veramist spray 27.5 mcg daily
```

กำกับ

```text
Veramist → DRUG
27.5 mcg → STRENGTH
daily → FREQUENCY
```

---

## Example 2

```text
Keflex one tablet q.i.d.
```

กำกับ

```text
Keflex → DRUG
one tablet → DOSAGE
q.i.d. → FREQUENCY
```

---

## Example 3

```text
30 mL of 0.5% Marcaine
```

กำกับ

```text
30 mL → DOSAGE
0.5% → STRENGTH
Marcaine → DRUG
```

---

## Example 4

```text
Omeprazole 20 mg daily
```

กำกับ

```text
Omeprazole → DRUG
20 mg → STRENGTH
daily → FREQUENCY
```

---

## Example 5

```text
500 units heparin IV
```

กำกับ

```text
500 units → DOSAGE
heparin → DRUG
IV → ROUTE
```

---

# ถ้าข้อความไม่มียา

ไม่จำเป็นต้อง Label อะไร

ตัวอย่าง

```text
nasolacrimal massage 2-3 times daily
```

แม้จะมีคำว่า

```text
2-3 times daily
```

แต่เป็นความถี่ของ procedure ไม่ใช่ยา

ดังนั้นไม่ต้องกำกับ

---

# หลัง Annotation เสร็จ

## 7. Export จาก Doccano

ไปที่

```text
Dataset
→ Actions
→ Export Dataset
```

เลือก

```text
JSONL
```

ถ้ายังไม่ได้ใช้ระบบ Approve

ไม่ต้องติ๊ก

```text
Export only approved documents
```

จากนั้นกด Export

---

## 8. ตั้งชื่อไฟล์ Export

ตั้งชื่อไฟล์ตามช่วงที่ตัวเองได้รับ

ตัวอย่าง

คนทำ 1-200

```text
annotated_001_200.jsonl
```

คนทำ 201-400

```text
annotated_201_400.jsonl
```

---

## 9. นำไฟล์ Export มาใส่ในโปรเจกต์

เอาไฟล์ไว้ที่

```text
data/mtsamples/annotated_parts/
```

ตัวอย่าง

```text
data/mtsamples/annotated_parts/annotated_001_200.jsonl
data/mtsamples/annotated_parts/annotated_201_400.jsonl
```

---

# Push งานขึ้น GitHub

## 10. ตรวจสอบไฟล์ก่อน

```bash
git status
```

ดูว่าไฟล์ที่เปลี่ยนถูกต้องหรือไม่

---

## 11. Add เฉพาะไฟล์ของตัวเอง

ตัวอย่าง

```bash
git add data/mtsamples/annotated_parts/annotated_201_400.jsonl
```

ไม่แนะนำให้ใช้

```bash
git add .
```

ถ้าไม่จำเป็น เพราะอาจเพิ่มไฟล์อื่นที่ไม่เกี่ยวข้องขึ้นไปด้วย

---

## 12. Commit

ตัวอย่าง

```bash
git commit -m "Add annotation 201-400"
```

---

## 13. Push Branch

ครั้งแรก

```bash
git push -u origin annotation-201-400
```

ถ้ายังทำ branch เดิมต่อ ครั้งต่อไปใช้

```bash
git push
```

---

# Pull Request

หลังจาก Push แล้ว เข้า GitHub

จะเห็นปุ่ม

```text
Compare & pull request
```

กดเข้าไป

ตรวจสอบว่า

```text
base: main
compare: annotation-201-400
```

จากนั้นกด

```text
Create pull request
```

รอเจ้าของ Repository ตรวจสอบและ Merge

---

# หลัง Merge

กลับมาที่เครื่องแล้วรัน

```bash
git checkout main
git pull origin main
```

ไฟล์ของเพื่อนที่ Merge แล้วจะลงมาในเครื่อง

---

# การรวมไฟล์ Annotation

เมื่อทุกคนทำเสร็จแล้ว จะได้ประมาณ

```text
annotated_001_200.jsonl
annotated_201_400.jsonl
```

รวมไฟล์บน Mac ได้ด้วย

```bash
cat data/mtsamples/annotated_parts/*.jsonl > data/mtsamples/mtsamples_annotated_all.jsonl
```

ตรวจจำนวนบรรทัด

```bash
wc -l data/mtsamples/mtsamples_annotated_all.jsonl
```

ถ้ากำกับทั้งหมด 400 ข้อความ ควรได้ประมาณ

```text
400
```

---

# การใช้ Dataset หลัง Annotation

ข้อมูลที่มี Ground Truth จะถูกนำไปแบ่งเป็น

```text
Train
Validation
Test
```

ทุกส่วนต้องมี Annotation เพราะต้องมีคำตอบจริงไว้เทียบกับผลที่โมเดลทำนาย

ตัวอย่าง ถ้ามี 400 ข้อความ

```text
Train       280
Validation   60
Test         60
```

ประมาณสัดส่วน

```text
70 / 15 / 15
```

---

# ข้อมูลที่ไม่ได้ Annotate

ข้อมูล MTSamples ที่เหลือและยังไม่มี Ground Truth

ไม่ควรเรียกว่า Test Set

ให้เรียกว่า

```text
Unlabeled Data
```

หรือ

```text
External Unlabeled Data
```

ข้อมูลส่วนนี้สามารถนำไปให้โมเดลลอง Prediction ได้

แต่ไม่สามารถใช้คำนวณ

```text
Precision
Recall
F1-score
```

ได้ เพราะไม่มี Ground Truth

---

# Workflow เวลาเริ่มทำงานวันใหม่

ก่อนเริ่มทำงาน

```bash
git checkout main
git pull origin main
```

ถ้าเป็นงานใหม่ให้สร้าง branch ใหม่

```bash
git checkout -b ชื่อ-branch
```

ตัวอย่าง

```bash
git checkout -b annotation-001-200
```

---

# ถ้าจะทำ Branch เดิมต่อ

ไม่ต้องสร้าง branch ใหม่

เช็ก branch ที่มีอยู่

```bash
git branch
```

จากนั้นเข้า branch เดิม

```bash
git checkout annotation-201-400
```

แล้วทำงานต่อได้เลย

---

# ข้อควรระวัง

ห้าม

```text
- แก้ไฟล์ Annotation ของคนอื่น
- ทุกคนแก้ไฟล์เดียวกัน
- Push งานตรง main โดยไม่จำเป็น
- เปลี่ยนชื่อ Labels เอง
- เดา Annotation ที่ไม่มีอยู่ในข้อความ
- Commit Python environment ขึ้น GitHub
```

---

# Python Environment

ไม่ควร Push โฟลเดอร์เหล่านี้ขึ้น GitHub

```text
doccano_env/
.venv/
.venv-doccano/
__pycache__/
.DS_Store
```

ควรสร้างไฟล์ `.gitignore`

แล้วใส่

```gitignore
doccano_env/
.venv/
.venv-doccano/
__pycache__/
*.pyc
.DS_Store
```

---

# Workflow สรุป

```text
git checkout main
        ↓
git pull origin main
        ↓
สร้าง Branch
        ↓
Import JSONL เข้า Doccano
        ↓
Annotate
        ↓
Export JSONL
        ↓
ใส่ไฟล์ใน annotated_parts
        ↓
git add
        ↓
git commit
        ↓
git push
        ↓
Pull Request
        ↓
Merge เข้า main
        ↓
รวมไฟล์ Annotation
        ↓
ตรวจ Annotation
        ↓
แปลงเป็น BIO
        ↓
แบ่ง Train / Validation / Test
        ↓
Train Model
```

---

# Labels ที่ใช้

ใช้เฉพาะ

```text
DRUG
STRENGTH
DOSAGE
FREQUENCY
ROUTE
```

ถ้ามีกรณีที่ไม่แน่ใจเรื่องการกำกับข้อมูล ให้ถามในกลุ่มก่อน เพื่อให้ทุกคนใช้ Annotation Guideline เดียวกัน
