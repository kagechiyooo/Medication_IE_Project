import pandas as pd
import re

# =========================
# 1. กำหนดไฟล์
# =========================

input_file = "data/mtsamples/mtsamples.csv"
output_file = "data/mtsamples/mtsamples_medication_clean.csv"


# =========================
# 2. อ่าน Dataset
# =========================

df = pd.read_csv(input_file)

# เอาเฉพาะแถวที่มี transcription
df = df.dropna(subset=["transcription"])

print("จำนวน transcription ทั้งหมด:", len(df))


# =========================
# 3. Pattern สำหรับหาข้อมูลยา
# =========================

# รูปแบบที่มักเจอในข้อมูลยา
medication_pattern = re.compile(
    r"("
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?)\b"
    r"|\btablet(?:s)?\b"
    r"|\bcapsule(?:s)?\b"
    r"|\bBID\b"
    r"|\bTID\b"
    r"|\bQID\b"
    r"|\bQHS\b"
    r"|\bPRN\b"
    r"|\bonce daily\b"
    r"|\btwice daily\b"
    r"|\bthree times daily\b"
    r"|\bevery \d+ hours\b"
    r"|\bp\.?o\.?\b"
    r"|\boral(?:ly)?\b"
    r")",
    re.IGNORECASE
)


# คำที่ช่วยบอกว่าเป็นบริบทการใช้ยา
med_context_pattern = re.compile(
    r"\b("
    r"take|takes|taking|"
    r"medication|medications|"
    r"prescribed|prescription|"
    r"start|started|"
    r"continue|continues|"
    r"use|uses|using|"
    r"administered|"
    r"tablet|tablets|"
    r"spray|inhaler|ointment|cream|"
    r"daily|bid|tid|qid|qhs|prn"
    r")\b",
    re.IGNORECASE
)

# คำที่มักไม่ใช่การใช้ยา
exclude_pattern = re.compile(
    r"\b("
    r"blood loss|"
    r"estimated blood loss|"
    r"balloon|"
    r"aspirated|"
    r"urine|"
    r"sterile water|"
    r"fat|"
    r"foley|"
    r"irrigation|"
    r"drain|"
    r"bladder tolerated|"
    r"specimen|"
    r"implant|"
    r"thyroid gland|"
    r"prostate including capsule|"
    r"seminal vesicles"
    r")\b",
    re.IGNORECASE
)


# =========================
# 4. ตัด Clinical note
#    ออกเป็นข้อความสั้น ๆ
# =========================

results = []

for _, row in df.iterrows():

    text = str(row["transcription"])

    # แบ่งข้อความตาม . หรือ ,
    segments = re.split(r"(?<=[.,])\s+", text)

    for segment in segments:

        segment = segment.strip()

        # ไม่เอาข้อความสั้นหรือยาวเกินไป
        if len(segment) < 10 or len(segment) > 500:
            continue

        # ต้องมี pattern ที่เกี่ยวกับยา
        has_med_pattern = bool(
            medication_pattern.search(segment)
        )

        # ต้องมีบริบทเกี่ยวกับยา
        has_context = bool(
            med_context_pattern.search(segment)
        )

        # ตัดข้อความที่ชัดเจนว่าไม่ใช่ยา
        excluded = bool(
            exclude_pattern.search(segment)
        )

        if excluded:
            continue

        if has_med_pattern and has_context:

            results.append({
                "medical_specialty": row.get(
                    "medical_specialty",
                    ""
                ),
                "text": segment
            })


# =========================
# 5. สร้าง DataFrame ใหม่
# =========================

result_df = pd.DataFrame(results)

# ลบข้อความซ้ำ
result_df = result_df.drop_duplicates(
    subset=["text"]
)

# รีเซ็ต index
result_df = result_df.reset_index(drop=True)


# =========================
# 6. บันทึกไฟล์
# =========================

result_df.to_csv(
    output_file,
    index=False
)


# =========================
# 7. แสดงผล
# =========================

print("จำนวนข้อความที่เกี่ยวกับยา:", len(result_df))
print("บันทึกไว้ที่:", output_file)

print("\n===== ตัวอย่าง 30 รายการ =====")
print("-" * 70)

for i, text in enumerate(
    result_df["text"].head(30),
    start=1
):
    print(f"{i}. {text}")