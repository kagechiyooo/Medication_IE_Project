import pandas as pd
import json

input_file = "data/mtsamples/mtsamples_medication_clean.csv"
output_file = "data/mtsamples/mtsamples_doccano.jsonl"

df = pd.read_csv(input_file)

count = 0

with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():

        text = str(row["text"]).strip()

        if not text:
            continue

        item = {
            "text": text,
            "label": []
        }

        f.write(
            json.dumps(item, ensure_ascii=False) + "\n"
        )

        count += 1

print("สร้างไฟล์สำหรับ Doccano เรียบร้อย")
print("จำนวนข้อความ:", count)
print("ไฟล์:", output_file)