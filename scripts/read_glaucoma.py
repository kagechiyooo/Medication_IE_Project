import json
import re
from collections import Counter

file_path = "data/glaucoma/Annotations/Combined_annotations.jsonl"

label_counter = Counter()

total_lines = 0
valid_lines = 0
invalid_lines = 0
lines_with_labels = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):

        line = line.strip()

        if not line:
            continue

        total_lines += 1

        # แก้ id ที่ถูกปิดบังด้วย *****
        line = re.sub(
            r'"id"\s*:\s*\*+',
            '"id": null',
            line
        )

        try:
            data = json.loads(line)
            valid_lines += 1

        except json.JSONDecodeError as e:
            invalid_lines += 1

            print(f"อ่านไม่ได้ที่บรรทัด {line_number}")
            print(line[:150])
            print("Error:", e)
            print("-" * 50)

            continue

        labels = data.get("labels", [])

        if labels:
            lines_with_labels += 1

        for label in labels:
            if len(label) >= 3:
                label_name = label[2]
                label_counter[label_name] += 1


print("\n===== สรุป Dataset =====")
print("จำนวนบรรทัดทั้งหมด:", total_lines)
print("บรรทัดที่อ่านได้:", valid_lines)
print("บรรทัดที่อ่านไม่ได้:", invalid_lines)
print("ข้อความที่มี Label:", lines_with_labels)

print("\n===== จำนวน Label แต่ละประเภท =====")

for label, count in label_counter.most_common():
    print(f"{label:25} {count}")