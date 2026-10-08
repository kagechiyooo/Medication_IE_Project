import json
import re

input_file = "data/glaucoma/Annotations/Combined_annotations.jsonl"
output_file = "data/glaucoma/glaucoma_selected.jsonl"

wanted_labels = {
    "DRUG",
    "STRENGTH",
    "DOSAGE",
    "FREQUENCY",
    "ROUTE"
}

kept_lines = 0
kept_entities = 0
invalid_lines = 0

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line_number, line in enumerate(fin, start=1):

        line = line.strip()

        if not line:
            continue

        # แก้ id ที่เป็น *****
        line = re.sub(
            r'"id"\s*:\s*\*+',
            '"id": null',
            line
        )

        try:
            data = json.loads(line)

        except json.JSONDecodeError:
            invalid_lines += 1
            continue

        new_labels = []

        for label in data.get("labels", []):
            if len(label) >= 3 and label[2] in wanted_labels:
                new_labels.append(label)

        # เก็บเฉพาะข้อความที่มี label ที่เราสนใจ
        if new_labels:

            new_data = {
                "text": data["text"],
                "labels": new_labels
            }

            fout.write(
                json.dumps(
                    new_data,
                    ensure_ascii=False
                ) + "\n"
            )

            kept_lines += 1
            kept_entities += len(new_labels)


print("สร้างไฟล์เรียบร้อย")
print("จำนวนข้อความที่เก็บ:", kept_lines)
print("จำนวน Entity ที่เก็บ:", kept_entities)
print("บรรทัดที่อ่านไม่ได้:", invalid_lines)
print("ไฟล์:", output_file)