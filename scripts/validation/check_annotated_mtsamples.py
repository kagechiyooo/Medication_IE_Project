import json

input_file = "data/mtsamples/mtsamples_annotated_001_400.jsonl"
output_file = "data/mtsamples/mtsamples_annotated_001_400_checked.jsonl"

VALID_LABELS = {
    "DRUG",
    "STRENGTH",
    "DOSAGE",
    "FREQUENCY",
    "ROUTE"
}

records = []
texts = set()

duplicate_count = 0
empty_label_count = 0
invalid_labels = []

with open(input_file, "r", encoding="utf-8") as f:
    for new_id, line in enumerate(f, start=1):
        item = json.loads(line)

        # เปลี่ยน ID ใหม่ให้เป็น 1-400
        item["id"] = new_id

        text = item.get("text", "")
        labels = item.get("label", [])

        # เช็กข้อความซ้ำ
        if text in texts:
            duplicate_count += 1
        else:
            texts.add(text)

        # เช็กข้อความที่ไม่มี annotation
        if not labels:
            empty_label_count += 1

        # เช็กชื่อ label
        for label in labels:
            label_name = label[2]

            if label_name not in VALID_LABELS:
                invalid_labels.append(
                    (new_id, label_name)
                )

        records.append(item)


with open(output_file, "w", encoding="utf-8") as f:
    for item in records:
        f.write(
            json.dumps(item, ensure_ascii=False)
            + "\n"
        )


print("===== CHECK RESULT =====")
print("Total:", len(records))
print("Duplicate text:", duplicate_count)
print("No annotation:", empty_label_count)
print("Invalid labels:", invalid_labels)
print("Saved:", output_file)