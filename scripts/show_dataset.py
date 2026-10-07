import json

file_path = "data/glaucoma/glaucoma_selected.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)

        print(f"\n===== ตัวอย่างที่ {i + 1} =====")
        print("ข้อความ:")
        print(data["text"])

        print("\nAnnotation:")
        for start, end, label in data["labels"]:
            entity = data["text"][start:end]
            print(f"{entity:20} -> {label}")

        if i == 4:
            break