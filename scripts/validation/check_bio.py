import json

file_path = "data/glaucoma/glaucoma_bio.jsonl"

with open(file_path, "r", encoding="utf-8") as f:

    for i, line in enumerate(f):

        data = json.loads(line)

        print("\nตัวอย่างที่", i + 1)
        print("ข้อความ:", data["text"])

        for token, label in zip(
            data["tokens"],
            data["ner_tags"]
        ):
            print(f"{token:20} {label}")

        print("-" * 50)

        if i == 4:
            break