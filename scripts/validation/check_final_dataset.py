import json
from collections import Counter

FILES = {
    "train": "data/train_final.jsonl",
    "validation": "data/validation_final.jsonl",
    "test": "data/test_final.jsonl"
}

ENTITY_TYPES = [
    "DRUG",
    "STRENGTH",
    "DOSAGE",
    "FREQUENCY",
    "ROUTE"
]

for split_name, path in FILES.items():

    total = 0
    mismatch = 0
    counter = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:

            data = json.loads(line)

            tokens = data["tokens"]
            tags = data["ner_tags"]

            total += 1

            if len(tokens) != len(tags):
                mismatch += 1
                print(
                    f"[ERROR] {split_name} "
                    f"tokens={len(tokens)} "
                    f"tags={len(tags)}"
                )

            for tag in tags:
                if tag.startswith("B-"):
                    counter[tag] += 1

    print(f"\n===== {split_name.upper()} =====")
    print("จำนวนข้อความ:", total)
    print("tokens != ner_tags:", mismatch)

    for entity in ENTITY_TYPES:
        print(
            f"B-{entity}:",
            counter[f"B-{entity}"]
        )