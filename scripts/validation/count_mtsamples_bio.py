import json
from collections import Counter

input_file = "data/glaucoma/glaucoma_bio.jsonl"

counter = Counter()

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        for tag in data["ner_tags"]:
            if tag.startswith("B-"):
                counter[tag] += 1

print("===== GLAUCOMA ENTITY COUNT =====")

for label in [
    "B-DRUG",
    "B-STRENGTH",
    "B-DOSAGE",
    "B-FREQUENCY",
    "B-ROUTE"
]:
    print(f"{label}: {counter[label]}")

print("Total entities:", sum(counter.values()))