import json
from collections import Counter

input_file = "data/mtsamples/bio/mtsamples_annotated_401_800_bio.jsonl"

counter = Counter()

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        for tag in data["ner_tags"]:
            if tag.startswith("B-"):
                counter[tag] += 1

print("===== MTSAMPLES 401-800 ENTITY COUNT =====")

for label in [
    "B-DRUG",
    "B-STRENGTH",
    "B-DOSAGE",
    "B-FREQUENCY",
    "B-ROUTE"
]:
    print(f"{label}: {counter[label]}")

print("Total entities:", sum(counter.values()))