import json
from collections import Counter

file_path = "data/glaucoma/glaucoma_bio.jsonl"

counter = Counter()
total_tokens = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        for label in data["ner_tags"]:
            counter[label] += 1
            total_tokens += 1

print("===== จำนวน BIO Labels =====")

for label, count in counter.most_common():
    print(f"{label:20} {count}")

print()
print("จำนวน token ทั้งหมด:", total_tokens)