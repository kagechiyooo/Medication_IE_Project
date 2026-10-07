import json
import random
from collections import Counter

input_file = "data/glaucoma/glaucoma_bio.jsonl"

train_file = "data/glaucoma/train.jsonl"
val_file = "data/glaucoma/validation.jsonl"
test_file = "data/glaucoma/test.jsonl"

random.seed(42)

data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

random.shuffle(data)

n = len(data)

train_end = int(n * 0.70)
val_end = train_end + int(n * 0.15)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]


def save_jsonl(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )


def count_labels(dataset):
    counter = Counter()

    for item in dataset:
        for label in item["ner_tags"]:
            if label != "O":
                counter[label] += 1

    return counter


save_jsonl(train_data, train_file)
save_jsonl(val_data, val_file)
save_jsonl(test_data, test_file)

print("===== จำนวนข้อความ =====")
print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Test:", len(test_data))

print("\n===== Train Labels =====")
for label, count in count_labels(train_data).most_common():
    print(f"{label:20} {count}")

print("\n===== Validation Labels =====")
for label, count in count_labels(val_data).most_common():
    print(f"{label:20} {count}")

print("\n===== Test Labels =====")
for label, count in count_labels(test_data).most_common():
    print(f"{label:20} {count}")