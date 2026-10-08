import json
import random
from collections import Counter

INPUT_FILE = "data/mtsamples/mtsamples_annotated_001_400_bio.jsonl"

TRAIN_FILE = "data/mtsamples/mtsamples_train.jsonl"
VAL_FILE = "data/mtsamples/mtsamples_val.jsonl"
TEST_FILE = "data/mtsamples/mtsamples_test.jsonl"

ENTITY_TYPES = [
    "DRUG",
    "STRENGTH",
    "DOSAGE",
    "FREQUENCY",
    "ROUTE"
]

TRAIN_SIZE = 280
VAL_SIZE = 60
TEST_SIZE = 60

SEED = 42
N_TRIALS = 10000


def get_entity_counts(record):
    counts = Counter()

    for tag in record["ner_tags"]:
        if tag.startswith("B-"):
            counts[tag[2:]] += 1

    return counts


def count_split(records):
    counts = Counter()

    for record in records:
        counts.update(record["_counts"])

    return counts


# อ่านข้อมูล
records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        item["_counts"] = get_entity_counts(item)
        records.append(item)


# จำนวน entity ทั้งหมด
total_counts = count_split(records)

targets = {
    "train": {
        e: total_counts[e] * 0.70
        for e in ENTITY_TYPES
    },
    "val": {
        e: total_counts[e] * 0.15
        for e in ENTITY_TYPES
    },
    "test": {
        e: total_counts[e] * 0.15
        for e in ENTITY_TYPES
    }
}


def split_score(train, val, test):
    split_records = {
        "train": train,
        "val": val,
        "test": test
    }

    score = 0

    for split_name, subset in split_records.items():

        counts = count_split(subset)

        for entity in ENTITY_TYPES:

            target = targets[split_name][entity]

            if target > 0:
                error = abs(
                    counts[entity] - target
                ) / target

                # ลงโทษความต่างแบบกำลังสอง
                score += error ** 2

    return score


random.seed(SEED)

best_score = float("inf")
best_train = None
best_val = None
best_test = None

indices = list(range(len(records)))

for trial in range(N_TRIALS):

    random.shuffle(indices)

    train_idx = indices[:TRAIN_SIZE]
    val_idx = indices[
        TRAIN_SIZE:
        TRAIN_SIZE + VAL_SIZE
    ]
    test_idx = indices[
        TRAIN_SIZE + VAL_SIZE:
    ]

    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    test = [records[i] for i in test_idx]

    score = split_score(
        train,
        val,
        test
    )

    if score < best_score:
        best_score = score
        best_train = train
        best_val = val
        best_test = test


def save_jsonl(records, path):

    with open(path, "w", encoding="utf-8") as f:

        for item in records:

            output = {
                k: v
                for k, v in item.items()
                if k != "_counts"
            }

            f.write(
                json.dumps(
                    output,
                    ensure_ascii=False
                ) + "\n"
            )


save_jsonl(best_train, TRAIN_FILE)
save_jsonl(best_val, VAL_FILE)
save_jsonl(best_test, TEST_FILE)


print("===== TOTAL =====")

for entity in ENTITY_TYPES:
    print(
        f"B-{entity}:",
        total_counts[entity]
    )


print("\n===== SPLIT RESULT =====")

for name, subset in [
    ("TRAIN", best_train),
    ("VAL", best_val),
    ("TEST", best_test)
]:

    counts = count_split(subset)

    print(f"\n{name}")
    print("จำนวนข้อความ:", len(subset))

    for entity in ENTITY_TYPES:
        print(
            f"B-{entity}:",
            counts[entity]
        )


print("\nBest score:", best_score)

print("\nไฟล์ที่สร้าง:")
print(TRAIN_FILE)
print(VAL_FILE)
print(TEST_FILE)