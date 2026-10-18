import json
import random
from collections import Counter
from pathlib import Path

INPUT_FILE = "data/mtsamples/bio/mtsamples_annotated_001_800_bio.jsonl"

OUTPUT_DIR = Path("data/mtsamples/experiment_splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILE = OUTPUT_DIR / "test_15.jsonl"
REMAINING_FILE = OUTPUT_DIR / "remaining_85.jsonl"

ENTITY_TYPES = [
    "DRUG",
    "STRENGTH",
    "DOSAGE",
    "FREQUENCY",
    "ROUTE",
]

SEED = 42
N_TRIALS = 50000

TEST_SIZE = 120

# เป้าหมายเพื่อให้พอรวมกับ Glaucoma แล้ว
# Final Test มี minority entity ประมาณ 100
TARGETS = {
    "DRUG": 123,
    "STRENGTH": 91,
    "DOSAGE": 96,
    "FREQUENCY": 94,
    "ROUTE": 27,
}


def get_entity_counts(record):
    counts = Counter()

    for tag in record["ner_tags"]:
        if tag.startswith("B-"):
            entity = tag[2:]

            if entity in ENTITY_TYPES:
                counts[entity] += 1

    return counts


def count_entities(records):
    counts = Counter()

    for record in records:
        counts.update(record["_counts"])

    return counts


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            output = {
                key: value
                for key, value in record.items()
                if key != "_counts"
            }

            f.write(
                json.dumps(
                    output,
                    ensure_ascii=False
                ) + "\n"
            )


# =========================
# อ่านข้อมูล
# =========================

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            item["_counts"] = get_entity_counts(item)
            records.append(item)


total_counts = count_entities(records)

print("===== MTSAMPLES TOTAL =====")
print("Texts:", len(records))

for entity in ENTITY_TYPES:
    print(
        f"{entity:10s}: "
        f"{total_counts[entity]}"
    )


# =========================
# Score
# =========================

def test_score(test_records):
    counts = count_entities(test_records)

    score = 0

    for entity in ENTITY_TYPES:
        target = TARGETS[entity]

        error = abs(
            counts[entity] - target
        ) / target

        score += error ** 2

    # ลงโทษแรง ถ้า minority entity ต่ำกว่าเป้าหมาย
    if counts["DOSAGE"] < TARGETS["DOSAGE"]:
        score += (
            TARGETS["DOSAGE"]
            - counts["DOSAGE"]
        ) * 0.05

    if counts["STRENGTH"] < TARGETS["STRENGTH"]:
        score += (
            TARGETS["STRENGTH"]
            - counts["STRENGTH"]
        ) * 0.03

    return score


# =========================
# สุ่มหลายรอบ
# =========================

rng = random.Random(SEED)

indices = list(range(len(records)))

best_score = float("inf")
best_test_indices = None

for trial in range(N_TRIALS):

    rng.shuffle(indices)

    candidate_indices = indices[:TEST_SIZE]

    candidate_test = [
        records[i]
        for i in candidate_indices
    ]

    score = test_score(candidate_test)

    if score < best_score:
        best_score = score
        best_test_indices = candidate_indices.copy()


# =========================
# แบ่ง Test / Remaining
# =========================

test_index_set = set(best_test_indices)

test_records = [
    records[i]
    for i in range(len(records))
    if i in test_index_set
]

remaining_records = [
    records[i]
    for i in range(len(records))
    if i not in test_index_set
]


save_jsonl(test_records, TEST_FILE)
save_jsonl(remaining_records, REMAINING_FILE)


# =========================
# Report
# =========================

test_counts = count_entities(test_records)
remaining_counts = count_entities(remaining_records)

print("\n===== TEST 15% =====")
print("Texts:", len(test_records))

for entity in ENTITY_TYPES:
    print(
        f"{entity:10s}: "
        f"{test_counts[entity]} "
        f"(target ≈ {TARGETS[entity]})"
    )


print("\n===== REMAINING 85% =====")
print("Texts:", len(remaining_records))

for entity in ENTITY_TYPES:
    print(
        f"{entity:10s}: "
        f"{remaining_counts[entity]}"
    )


print("\nBest score:", best_score)

print("\nSaved:")
print(TEST_FILE)
print(REMAINING_FILE)