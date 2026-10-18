import json
import hashlib
from pathlib import Path
from collections import Counter


ENTITY_TYPES = [
    "DRUG",
    "STRENGTH",
    "DOSAGE",
    "FREQUENCY",
    "ROUTE",
]

DATASETS = {
    "final1": {
        "train": "data/final1/train.jsonl",
        "validation": "data/final1/validation.jsonl",
        "test": "data/final1/test.jsonl",
    },
    "final2": {
        "train": "data/final2/train.jsonl",
        "validation": "data/final2/validation.jsonl",
        "test": "data/final2/test.jsonl",
    },
    "final3": {
        "train": "data/final3/train.jsonl",
        "validation": "data/final3/validation.jsonl",
        "test": "data/final3/test.jsonl",
    },
}


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(
                        f"❌ JSON error: {path} "
                        f"line {line_number}: {e}"
                    )

    return records


def record_key(record):
    """
    ใช้ tokens + ner_tags เป็นตัวระบุ record
    เพื่อเช็กว่า record เดียวกันไปอยู่ข้าม split หรือไม่
    """
    data = {
        "tokens": record.get("tokens", []),
        "ner_tags": record.get("ner_tags", []),
    }

    text = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def count_entities(records):
    counts = Counter()

    for record in records:
        for tag in record.get("ner_tags", []):
            if tag.startswith("B-"):
                entity = tag[2:]

                if entity in ENTITY_TYPES:
                    counts[entity] += 1

    return counts


def check_alignment(records, path):
    errors = 0

    for i, record in enumerate(records, start=1):

        tokens = record.get("tokens", [])
        tags = record.get("ner_tags", [])

        if len(tokens) != len(tags):
            errors += 1

            print(
                f"❌ Alignment error: "
                f"{path} row {i}"
            )

    return errors


def get_keys(records):
    return {
        record_key(record)
        for record in records
    }


all_data = {}


# ==========================================
# 1. LOAD + BASIC CHECK
# ==========================================

for dataset_name, paths in DATASETS.items():

    print("\n" + "=" * 50)
    print(dataset_name.upper())
    print("=" * 50)

    all_data[dataset_name] = {}

    total_texts = 0

    for split_name, path in paths.items():

        records = load_jsonl(path)

        all_data[dataset_name][split_name] = records

        total_texts += len(records)

        counts = count_entities(records)

        alignment_errors = check_alignment(
            records,
            path
        )

        print(f"\n{split_name.upper()}")
        print("Texts:", len(records))

        for entity in ENTITY_TYPES:
            print(
                f"{entity:10s}: "
                f"{counts[entity]}"
            )

        print(
            "Alignment errors:",
            alignment_errors
        )

    print(
        "\nTOTAL TEXTS:",
        total_texts
    )

    if total_texts == 2668:
        print("✅ Total = 2668")
    else:
        print(
            "❌ Total should be 2668"
        )


# ==========================================
# 2. CHECK OVERLAP
# ==========================================

print("\n" + "=" * 50)
print("CHECK TRAIN / VALIDATION / TEST OVERLAP")
print("=" * 50)

for dataset_name in DATASETS:

    train_keys = get_keys(
        all_data[dataset_name]["train"]
    )

    val_keys = get_keys(
        all_data[dataset_name]["validation"]
    )

    test_keys = get_keys(
        all_data[dataset_name]["test"]
    )

    train_val_overlap = (
        train_keys & val_keys
    )

    train_test_overlap = (
        train_keys & test_keys
    )

    val_test_overlap = (
        val_keys & test_keys
    )

    print(f"\n{dataset_name}")

    print(
        "Train ∩ Validation:",
        len(train_val_overlap)
    )

    print(
        "Train ∩ Test:",
        len(train_test_overlap)
    )

    print(
        "Validation ∩ Test:",
        len(val_test_overlap)
    )

    if (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    ):
        print(
            "✅ No split overlap"
        )
    else:
        print(
            "❌ Found split overlap"
        )


# ==========================================
# 3. CHECK TEST SAME ACROSS ALL DATASETS
# ==========================================

print("\n" + "=" * 50)
print("CHECK SAME TEST SET")
print("=" * 50)

test1 = all_data["final1"]["test"]
test2 = all_data["final2"]["test"]
test3 = all_data["final3"]["test"]

test1_keys = get_keys(test1)
test2_keys = get_keys(test2)
test3_keys = get_keys(test3)


if (
    test1_keys == test2_keys
    and test1_keys == test3_keys
):
    print(
        "✅ final1 / final2 / final3 "
        "use the SAME Test set"
    )
else:
    print(
        "❌ Test sets are different"
    )


print(
    "final1 Test:",
    len(test1_keys)
)

print(
    "final2 Test:",
    len(test2_keys)
)

print(
    "final3 Test:",
    len(test3_keys)
)


# ==========================================
# 4. FINAL RESULT
# ==========================================

print("\n" + "=" * 50)
print("VALIDATION COMPLETE")
print("=" * 50)