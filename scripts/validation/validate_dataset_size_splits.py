"""Validate only the nested dataset-size experiment outputs."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ENTITY_TYPES = ("DRUG", "STRENGTH", "DOSAGE", "FREQUENCY", "ROUTE")
SIZES = (50, 75, 100)
EXPECTED = {
    50: {"train": 794, "validation": 340, "glaucoma": 794, "mtsamples": 340},
    75: {"train": 1191, "validation": 510, "glaucoma": 1191, "mtsamples": 510},
    100: {"train": 1588, "validation": 680, "glaucoma": 1588, "mtsamples": 680},
}
ROOT = Path("data/dataset_size")
GLAUCOMA_REMAINING = Path("data/glaucoma/experiment_splits/remaining_85.jsonl")
MTSAMPLES_REMAINING = Path("data/mtsamples/experiment_splits/remaining_85.jsonl")
GLAUCOMA_TEST = Path("data/glaucoma/experiment_splits/test_15.jsonl")
MTSAMPLES_TEST = Path("data/mtsamples/experiment_splits/test_15.jsonl")


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def record_key(record):
    payload = {
        "tokens": record.get("tokens", []),
        "ner_tags": record.get("ner_tags", []),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def key_counter(records):
    return Counter(record_key(record) for record in records)


def entity_counts(records):
    return Counter(
        tag[2:]
        for record in records
        for tag in record.get("ner_tags", [])
        if tag.startswith("B-") and tag[2:] in ENTITY_TYPES
    )


def alignment_errors(records):
    return [
        index
        for index, record in enumerate(records, 1)
        if len(record.get("tokens", [])) != len(record.get("ner_tags", []))
    ]


def multiset_subset(left, right):
    return all(count <= right[key] for key, count in left.items())


def report_entities(records):
    counts = entity_counts(records)
    total = sum(counts.values())
    for entity in ENTITY_TYPES:
        percentage = 100.0 * counts[entity] / total if total else 0.0
        print(f"    {entity:10s} {counts[entity]:5d}  {percentage:7.3f}%")


def identify_source_counts(pool, glaucoma_keys, mtsamples_keys):
    counts = Counter()
    unknown = []
    for record in pool:
        key = record_key(record)
        if key in glaucoma_keys and key not in mtsamples_keys:
            counts["glaucoma"] += 1
        elif key in mtsamples_keys and key not in glaucoma_keys:
            counts["mtsamples"] += 1
        else:
            unknown.append(key)
    return counts, unknown


def main():
    failures = []
    glaucoma_remaining = load_jsonl(GLAUCOMA_REMAINING)
    mtsamples_remaining = load_jsonl(MTSAMPLES_REMAINING)
    locked_test = load_jsonl(GLAUCOMA_TEST) + load_jsonl(MTSAMPLES_TEST)
    shared_test = load_jsonl(ROOT / "test_15.jsonl")
    glaucoma_keys = set(key_counter(glaucoma_remaining))
    mtsamples_keys = set(key_counter(mtsamples_remaining))

    print("===== SHARED TEST =====")
    print("Records:", len(shared_test))
    test_matches = key_counter(shared_test) == key_counter(locked_test)
    print("Matches locked Test (including multiplicity):", "PASS" if test_matches else "FAIL")
    if len(shared_test) != 400:
        failures.append(f"Shared Test has {len(shared_test)} records, expected 400")
    if not test_matches:
        failures.append("Shared Test differs from the two locked Test files")

    data = {}
    test_counter = key_counter(shared_test)
    glaucoma_full_counter = key_counter(glaucoma_remaining)

    for size in SIZES:
        train = load_jsonl(ROOT / f"dataset_{size}" / "train.jsonl")
        validation = load_jsonl(ROOT / f"dataset_{size}" / "validation.jsonl")
        pool = train + validation
        data[size] = {"train": train, "validation": validation, "pool": pool}

        print(f"\n===== DATASET {size}% =====")
        print(f"Pool: {len(pool)} | Train: {len(train)} | Validation: {len(validation)}")
        for split_name, records in (("Train", train), ("Validation", validation)):
            errors = alignment_errors(records)
            print(f"  {split_name}: alignment errors = {len(errors)}")
            report_entities(records)
            if errors:
                failures.append(f"Dataset {size} {split_name}: alignment rows {errors[:10]}")

        expected = EXPECTED[size]
        if len(train) != expected["train"]:
            failures.append(f"Dataset {size} Train={len(train)}, expected {expected['train']}")
        if len(validation) != expected["validation"]:
            failures.append(
                f"Dataset {size} Validation={len(validation)}, expected {expected['validation']}"
            )

        sources, unknown = identify_source_counts(pool, glaucoma_keys, mtsamples_keys)
        print(
            f"  Sources: Glaucoma={sources['glaucoma']}, "
            f"MTSamples={sources['mtsamples']}, unknown/ambiguous={len(unknown)}"
        )
        if unknown:
            failures.append(f"Dataset {size}: {len(unknown)} unknown/ambiguous source records")
        for source in ("glaucoma", "mtsamples"):
            if sources[source] != expected[source]:
                failures.append(
                    f"Dataset {size} {source}={sources[source]}, expected {expected[source]}"
                )

        counters = {
            "train": key_counter(train),
            "validation": key_counter(validation),
        }
        overlaps = {
            "Train ∩ Validation": set(counters["train"]) & set(counters["validation"]),
            "Train ∩ Test": set(counters["train"]) & set(test_counter),
            "Validation ∩ Test": set(counters["validation"]) & set(test_counter),
        }
        print("  Overlap:", ", ".join(f"{name}={len(keys)}" for name, keys in overlaps.items()))
        if any(overlaps.values()):
            failures.append(f"Dataset {size}: split overlap detected")

        # Every selected Glaucoma duplicate key must include the complete source
        # group, and the complete group must occur in only Train or Validation.
        train_counter = counters["train"]
        validation_counter = counters["validation"]
        duplicate_errors = []
        for key, full_count in glaucoma_full_counter.items():
            train_count = train_counter[key]
            validation_count = validation_counter[key]
            selected_count = train_count + validation_count
            if selected_count not in (0, full_count) or (train_count and validation_count):
                duplicate_errors.append(key)
        print("  Glaucoma duplicate leakage/errors:", len(duplicate_errors))
        if duplicate_errors:
            failures.append(f"Dataset {size}: {len(duplicate_errors)} duplicate-group errors")

    print("\n===== NESTED POOLS (MULTISET) =====")
    pool_counters = {size: key_counter(data[size]["pool"]) for size in SIZES}
    nested_50_75 = multiset_subset(pool_counters[50], pool_counters[75])
    nested_75_100 = multiset_subset(pool_counters[75], pool_counters[100])
    print("DatasetPool50 ⊆ DatasetPool75:", "PASS" if nested_50_75 else "FAIL")
    print("DatasetPool75 ⊆ DatasetPool100:", "PASS" if nested_75_100 else "FAIL")
    if not nested_50_75:
        failures.append("DatasetPool50 is not a multiset subset of DatasetPool75")
    if not nested_75_100:
        failures.append("DatasetPool75 is not a multiset subset of DatasetPool100")

    print("\n===== RESULT =====")
    if failures:
        print("FAIL")
        for failure in failures:
            print(" -", failure)
        return 1
    print("PASS: all dataset-size validation checks succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
