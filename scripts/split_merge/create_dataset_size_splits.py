"""Create nested 50%, 75%, and 100% train/validation dataset pools.

The locked 15% test data is copied to one shared file and is never used while
selecting a pool or balancing train/validation.  Glaucoma records with the same
``tokens`` and ``ner_tags`` are always handled as one indivisible group.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ENTITY_TYPES = ("DRUG", "STRENGTH", "DOSAGE", "FREQUENCY", "ROUTE")
ENTITY_WEIGHTS = {
    "DRUG": 1.0,
    "STRENGTH": 4.0,
    "DOSAGE": 6.0,
    "FREQUENCY": 1.0,
    "ROUTE": 1.0,
}
SEED = 42
N_TRIALS = 30_000

GLAUCOMA_REMAINING = Path("data/glaucoma/experiment_splits/remaining_85.jsonl")
MTSAMPLES_REMAINING = Path("data/mtsamples/experiment_splits/remaining_85.jsonl")
GLAUCOMA_TEST = Path("data/glaucoma/experiment_splits/test_15.jsonl")
MTSAMPLES_TEST = Path("data/mtsamples/experiment_splits/test_15.jsonl")
OUTPUT_ROOT = Path("data/dataset_size")

SOURCE_TARGETS = {
    50: {"glaucoma": 794, "mtsamples": 340},
    75: {"glaucoma": 1191, "mtsamples": 510},
    100: {"glaucoma": 1588, "mtsamples": 680},
}
VALIDATION_TARGETS = {
    50: {"glaucoma": 238, "mtsamples": 102},
    75: {"glaucoma": 357, "mtsamples": 153},
    100: {"glaucoma": 476, "mtsamples": 204},
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_key(record):
    payload = {
        "tokens": record.get("tokens", []),
        "ner_tags": record.get("ner_tags", []),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def entity_counts(record):
    return Counter(
        tag[2:]
        for tag in record.get("ner_tags", [])
        if tag.startswith("B-") and tag[2:] in ENTITY_TYPES
    )


def total_entity_counts(records):
    result = Counter()
    for record in records:
        result.update(entity_counts(record))
    return result


def make_units(records, group_duplicates):
    """Return selection units while retaining the source records' order."""
    if not group_duplicates:
        return [
            {
                "indices": (index,),
                "size": 1,
                "counts": entity_counts(record),
            }
            for index, record in enumerate(records)
        ]

    grouped_indices = defaultdict(list)
    for index, record in enumerate(records):
        grouped_indices[record_key(record)].append(index)

    units = []
    for indices in grouped_indices.values():
        counts = Counter()
        for index in indices:
            counts.update(entity_counts(records[index]))
        units.append(
            {"indices": tuple(indices), "size": len(indices), "counts": counts}
        )
    return units


def distribution_score(counts, targets):
    score = 0.0
    for entity in ENTITY_TYPES:
        target = targets[entity]
        if target:
            relative_error = (counts[entity] - target) / target
            score += ENTITY_WEIGHTS[entity] * relative_error * relative_error
    return score


def choose_units(units, target_size, entity_targets, seed):
    """Repeated seeded search for an exact-size, entity-balanced unit subset."""
    if target_size == sum(unit["size"] for unit in units):
        return list(range(len(units))), 0.0

    rng = random.Random(seed)
    order = list(range(len(units)))
    best_indices = None
    best_score = float("inf")

    for _ in range(N_TRIALS):
        rng.shuffle(order)
        selected = []
        size = 0
        counts = Counter()

        for unit_index in order:
            unit = units[unit_index]
            if size + unit["size"] <= target_size:
                selected.append(unit_index)
                size += unit["size"]
                counts.update(unit["counts"])
                if size == target_size:
                    break

        if size != target_size:
            continue

        score = distribution_score(counts, entity_targets)
        if score < best_score:
            best_indices = selected.copy()
            best_score = score

    if best_indices is None:
        raise RuntimeError(
            f"Could not reach exact size {target_size} without splitting a group"
        )
    return best_indices, best_score


def select_records(records, target_size, reference_counts, reference_size, seed,
                   group_duplicates):
    units = make_units(records, group_duplicates)
    ratio = target_size / reference_size
    targets = {entity: reference_counts[entity] * ratio for entity in ENTITY_TYPES}
    chosen, score = choose_units(units, target_size, targets, seed)
    chosen_record_indices = {
        record_index
        for unit_index in chosen
        for record_index in units[unit_index]["indices"]
    }
    selected = [
        record for index, record in enumerate(records) if index in chosen_record_indices
    ]
    return selected, score


def split_train_validation(records, validation_size, seed, group_duplicates):
    counts = total_entity_counts(records)
    validation, score = select_records(
        records,
        validation_size,
        counts,
        len(records),
        seed,
        group_duplicates,
    )
    validation_keys = Counter(record_key(record) for record in validation)
    train = []
    remaining_validation = validation_keys.copy()
    for record in records:
        key = record_key(record)
        if remaining_validation[key]:
            remaining_validation[key] -= 1
        else:
            train.append(record)
    return train, validation, score


def print_split_summary(size, train, validation, source_counts, scores):
    print(f"\n===== DATASET {size}% =====")
    print(
        "Pool:", len(train) + len(validation),
        f"(Glaucoma {source_counts['glaucoma']}, MTSamples {source_counts['mtsamples']})",
    )
    print("Selection scores:", ", ".join(f"{name}={value:.8f}" for name, value in scores.items()))
    for name, records in (("TRAIN", train), ("VALIDATION", validation)):
        counts = total_entity_counts(records)
        print(f"{name}: {len(records)}")
        print("  " + ", ".join(f"{entity}={counts[entity]}" for entity in ENTITY_TYPES))


def main():
    glaucoma = load_jsonl(GLAUCOMA_REMAINING)
    mtsamples = load_jsonl(MTSAMPLES_REMAINING)
    glaucoma_test = load_jsonl(GLAUCOMA_TEST)
    mtsamples_test = load_jsonl(MTSAMPLES_TEST)

    expected_inputs = (1588, 680, 280, 120)
    actual_inputs = (len(glaucoma), len(mtsamples), len(glaucoma_test), len(mtsamples_test))
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"Unexpected input sizes: {actual_inputs}, expected {expected_inputs}")

    glaucoma_reference_counts = total_entity_counts(glaucoma)
    mtsamples_reference_counts = total_entity_counts(mtsamples)

    # Build downward so every smaller source pool is a multiset subset of the
    # previous one.  Targets still refer to the full remaining source data.
    source_pools = {
        100: {"glaucoma": glaucoma, "mtsamples": mtsamples},
    }
    pool_scores = {100: {"glaucoma_pool": 0.0, "mtsamples_pool": 0.0}}
    previous_size = 100
    for size in (75, 50):
        g_pool, g_score = select_records(
            source_pools[previous_size]["glaucoma"],
            SOURCE_TARGETS[size]["glaucoma"],
            glaucoma_reference_counts,
            len(glaucoma),
            SEED + size,
            group_duplicates=True,
        )
        m_pool, m_score = select_records(
            source_pools[previous_size]["mtsamples"],
            SOURCE_TARGETS[size]["mtsamples"],
            mtsamples_reference_counts,
            len(mtsamples),
            SEED + 100 + size,
            group_duplicates=False,
        )
        source_pools[size] = {"glaucoma": g_pool, "mtsamples": m_pool}
        pool_scores[size] = {
            "glaucoma_pool": g_score,
            "mtsamples_pool": m_score,
        }
        previous_size = size

    for size in (50, 75, 100):
        g_train, g_validation, g_score = split_train_validation(
            source_pools[size]["glaucoma"],
            VALIDATION_TARGETS[size]["glaucoma"],
            SEED + 200 + size,
            group_duplicates=True,
        )
        m_train, m_validation, m_score = split_train_validation(
            source_pools[size]["mtsamples"],
            VALIDATION_TARGETS[size]["mtsamples"],
            SEED + 300 + size,
            group_duplicates=False,
        )
        train = g_train + m_train
        validation = g_validation + m_validation
        output_dir = OUTPUT_ROOT / f"dataset_{size}"
        save_jsonl(train, output_dir / "train.jsonl")
        save_jsonl(validation, output_dir / "validation.jsonl")
        scores = dict(pool_scores[size])
        scores.update({"glaucoma_validation": g_score, "mtsamples_validation": m_score})
        print_split_summary(
            size,
            train,
            validation,
            SOURCE_TARGETS[size],
            scores,
        )

    shared_test = glaucoma_test + mtsamples_test
    save_jsonl(shared_test, OUTPUT_ROOT / "test_15.jsonl")
    print(f"\nSHARED TEST: {len(shared_test)} (Glaucoma 280, MTSamples 120)")


if __name__ == "__main__":
    main()
