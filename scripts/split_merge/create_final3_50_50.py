import json
import random
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


ENTITY_TYPES = [
    "DRUG",
    "STRENGTH",
    "DOSAGE",
    "FREQUENCY",
    "ROUTE",
]

SEED = 42
N_TRIALS = 30000


GLAUCOMA_REMAINING = (
    "data/glaucoma/experiment_splits/remaining_85.jsonl"
)

MTSAMPLES_REMAINING = (
    "data/mtsamples/experiment_splits/remaining_85.jsonl"
)

GLAUCOMA_TEST = (
    "data/glaucoma/experiment_splits/test_15.jsonl"
)

MTSAMPLES_TEST = (
    "data/mtsamples/experiment_splits/test_15.jsonl"
)


# final3 = 50/50
GLAUCOMA_TRAIN_SIZE = 794
GLAUCOMA_VAL_SIZE = 794

MTSAMPLES_TRAIN_SIZE = 340
MTSAMPLES_VAL_SIZE = 340


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
        counts.update(
            get_entity_counts(record)
        )

    return counts


def record_key(record):

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


def load_jsonl(path):

    records = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def save_jsonl(records, path):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


def split_glaucoma_group_aware(
    records,
    val_size
):

    groups_dict = defaultdict(list)

    for record in records:

        key = record_key(record)

        groups_dict[key].append(
            record
        )

    groups = list(
        groups_dict.values()
    )

    total_counts = count_entities(
        records
    )

    val_ratio = (
        val_size
        / len(records)
    )

    targets = {
        entity:
        total_counts[entity]
        * val_ratio
        for entity in ENTITY_TYPES
    }

    weights = {
        "DRUG": 1,
        "STRENGTH": 4,
        "DOSAGE": 6,
        "FREQUENCY": 1,
        "ROUTE": 1,
    }

    def score(selected_groups):

        selected_records = []

        for group in selected_groups:
            selected_records.extend(
                group
            )

        counts = count_entities(
            selected_records
        )

        result = 0

        for entity in ENTITY_TYPES:

            target = targets[entity]

            if target > 0:

                error = abs(
                    counts[entity]
                    - target
                ) / target

                result += (
                    weights[entity]
                    * error ** 2
                )

        return result


    rng = random.Random(SEED)

    best_score = float("inf")
    best_val_groups = None


    for trial in range(N_TRIALS):

        shuffled = groups.copy()

        rng.shuffle(shuffled)

        selected = []
        current_size = 0

        for group in shuffled:

            group_size = len(group)

            if (
                current_size
                + group_size
                <= val_size
            ):

                selected.append(
                    group
                )

                current_size += (
                    group_size
                )

            if current_size == val_size:
                break


        if current_size != val_size:
            continue


        current_score = score(
            selected
        )


        if current_score < best_score:

            best_score = current_score

            best_val_groups = (
                selected.copy()
            )


    if best_val_groups is None:

        raise RuntimeError(
            "Cannot create exact "
            f"Glaucoma validation size "
            f"{val_size}"
        )


    val_keys = set()

    for group in best_val_groups:

        key = record_key(
            group[0]
        )

        val_keys.add(key)


    train_records = []
    val_records = []


    for record in records:

        key = record_key(record)

        if key in val_keys:

            val_records.append(
                record
            )

        else:

            train_records.append(
                record
            )


    return (
        train_records,
        val_records,
        best_score
    )


def split_mtsamples(
    records,
    train_size,
    val_size
):

    total_counts = count_entities(
        records
    )

    val_ratio = (
        val_size
        / len(records)
    )

    targets = {
        entity:
        total_counts[entity]
        * val_ratio
        for entity in ENTITY_TYPES
    }

    weights = {
        "DRUG": 1,
        "STRENGTH": 4,
        "DOSAGE": 6,
        "FREQUENCY": 1,
        "ROUTE": 1,
    }

    def score(val_records):

        counts = count_entities(
            val_records
        )

        result = 0

        for entity in ENTITY_TYPES:

            target = targets[entity]

            if target > 0:

                error = abs(
                    counts[entity]
                    - target
                ) / target

                result += (
                    weights[entity]
                    * error ** 2
                )

        return result


    rng = random.Random(SEED)

    indices = list(
        range(len(records))
    )

    best_score = float("inf")
    best_val_indices = None


    for trial in range(N_TRIALS):

        rng.shuffle(indices)

        val_indices = (
            indices[:val_size]
        )

        val_records = [
            records[i]
            for i in val_indices
        ]

        current_score = score(
            val_records
        )

        if current_score < best_score:

            best_score = current_score

            best_val_indices = (
                val_indices.copy()
            )


    val_index_set = set(
        best_val_indices
    )

    train_records = [
        records[i]
        for i in range(len(records))
        if i not in val_index_set
    ]

    val_records = [
        records[i]
        for i in range(len(records))
        if i in val_index_set
    ]


    if len(train_records) != train_size:

        raise RuntimeError(
            "Wrong MTSamples train size"
        )


    return (
        train_records,
        val_records,
        best_score
    )


glaucoma = load_jsonl(
    GLAUCOMA_REMAINING
)

mtsamples = load_jsonl(
    MTSAMPLES_REMAINING
)


(
    glaucoma_train,
    glaucoma_val,
    glaucoma_score
) = split_glaucoma_group_aware(
    glaucoma,
    GLAUCOMA_VAL_SIZE
)


(
    mtsamples_train,
    mtsamples_val,
    mtsamples_score
) = split_mtsamples(
    mtsamples,
    MTSAMPLES_TRAIN_SIZE,
    MTSAMPLES_VAL_SIZE
)


save_jsonl(
    glaucoma_train,
    "data/glaucoma/experiment_splits/train_50.jsonl"
)

save_jsonl(
    glaucoma_val,
    "data/glaucoma/experiment_splits/validation_50.jsonl"
)

save_jsonl(
    mtsamples_train,
    "data/mtsamples/experiment_splits/train_50.jsonl"
)

save_jsonl(
    mtsamples_val,
    "data/mtsamples/experiment_splits/validation_50.jsonl"
)


glaucoma_test = load_jsonl(
    GLAUCOMA_TEST
)

mtsamples_test = load_jsonl(
    MTSAMPLES_TEST
)


final_train = (
    glaucoma_train
    + mtsamples_train
)

final_val = (
    glaucoma_val
    + mtsamples_val
)

final_test = (
    glaucoma_test
    + mtsamples_test
)


save_jsonl(
    final_train,
    "data/final3/train.jsonl"
)

save_jsonl(
    final_val,
    "data/final3/validation.jsonl"
)

save_jsonl(
    final_test,
    "data/final3/test.jsonl"
)


train_keys = {
    record_key(record)
    for record in final_train
}

val_keys = {
    record_key(record)
    for record in final_val
}

test_keys = {
    record_key(record)
    for record in final_test
}


print(
    "===== FINAL3 50/50 ====="
)


for name, subset in [
    ("TRAIN", final_train),
    ("VALIDATION", final_val),
    ("TEST", final_test),
]:

    counts = count_entities(
        subset
    )

    print(
        f"\n{name}"
    )

    print(
        "Texts:",
        len(subset)
    )

    for entity in ENTITY_TYPES:

        print(
            f"{entity:10s}: "
            f"{counts[entity]}"
        )


print(
    "\n===== DUPLICATE LEAKAGE CHECK ====="
)

print(
    "Train ∩ Validation:",
    len(train_keys & val_keys)
)

print(
    "Train ∩ Test:",
    len(train_keys & test_keys)
)

print(
    "Validation ∩ Test:",
    len(val_keys & test_keys)
)


if (
    len(train_keys & val_keys) == 0
    and len(train_keys & test_keys) == 0
    and len(val_keys & test_keys) == 0
):

    print(
        "✅ No duplicate leakage"
    )

else:

    print(
        "❌ Duplicate leakage found"
    )


print(
    "\nScores"
)

print(
    "Glaucoma:",
    glaucoma_score
)

print(
    "MTSamples:",
    mtsamples_score
)