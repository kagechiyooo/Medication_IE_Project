import json
import random
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


INPUT_FILE = "data/glaucoma/glaucoma_bio.jsonl"

OUTPUT_DIR = Path(
    "data/glaucoma/experiment_splits"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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
N_TRIALS = 30000

# ต้องการ Test = 280 แถว
TEST_SIZE = 280


def get_entity_counts(record):
    counts = Counter()

    for tag in record["ner_tags"]:
        if tag.startswith("B-"):
            entity = tag[2:]

            if entity in ENTITY_TYPES:
                counts[entity] += 1

    return counts


def record_key(record):
    """
    record ที่ tokens + ner_tags เหมือนกัน
    ถือเป็น duplicate group เดียวกัน
    """

    data = {
        "tokens": record.get(
            "tokens",
            []
        ),
        "ner_tags": record.get(
            "ner_tags",
            []
        ),
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
        counts.update(
            record["_counts"]
        )

    return counts


def save_jsonl(records, path):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            output = {
                key: value
                for key, value
                in record.items()
                if key not in [
                    "_counts",
                    "_group_key",
                ]
            }

            f.write(
                json.dumps(
                    output,
                    ensure_ascii=False
                ) + "\n"
            )


# =====================================
# 1. LOAD DATA
# =====================================

records = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if line.strip():

            item = json.loads(line)

            item["_counts"] = (
                get_entity_counts(item)
            )

            item["_group_key"] = (
                record_key(item)
            )

            records.append(item)


print(
    "===== GLAUCOMA TOTAL ====="
)

print(
    "Texts:",
    len(records)
)


# =====================================
# 2. GROUP DUPLICATES
# =====================================

group_dict = defaultdict(list)

for record in records:

    group_dict[
        record["_group_key"]
    ].append(record)


groups = list(
    group_dict.values()
)


print(
    "Unique groups:",
    len(groups)
)

print(
    "Duplicate rows:",
    len(records) - len(groups)
)


# =====================================
# 3. GROUP INFORMATION
# =====================================

group_info = []

for group in groups:

    entity_counts = (
        count_entities(group)
    )

    group_info.append({
        "records": group,
        "size": len(group),
        "counts": entity_counts,
    })


total_counts = (
    count_entities(records)
)


print(
    "\n===== ENTITY TOTAL ====="
)

for entity in ENTITY_TYPES:

    print(
        f"{entity:10s}: "
        f"{total_counts[entity]}"
    )


# =====================================
# 4. TARGET ENTITY COUNTS
# =====================================

TEST_RATIO = (
    TEST_SIZE / len(records)
)

targets = {
    entity:
    total_counts[entity]
    * TEST_RATIO
    for entity in ENTITY_TYPES
}


weights = {
    "DRUG": 1,
    "STRENGTH": 4,
    "DOSAGE": 6,
    "FREQUENCY": 1,
    "ROUTE": 1,
}


# =====================================
# 5. SCORE
# =====================================

def calculate_score(
    selected_groups
):

    size = sum(
        group["size"]
        for group
        in selected_groups
    )

    # size ต้องใกล้ 280 มากที่สุด
    size_error = abs(
        size - TEST_SIZE
    )

    counts = Counter()

    for group in selected_groups:
        counts.update(
            group["counts"]
        )

    entity_score = 0

    for entity in ENTITY_TYPES:

        target = targets[entity]

        if target > 0:

            error = abs(
                counts[entity]
                - target
            ) / target

            entity_score += (
                weights[entity]
                * error ** 2
            )

    # ให้จำนวนข้อความสำคัญมาก
    return (
        size_error * 100
        + entity_score
    )


# =====================================
# 6. RANDOM SEARCH
# =====================================

rng = random.Random(SEED)

best_score = float("inf")
best_groups = None


for trial in range(N_TRIALS):

    shuffled = groups.copy()

    rng.shuffle(shuffled)

    selected = []
    current_size = 0

    for group in shuffled:

        group_size = len(group)

        # ถ้าใส่แล้วไม่เกิน 280
        # ให้เลือกก่อน
        if (
            current_size + group_size
            <= TEST_SIZE
        ):

            selected.append({
                "records": group,
                "size": group_size,
                "counts":
                count_entities(group),
            })

            current_size += (
                group_size
            )

        if current_size == TEST_SIZE:
            break

    # ถ้าไม่ครบ 280
    # ข้าม trial นี้
    if current_size != TEST_SIZE:
        continue

    score = calculate_score(
        selected
    )

    if score < best_score:

        best_score = score
        best_groups = selected


if best_groups is None:
    raise RuntimeError(
        "ไม่สามารถสร้าง Test "
        "จำนวน 280 แถวได้"
    )


# =====================================
# 7. CREATE TEST / REMAINING
# =====================================

test_group_keys = set()

for group in best_groups:

    for record in group["records"]:

        test_group_keys.add(
            record["_group_key"]
        )


test_records = []

remaining_records = []


for record in records:

    if (
        record["_group_key"]
        in test_group_keys
    ):
        test_records.append(
            record
        )

    else:
        remaining_records.append(
            record
        )


# =====================================
# 8. SAVE
# =====================================

save_jsonl(
    test_records,
    TEST_FILE
)

save_jsonl(
    remaining_records,
    REMAINING_FILE
)


# =====================================
# 9. REPORT
# =====================================

test_counts = (
    count_entities(
        test_records
    )
)

remaining_counts = (
    count_entities(
        remaining_records
    )
)


print(
    "\n===== TEST ====="
)

print(
    "Texts:",
    len(test_records)
)

for entity in ENTITY_TYPES:

    print(
        f"{entity:10s}: "
        f"{test_counts[entity]} "
        f"(target ≈ "
        f"{targets[entity]:.1f})"
    )


print(
    "\n===== REMAINING ====="
)

print(
    "Texts:",
    len(remaining_records)
)

for entity in ENTITY_TYPES:

    print(
        f"{entity:10s}: "
        f"{remaining_counts[entity]}"
    )


print(
    "\nBest score:",
    best_score
)


# =====================================
# 10. CHECK DUPLICATE LEAKAGE
# =====================================

test_keys = {
    record["_group_key"]
    for record
    in test_records
}

remaining_keys = {
    record["_group_key"]
    for record
    in remaining_records
}

overlap = (
    test_keys
    & remaining_keys
)


print(
    "\n===== DUPLICATE CHECK ====="
)

print(
    "Test ∩ Remaining:",
    len(overlap)
)

if len(overlap) == 0:

    print(
        "✅ No duplicate leakage"
    )

else:

    print(
        "❌ Duplicate leakage found"
    )


print(
    "\nSaved:"
)

print(TEST_FILE)
print(REMAINING_FILE)