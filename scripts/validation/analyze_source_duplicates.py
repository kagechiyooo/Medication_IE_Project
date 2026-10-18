import json
import hashlib
from collections import Counter, defaultdict


FILES = {
    "glaucoma": "data/glaucoma/glaucoma_bio.jsonl",
    "mtsamples": "data/mtsamples/bio/mtsamples_annotated_001_800_bio.jsonl",
}


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def record_key(record):
    """
    ใช้ tokens + ner_tags เป็นตัวระบุว่า
    record เหมือนกันหรือไม่
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


def analyze_dataset(name, records):

    groups = defaultdict(list)

    for index, record in enumerate(records):
        key = record_key(record)
        groups[key].append(index)

    duplicate_groups = {
        key: indices
        for key, indices in groups.items()
        if len(indices) > 1
    }

    duplicate_extra_rows = sum(
        len(indices) - 1
        for indices in duplicate_groups.values()
    )

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    print("Total rows:", len(records))
    print("Unique records:", len(groups))
    print(
        "Duplicate groups:",
        len(duplicate_groups)
    )
    print(
        "Extra duplicate rows:",
        duplicate_extra_rows
    )

    print("\nDuplicate group sizes:")

    size_counts = Counter(
        len(indices)
        for indices in duplicate_groups.values()
    )

    if not size_counts:
        print("None")
    else:
        for size in sorted(size_counts):
            print(
                f"{size} copies : "
                f"{size_counts[size]} groups"
            )

    # แสดงตัวอย่าง duplicate 5 กลุ่มแรก
    print("\nExample duplicates:")

    shown = 0

    for key, indices in duplicate_groups.items():

        if shown >= 5:
            break

        first_record = records[
            indices[0]
        ]

        text = first_record.get(
            "text",
            ""
        )

        if not text:
            text = " ".join(
                first_record.get(
                    "tokens",
                    []
                )
            )

        print(
            f"\nRows: "
            f"{[i + 1 for i in indices]}"
        )

        print(
            "Text:",
            text[:300]
        )

        shown += 1

    return groups


# ========================================
# LOAD
# ========================================

glaucoma = load_jsonl(
    FILES["glaucoma"]
)

mtsamples = load_jsonl(
    FILES["mtsamples"]
)


# ========================================
# ANALYZE EACH SOURCE
# ========================================

glaucoma_groups = analyze_dataset(
    "glaucoma",
    glaucoma
)

mtsamples_groups = analyze_dataset(
    "mtsamples",
    mtsamples
)


# ========================================
# CHECK CROSS-SOURCE DUPLICATES
# ========================================

print("\n" + "=" * 60)
print("CROSS-SOURCE DUPLICATES")
print("=" * 60)

common_keys = (
    set(glaucoma_groups.keys())
    &
    set(mtsamples_groups.keys())
)

print(
    "Same records in both datasets:",
    len(common_keys)
)

if common_keys:

    print(
        "\nExample cross-source duplicates:"
    )

    shown = 0

    for key in common_keys:

        if shown >= 5:
            break

        g_index = glaucoma_groups[key][0]
        m_index = mtsamples_groups[key][0]

        g_record = glaucoma[g_index]
        m_record = mtsamples[m_index]

        text = g_record.get(
            "text",
            ""
        )

        if not text:
            text = " ".join(
                g_record.get(
                    "tokens",
                    []
                )
            )

        print(
            f"\nGlaucoma row: "
            f"{g_index + 1}"
        )

        print(
            f"MTSamples row: "
            f"{m_index + 1}"
        )

        print(
            "Text:",
            text[:300]
        )

        shown += 1


# ========================================
# TOTAL
# ========================================

print("\n" + "=" * 60)
print("TOTAL")
print("=" * 60)

total_rows = (
    len(glaucoma)
    + len(mtsamples)
)

all_keys = (
    set(glaucoma_groups.keys())
    |
    set(mtsamples_groups.keys())
)

print(
    "Total rows:",
    total_rows
)

print(
    "Unique records across both sources:",
    len(all_keys)
)

print(
    "Extra duplicate rows overall:",
    total_rows - len(all_keys)
)