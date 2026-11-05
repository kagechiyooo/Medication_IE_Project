import json

FILES = {
    "train": [
    "archive/initial_split/data/glaucoma/splits/train.jsonl",
    "archive/initial_split/data/mtsamples/splits/mtsamples_train_001_800.jsonl"
    ],
    "validation": [
        "archive/initial_split/data/glaucoma/splits/validation.jsonl",
        "archive/initial_split/data/mtsamples/splits/mtsamples_val_001_800.jsonl"
    ],
    "test": [
        "archive/initial_split/data/glaucoma/splits/test.jsonl",
        "archive/initial_split/data/mtsamples/splits/mtsamples_test_001_800.jsonl"
    ]
}

OUTPUTS = {
    "train": "archive/initial_split/data/final/train.jsonl",
    "validation": "archive/initial_split/data/final/validation.jsonl",
    "test": "archive/initial_split/data/final/test.jsonl"
}


def merge_jsonl(input_files, output_file):
    records = []

    for path in input_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    with open(output_file, "w", encoding="utf-8") as f:
        for item in records:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )

    return len(records)


print("===== MERGE RESULT =====")

for split_name in ["train", "validation", "test"]:

    count = merge_jsonl(
        FILES[split_name],
        OUTPUTS[split_name]
    )

    print(
        f"{split_name}: {count} -> {OUTPUTS[split_name]}"
    )