import json

FILES = {
    "train": [
    "data/glaucoma/splits/train.jsonl",
    "data/mtsamples/splits/mtsamples_train_001_800.jsonl"
    ],
    "validation": [
        "data/glaucoma/splits/validation.jsonl",
        "data/mtsamples/splits/mtsamples_val_001_800.jsonl"
    ],
    "test": [
        "data/glaucoma/splits/test.jsonl",
        "data/mtsamples/splits/mtsamples_test_001_800.jsonl"
    ]
}

OUTPUTS = {
    "train": "data/final/train.jsonl",
    "validation": "data/final/validation.jsonl",
    "test": "data/final/test.jsonl"
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