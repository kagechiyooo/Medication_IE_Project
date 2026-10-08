import json

FILES = {
    "train": [
        "data/glaucoma/train.jsonl",
        "data/mtsamples/mtsamples_train.jsonl"
    ],
    "validation": [
        "data/glaucoma/validation.jsonl",
        "data/mtsamples/mtsamples_val.jsonl"
    ],
    "test": [
        "data/glaucoma/test.jsonl",
        "data/mtsamples/mtsamples_test.jsonl"
    ]
}

OUTPUTS = {
    "train": "data/train_final.jsonl",
    "validation": "data/validation_final.jsonl",
    "test": "data/test_final.jsonl"
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