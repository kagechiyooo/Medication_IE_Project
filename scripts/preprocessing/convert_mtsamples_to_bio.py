import json
import re

input_file = "data/mtsamples/mtsamples_annotated_001_400_checked.jsonl"
output_file = "data/mtsamples/mtsamples_annotated_001_400_bio.jsonl"


def tokenize_with_spans(text):
    """
    แบ่งข้อความเป็น token แบบเดียวกับ Glaucoma
    และเก็บตำแหน่ง start/end
    """
    tokens = []

    for match in re.finditer(r"\S+", text):
        tokens.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    return tokens


def get_bio_labels(tokens, entities):
    bio_labels = ["O"] * len(tokens)

    for entity_start, entity_end, entity_type in entities:

        first_token = True

        for i, token in enumerate(tokens):

            token_start = token["start"]
            token_end = token["end"]

            overlaps = (
                token_start < entity_end
                and token_end > entity_start
            )

            if overlaps:
                if first_token:
                    bio_labels[i] = f"B-{entity_type}"
                    first_token = False
                else:
                    bio_labels[i] = f"I-{entity_type}"

    return bio_labels


count = 0

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:

        data = json.loads(line)

        text = data["text"]

        # Doccano export ใช้ key "label"
        entities = data.get("label", [])

        tokens_info = tokenize_with_spans(text)

        tokens = [
            token["text"]
            for token in tokens_info
        ]

        bio_labels = get_bio_labels(
            tokens_info,
            entities
        )

        output = {
            "id": data.get("id"),
            "text": text,
            "tokens": tokens,
            "ner_tags": bio_labels
        }

        fout.write(
            json.dumps(
                output,
                ensure_ascii=False
            ) + "\n"
        )

        count += 1


print("แปลง MTSamples เป็น BIO เรียบร้อย")
print("จำนวนข้อความ:", count)
print("ไฟล์:", output_file)