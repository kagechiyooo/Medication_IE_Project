import json
from pathlib import Path

input_file = Path("data/mtsamples/mtsamples_doccano.jsonl")
output_dir = Path("data/mtsamples/annotation_parts")

output_dir.mkdir(parents=True, exist_ok=True)

# อ่านข้อมูลทั้งหมด
data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print("จำนวนข้อมูลทั้งหมด:", len(data))

# แบ่งช่วงละ 200
chunk_size = 200

for start in range(0, len(data), chunk_size):
    end = min(start + chunk_size, len(data))

    part = data[start:end]

    # ใช้เลขแบบคนอ่านง่าย เริ่มที่ 1
    start_no = start + 1
    end_no = end

    output_file = output_dir / f"mtsamples_{start_no:03d}_{end_no:03d}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for item in part:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"สร้าง {output_file.name} "
        f"จำนวน {len(part)} ข้อความ"
    )

print("\nเสร็จเรียบร้อย")
print("โฟลเดอร์:", output_dir)