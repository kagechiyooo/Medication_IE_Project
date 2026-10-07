import pandas as pd

file_path = "data/mtsamples/mtsamples.csv"

df = pd.read_csv(file_path)

print("จำนวนข้อมูลทั้งหมด:", len(df))
print()

print("ชื่อคอลัมน์:")
print(df.columns.tolist())
print()

print("จำนวน transcription ที่ไม่ว่าง:")
print(df["transcription"].notna().sum())
print()

print("ตัวอย่าง transcription 3 รายการ:")
print("-" * 60)

count = 0

for text in df["transcription"].dropna():
    print(text[:500])
    print("-" * 60)

    count += 1

    if count == 3:
        break