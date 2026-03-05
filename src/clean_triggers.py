import os
import pandas as pd
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

CLEAN_PROMPT = """다음은 유튜브 쿠션 파운데이션 리뷰에서 추출한 '리스크 트리거' 문장들입니다.
아래 기준으로 정제하고, 정제된 텍스트만 출력하세요 (설명·번호 없이).

정제 기준:
1. 오타·맞춤법·비문 수정 (예: "재형" → "제형", "장각질" → "잔각질")
2. 문장 앞뒤 불필요한 추임새 제거 ("어..", "음..", "그니까" 등)
3. 유튜버 구어체와 '조건부 불안 뉘앙스'는 반드시 살릴 것 (딱딱한 보고서체 금지)
4. `|||` 구분자 그대로 유지, 각 토막이 독립 리스크를 나타내도록 정리
5. 문장 중간의 맥락 없는 노이즈 구절은 삭제 (예: 비교 브랜드 언급·관계없는 설명은 핵심만 추출)

제품: {product_name}

원본:
{text}"""

def clean_triggers(text: str, product_name: str) -> str:
    if not text or pd.isna(text) or str(text).strip() == "":
        return text
    prompt = CLEAN_PROMPT.format(product_name=product_name, text=str(text))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()

INPUT  = "outputs/product_detail_summary.csv"
OUTPUT = "outputs/cleaned_product_summary.csv"

df = pd.read_csv(INPUT)
originals = df["triggers"].copy()

cleaned = []
total = len(df)
for i, (_, row) in enumerate(df.iterrows()):
    result = clean_triggers(row["triggers"], row["product_name"])
    cleaned.append(result)
    print(f"  [{i+1:02d}/{total}] {row['product_name']} 완료", flush=True)

df["triggers"] = cleaned
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\n파일 저장 완료!")
print(f"→ {OUTPUT}\n")
print("=" * 60)
print("전/후 비교 샘플 (5개)")
print("=" * 60)

# 트리거가 길이 있는 첫 5개 행 선택
sample_indices = [i for i in range(total) if pd.notna(originals.iloc[i]) and len(str(originals.iloc[i])) > 10][:5]

for rank, idx in enumerate(sample_indices, 1):
    name  = df.iloc[idx]["product_name"]
    orig  = str(originals.iloc[idx])
    clean = str(cleaned[idx])
    # 첫 번째 ||| 토막만 보여줌
    orig_first  = orig.split("|||")[0].strip()
    clean_first = clean.split("|||")[0].strip()
    print(f"\n[{rank}] {name}")
    print(f"  【전】 {orig_first[:180]}")
    print(f"  【후】 {clean_first[:180]}")
