"""팝업 섹션이 비어있는 리포트만 선택적으로 재처리"""
import os, re, json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
client  = genai.Client(api_key=API_KEY)

ORIG_PATH    = "outputs/product_detail_summary.csv"
CLEANED_PATH = "outputs/cleaned_product_summary.csv"
OUT_DIR      = Path("prototype_results")

# ── 빈 팝업 파일 탐지 ─────────────────────────────────────────────────────────
broken_files = []
for f in sorted(OUT_DIR.glob("*.md")):
    text = f.read_text(encoding="utf-8")
    m = re.search(r"### 구매 전 팝업 경고\n(.*?)\n### 트리거", text, re.DOTALL)
    if m and not m.group(1).strip():
        broken_files.append(f)

broken_names = {f.stem.replace("_", " ") for f in broken_files}
print(f"재처리 대상: {len(broken_files)}개")
for f in broken_files:
    print(f"  - {f.name}")

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
orig    = pd.read_csv(ORIG_PATH,    encoding="utf-8-sig")
cleaned = pd.read_csv(CLEANED_PATH, encoding="utf-8-sig")
merged  = orig.merge(
    cleaned[["product_name", "triggers"]].rename(columns={"triggers": "cleaned_triggers"}),
    on="product_name", how="left"
)
merged["final_triggers"] = merged["cleaned_triggers"].fillna(merged["triggers"])

# ── JSON 프롬프트 ─────────────────────────────────────────────────────────────
PROMPT = """당신은 쿠션 파운데이션 리스크 분석 전문가입니다.
아래 트리거 문장들만 참고하여 아래 JSON을 **반드시** 완성하세요.
마크다운 코드블럭 없이 JSON만 출력하세요.

[규칙]
- 트리거 문장에 없는 내용은 절대 추가하지 마세요.
- comment: 50자 이내, 가장 치명적인 리스크 1~2개 중심 한 줄 요약.
- popup_friendly : 친절형. "💛 잠깐, 이런 분들은 확인해보세요!" 로 시작. 불릿 2~3개.
- popup_warning  : 경고형. "🚨 구매 전 반드시 확인하세요!" 로 시작. 불릿 2~3개.
- popup_info     : 정보전달형. "ℹ️ 이런 분들께 맞는 제품이에요." 로 시작. 불릿 2~3개.

출력 형식:
{{
  "comment": "...",
  "popup_friendly": "💛 잠깐, 이런 분들은 확인해보세요!\\n- ...\\n- ...",
  "popup_warning": "🚨 구매 전 반드시 확인하세요!\\n- ...\\n- ...",
  "popup_info": "ℹ️ 이런 분들께 맞는 제품이에요.\\n- ...\\n- ..."
}}

제품명: {product_name}
트리거 목록:
{triggers}"""

def call_gemini(pname, triggers_text):
    prompt = PROMPT.format(product_name=pname, triggers=triggers_text)
    resp   = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text   = resp.text.strip()
    text   = re.sub(r"^```(?:json)?\s*", "", text)
    text   = re.sub(r"\s*```$",          "", text)
    data   = json.loads(text)
    return (data["comment"],
            data["popup_friendly"],
            data["popup_warning"],
            data["popup_info"])

# ── 재처리 & 파일 덮어쓰기 ───────────────────────────────────────────────────
fixed = []
for _, row in merged.iterrows():
    pname = str(row["product_name"]).strip()
    safe  = re.sub(r'[\\/:*?"<>| ]', '_', pname)
    fpath = OUT_DIR / f"{safe}.md"

    if fpath not in broken_files:
        continue

    triggers_raw  = str(row["final_triggers"]) if pd.notna(row["final_triggers"]) else ""
    trigger_list  = [t.strip() for t in triggers_raw.split("|||") if t.strip()]
    triggers_text = "\n".join(f"- {t}" for t in trigger_list)

    comment, popup_friendly, popup_warning, popup_info = call_gemini(pname, triggers_text)

    # 기존 파일의 섹션만 교체
    old_text = fpath.read_text(encoding="utf-8")

    popup_block = (
        f"### 구매 전 팝업 경고\n\n"
        f"**[친절형]**\n{popup_friendly}\n\n"
        f"**[경고형]**\n{popup_warning}\n\n"
        f"**[정보전달형]**\n{popup_info}\n"
    )
    new_text = re.sub(
        r"### 구매 전 팝업 경고\n.*?\n(### 트리거)",
        popup_block + r"\n\1",
        old_text,
        flags=re.DOTALL
    )
    # 핵심 코멘트도 교체
    new_text = re.sub(
        r"(### 핵심 코멘트\n> ).*",
        rf"\g<1>{comment}",
        new_text
    )
    fpath.write_text(new_text, encoding="utf-8")
    fixed.append(pname)
    print(f"  ✓ {pname} 수정 완료", flush=True)

# ── 검증 ─────────────────────────────────────────────────────────────────────
print(f"\n총 {len(fixed)}개 파일 수정 완료\n")
print("── 팝업 섹션 채움 여부 검증 ──")
all_ok = True
for f in broken_files:
    text = f.read_text(encoding="utf-8")
    m    = re.search(r"### 구매 전 팝업 경고\n(.*?)\n### 트리거", text, re.DOTALL)
    ok   = m and m.group(1).strip()
    status = "✅ 채워짐" if ok else "❌ 여전히 비어있음"
    if not ok:
        all_ok = False
    print(f"  {status}  {f.name}")

print("\n" + ("모든 팝업 정상 생성 완료!" if all_ok else "일부 파일 재확인 필요"))
