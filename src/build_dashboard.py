import os
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# ── 설정 ──────────────────────────────────────────────────────────────────────
load_dotenv()  # .env 파일 자동 로드
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError(".env 또는 환경변수에 GOOGLE_API_KEY / GEMINI_API_KEY가 없습니다.")
client = genai.Client(api_key=API_KEY)

ORIG_PATH    = "outputs/product_detail_summary.csv"
CLEANED_PATH = "outputs/cleaned_product_summary.csv"
OUT_DIR      = Path("prototype_results")
MASTER_PATH  = Path("master_dashboard.md")

OUT_DIR.mkdir(exist_ok=True)

# ── 데이터 로드 & 병합 ────────────────────────────────────────────────────────
orig    = pd.read_csv(ORIG_PATH,    encoding="utf-8-sig")
cleaned = pd.read_csv(CLEANED_PATH, encoding="utf-8-sig")

# cleaned의 triggers 컬럼을 cleaned_triggers로 rename 후 병합
merged = orig.merge(
    cleaned[["product_name", "triggers"]].rename(columns={"triggers": "cleaned_triggers"}),
    on="product_name", how="left"
)
# cleaned_triggers가 없으면 원본 triggers 사용
merged["final_triggers"] = merged["cleaned_triggers"].fillna(merged["triggers"])

# ── Gemini 프롬프트 (JSON 강제) ───────────────────────────────────────────────
COMMENT_PROMPT = """당신은 쿠션 파운데이션 리스크 분석 전문가입니다.
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

def call_gemini(product_name: str, triggers: str) -> tuple[str, str, str, str]:
    import json
    prompt = COMMENT_PROMPT.format(product_name=product_name, triggers=triggers)
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = resp.text.strip()
    # 마크다운 코드블럭 제거
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        comment         = data.get("comment", "")
        popup_friendly  = data.get("popup_friendly", "")
        popup_warning   = data.get("popup_warning", "")
        popup_info      = data.get("popup_info", "")
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 텍스트 전체를 fallback으로
        comment        = text[:80]
        popup_friendly = popup_warning = popup_info = text
    return comment, popup_friendly, popup_warning, popup_info

def score_to_stars(score: float) -> str:
    full  = int(score)
    half  = 1 if (score - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty

def risk_level(trigger_count: int) -> str:
    if trigger_count <= 4:  return "🟢 낮음"
    if trigger_count <= 8:  return "🟡 보통"
    return "🔴 높음"

# ── 제품별 리포트 생성 ────────────────────────────────────────────────────────
report_rows = []   # master dashboard용

for _, row in merged.iterrows():
    pname    = str(row["product_name"]).strip()
    brand    = str(row["brand"]).strip()
    triggers_raw = str(row["final_triggers"]) if pd.notna(row["final_triggers"]) else ""

    # ||| 로 분리된 트리거 목록
    trigger_list = [t.strip() for t in triggers_raw.split("|||") if t.strip()]
    n_triggers   = len(trigger_list)

    avg_score = float(row["avg_score"]) if pd.notna(row["avg_score"]) else 0.0
    stars     = score_to_stars(avg_score)
    risk      = risk_level(n_triggers)

    # 트리거 2개 이하 → 데이터 부족
    if n_triggers <= 2:
        comment        = "데이터 부족 — 트리거 문장이 2개 이하입니다."
        popup_friendly = popup_warning = popup_info = "- 분석에 필요한 리뷰 데이터가 부족합니다."
        insufficient = True
    else:
        triggers_text = "\n".join(f"- {t}" for t in trigger_list)
        comment, popup_friendly, popup_warning, popup_info = call_gemini(pname, triggers_text)
        insufficient = False

    print(f"  ✓ {pname}", flush=True)

    # ── 개별 리포트 ────────────────────────────────────────────────────────
    safe_name = re.sub(r'[\\/:*?"<>| ]', '_', pname)
    report_path = OUT_DIR / f"{safe_name}.md"

    score_dist = str(row.get("score_distribution","")).strip()
    strengths  = str(row.get("top_strengths","")).strip()
    weaknesses = str(row.get("top_weaknesses","")).strip()
    audience   = str(row.get("target_audience","")).strip()
    mention    = int(row["mention_count"]) if pd.notna(row["mention_count"]) else 0

    trigger_md = "\n".join(f"{i+1}. {t}" for i, t in enumerate(trigger_list)) if trigger_list else "_없음_"

    report_md = f"""# {pname}
> **브랜드**: {brand} | **언급 수**: {mention}건 | **평균 점수**: {avg_score:.2f} {stars}

---

## 📊 기본 정보
| 항목 | 내용 |
|------|------|
| 점수 분포 | {score_dist} |
| 강점 | {strengths} |
| 약점 | {weaknesses} |
| 추천 타겟 | {audience} |

---

## ⚠️ 리스크 분석
**리스크 레벨**: {risk} (트리거 {n_triggers}개)

{"> 🔕 **데이터 부족**: 분석에 필요한 트리거 문장이 2개 이하입니다." if insufficient else ""}

### 핵심 코멘트
> {comment}

### 구매 전 팝업 경고

**[친절형]**
{popup_friendly}

**[경고형]**
{popup_warning}

**[정보전달형]**
{popup_info}

### 트리거 전체 목록
{trigger_md}
"""
    report_path.write_text(report_md, encoding="utf-8")

    report_rows.append({
        "brand":       brand,
        "product":     pname,
        "score":       f"{avg_score:.2f} {stars}",
        "risk":        risk,
        "n_triggers":  n_triggers,
        "comment":     comment,
        "report_file": f"prototype_results/{safe_name}.md",
    })

# ── master_dashboard.md ───────────────────────────────────────────────────────
rows_md = "\n".join(
    f"| [{r['product']}]({r['report_file']}) | {r['brand']} | {r['score']} | {r['risk']} | {r['n_triggers']} | {r['comment'][:40]}... |"
    for r in report_rows
)

master_md = f"""# 🛡️ Risk-Guard Dashboard

> 쿠션 파운데이션 리스크 분석 대시보드 | 총 {len(report_rows)}개 제품

---

## 제품 목록

| 제품명 | 브랜드 | 평균 점수 | 리스크 | 트리거 수 | 핵심 코멘트 |
|--------|--------|-----------|--------|-----------|------------|
{rows_md}

---

## 리스크 레벨 기준
- 🟢 **낮음**: 트리거 4개 이하
- 🟡 **보통**: 트리거 5~8개
- 🔴 **높음**: 트리거 9개 이상

_Generated by Risk-Guard Pipeline_
"""
MASTER_PATH.write_text(master_md, encoding="utf-8")

# ── 완료 보고 ─────────────────────────────────────────────────────────────────
print(f"\n총 {len(report_rows)}개의 리포트 생성 완료\n")
print("📁 파일 구조")
print(f"  master_dashboard.md")
print(f"  prototype_results/")
for r in report_rows:
    flag = "⚠️" if r["n_triggers"] <= 2 else "  "
    print(f"  {flag}  {Path(r['report_file']).name}")
