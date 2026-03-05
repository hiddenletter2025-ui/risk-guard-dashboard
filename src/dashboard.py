import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ── 경로 기준점: dashboard.py 위치에 관계없이 프로젝트 루트를 찾는다 ─────────
# src/dashboard.py  →  .parent = src/  →  .parent.parent = project root
BASE_DIR = Path(__file__).resolve().parent.parent

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk-Guard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* 사이드바 배경 */
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.82rem; }

/* 메트릭 카드 */
[data-testid="stMetric"] {
    background: #1e2130;
    border-radius: 12px;
    padding: 12px 16px;
}

/* 트리거 expander 스타일 */
[data-testid="stExpander"] summary {
    font-size: 0.88rem;
    color: #37474f;
}

/* 리스크 뱃지 */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    color: white;
}
.badge-high   { background: #e53935; }
.badge-medium { background: #fb8c00; }
.badge-low    { background: #43a047; }
.badge-none   { background: #757575; }

/* 제품 라디오 */
div[role="radiogroup"] label { padding: 4px 0; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
@st.cache_data
def load_df():
    df = pd.read_csv(BASE_DIR / "outputs" / "cleaned_product_summary.csv", encoding="utf-8-sig")
    df["trigger_count"] = pd.to_numeric(df["trigger_count"], errors="coerce").fillna(0).astype(int)
    df["mention_count"] = pd.to_numeric(df["mention_count"], errors="coerce").fillna(0).astype(int)
    df["avg_score"]     = pd.to_numeric(df["avg_score"],     errors="coerce").fillna(0.0)
    df["ad_count"]      = pd.to_numeric(df["ad_count"],      errors="coerce").fillna(0).astype(int)
    return df

@st.cache_data
def load_report(product_name: str) -> dict:
    """마크다운에서 AI 생성 섹션(comment, popup, trigger list) 파싱"""
    safe = re.sub(r'[\\/:*?"<>| ]', "_", product_name)
    path = BASE_DIR / "prototype_results" / f"{safe}.md"
    if not path.exists():
        return {"comment": "", "popup_type": "none", "triggers": []}

    text = path.read_text(encoding="utf-8")

    # 핵심 코멘트
    m = re.search(r"### 핵심 코멘트\n> (.+)", text)
    comment = m.group(1).strip() if m else ""

    # 팝업 형식 판별 - 신형(3종)
    if "**[친절형]**" in text:
        m1 = re.search(r"\*\*\[친절형\]\*\*\n(.*?)\n\n\*\*\[경고형\]", text, re.DOTALL)
        m2 = re.search(r"\*\*\[경고형\]\*\*\n(.*?)\n\n\*\*\[정보전달형\]", text, re.DOTALL)
        m3 = re.search(r"\*\*\[정보전달형\]\*\*\n(.*?)\n\n###", text, re.DOTALL)
        popup = {
            "type":     "new",
            "friendly": m1.group(1).strip() if m1 else "",
            "warning":  m2.group(1).strip() if m2 else "",
            "info":     m3.group(1).strip() if m3 else "",
        }
    else:
        # 구형(단일) 또는 데이터부족
        m = re.search(r"### 구매 전 팝업 경고\n(.*?)\n### 트리거", text, re.DOTALL)
        body = m.group(1).strip() if m else ""
        popup = {"type": "legacy", "body": body}

    # 트리거 목록
    m = re.search(r"### 트리거 전체 목록\n(.*?)$", text, re.DOTALL)
    triggers = []
    if m:
        for line in m.group(1).strip().splitlines():
            t = re.sub(r"^\d+\.\s*", "", line.strip())
            if t:
                triggers.append(t)

    return {"comment": comment, "popup": popup, "triggers": triggers}


def risk_label(n: int) -> tuple[str, str]:
    """(이모지+텍스트, css클래스)"""
    if n <= 2:  return "⚪ 데이터 부족", "badge-none"
    if n <= 4:  return "🟢 낮음",       "badge-low"
    if n <= 8:  return "🟡 보통",       "badge-medium"
    return           "🔴 높음",          "badge-high"

def parse_score_dist(dist_str: str) -> dict[int, int]:
    result = {}
    for part in str(dist_str).split("/"):
        m = re.search(r"(\d+)점:(\d+)건", part)
        if m:
            result[int(m.group(1))] = int(m.group(2))
    return result

def stars(score: float) -> str:
    filled = int(round(score * 2)) // 2
    half   = int(round(score * 2)) % 2
    return "★" * filled + ("½" if half else "") + "☆" * (5 - filled - half)


# ── 데이터 준비 ───────────────────────────────────────────────────────────────
df = load_df()

RISK_ORDER = {"🔴 높음": 0, "🟡 보통": 1, "🟢 낮음": 2, "⚪ 데이터 부족": 3}
df["_risk_label"] = df["trigger_count"].apply(lambda n: risk_label(n)[0])
df["_risk_order"] = df["_risk_label"].map(RISK_ORDER)


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Risk-Guard")
    st.caption("쿠션 파운데이션 리스크 분석 대시보드")
    st.divider()

    search    = st.text_input("🔍 제품 / 브랜드 검색", placeholder="예: 클리오, 커버")
    sel_brand = st.selectbox("브랜드", ["전체"] + sorted(df["brand"].dropna().unique()))
    sel_risk  = st.selectbox("리스크 레벨", ["전체", "🔴 높음", "🟡 보통", "🟢 낮음", "⚪ 데이터 부족"])
    sort_by   = st.selectbox("정렬 기준", ["리스크 높은 순", "평균 점수 높은 순", "언급 수 많은 순"])

    # 필터 적용
    fdf = df.copy()
    if search:
        mask = (fdf["product_name"].str.contains(search, case=False, na=False) |
                fdf["brand"].str.contains(search, case=False, na=False))
        fdf = fdf[mask]
    if sel_brand != "전체":
        fdf = fdf[fdf["brand"] == sel_brand]
    if sel_risk != "전체":
        fdf = fdf[fdf["_risk_label"] == sel_risk]

    # 정렬
    if sort_by == "리스크 높은 순":
        fdf = fdf.sort_values(["_risk_order", "avg_score"], ascending=[True, False])
    elif sort_by == "평균 점수 높은 순":
        fdf = fdf.sort_values("avg_score", ascending=False)
    else:
        fdf = fdf.sort_values("mention_count", ascending=False)

    st.divider()
    st.caption(f"**{len(fdf)}개** 제품")

    if fdf.empty:
        st.warning("검색 결과가 없습니다.")
        st.stop()

    product_list = fdf["product_name"].tolist()
    sel_product  = st.radio(
        "제품 선택",
        product_list,
        format_func=lambda x: x,
        label_visibility="collapsed",
    )


# ── 메인 화면 ─────────────────────────────────────────────────────────────────
row    = df[df["product_name"] == sel_product].iloc[0]
report = load_report(sel_product)
rlabel, rbadge = risk_label(int(row["trigger_count"]))

# 헤더
st.markdown(
    f"## {sel_product} &nbsp;"
    f'<span class="badge {rbadge}">{rlabel}</span>',
    unsafe_allow_html=True,
)
st.caption(f"브랜드: **{row['brand']}**")
st.divider()

# 메트릭 4개
c1, c2, c3, c4 = st.columns(4)
c1.metric("📝 언급 수",   f"{int(row['mention_count'])}건")
c2.metric("⭐ 평균 점수", f"{row['avg_score']:.2f}  {stars(row['avg_score'])}")
c3.metric("⚠️ 트리거 수", f"{int(row['trigger_count'])}개")
c4.metric("📣 광고 포함", f"{int(row['ad_count'])}건")

st.divider()

# ── 2단 레이아웃 ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ── 왼쪽: 기본 정보 ───────────────────────────────────────────────────────────
with left:
    st.subheader("📊 기본 정보")

    # 점수 분포 차트
    scores = parse_score_dist(row["score_distribution"])
    if scores:
        bar_colors = [
            "#e53935" if k <= 2 else "#fb8c00" if k == 3 else "#43a047"
            for k in sorted(scores)
        ]
        fig = go.Figure(go.Bar(
            x=[f"{k}점" for k in sorted(scores)],
            y=[scores[k] for k in sorted(scores)],
            marker_color=bar_colors,
            text=[f"{v}건" for v in [scores[k] for k in sorted(scores)]],
            textposition="outside",
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(visible=False),
            xaxis=dict(tickfont=dict(size=13)),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 강점
    with st.expander("✅ 강점", expanded=True):
        for item in str(row["top_strengths"]).split(","):
            clean = re.sub(r"\(\d+\)", "", item).strip()
            if clean and clean != "nan":
                st.markdown(f"- {clean}")

    # 약점
    with st.expander("❌ 약점", expanded=True):
        for item in str(row["top_weaknesses"]).split(","):
            clean = re.sub(r"\(\d+\)", "", item).strip()
            if clean and clean != "nan":
                st.markdown(f"- {clean}")

    # 추천 타겟
    with st.expander("🎯 추천 타겟"):
        for item in str(row["target_audience"]).split(","):
            clean = re.sub(r"\(\d+\)", "", item).strip()
            if clean and clean != "nan":
                st.markdown(f"- {clean}")


# ── 오른쪽: 리스크 분석 ──────────────────────────────────────────────────────
with right:
    st.subheader("⚠️ 리스크 분석")

    # 핵심 코멘트
    if report.get("comment"):
        st.info(f"💬 **핵심 코멘트**\n\n{report['comment']}")

    # 팝업 경고 섹션
    st.markdown("**🚨 구매 전 팝업 경고**")

    popup = report.get("popup", {})

    if popup.get("type") == "new":
        tab1, tab2, tab3 = st.tabs(["💛 친절형", "🚨 경고형", "ℹ️ 정보전달형"])
        with tab1:
            st.success(popup.get("friendly", ""))
        with tab2:
            st.error(popup.get("warning", ""))
        with tab3:
            st.info(popup.get("info", ""))

    elif popup.get("type") == "legacy" and popup.get("body"):
        st.warning(popup["body"])
    else:
        st.caption("⚪ 데이터 부족으로 팝업 문구를 생성할 수 없습니다.")

    # 트리거 목록
    st.markdown("---")
    triggers = report.get("triggers", [])
    st.markdown(f"**🎯 리스크 트리거 ({len(triggers)}개)**")

    if triggers:
        for i, t in enumerate(triggers, 1):
            with st.expander(f"#{i}  {t[:40]}…" if len(t) > 40 else f"#{i}  {t}"):
                st.markdown(t)
    else:
        st.caption("트리거 없음")
