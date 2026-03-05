# =============================================================================
# 📊 제품별 공통 장단점 · 추천 대상 · Trigger 분석 리포트
# =============================================================================
# 행(Row) = 개별 제품명
# 컬럼 = 언급횟수, 평균점수, 주요장점 TOP3, 주요단점 TOP2, 추천대상, trigger 리스트
#
# 필터: 언급 3회 이상 제품만 분석
# =============================================================================

import os
import re
import base64
import io
from collections import Counter
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =============================================================================
# 0. 한글 폰트
# =============================================================================
def setup_korean_font():
    candidates = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()

# =============================================================================
# 1. 설정
# =============================================================================
INPUT_PATH = "./outputs/cushion_summary_final.csv"
OUTPUT_HTML = "./outputs/product_detail_report.html"
OUTPUT_CSV = "./outputs/product_detail_summary.csv"

MIN_MENTIONS = 3  # 최소 언급 횟수

# =============================================================================
# 2. 데이터 로드
# =============================================================================
df = pd.read_csv(INPUT_PATH)
df['recommendation_score'] = pd.to_numeric(df['recommendation_score'], errors='coerce')
df['critical_weaknesses'] = df['critical_weaknesses'].fillna('없음')
df['trigger_sentence'] = df['trigger_sentence'].fillna('없음')
df['key_strengths'] = df['key_strengths'].fillna('')
df['target_audience'] = df['target_audience'].fillna('')
df['is_ad'] = df['is_ad'].fillna('N')

print(f"📂 데이터 로드: {len(df)}행")

# =============================================================================
# 3. 키워드 추출 함수
# =============================================================================
STOPWORDS = {
    '있는', '있음', '있고', '없음', '없는', '좋은', '좋음', '좋고',
    '되는', '됨', '하는', '함', '않는', '않음', '같은', '같음',
    '수', '것', '등', '및', '중', '때', '후', '약', '더', '매우',
    '정도', '느낌', '편', '때문', '가능', '부분', '사용', '제품',
}


def extract_top_keywords(series: pd.Series, top_n: int = 3) -> list:
    """
    쉼표 구분 텍스트에서 키워드 빈도 상위 N개 추출
    Returns: [('키워드', 횟수), ...]
    """
    all_items = []
    for text in series.dropna():
        text = str(text).strip()
        if text in ('없음', '', 'nan'):
            continue
        items = re.split(r'[,，、]', text)
        for item in items:
            item = item.strip()
            if len(item) >= 2 and item not in STOPWORDS:
                all_items.append(item)
    return Counter(all_items).most_common(top_n)


def extract_target_keywords(series: pd.Series, top_n: int = 5) -> list:
    """
    추천 대상 텍스트에서 핵심 키워드 추출
    쉼표 구분 + 공백 분리 혼합 처리
    """
    all_items = []
    for text in series.dropna():
        text = str(text).strip()
        if text in ('없음', '', 'nan'):
            continue
        # 쉼표로 먼저 나누고
        chunks = re.split(r'[,，、/]', text)
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) >= 2 and chunk not in STOPWORDS:
                all_items.append(chunk)
    return Counter(all_items).most_common(top_n)


def collect_triggers(series: pd.Series) -> list:
    """유효한 trigger_sentence만 수집"""
    triggers = []
    for text in series.dropna():
        text = str(text).strip()
        if text not in ('없음', '', 'nan') and len(text) > 10:
            triggers.append(text)
    return triggers


# =============================================================================
# 4. 제품별 분석 실행
# =============================================================================
print(f"\n{'='*60}")
print(f"📊 제품별 상세 분석 (언급 {MIN_MENTIONS}회 이상)")
print(f"{'='*60}")

# 3회 이상 언급 제품 필터
product_counts = df['product_name'].value_counts()
valid_products = product_counts[product_counts >= MIN_MENTIONS].index.tolist()
print(f"  분석 대상: {len(valid_products)}개 제품 (전체 {df['product_name'].nunique()}개 중)")

product_rows = []

for product in valid_products:
    grp = df[df['product_name'] == product]

    mention_count = len(grp)
    avg_score = grp['recommendation_score'].mean()
    score_std = grp['recommendation_score'].std()
    brand = grp['brand_name'].iloc[0] if 'brand_name' in grp.columns else product.split()[0]

    # 장점 TOP 3
    top_strengths = extract_top_keywords(grp['key_strengths'], top_n=3)
    strengths_str = ', '.join([f"{kw}({cnt})" for kw, cnt in top_strengths]) if top_strengths else '없음'

    # 단점 TOP 2
    top_weaknesses = extract_top_keywords(grp['critical_weaknesses'], top_n=2)
    weaknesses_str = ', '.join([f"{kw}({cnt})" for kw, cnt in top_weaknesses]) if top_weaknesses else '없음'

    # 추천 대상 TOP 5
    top_targets = extract_target_keywords(grp['target_audience'], top_n=5)
    target_str = ', '.join([f"{kw}({cnt})" for kw, cnt in top_targets]) if top_targets else '없음'

    # Trigger 수집
    triggers = collect_triggers(grp['trigger_sentence'])
    trigger_count = len(triggers)

    # 광고 비율
    ad_count = (grp['is_ad'] == 'Y').sum()

    # 점수 분포
    score_dist = grp['recommendation_score'].value_counts().sort_index()
    score_dist_str = ' / '.join([f"{int(k)}점:{int(v)}건" for k, v in score_dist.items()])

    product_rows.append({
        'brand': brand,
        'product_name': product,
        'mention_count': mention_count,
        'avg_score': round(avg_score, 2),
        'score_std': round(score_std, 2) if pd.notna(score_std) else 0,
        'score_distribution': score_dist_str,
        'top_strengths': strengths_str,
        'top_weaknesses': weaknesses_str,
        'target_audience': target_str,
        'trigger_count': trigger_count,
        'triggers': ' ||| '.join(triggers) if triggers else '없음',
        'ad_count': ad_count,
        # 원본 리스트 (HTML 리포트용)
        '_strengths_list': top_strengths,
        '_weaknesses_list': top_weaknesses,
        '_target_list': top_targets,
        '_triggers_raw': triggers,
    })

# DataFrame 생성
summary_df = pd.DataFrame(product_rows)
summary_df = summary_df.sort_values('mention_count', ascending=False).reset_index(drop=True)

# CSV 저장 (내부 리스트 컬럼 제외)
csv_cols = [c for c in summary_df.columns if not c.startswith('_')]
summary_df[csv_cols].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n💾 CSV 저장: {OUTPUT_CSV}")

# 콘솔 출력
print(f"\n{'─'*60}")
for _, row in summary_df.head(5).iterrows():
    print(f"\n🧴 {row['product_name']} (언급 {row['mention_count']}회, 평균 {row['avg_score']}점)")
    print(f"   장점: {row['top_strengths']}")
    print(f"   단점: {row['top_weaknesses']}")
    print(f"   대상: {row['target_audience']}")
    print(f"   Trigger: {row['trigger_count']}개 | 광고: {row['ad_count']}건")


# =============================================================================
# 5. 차트 생성 함수
# =============================================================================
COLORS = {
    'primary': '#6C5CE7', 'secondary': '#00B894', 'accent': '#FD79A8',
    'warning': '#FDCB6E', 'dark': '#2D3436',
    'palette': ['#6C5CE7', '#00B894', '#FD79A8', '#FDCB6E', '#0984E3',
                '#E17055', '#00CEC9', '#A29BFE', '#FAB1A0', '#81ECEC'],
    'score_colors': {1: '#FF6B6B', 2: '#FFA502', 3: '#FDCB6E', 4: '#00B894', 5: '#6C5CE7'},
}

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def make_product_chart(row) -> str:
    """개별 제품의 미니 차트: 점수 분포 + 장점 바"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

    # 왼쪽: 점수 분포 바
    ax1 = axes[0]
    dist = row['score_distribution']
    scores, counts = [], []
    for item in dist.split(' / '):
        parts = item.split(':')
        s = int(parts[0].replace('점', ''))
        c = int(parts[1].replace('건', ''))
        scores.append(s)
        counts.append(c)

    bar_colors = [COLORS['score_colors'].get(s, '#DDD') for s in scores]
    ax1.bar([str(s) for s in scores], counts, color=bar_colors, width=0.6, edgecolor='white')
    ax1.set_title('점수 분포', fontsize=11, fontweight='bold')
    ax1.set_xlabel('점수', fontsize=9)
    ax1.set_ylabel('건수', fontsize=9)
    for i, c in enumerate(counts):
        ax1.text(i, c + 0.1, str(c), ha='center', fontsize=10, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 오른쪽: 장점 키워드 바
    ax2 = axes[1]
    strengths = row['_strengths_list']
    if strengths:
        labels = [kw for kw, _ in strengths]
        vals = [cnt for _, cnt in strengths]
        bars = ax2.barh(range(len(labels)), vals, color=COLORS['secondary'], height=0.5)
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=10)
        ax2.invert_yaxis()
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     str(v), va='center', fontsize=10, fontweight='bold')
    ax2.set_title('주요 장점 빈도', fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(f"{row['product_name']}", fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig_to_base64(fig)


# =============================================================================
# 6. HTML 리포트 생성
# =============================================================================
print("\n📝 HTML 리포트 생성 중...")

now = datetime.now().strftime("%Y-%m-%d %H:%M")

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>제품별 상세 분석 리포트</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap" rel="stylesheet">
<style>
:root {{
  --primary: #6C5CE7; --secondary: #00B894; --accent: #FD79A8;
  --warning: #FDCB6E; --dark: #2D3436; --light: #F8F9FA;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: 'Noto Sans KR', sans-serif;
  background: #F5F3FF;
  color: var(--dark);
  line-height: 1.7;
}}
.header {{
  background: linear-gradient(135deg, #6C5CE7, #A29BFE);
  color: white;
  padding: 50px 40px 35px;
  text-align: center;
}}
.header h1 {{ font-size: 2rem; font-weight: 900; margin-bottom: 6px; }}
.header p {{ font-size: 0.92rem; opacity: 0.85; }}

.container {{ max-width: 1100px; margin: 30px auto; padding: 0 20px 60px; }}

/* ── 제품 카드 ── */
.product-card {{
  background: white;
  border-radius: 18px;
  padding: 28px 32px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  border-left: 5px solid var(--primary);
}}
.product-card.top5 {{ border-left-color: var(--secondary); }}

.card-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 16px;
}}
.card-title {{
  font-size: 1.25rem;
  font-weight: 800;
}}
.card-title .brand {{
  color: #999;
  font-weight: 400;
  font-size: 0.85rem;
  margin-left: 6px;
}}
.card-meta {{
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}}
.meta-badge {{
  display: inline-block;
  padding: 4px 14px;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 600;
}}
.meta-badge.mentions {{ background: #EDE7FF; color: var(--primary); }}
.meta-badge.score {{ background: #E8FFF3; color: var(--secondary); }}
.meta-badge.std {{ background: #FFF3E0; color: #E17055; }}
.meta-badge.ad {{ background: #FFE0EC; color: var(--accent); }}
.meta-badge.trigger {{ background: #FFF8E1; color: #F39C12; }}

/* ── 그리드 ── */
.card-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
}}
@media (max-width: 700px) {{ .card-grid {{ grid-template-columns: 1fr; }} }}

.card-section {{
  background: var(--light);
  border-radius: 12px;
  padding: 14px 18px;
}}
.card-section h4 {{
  font-size: 0.85rem;
  font-weight: 700;
  margin-bottom: 8px;
  color: #666;
}}

/* ── 키워드 태그 ── */
.kw-tag {{
  display: inline-block;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 0.8rem;
  margin: 2px 3px;
  font-weight: 500;
  color: white;
}}
.kw-tag.strength {{ background: var(--secondary); }}
.kw-tag.weakness {{ background: var(--accent); }}
.kw-tag.target {{ background: var(--primary); }}

.kw-count {{
  font-size: 0.7rem;
  opacity: 0.8;
  margin-left: 2px;
}}

/* ── Trigger 인용 ── */
.trigger-box {{
  background: #FFFCF0;
  border-left: 3px solid var(--warning);
  padding: 10px 14px;
  margin: 6px 0;
  border-radius: 0 8px 8px 0;
  font-size: 0.82rem;
  color: #555;
  line-height: 1.5;
}}

/* ── 차트 ── */
.chart-img {{
  width: 100%;
  max-width: 700px;
  display: block;
  margin: 12px auto;
  border-radius: 10px;
}}

/* ── 요약 테이블 ── */
.summary-section {{
  background: white;
  border-radius: 18px;
  padding: 28px 32px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}}
.summary-section h2 {{
  font-size: 1.2rem;
  font-weight: 800;
  margin-bottom: 16px;
}}
table.summary-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}}
table.summary-table th {{
  background: var(--light);
  padding: 10px 12px;
  text-align: left;
  font-weight: 700;
  border-bottom: 2px solid #E0E0E0;
  white-space: nowrap;
}}
table.summary-table td {{
  padding: 8px 12px;
  border-bottom: 1px solid #F0F0F0;
  vertical-align: top;
}}
table.summary-table tr:hover td {{ background: #FAFAFE; }}
.score-bar {{
  display: inline-block;
  height: 8px;
  border-radius: 4px;
  margin-right: 4px;
}}

.footer {{
  text-align: center;
  padding: 30px;
  color: #AAA;
  font-size: 0.78rem;
}}
</style>
</head>
<body>

<div class="header">
  <h1>🧴 제품별 상세 분석 리포트</h1>
  <p>개별 제품 단위 공통 장단점 · 추천 대상 · 판단 피로 Trigger | {now}</p>
  <p style="margin-top:6px;">분석 대상: {len(valid_products)}개 제품 (언급 {MIN_MENTIONS}회 이상) | 총 {len(df)}건 리뷰 / {df['video_id'].nunique()}개 영상</p>
</div>

<div class="container">
"""

# ── 요약 테이블 ──
html += """
<div class="summary-section">
  <h2>📋 전체 요약 테이블</h2>
  <div style="overflow-x:auto;">
  <table class="summary-table">
    <tr>
      <th>#</th><th>제품명</th><th>언급</th><th>평균</th><th>편차</th>
      <th>주요 장점 TOP 3</th><th>주요 단점 TOP 2</th><th>추천 대상</th>
      <th>Trigger</th><th>광고</th>
    </tr>
"""

for i, row in summary_df.iterrows():
    # 점수 색상
    sc = row['avg_score']
    sc_color = '#00B894' if sc >= 4 else '#FDCB6E' if sc >= 3 else '#FD79A8'

    # 장점 태그
    str_tags = ''
    for kw, cnt in row['_strengths_list']:
        str_tags += f'<span class="kw-tag strength">{kw}<span class="kw-count">({cnt})</span></span>'
    if not str_tags:
        str_tags = '<span style="color:#aaa">없음</span>'

    # 단점 태그
    weak_tags = ''
    for kw, cnt in row['_weaknesses_list']:
        weak_tags += f'<span class="kw-tag weakness">{kw}<span class="kw-count">({cnt})</span></span>'
    if not weak_tags:
        weak_tags = '<span style="color:#aaa">없음</span>'

    # 추천 대상 (상위 3개만)
    tgt = ', '.join([kw for kw, _ in row['_target_list'][:3]]) or '없음'

    html += f"""
    <tr>
      <td>{i+1}</td>
      <td><b>{row['product_name']}</b></td>
      <td style="text-align:center">{row['mention_count']}</td>
      <td style="text-align:center;color:{sc_color};font-weight:700">{sc}</td>
      <td style="text-align:center">{row['score_std']}</td>
      <td>{str_tags}</td>
      <td>{weak_tags}</td>
      <td style="font-size:0.78rem">{tgt}</td>
      <td style="text-align:center">{row['trigger_count']}</td>
      <td style="text-align:center">{row['ad_count']}</td>
    </tr>"""

html += """
  </table>
  </div>
</div>
"""

# ── 개별 제품 카드 (상위 20개) ──
print("📈 제품별 차트 생성 중...")
card_count = min(20, len(summary_df))

for i, (_, row) in enumerate(summary_df.head(card_count).iterrows()):
    is_top5 = i < 5
    card_class = 'product-card top5' if is_top5 else 'product-card'

    # 차트 생성
    chart_b64 = make_product_chart(row)

    # 장점 태그
    str_tags = ''
    for kw, cnt in row['_strengths_list']:
        str_tags += f'<span class="kw-tag strength">{kw} <span class="kw-count">({cnt})</span></span> '

    # 단점 태그
    weak_tags = ''
    for kw, cnt in row['_weaknesses_list']:
        weak_tags += f'<span class="kw-tag weakness">{kw} <span class="kw-count">({cnt})</span></span> '
    if not weak_tags.strip():
        weak_tags = '<span style="color:#ccc;">언급 없음</span>'

    # 추천 대상 태그
    tgt_tags = ''
    for kw, cnt in row['_target_list']:
        tgt_tags += f'<span class="kw-tag target" style="opacity:{min(1, 0.5 + cnt/max(1, row["_target_list"][0][1])*0.5)}">{kw} <span class="kw-count">({cnt})</span></span> '

    # Trigger 인용 (최대 3개)
    trigger_html = ''
    for t in row['_triggers_raw'][:3]:
        t_short = t[:180] + '...' if len(t) > 180 else t
        trigger_html += f'<div class="trigger-box">"{t_short}"</div>'
    if not trigger_html:
        trigger_html = '<div style="color:#ccc; font-size:0.82rem;">Trigger 문장 없음</div>'

    html += f"""
<div class="{card_class}">
  <div class="card-header">
    <div class="card-title">
      {'🏆 ' if is_top5 else ''}{row['product_name']}
      <span class="brand">{row['brand']}</span>
    </div>
    <div class="card-meta">
      <span class="meta-badge mentions">언급 {row['mention_count']}회</span>
      <span class="meta-badge score">평균 {row['avg_score']}점</span>
      <span class="meta-badge std">편차 {row['score_std']}</span>
      {'<span class="meta-badge ad">광고 ' + str(row['ad_count']) + '건</span>' if row['ad_count'] > 0 else ''}
      <span class="meta-badge trigger">Trigger {row['trigger_count']}개</span>
    </div>
  </div>

  <img class="chart-img" src="data:image/png;base64,{chart_b64}">

  <div class="card-grid">
    <div class="card-section">
      <h4>✅ 공통 장점 TOP 3</h4>
      {str_tags if str_tags.strip() else '<span style="color:#ccc;">없음</span>'}
    </div>
    <div class="card-section">
      <h4>❌ 공통 단점 TOP 2</h4>
      {weak_tags}
    </div>
  </div>

  <div class="card-grid">
    <div class="card-section">
      <h4>👤 추천 대상</h4>
      {tgt_tags if tgt_tags.strip() else '<span style="color:#ccc;">없음</span>'}
    </div>
    <div class="card-section">
      <h4>⚡ 판단 피로 Trigger ({row['trigger_count']}개)</h4>
      {trigger_html}
    </div>
  </div>
</div>
"""
    print(f"  ✅ [{i+1}/{card_count}] {row['product_name']}")

# ── Footer ──
html += f"""
</div>
<div class="footer">
  📊 Product Detail Report | {now} | {len(valid_products)}개 제품 / {len(df)}건 리뷰 / {df['video_id'].nunique()}영상
</div>
</body>
</html>"""

# 저장
with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n{'='*60}")
print(f"🎉 완료!")
print(f"  📄 HTML 리포트: {OUTPUT_HTML}")
print(f"  📊 CSV 요약: {OUTPUT_CSV}")
print(f"  📦 분석 제품: {len(valid_products)}개 ({card_count}개 카드 생성)")
print(f"{'='*60}")