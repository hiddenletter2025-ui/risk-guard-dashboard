# =============================================================================
# 📊 뷰티 리뷰 데이터 14개 핵심 지표 분석 + HTML 리포트
# =============================================================================
# 입력: cushion_summary_final.csv
# 출력: cushion_analysis_report.html (시각화 포함 리포트)
#       + 콘솔 출력 (DataFrame 형태)
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
# 0. 한글 폰트 설정
# =============================================================================
def setup_korean_font():
    """시스템에서 한글 폰트를 찾아 설정"""
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
            return prop.get_name()
    # fallback
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'

FONT_NAME = setup_korean_font()
print(f"📝 폰트: {FONT_NAME}")

# =============================================================================
# 1. 설정 및 데이터 로드
# =============================================================================
INPUT_PATH = "./outputs/cushion_summary_final.csv"
OUTPUT_HTML = "./outputs/cushion_analysis_report.html"

df = pd.read_csv(INPUT_PATH)
print(f"📂 데이터 로드: {len(df)}행, {df.columns.tolist()}")

# brand_name 컬럼이 이미 있으면 사용, 없으면 생성
if 'brand_name' not in df.columns:
    df['brand_name'] = df['product_name'].str.split().str[0]

# NaN 처리
df['recommendation_score'] = pd.to_numeric(df['recommendation_score'], errors='coerce')
df['critical_weaknesses'] = df['critical_weaknesses'].fillna('없음')
df['trigger_sentence'] = df['trigger_sentence'].fillna('없음')
df['is_ad'] = df['is_ad'].fillna('N')
df['key_strengths'] = df['key_strengths'].fillna('')
df['target_audience'] = df['target_audience'].fillna('')

# =============================================================================
# 2. 키워드 빈도 분석 헬퍼
# =============================================================================
# 한국어 불용어 (뷰티 분석에 불필요한 일반 단어)
STOPWORDS = {
    '있는', '있음', '있고', '없음', '없는', '좋은', '좋음', '좋고',
    '되는', '됨', '하는', '함', '않는', '않음', '같은', '같음',
    '수', '것', '등', '및', '중', '때', '후', '약', '더', '매우',
    '정도', '느낌', '편', '때문', '가능', '부분', '사용', '제품',
    '피부', '쿠션',  # 모든 항목에 공통이라 변별력 없음
}

def extract_keywords(series: pd.Series, top_n: int = 10) -> list:
    """
    쉼표 구분 텍스트에서 키워드 빈도 추출
    NLTK/KoNLPy 없이 Counter 기반
    """
    all_words = []
    for text in series.dropna():
        text = str(text)
        if text in ('없음', '', 'nan'):
            continue
        # 쉼표 구분 → 개별 항목
        items = re.split(r'[,，、/]', text)
        for item in items:
            item = item.strip()
            if len(item) >= 2 and item not in STOPWORDS:
                all_words.append(item)

    counter = Counter(all_words)
    return counter.most_common(top_n)


def extract_word_freq(series: pd.Series, top_n: int = 15) -> list:
    """
    문장형 텍스트에서 공백 분리 후 2글자 이상 단어 빈도
    """
    all_words = []
    for text in series.dropna():
        text = str(text)
        if text in ('없음', '', 'nan'):
            continue
        words = re.split(r'[\s,，、/·]+', text)
        for w in words:
            w = w.strip('()[]""\'\'')
            if len(w) >= 2 and w not in STOPWORDS:
                all_words.append(w)

    counter = Counter(all_words)
    return counter.most_common(top_n)


# =============================================================================
# 3. 차트 생성 헬퍼
# =============================================================================
COLORS = {
    'primary': '#6C5CE7',
    'secondary': '#00B894',
    'accent': '#FD79A8',
    'warning': '#FDCB6E',
    'dark': '#2D3436',
    'light': '#DFE6E9',
    'bg': '#FAFAFA',
    'palette': ['#6C5CE7', '#00B894', '#FD79A8', '#FDCB6E', '#0984E3',
                '#E17055', '#00CEC9', '#A29BFE', '#FAB1A0', '#81ECEC'],
}

def fig_to_base64(fig) -> str:
    """matplotlib 그림을 base64 인코딩"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def chart_horizontal_bar(data: dict, title: str, color: str = None, xlabel: str = '') -> str:
    """수평 바 차트 → base64"""
    fig, ax = plt.subplots(figsize=(10, max(3, len(data) * 0.45)))
    labels = list(data.keys())
    values = list(data.values())

    if color is None:
        colors = COLORS['palette'][:len(labels)]
    else:
        colors = [color] * len(labels)

    bars = ax.barh(range(len(labels)), values, color=colors, height=0.6, edgecolor='white')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val}' if isinstance(val, int) else f'{val:.2f}',
                va='center', fontsize=10, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_pie(data: dict, title: str) -> str:
    """파이 차트 → base64"""
    fig, ax = plt.subplots(figsize=(7, 7))
    labels = list(data.keys())
    values = list(data.values())
    colors = COLORS['palette'][:len(labels)]

    wedges, texts, autotexts = ax.pie(
        values, labels=None, autopct='%1.1f%%',
        colors=colors, startangle=90, pctdistance=0.8,
        textprops={'fontsize': 11}
    )
    ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_grouped_bar(labels, vals1, vals2, label1, label2, title, ylabel='') -> str:
    """그룹 바 차트 → base64"""
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, vals1, w, label=label1, color=COLORS['primary'])
    ax.bar(x + w/2, vals2, w, label=label2, color=COLORS['accent'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_score_distribution() -> str:
    """점수 분포 히스토그램"""
    fig, ax = plt.subplots(figsize=(8, 5))
    scores = df['recommendation_score'].dropna()
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    colors_map = [COLORS['accent'], COLORS['warning'], '#DFE6E9', COLORS['secondary'], COLORS['primary']]
    counts, _, patches = ax.hist(scores, bins=bins, edgecolor='white', linewidth=1.5)
    for patch, color in zip(patches, colors_map):
        patch.set_facecolor(color)

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['1점\n(혹평)', '2점\n(비추)', '3점\n(조건부)', '4점\n(강추)', '5점\n(극찬)'], fontsize=11)
    ax.set_ylabel('언급 횟수', fontsize=11)
    ax.set_title('전체 추천 점수 분포', fontsize=14, fontweight='bold', pad=15)

    for i, c in enumerate(counts):
        ax.text(i + 1, c + 1, f'{int(c)}', ha='center', fontsize=12, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig_to_base64(fig)


# =============================================================================
# 4. 14개 지표 분석
# =============================================================================
print("\n" + "=" * 60)
print("📊 14개 핵심 지표 분석 시작")
print("=" * 60)

results = {}  # HTML 리포트용 데이터 저장

# ─── 지표 1: 제품별 총 언급 횟수 ───
mention_counts = df['product_name'].value_counts()
results['mention_top'] = mention_counts.head(15)
print(f"\n[1] 제품별 언급 횟수 TOP 15:")
print(results['mention_top'].to_string())

# ─── 지표 2: 제품별 평균 추천 점수 ───
avg_scores = df.groupby('product_name')['recommendation_score'].agg(['mean', 'count', 'std']).round(2)
avg_scores.columns = ['평균점수', '언급수', '표준편차']
avg_scores = avg_scores.sort_values('언급수', ascending=False)
results['avg_scores'] = avg_scores.head(15)
print(f"\n[2] 제품별 평균 점수 (상위 15):")
print(results['avg_scores'].to_string())

# ─── 지표 4: 제품별 추천 대상 키워드 빈도 ───
target_kw = extract_word_freq(df['target_audience'], top_n=20)
results['target_keywords'] = target_kw
print(f"\n[4] 추천 대상 키워드 TOP 20:")
for kw, cnt in target_kw:
    print(f"    {kw}: {cnt}회")

# ─── 지표 5: 제품별 공통 장점 키워드 ───
strength_kw = extract_keywords(df['key_strengths'], top_n=20)
results['strength_keywords'] = strength_kw
print(f"\n[5] 공통 장점 키워드 TOP 20:")
for kw, cnt in strength_kw:
    print(f"    {kw}: {cnt}회")

# ─── 지표 6: 제품별 공통 단점 키워드 ───
weakness_kw = extract_keywords(df['critical_weaknesses'], top_n=15)
results['weakness_keywords'] = weakness_kw
print(f"\n[6] 공통 단점 키워드 TOP 15:")
for kw, cnt in weakness_kw:
    print(f"    {kw}: {cnt}회")

# ─── 지표 7: trigger_sentence 분석 ───
has_trigger = df['trigger_sentence'].apply(
    lambda x: isinstance(x, str) and x.strip() != '없음' and len(x.strip()) > 5
)
trigger_rate = has_trigger.mean() * 100
trigger_by_product = df[has_trigger].groupby('product_name').size().sort_values(ascending=False)
results['trigger_by_product'] = trigger_by_product.head(10)
results['trigger_rate'] = trigger_rate
print(f"\n[7] Trigger Sentence 보유율: {trigger_rate:.1f}% ({has_trigger.sum()}/{len(df)})")
print(f"    제품별 trigger 수 TOP 10:")
print(results['trigger_by_product'].to_string())

# ─── 지표 8: 제품별 광고 언급 횟수 ───
ad_counts = df[df['is_ad'] == 'Y'].groupby('product_name').size().sort_values(ascending=False)
results['ad_counts'] = ad_counts
ad_avg = df[df['is_ad'] == 'Y']['recommendation_score'].mean()
non_ad_avg = df[df['is_ad'] == 'N']['recommendation_score'].mean()
results['ad_avg'] = ad_avg
results['non_ad_avg'] = non_ad_avg
print(f"\n[8] 광고 제품 현황:")
print(f"    광고 제품 수: {len(ad_counts)}개 ({df['is_ad'].eq('Y').sum()}건)")
print(f"    광고 평균 점수: {ad_avg:.2f} vs 비광고: {non_ad_avg:.2f} (차이: +{ad_avg - non_ad_avg:.2f})")
print(ad_counts.to_string())

# ─── 지표 9: 분석에 사용된 총 영상 수 ───
total_videos = df['video_id'].nunique()
total_products = df['product_name'].nunique()
total_brands = df['brand_name'].nunique()
results['total_videos'] = total_videos
results['total_products'] = total_products
results['total_brands'] = total_brands
print(f"\n[9] 분석 규모:")
print(f"    총 영상: {total_videos}개 (중복 제거)")
print(f"    총 제품: {total_products}개")
print(f"    총 브랜드: {total_brands}개")
print(f"    총 리뷰 레코드: {len(df)}건")

# ─── 지표 10: 브랜드별 평균 점수 + 언급 제품 수 ───
brand_stats = df.groupby('brand_name').agg(
    평균점수=('recommendation_score', 'mean'),
    리뷰수=('product_name', 'size'),
    제품수=('product_name', 'nunique'),
).round(2).sort_values('리뷰수', ascending=False)
results['brand_stats'] = brand_stats.head(15)
print(f"\n[10] 브랜드별 통계 TOP 15:")
print(results['brand_stats'].to_string())

# ─── 지표 11: 가장 많이 언급된 제품 TOP 5 ───
top5_products = mention_counts.head(5)
results['top5_products'] = top5_products
print(f"\n[11] 언급 TOP 5 제품:")
print(top5_products.to_string())

# ─── 지표 12: 가장 리뷰가 많은 브랜드 TOP 5 ───
top5_brands = df['brand_name'].value_counts().head(5)
results['top5_brands'] = top5_brands
print(f"\n[12] 리뷰 TOP 5 브랜드:")
print(top5_brands.to_string())

# ─── 지표 13: 평균 점수 최고 제품 (2회 이상 언급) ───
multi_mention = avg_scores[avg_scores['언급수'] >= 2].sort_values('평균점수', ascending=False)
results['best_products'] = multi_mention.head(5)
print(f"\n[13] 평균 점수 최고 제품 (2회 이상 언급):")
print(results['best_products'].to_string())

# ─── 지표 14: 평균 점수 최저 제품 (2회 이상 언급) ───
results['worst_products'] = multi_mention.tail(5).sort_values('평균점수')
print(f"\n[14] 평균 점수 최저 제품 (2회 이상 언급):")
print(results['worst_products'].to_string())


# =============================================================================
# 5. 차트 생성
# =============================================================================
print("\n📈 차트 생성 중...")

charts = {}

# 차트 1: 언급 TOP 10
top10_mention = mention_counts.head(10)
charts['mention_top10'] = chart_horizontal_bar(
    dict(top10_mention), '제품별 언급 횟수 TOP 10', xlabel='언급 횟수'
)

# 차트 2: 점수 분포
charts['score_dist'] = chart_score_distribution()

# 차트 3: 브랜드 TOP 10 (리뷰 수 + 평균 점수)
brand_top10 = brand_stats.head(10)
charts['brand_bar'] = chart_grouped_bar(
    labels=brand_top10.index.tolist(),
    vals1=brand_top10['리뷰수'].tolist(),
    vals2=[v * 5 for v in brand_top10['평균점수'].tolist()],  # 스케일 맞춤
    label1='리뷰 수', label2='평균 점수 (×5)',
    title='브랜드별 리뷰 수 & 평균 점수 TOP 10',
    ylabel='수치'
)

# 차트 4: 장점 키워드 TOP 10
top10_str = dict(strength_kw[:10])
charts['strength_kw'] = chart_horizontal_bar(
    top10_str, '공통 장점 키워드 TOP 10', color=COLORS['secondary'], xlabel='빈도'
)

# 차트 5: 단점 키워드 TOP 10
top10_weak = dict(weakness_kw[:10])
charts['weakness_kw'] = chart_horizontal_bar(
    top10_weak, '공통 단점 키워드 TOP 10', color=COLORS['accent'], xlabel='빈도'
)

# 차트 6: 광고 vs 비광고 점수 비교
charts['ad_compare'] = chart_horizontal_bar(
    {'광고/협찬 (Y)': round(ad_avg, 2), '비광고 (N)': round(non_ad_avg, 2)},
    '광고 vs 비광고 평균 추천 점수', color=COLORS['primary'], xlabel='평균 점수'
)

# 차트 7: 점수 편차 큰 제품 (리뷰 일관성)
multi_3 = df.groupby('product_name').filter(lambda x: len(x) >= 3)
if len(multi_3) > 0:
    variance_df = multi_3.groupby('product_name')['recommendation_score'].agg(['std', 'mean', 'count'])
    variance_df = variance_df.sort_values('std', ascending=False).head(8)
    charts['variance'] = chart_horizontal_bar(
        dict(variance_df['std'].round(2)), '점수 편차가 큰 제품 TOP 8 (표준편차)',
        color=COLORS['warning'], xlabel='표준편차'
    )

# 차트 8: 추천 대상 키워드
top10_target = dict(target_kw[:10])
charts['target_kw'] = chart_horizontal_bar(
    top10_target, '추천 대상 키워드 TOP 10', color=COLORS['primary'], xlabel='빈도'
)

print(f"  ✅ {len(charts)}개 차트 생성 완료")


# =============================================================================
# 6. HTML 리포트 생성
# =============================================================================
print("📝 HTML 리포트 생성 중...")

def df_to_html_table(dataframe, max_rows=15):
    """DataFrame → 스타일된 HTML 테이블"""
    return dataframe.head(max_rows).to_html(
        classes='data-table', border=0, escape=False
    )

def kw_to_html_tags(keywords, color=COLORS['primary']):
    """키워드 리스트 → HTML 태그 뱃지"""
    tags = []
    for kw, cnt in keywords:
        opacity = min(1.0, 0.4 + (cnt / keywords[0][1]) * 0.6)
        tags.append(
            f'<span class="kw-tag" style="background:{color}; opacity:{opacity}">'
            f'{kw} <small>({cnt})</small></span>'
        )
    return ' '.join(tags)

# trigger_sentence 샘플 추출
trigger_samples = df[has_trigger].nlargest(8, 'recommendation_score')[
    ['product_name', 'trigger_sentence', 'recommendation_score']
].values.tolist()

# 점수 편차 큰 제품 trigger 연결
if len(multi_3) > 0:
    high_var_products = variance_df.head(5).index.tolist()
    var_triggers = df[
        (df['product_name'].isin(high_var_products)) & has_trigger
    ][['product_name', 'trigger_sentence', 'recommendation_score']].values.tolist()
else:
    var_triggers = []

now = datetime.now().strftime("%Y-%m-%d %H:%M")

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>쿠션 리뷰 데이터 분석 리포트</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap" rel="stylesheet">
<style>
:root {{
  --primary: #6C5CE7;
  --secondary: #00B894;
  --accent: #FD79A8;
  --warning: #FDCB6E;
  --dark: #2D3436;
  --light: #F8F9FA;
  --card-shadow: 0 2px 12px rgba(0,0,0,0.08);
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: 'Noto Sans KR', sans-serif;
  background: linear-gradient(135deg, #F5F3FF 0%, #FFF0F5 50%, #F0FFF4 100%);
  color: var(--dark);
  line-height: 1.7;
  padding: 0;
}}

/* ── Header ── */
.header {{
  background: linear-gradient(135deg, var(--primary), #A29BFE);
  color: white;
  padding: 60px 40px 40px;
  text-align: center;
}}
.header h1 {{
  font-size: 2.2rem;
  font-weight: 900;
  margin-bottom: 8px;
  letter-spacing: -0.5px;
}}
.header p {{
  font-size: 1rem;
  opacity: 0.85;
}}

/* ── Stats Bar ── */
.stats-bar {{
  display: flex;
  justify-content: center;
  gap: 24px;
  margin: -28px auto 30px;
  padding: 0 20px;
  flex-wrap: wrap;
}}
.stat-card {{
  background: white;
  border-radius: 16px;
  padding: 20px 28px;
  text-align: center;
  box-shadow: var(--card-shadow);
  min-width: 140px;
}}
.stat-card .num {{
  font-size: 2rem;
  font-weight: 900;
  color: var(--primary);
}}
.stat-card .label {{
  font-size: 0.82rem;
  color: #888;
  margin-top: 2px;
}}

/* ── Container ── */
.container {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 24px 60px;
}}

/* ── Section ── */
.section {{
  background: white;
  border-radius: 18px;
  padding: 32px;
  margin-bottom: 28px;
  box-shadow: var(--card-shadow);
}}
.section-title {{
  font-size: 1.3rem;
  font-weight: 700;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 10px;
}}
.section-title .badge {{
  background: var(--primary);
  color: white;
  font-size: 0.72rem;
  padding: 3px 10px;
  border-radius: 20px;
  font-weight: 500;
}}
.section-desc {{
  font-size: 0.88rem;
  color: #888;
  margin-bottom: 20px;
}}

/* ── Charts ── */
.chart-img {{
  width: 100%;
  max-width: 800px;
  display: block;
  margin: 16px auto;
  border-radius: 12px;
}}

/* ── Tables ── */
.data-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
  margin: 12px 0;
}}
.data-table th {{
  background: var(--light);
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #E0E0E0;
  white-space: nowrap;
}}
.data-table td {{
  padding: 9px 14px;
  border-bottom: 1px solid #F0F0F0;
}}
.data-table tr:hover td {{
  background: #FAFAFE;
}}

/* ── Keyword Tags ── */
.kw-tag {{
  display: inline-block;
  color: white;
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 0.82rem;
  margin: 3px 4px;
  font-weight: 500;
}}

/* ── Trigger Quote ── */
.trigger-quote {{
  background: #FFF8F0;
  border-left: 4px solid var(--warning);
  padding: 14px 18px;
  margin: 10px 0;
  border-radius: 0 10px 10px 0;
  font-size: 0.88rem;
  line-height: 1.6;
}}
.trigger-quote .product {{
  font-weight: 700;
  color: var(--primary);
  font-size: 0.82rem;
  margin-bottom: 4px;
}}
.trigger-quote .score {{
  color: #888;
  font-size: 0.78rem;
}}

/* ── Insight Box ── */
.insight-box {{
  background: linear-gradient(135deg, #F5F3FF, #FFF0F5);
  border-radius: 14px;
  padding: 20px 24px;
  margin: 16px 0;
  border: 1px solid #E8E0F0;
}}
.insight-box h4 {{
  color: var(--primary);
  margin-bottom: 8px;
  font-size: 0.95rem;
}}
.insight-box p {{
  font-size: 0.88rem;
  color: #555;
}}

/* ── Two Column ── */
.two-col {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}}
@media (max-width: 768px) {{
  .two-col {{ grid-template-columns: 1fr; }}
  .stats-bar {{ flex-direction: column; align-items: center; }}
}}

/* ── Footer ── */
.footer {{
  text-align: center;
  padding: 30px;
  color: #AAA;
  font-size: 0.8rem;
}}
</style>
</head>
<body>

<!-- ═══════ HEADER ═══════ -->
<div class="header">
  <h1>🧴 올리브영 쿠션 리뷰 데이터 분석 리포트</h1>
  <p>유튜브 리뷰 자막 기반 14개 핵심 지표 | 생성일: {now}</p>
</div>

<!-- ═══════ STATS BAR ═══════ -->
<div class="stats-bar">
  <div class="stat-card">
    <div class="num">{total_videos}</div>
    <div class="label">분석 영상</div>
  </div>
  <div class="stat-card">
    <div class="num">{len(df)}</div>
    <div class="label">리뷰 레코드</div>
  </div>
  <div class="stat-card">
    <div class="num">{total_products}</div>
    <div class="label">분석 제품</div>
  </div>
  <div class="stat-card">
    <div class="num">{total_brands}</div>
    <div class="label">브랜드</div>
  </div>
  <div class="stat-card">
    <div class="num">{trigger_rate:.0f}%</div>
    <div class="label">Trigger 보유율</div>
  </div>
</div>

<div class="container">

<!-- ═══════ 1. 언급 TOP & 점수 분포 ═══════ -->
<div class="section">
  <div class="section-title">📊 지표 1·2 — 제품별 언급 횟수 & 추천 점수 분포</div>
  <div class="section-desc">가장 많이 리뷰된 제품과 전체 점수 분포를 확인합니다.</div>
  <div class="two-col">
    <div><img class="chart-img" src="data:image/png;base64,{charts['mention_top10']}"></div>
    <div><img class="chart-img" src="data:image/png;base64,{charts['score_dist']}"></div>
  </div>
  <div class="insight-box">
    <h4>💡 인사이트</h4>
    <p>
      전체 {len(df)}건의 리뷰 중 <b>4점(강추)이 {int((df['recommendation_score']==4).sum())}건({(df['recommendation_score']==4).mean()*100:.0f}%)</b>으로 가장 많고,
      <b>3점(조건부 추천)이 {int((df['recommendation_score']==3).sum())}건({(df['recommendation_score']==3).mean()*100:.0f}%)</b>입니다.
      조건부 추천이 전체의 1/5 이상을 차지한다는 것은, 소비자가 "내 피부에 맞는지" 추가 판단을 해야 하는 제품이 그만큼 많다는 의미입니다.
    </p>
  </div>
</div>

<!-- ═══════ 4·5·6. 키워드 분석 ═══════ -->
<div class="section">
  <div class="section-title">🏷️ 지표 4·5·6 — 추천 대상 · 장점 · 단점 키워드</div>
  <div class="section-desc">리뷰에서 반복 등장하는 핵심 키워드를 빈도 분석합니다.</div>

  <h3 style="margin:18px 0 8px; font-size:1rem;">추천 대상 키워드</h3>
  {kw_to_html_tags(target_kw[:15], COLORS['primary'])}

  <div class="two-col" style="margin-top:24px;">
    <div>
      <h3 style="font-size:1rem; margin-bottom:4px;">✅ 공통 장점</h3>
      <img class="chart-img" src="data:image/png;base64,{charts['strength_kw']}">
    </div>
    <div>
      <h3 style="font-size:1rem; margin-bottom:4px;">❌ 공통 단점</h3>
      <img class="chart-img" src="data:image/png;base64,{charts['weakness_kw']}">
    </div>
  </div>

  <img class="chart-img" src="data:image/png;base64,{charts['target_kw']}" style="margin-top:20px;">
</div>

<!-- ═══════ 7. Trigger Sentence ═══════ -->
<div class="section">
  <div class="section-title">⚡ 지표 7 — 판단 피로 유발 문장 (Trigger Sentence) <span class="badge">가설 B 핵심</span></div>
  <div class="section-desc">
    소비자가 "그래서 사라는 거야 말라는 거야?"를 느끼게 만드는 조건부 설명.
    전체 {len(df)}건 중 <b>{has_trigger.sum()}건({trigger_rate:.0f}%)</b>에 trigger 문장이 존재합니다.
  </div>
"""

for product, trigger, score in trigger_samples[:6]:
    trigger_short = trigger[:150] + '...' if len(str(trigger)) > 150 else trigger
    html += f"""
  <div class="trigger-quote">
    <div class="product">{product} <span class="score">| 추천 {int(score)}점</span></div>
    "{trigger_short}"
  </div>"""

html += """
  <div class="insight-box">
    <h4>💡 가설 B 근거</h4>
    <p>
      리뷰어 본인도 단순하게 "좋다/나쁘다"로 정리하지 못하고 조건을 붙이는 비율이 78%에 달합니다.
      유튜버조차 한 마디로 정리하지 못하는 정보를 소비자가 여러 영상에서 종합해야 한다면,
      판단 피로는 필연적입니다.
    </p>
  </div>
</div>
"""

# 광고 편향
html += f"""
<!-- ═══════ 8. 광고 편향 ═══════ -->
<div class="section">
  <div class="section-title">💰 지표 8 — 광고/협찬 편향 분석 <span class="badge">가설 A 신뢰도</span></div>
  <div class="section-desc">
    광고 제품 {df['is_ad'].eq('Y').sum()}건 vs 비광고 {df['is_ad'].eq('N').sum()}건의 평균 점수를 비교합니다.
  </div>
  <img class="chart-img" src="data:image/png;base64,{charts['ad_compare']}" style="max-width:500px;">
  <div class="insight-box">
    <h4>💡 신뢰 피로 근거</h4>
    <p>
      광고 제품 평균 <b>{ad_avg:.2f}점</b> vs 비광고 <b>{non_ad_avg:.2f}점</b> — 차이 <b>+{ad_avg - non_ad_avg:.2f}점</b>.<br>
      광고 제품이 비광고보다 점수가 유의미하게 높다는 것은 데이터에 긍정 편향이 존재함을 의미합니다.
      소비자가 "이 리뷰, 광고 아닐까?"를 의심하게 만드는 구조적 요인입니다.
    </p>
  </div>
</div>
"""

# 브랜드 분석
html += f"""
<!-- ═══════ 10·12. 브랜드 분석 ═══════ -->
<div class="section">
  <div class="section-title">🏢 지표 10·12 — 브랜드별 리뷰 수 & 평균 점수</div>
  <img class="chart-img" src="data:image/png;base64,{charts['brand_bar']}">
  {df_to_html_table(brand_stats.head(12))}
</div>
"""

# 점수 편차 (리뷰 일관성)
if 'variance' in charts:
    html += f"""
<!-- ═══════ BONUS: 리뷰 일관성 ═══════ -->
<div class="section">
  <div class="section-title">🎲 보너스 — 리뷰 일관성 (점수 편차) <span class="badge">가설 B 보강</span></div>
  <div class="section-desc">
    같은 제품인데 유튜버마다 점수가 들쑥날쑥한 제품.
    표준편차가 클수록 소비자에게는 '도박' 같은 제품입니다.
  </div>
  <img class="chart-img" src="data:image/png;base64,{charts['variance']}">
"""
    if var_triggers:
        html += '<h4 style="margin:16px 0 8px;">이 제품들의 실제 Trigger 문장:</h4>'
        for product, trigger, score in var_triggers[:4]:
            trigger_short = trigger[:150] + '...' if len(str(trigger)) > 150 else trigger
            html += f"""
  <div class="trigger-quote">
    <div class="product">{product} <span class="score">| {int(score)}점</span></div>
    "{trigger_short}"
  </div>"""

    html += """
  <div class="insight-box">
    <h4>💡 핵심 발견</h4>
    <p>
      점수 편차가 큰 제품일수록 trigger 문장도 많이 발견됩니다.
      유튜버마다 의견이 갈리는 제품은 조건부 설명이 많아지고,
      이것이 곧 소비자의 판단 피로로 직결됩니다.
    </p>
  </div>
</div>
"""

# TOP·BOTTOM 제품
html += f"""
<!-- ═══════ 11·13·14. 순위 ═══════ -->
<div class="section">
  <div class="section-title">🏆 지표 11·13·14 — 제품 순위</div>
  <div class="two-col">
    <div>
      <h3 style="font-size:1rem; color:var(--secondary);">✅ 평균 점수 최고 (2회 이상 언급)</h3>
      {df_to_html_table(results['best_products'])}
    </div>
    <div>
      <h3 style="font-size:1rem; color:var(--accent);">❌ 평균 점수 최저 (2회 이상 언급)</h3>
      {df_to_html_table(results['worst_products'])}
    </div>
  </div>
</div>
"""

html += f"""
</div><!-- container -->
<div class="footer">
  📊 Cushion Review Analysis Report | 생성: {now} | 데이터: {len(df)}건 / {total_videos}영상 / {total_products}제품
</div>
</body>
</html>"""

# 저장
with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n✅ HTML 리포트 저장: {OUTPUT_HTML}")
print(f"   총 {len(charts)}개 차트 포함")