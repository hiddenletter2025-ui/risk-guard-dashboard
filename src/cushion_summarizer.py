# =============================================================================
# 📊 유튜브 쿠션 리뷰 자막 → 11항목 구조화 요약기 v3 (수정본)
# =============================================================================
# ✅ response_mime_type='application/json' → 코드블록 파싱 문제 해결
# ✅ 잘린 JSON 자동 복구 로직 → 긴 응답 잘림 문제 해결
# ✅ try/except 이중 파싱 버그 수정
#
# pip install google-genai pandas tqdm python-dotenv
# =============================================================================

import os
import re
import json
import time
import traceback
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from google import genai
from google.genai import types

from dotenv import load_dotenv

# =============================================================================
# 1. 설정
# =============================================================================

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ 에러: .env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    exit()

# 파일 경로
INPUT_PATH = "outputs/cushion_raw_data_final.csv"
OUTPUT_PATH = "outputs/cushion_summary_final.csv"

# 모델 설정
MODEL_NAME = "gemini-2.5-flash"
MAX_TRANSCRIPT_CHARS = 28000

# API 호출 설정
MAX_RETRIES = 5
BASE_DELAY = 10
DELAY_BETWEEN_CALLS = 4

# 요약 대상 필터
SKIP_STATUSES = ["숏폼 제외", "메타데이터 수집 실패"]

# ★ True면 기존 결과 백업 후 전체 재실행
FRESH_START = True


# =============================================================================
# 2. 프롬프트
# =============================================================================

SYSTEM_PROMPT = """\
당신은 한국 뷰티 유튜브 리뷰 영상을 분석하는 전문가입니다.
영상 자막을 읽고, 리뷰에서 언급되는 **각 제품별로** 아래 11개 항목을 추출하세요.

## 추출 항목

1. **video_id**: 제공된 video_id를 그대로 사용
2. **product_name**: 정확한 제품명 (브랜드 + 제품라인). 예: "에스쁘아 비 글로우 쿠션"
3. **recommendation_score**: 리뷰어의 추천 강도 (1~5점 정수)
   - 5: 극찬 (인생템, 무조건 사세요)
   - 4: 강추 (매우 좋음, 적극 추천)
   - 3: 괜찮음 (장단점 공존, 조건부 추천)
   - 2: 비추 (단점이 크지만 일부 장점)
   - 1: 혹평 (사지 마세요, 최악)
4. **reason_for_selection**: 리뷰어가 이 제품을 선정/리뷰한 이유 (반드시 30자 이내로 요약)
5. **target_audience**: 이 제품이 적합한 주 타겟 (피부 타입, 상황, 계절 등). 예: "건성, 겨울철, 촉촉한 마감 선호"
6. **key_strengths**: 핵심 장점 (최대 3개, 쉼표 구분)
7. **critical_weaknesses**: 치명적 단점 (최대 2개, 쉼표 구분). 없으면 "없음"
8. **comparison_evidence**: 다른 제품과의 비교 발언이나 수치적 근거. 없으면 "없음"
9. **trigger_sentence**: ⭐ 가설 B용 트리거 문장 — 소비자가 판단 피로를 느낄 만한 복잡한 조건부 설명을 자막 원문 그대로 추출. 없으면 "없음"
10. **is_ad**: 광고/협찬 여부 ("광고", "협찬", "없음", "불명확" 중 택1)
11. **sold_at_oliveyoung**: 올리브영 판매 여부 ("예", "아니오", "불명확" 중 택1)

## 핵심 규칙
- 하나의 영상에서 여러 제품이 리뷰되면 제품마다 별도 객체로 분리
- 자막에서 명시적으로 언급된 정보만 추출하고 추측하지 마세요
- trigger_sentence는 원문 그대로 복사하세요
- 각 항목의 값은 최대한 간결하게 작성하세요
"""

USER_PROMPT_TEMPLATE = """\
## 분석 대상 영상

- video_id: {video_id}
- 영상 제목: {title}
- 채널명: {channel}

## 자막 전문

{transcript}

---

위 자막에서 언급된 각 제품별로 11개 항목을 추출하여 JSON 배열로 응답하세요.
"""

RESPONSE_SCHEMA = {
    'type': 'ARRAY',
    'items': {
        'type': 'OBJECT',
        'properties': {
            'video_id':              {'type': 'STRING'},
            'product_name':          {'type': 'STRING'},
            'recommendation_score':  {'type': 'INTEGER'},
            'reason_for_selection':  {'type': 'STRING'},
            'target_audience':       {'type': 'STRING'},
            'key_strengths':         {'type': 'STRING'},
            'critical_weaknesses':   {'type': 'STRING'},
            'comparison_evidence':   {'type': 'STRING'},
            'trigger_sentence':      {'type': 'STRING'},
            'is_ad':                 {'type': 'STRING'},
            'sold_at_oliveyoung':    {'type': 'STRING'},
        },
        'required': [
            'video_id', 'product_name', 'recommendation_score',
            'reason_for_selection', 'target_audience',
            'key_strengths', 'critical_weaknesses',
            'comparison_evidence', 'trigger_sentence',
            'is_ad', 'sold_at_oliveyoung',
        ],
    },
}


# =============================================================================
# 3. 유틸리티
# =============================================================================

def truncate_transcript(text: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    if not text or not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    front = int(max_chars * 0.6)
    back = max_chars - front
    return (
        text[:front]
        + f"\n\n[... 중간 {len(text) - max_chars}자 생략 ...]\n\n"
        + text[-back:]
    )


def repair_truncated_json(text: str) -> list:
    """
    토큰 제한으로 잘린 JSON 배열을 복구한다.
    마지막으로 완전히 닫힌 {...} 객체까지만 살리고 ] 로 닫아줌.

    예: [{"a":1},{"b":2},{"c":3  ← 여기서 잘림
    →   [{"a":1},{"b":2}]        ← 완전한 2개만 살림
    """
    text = text.strip()

    # 이미 정상이면 그대로 파싱
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    # '[' 로 시작하는지 확인
    if not text.startswith('['):
        return []

    # 마지막으로 완전히 닫힌 '}' 위치 찾기
    depth = 0
    last_complete_end = -1
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_complete_end = i

    if last_complete_end <= 0:
        return []

    # 마지막 완전한 객체까지 자르고 배열 닫기
    repaired = text[:last_complete_end + 1].rstrip(',').rstrip() + ']'

    try:
        result = json.loads(repaired)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        return []


def load_completed_ids(filepath: str) -> set:
    if not os.path.exists(filepath):
        return set()
    try:
        df = pd.read_csv(filepath)
        if 'video_id' in df.columns:
            return set(df['video_id'].dropna().astype(str).str.strip())
    except Exception:
        pass
    return set()


def save_checkpoint(new_rows: list, filepath: str):
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(filepath, index=False, encoding='utf-8-sig')


def make_empty_row(video_id, status, title="", channel=""):
    return {
        'video_id': video_id, 'product_name': '',
        'recommendation_score': 0, 'reason_for_selection': '',
        'target_audience': '', 'key_strengths': '',
        'critical_weaknesses': '', 'comparison_evidence': '',
        'trigger_sentence': '', 'is_ad': '불명확',
        'sold_at_oliveyoung': '불명확', '_status': status,
        'source_title': str(title)[:80], 'source_channel': str(channel),
    }


# =============================================================================
# 4. Gemini API
# =============================================================================

class GeminiSummarizer:

    def __init__(self, api_key: str, model_name: str = MODEL_NAME):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.total_calls = 0
        self.total_errors = 0
        print(f"  🤖 모델: {model_name}")
        print(f"  🔑 API 클라이언트 초기화 완료")
        print(f"  📋 JSON 강제 모드 + 잘린 JSON 자동 복구")

    def summarize(self, video_id: str, title: str,
                  channel: str, transcript: str) -> list:

        transcript = truncate_transcript(transcript)
        if not transcript or len(transcript.strip()) < 50:
            return [make_empty_row(video_id, '자막 부족으로 스킵', title, channel)]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            video_id=video_id,
            title=title or "(제목 없음)",
            channel=channel or "(채널 불명)",
            transcript=transcript,
        )

        last_error = None
        raw_text = ""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # ── API 호출 ──
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=8192,
                        response_mime_type='application/json',
                    ),
                )
                self.total_calls += 1
                raw_text = response.text

                # ── JSON 파싱 (정상 → 잘림 복구 순서) ──
                products = None

                # 1차: 정상 파싱 시도
                try:
                    parsed = json.loads(raw_text)
                    if isinstance(parsed, dict):
                        products = [parsed]
                    elif isinstance(parsed, list):
                        products = parsed
                except json.JSONDecodeError:
                    pass

                # 2차: 잘린 JSON 자동 복구
                if products is None:
                    print(f"      🔧 잘린 JSON 복구 시도 (영상: {video_id})")
                    products = repair_truncated_json(raw_text)
                    if products:
                        print(f"      ✅ 복구 성공: {len(products)}개 제품 살림")

                # 둘 다 실패
                if not products:
                    last_error = f"파싱 실패 (응답 {len(raw_text)}자)"
                    print(f"      ⚠️  파싱 실패 (시도 {attempt}/{MAX_RETRIES}), "
                          f"응답길이: {len(raw_text)}자")
                    if attempt < MAX_RETRIES:
                        time.sleep(BASE_DELAY)
                    continue

                # ── 성공: 메타 데이터 추가 ──
                for p in products:
                    p['video_id'] = video_id
                    p['_status'] = '정상'
                    p['source_title'] = str(title)[:80]
                    p['source_channel'] = str(channel)

                return products

            except Exception as e:
                self.total_errors += 1
                err_msg = str(e)
                last_error = err_msg

                is_rate_limit = any(k in err_msg.lower() for k in [
                    'rate', 'quota', '429', 'resource_exhausted',
                    'too many', 'limit', 'retry'
                ])

                delay = BASE_DELAY * (2 ** (attempt - 1))
                if is_rate_limit:
                    delay = max(delay, 60)

                print(f"      ❌ API 오류 (시도 {attempt}/{MAX_RETRIES}): "
                      f"{err_msg[:80]}")
                print(f"         → {delay}초 대기 후 재시도")

                if attempt < MAX_RETRIES:
                    time.sleep(delay)

        return [make_empty_row(
            video_id, f'요약 실패: {str(last_error)[:100]}', title, channel
        )]


# =============================================================================
# 5. 메인 파이프라인
# =============================================================================

def run_summarizer():

    print("\n" + "=" * 65)
    print("📊 유튜브 쿠션 리뷰 자막 → 11항목 구조화 요약기 v3")
    print("   (google-genai SDK + JSON 강제 모드 + 잘림 복구)")
    print("=" * 65)
    start_time = datetime.now()

    # ── FRESH START ──
    if FRESH_START and os.path.exists(OUTPUT_PATH):
        backup = OUTPUT_PATH.replace('.csv', '_backup.csv')
        os.rename(OUTPUT_PATH, backup)
        print(f"\n  🗑️  FRESH_START=True → 백업({backup}) 후 처음부터 시작")

    # ── 원본 로드 ──
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 입력 파일 없음: {INPUT_PATH}")
        return

    raw_df = pd.read_csv(INPUT_PATH)
    print(f"\n  📂 원본 데이터: {len(raw_df)}개 영상 로드")

    for col in ['video_id', 'transcript']:
        if col not in raw_df.columns:
            print(f"❌ 필수 컬럼 '{col}' 없음")
            return

    # ── 필터링 ──
    target_df = raw_df.copy()

    if 'status' in target_df.columns:
        mask = ~target_df['status'].astype(str).apply(
            lambda s: any(skip in s for skip in SKIP_STATUSES)
        )
        ex = (~mask).sum()
        target_df = target_df[mask]
        if ex > 0:
            print(f"  🚫 상태 기반 제외: {ex}개")

    if 'duration_seconds' in target_df.columns:
        shorts = target_df['duration_seconds'].fillna(0).astype(int) <= 60
        ex = shorts.sum()
        target_df = target_df[~shorts]
        if ex > 0:
            print(f"  🚫 숏폼(≤60초) 제외: {ex}개")

    def has_valid_transcript(x):
        if not isinstance(x, str):
            return False
        x = x.strip()
        return len(x) >= 50 and not x.startswith('자막 없음') and not x.startswith('오류')

    valid = target_df['transcript'].apply(has_valid_transcript)
    ex = (~valid).sum()
    target_df = target_df[valid].copy()
    if ex > 0:
        print(f"  🚫 자막 없음/오류 제외: {ex}개")

    print(f"  ✅ 요약 대상: {len(target_df)}개 영상")

    # ── Resume ──
    completed_ids = load_completed_ids(OUTPUT_PATH)
    if completed_ids:
        before = len(target_df)
        target_df = target_df[~target_df['video_id'].astype(str).isin(completed_ids)]
        skipped = before - len(target_df)
        if skipped > 0:
            print(f"  ♻️  Resume: {skipped}개 완료 → {len(target_df)}개 남음")

    if target_df.empty:
        print("\n✅ 모든 영상이 이미 요약 완료되었습니다!")
        return pd.read_csv(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else pd.DataFrame()

    est_min = len(target_df) * (DELAY_BETWEEN_CALLS + 3) / 60
    print(f"  ⏱️  예상 소요: 약 {est_min:.0f}분")
    print("-" * 65)

    # ── 요약 루프 ──
    summarizer = GeminiSummarizer(api_key=API_KEY, model_name=MODEL_NAME)

    unsaved_buffer = []
    success_count = 0
    fail_count = 0
    total_products = 0

    progress = tqdm(
        target_df.iterrows(),
        total=len(target_df),
        desc="📝 요약 진행",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for idx, row in progress:
        video_id = str(row['video_id']).strip()
        title = str(row.get('title', ''))
        channel = str(row.get('channel_title', ''))
        transcript = str(row.get('transcript', ''))

        progress.set_postfix_str(f"{title[:20]}...")

        try:
            products = summarizer.summarize(
                video_id=video_id,
                title=title,
                channel=channel,
                transcript=transcript,
            )

            unsaved_buffer.extend(products)
            total_products += len(products)

            status = products[0].get('_status', '') if products else ''
            if '실패' in status or '스킵' in status:
                fail_count += 1
                tqdm.write(f"  ⚠️  [{video_id}] {status}")
            else:
                success_count += 1
                names = [p.get('product_name', '?') for p in products]
                tqdm.write(
                    f"  ✅ [{video_id}] {len(products)}개 제품: "
                    f"{', '.join(names[:3])}"
                )

        except Exception as e:
            fail_count += 1
            tqdm.write(f"  ❌ [{video_id}] 치명적 오류: {str(e)[:80]}")
            unsaved_buffer.append(
                make_empty_row(video_id, f'치명적 오류: {str(e)[:100]}', title, channel)
            )

        # 매 영상마다 체크포인트
        if unsaved_buffer:
            save_checkpoint(unsaved_buffer, OUTPUT_PATH)
            unsaved_buffer = []

        time.sleep(DELAY_BETWEEN_CALLS)

    # ── 결과 리포트 ──
    elapsed = datetime.now() - start_time
    final_df = pd.read_csv(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else pd.DataFrame()

    print("\n" + "=" * 65)
    print("🎉 요약 완료!")
    print("=" * 65)
    print(f"  📁 출력: {OUTPUT_PATH}")
    print(f"  📊 이번 세션:")
    print(f"     ├─ 처리: {success_count + fail_count}개")
    print(f"     ├─ 성공: {success_count}개")
    print(f"     ├─ 실패: {fail_count}개")
    print(f"     └─ 추출 제품: {total_products}개")
    print(f"  📦 누적: {len(final_df)}행")
    print(f"  🔑 API: {summarizer.total_calls}회 (오류 {summarizer.total_errors}회)")
    print(f"  ⏱️  소요: {elapsed.seconds // 60}분 {elapsed.seconds % 60}초")
    print("=" * 65)

    if not final_df.empty and '_status' in final_df.columns:
        print(f"\n📋 상태 분포:")
        print(final_df['_status'].value_counts().to_string())

    if not final_df.empty and 'product_name' in final_df.columns:
        products_col = final_df['product_name'].dropna()
        products_col = products_col[products_col.str.len() > 0]
        if len(products_col) > 0:
            print(f"\n🏷️  추출 제품 ({len(products_col)}개 중 상위 10개):")
            for name in products_col.head(10):
                print(f"     • {name}")

    return final_df


# =============================================================================
# 6. 실행
# =============================================================================

if __name__ == "__main__":
    df = run_summarizer()