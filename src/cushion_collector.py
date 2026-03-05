
# =============================================================================
# 🎬 유튜브 쿠션 리뷰 통합 데이터 수집기 v2
# =============================================================================
# Target ID 기반 수집 | Resume 이어받기 | 차단 방지 극대화 | 숏폼 자동 제외
#
# 필요 라이브러리:
#   pip install google-api-python-client yt-dlp requests pandas
#
# 필요 파일:
#   - data/target_list.csv        (video_id 컬럼)
#   - youtube.com_cookies.txt     (쿠키 파일)
# =============================================================================

import os
import re
import glob
import time
import random
import warnings
from datetime import datetime

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

import yt_dlp
import requests
import pandas as pd
from googleapiclient.discovery import build

from dotenv import load_dotenv

# =============================================================================
# 1. 설정 상수
# =============================================================================

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    print("❌ 에러: .env 파일에서 YOUTUBE_API_KEY를 찾을 수 없습니다.")
    exit()

# 파일 경로
TARGET_LIST_PATH = "data/target_list.csv"
OUTPUT_PATH = "outputs/cushion_raw_data_final.csv"
COOKIE_PATH = "youtube.com_cookies.txt"

# 수집 설정
COMMENTS_PER_VIDEO = 100
CHECKPOINT_INTERVAL = 5
MIN_DURATION_SECONDS = 60        # 60초 이하 숏폼 제외

# 차단 방지 설정
DELAY_MIN = 15                   # 영상 간 최소 대기(초)
DELAY_MAX = 30                   # 영상 간 최대 대기(초)
DELAY_BETWEEN_STEPS = 3          # 자막↔댓글 사이 대기(초)

# 최신 브라우저 User-Agent
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


# =============================================================================
# 2. 유틸리티 함수
# =============================================================================

def load_target_ids(filepath: str) -> list:
    """target_list.csv에서 video_id 리스트를 읽어옴"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ 타겟 리스트 파일이 없습니다: {filepath}\n"
            f"   'data/target_list.csv' 파일을 생성하고 video_id 컬럼을 추가하세요."
        )

    df = pd.read_csv(filepath)

    if 'video_id' not in df.columns:
        raise ValueError(
            f"❌ '{filepath}'에 'video_id' 컬럼이 없습니다.\n"
            f"   현재 컬럼: {list(df.columns)}"
        )

    ids = df['video_id'].dropna().astype(str).str.strip().tolist()
    ids = [vid for vid in ids if vid]
    print(f"  📋 타겟 리스트: {len(ids)}개 영상 로드 완료")
    return ids


def load_completed_ids(filepath: str) -> set:
    """이미 수집 완료된 video_id 세트를 반환 (Resume 기능)"""
    if not os.path.exists(filepath):
        return set()

    try:
        df = pd.read_csv(filepath)
        if 'video_id' in df.columns:
            completed = set(df['video_id'].dropna().astype(str).str.strip())
            print(f"  ♻️  기존 수집 데이터 발견: {len(completed)}개 완료됨")
            return completed
    except Exception as e:
        print(f"  ⚠️  기존 파일 읽기 실패 (새로 시작): {e}")

    return set()


def save_checkpoint(new_data: list, filepath: str, tag: str = ""):
    """
    신규 수집 데이터를 기존 CSV에 append 방식으로 중간 저장
    기존 파일이 있으면 이어붙이고, 없으면 새로 생성
    """
    if not new_data:
        return

    new_df = pd.DataFrame(new_data)

    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # 혹시 모를 중복 제거
        combined_df = combined_df.drop_duplicates(subset='video_id', keep='last')
    else:
        combined_df = new_df

    # 컬럼 순서 정리
    column_order = [
        'video_id', 'title', 'channel_title', 'published_at',
        'view_count', 'like_count', 'comment_count', 'duration_raw',
        'duration_seconds', 'description', 'transcript_source',
        'transcript', 'comments', 'status', 'collected_at'
    ]
    existing_cols = [c for c in column_order if c in combined_df.columns]
    combined_df = combined_df[existing_cols]

    combined_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    total = len(combined_df)
    print(f"  💾 [체크포인트{tag}] 누적 {total}개 → {filepath} 저장")


def parse_duration_to_seconds(duration_str: str) -> int:
    """ISO 8601 duration(PT1H2M3S)을 초 단위로 변환"""
    if not duration_str:
        return 0

    match = re.match(
        r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?',
        duration_str
    )
    if not match:
        return 0

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    return hours * 3600 + minutes * 60 + seconds


# =============================================================================
# 3. YouTube 메타데이터 + 댓글 수집 클래스
# =============================================================================

class YouTubeAPI:
    """YouTube Data API를 사용한 메타데이터 및 댓글 수집"""

    def __init__(self, api_key: str):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.quota_used = 0

    def get_video_details(self, video_id: str) -> dict:
        """단일 영상의 메타데이터 수집"""
        try:
            response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            ).execute()
            self.quota_used += 1

            items = response.get('items', [])
            if not items:
                return None

            item = items[0]
            snippet = item.get('snippet', {})
            stats = item.get('statistics', {})
            duration_raw = item.get('contentDetails', {}).get('duration', '')

            return {
                'video_id': video_id,
                'title': snippet.get('title', ''),
                'channel_title': snippet.get('channelTitle', ''),
                'published_at': snippet.get('publishedAt', ''),
                'description': snippet.get('description', '')[:500],
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0)),
                'duration_raw': duration_raw,
                'duration_seconds': parse_duration_to_seconds(duration_raw),
            }

        except Exception as e:
            print(f"    ⚠️  메타데이터 수집 실패: {str(e)[:80]}")
            return None

    def get_comments(self, video_id: str, max_comments: int = 100) -> str:
        """단일 영상의 상위 댓글을 ' ||| ' 구분 문자열로 반환"""
        comments = []
        next_page_token = None

        try:
            while len(comments) < max_comments:
                request_params = {
                    'part': 'snippet',
                    'videoId': video_id,
                    'maxResults': min(100, max_comments - len(comments)),
                    'order': 'relevance',
                    'textFormat': 'plainText',
                }
                if next_page_token:
                    request_params['pageToken'] = next_page_token

                response = self.youtube.commentThreads().list(
                    **request_params
                ).execute()
                self.quota_used += 1

                for item in response.get('items', []):
                    text = (
                        item['snippet']['topLevelComment']
                            ['snippet']['textDisplay']
                    )
                    text = re.sub(r'\s+', ' ', text).strip()
                    if text:
                        comments.append(text)

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break

                time.sleep(0.5)

        except Exception as e:
            err = str(e)
            if 'commentsDisabled' in err or '403' in err:
                return "댓글 비활성화"
            return f"댓글 수집 오류: {err[:80]}"

        if not comments:
            return "댓글 없음"

        return " ||| ".join(comments)


# =============================================================================
# 4. 자막 추출 클래스 (yt-dlp 기반, 차단 방지 강화)
# =============================================================================

class TranscriptExtractor:
    """
    yt-dlp 기반 한국어 자막 추출기
    차단 방지: User-Agent + 쿠키 + sleep
    """

    def __init__(self, cookie_path: str = None):
        self.cookie_path = cookie_path

        # 쿠키 파일 존재 확인
        if self.cookie_path and os.path.exists(self.cookie_path):
            print(f"  🍪 쿠키 파일 로드: {self.cookie_path}")
        else:
            print(f"  ⚠️  쿠키 파일 없음 — 차단 위험 증가")
            self.cookie_path = None

    def _clean_vtt(self, vtt_content: str) -> str:
        """VTT 파일의 타임라인/태그/중복 제거 → 순수 텍스트"""
        lines = vtt_content.split('\n')
        cleaned = []

        for line in lines:
            if any(k in line for k in ["WEBVTT", "Kind:", "Language:", "-->"]):
                continue
            if line.strip().isdigit() or not line.strip():
                continue

            clean = re.sub(r'<[^>]+>', '', line).strip()

            if clean and (not cleaned or cleaned[-1] != clean):
                cleaned.append(clean)

        return " ".join(cleaned).strip()

    def extract(self, video_id: str) -> tuple:
        """
        자막 추출
        Returns: (텍스트, 소스) — 소스: '수동 자막' | '자동 생성 자막' | '없음' | '오류'
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        temp = f"temp_{video_id}"

        ydl_opts = {
            'skip_download': True,
            'writesubs': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['ko'],
            'subtitlesformat': 'vtt',
            'outtmpl': temp,
            'quiet': True,
            'no_warnings': True,
            'format': None,
            # 차단 방지 옵션
            'http_headers': {'User-Agent': USER_AGENT},
            'socket_timeout': 30,
            'retries': 3,
            'extractor_retries': 3,
        }

        if self.cookie_path:
            ydl_opts['cookiefile'] = self.cookie_path

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                subs = info.get('subtitles', {})
                auto_subs = info.get('automatic_captions', {})

                ko_sub = None
                source = ""

                if 'ko' in subs:
                    ko_sub = subs['ko']
                    source = "수동 자막"
                elif 'ko' in auto_subs:
                    ko_sub = auto_subs['ko']
                    source = "자동 생성 자막"

                if not ko_sub:
                    return ("자막 없음", "없음")

                # VTT 포맷 URL 탐색
                vtt_url = None
                for fmt in ko_sub:
                    if fmt.get('ext') == 'vtt':
                        vtt_url = fmt['url']
                        break
                if not vtt_url:
                    vtt_url = ko_sub[0]['url']

                vtt_content = requests.get(
                    vtt_url,
                    verify=False,
                    headers={'User-Agent': USER_AGENT},
                    timeout=30
                ).text

                cleaned = self._clean_vtt(vtt_content)
                return (cleaned, source)

        except Exception as e:
            return (f"오류: {str(e)[:100]}", "오류")
        finally:
            for f in glob.glob(f"{temp}*"):
                try:
                    os.remove(f)
                except OSError:
                    pass


# =============================================================================
# 5. 통합 수집기
# =============================================================================

class CushionCollector:
    """
    Target ID 기반 통합 수집기

    특징:
    - target_list.csv 기반 수집 (검색 로직 없음)
    - Resume: 기존 결과 파일이 있으면 미수집 ID만 이어서 수집
    - 5개마다 체크포인트 저장
    - 영상당 15~30초 랜덤 대기 (차단 방지)
    - 60초 이하 숏폼 자동 제외
    """

    def __init__(self, api_key: str, cookie_path: str = None):
        self.api = YouTubeAPI(api_key)
        self.extractor = TranscriptExtractor(cookie_path)
        self.unsaved_buffer = []     # 아직 저장 안 된 신규 데이터

    def run(self,
            target_path: str = TARGET_LIST_PATH,
            output_path: str = OUTPUT_PATH,
            comments_per_video: int = COMMENTS_PER_VIDEO) -> pd.DataFrame:
        """메인 수집 파이프라인"""

        print("\n" + "=" * 60)
        print("🎬 유튜브 쿠션 리뷰 통합 수집기 v2")
        print("=" * 60)

        start_time = datetime.now()

        # ─── Step 1: 타겟 로드 + Resume 체크 ───
        all_ids = load_target_ids(target_path)
        completed_ids = load_completed_ids(output_path)

        pending_ids = [vid for vid in all_ids if vid not in completed_ids]

        if not pending_ids:
            print("\n✅ 모든 영상이 이미 수집 완료되었습니다!")
            if os.path.exists(output_path):
                return pd.read_csv(output_path)
            return pd.DataFrame()

        print(f"\n  📊 전체: {len(all_ids)}개 | 완료: {len(completed_ids)}개 | 남은 작업: {len(pending_ids)}개")
        est_min = len(pending_ids) * (DELAY_MIN + DELAY_MAX) / 2 / 60
        print(f"  ⏱️  예상 소요: 약 {est_min:.0f}분")
        print("-" * 60)

        # ─── Step 2: 수집 루프 ───
        total = len(pending_ids)
        success_t = 0   # 자막 성공
        success_c = 0   # 댓글 성공
        skipped = 0     # 숏폼 스킵

        for i, video_id in enumerate(pending_ids):
            status_parts = []

            print(f"\n  [{i + 1}/{total}] ID: {video_id}")

            # ── 메타데이터 수집 ──
            meta = self.api.get_video_details(video_id)

            if meta is None:
                print(f"    ❌ 메타데이터 수집 실패 — 건너뜀")
                record = {
                    'video_id': video_id, 'title': '', 'channel_title': '',
                    'published_at': '', 'description': '', 'view_count': 0,
                    'like_count': 0, 'comment_count': 0, 'duration_raw': '',
                    'duration_seconds': 0, 'transcript_source': '',
                    'transcript': '', 'comments': '',
                    'status': '메타데이터 수집 실패',
                    'collected_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                self.unsaved_buffer.append(record)
                self._maybe_checkpoint(i, total, output_path)
                self._wait(i, total)
                continue

            title = meta['title'][:35]
            dur_sec = meta['duration_seconds']
            print(f"    📌 {title}... ({dur_sec}초)")

            # ── 숏폼 필터 (60초 이하 제외) ──
            if dur_sec <= MIN_DURATION_SECONDS:
                print(f"    ⏭️  숏폼 제외 ({dur_sec}초 ≤ {MIN_DURATION_SECONDS}초)")
                skipped += 1
                meta['transcript_source'] = ''
                meta['transcript'] = ''
                meta['comments'] = ''
                meta['status'] = f'숏폼 제외 ({dur_sec}초)'
                meta['collected_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.unsaved_buffer.append(meta)
                self._maybe_checkpoint(i, total, output_path)
                self._wait(i, total)
                continue

            # ── 자막 추출 ──
            try:
                transcript_text, transcript_source = self.extractor.extract(video_id)
                meta['transcript'] = transcript_text
                meta['transcript_source'] = transcript_source

                if "오류" in transcript_text or "자막 없음" in transcript_text:
                    print(f"    📄 자막: ❌ {transcript_text[:50]}")
                    status_parts.append(f"자막실패:{transcript_text[:30]}")
                else:
                    print(f"    📄 자막: ✅ {transcript_source} ({len(transcript_text)}자)")
                    success_t += 1
            except Exception as e:
                meta['transcript'] = ''
                meta['transcript_source'] = '오류'
                status_parts.append(f"자막예외:{str(e)[:30]}")
                print(f"    📄 자막: ❌ 예외")

            # 자막↔댓글 사이 대기
            time.sleep(random.uniform(2, DELAY_BETWEEN_STEPS))

            # ── 댓글 수집 ──
            try:
                comments_text = self.api.get_comments(video_id, comments_per_video)
                meta['comments'] = comments_text

                if any(k in comments_text for k in ["오류", "비활성화", "없음"]):
                    print(f"    💬 댓글: ❌ {comments_text[:50]}")
                    status_parts.append(f"댓글:{comments_text[:30]}")
                else:
                    cnt = comments_text.count(' ||| ') + 1
                    print(f"    💬 댓글: ✅ {cnt}개")
                    success_c += 1
            except Exception as e:
                meta['comments'] = ''
                status_parts.append(f"댓글예외:{str(e)[:30]}")
                print(f"    💬 댓글: ❌ 예외")

            # ── 상태 기록 ──
            meta['status'] = " | ".join(status_parts) if status_parts else "정상"
            meta['collected_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.unsaved_buffer.append(meta)

            # ── 체크포인트 ──
            self._maybe_checkpoint(i, total, output_path)

            # ── 차단 방지 대기 ──
            self._wait(i, total)

        # ─── Step 3: 최종 저장 ───
        if self.unsaved_buffer:
            save_checkpoint(self.unsaved_buffer, output_path, tag=" 최종")
            self.unsaved_buffer = []

        # ─── Step 4: 결과 리포트 ───
        elapsed = datetime.now() - start_time

        if os.path.exists(output_path):
            final_df = pd.read_csv(output_path)
        else:
            final_df = pd.DataFrame()

        normal_count = sum(
            1 for _, row in final_df.iterrows()
            if str(row.get('status', '')) == '정상'
        ) if not final_df.empty else 0

        print("\n" + "=" * 60)
        print("🎉 수집 완료!")
        print("=" * 60)
        print(f"  📁 출력: {output_path}")
        print(f"  📊 이번 세션: {total}개 처리")
        print(f"     ├─ 자막 성공: {success_t}개")
        print(f"     ├─ 댓글 성공: {success_c}개")
        print(f"     ├─ 숏폼 제외: {skipped}개")
        print(f"     └─ 정상 수집: {normal_count}개 (누적)")
        print(f"  📦 총 누적 데이터: {len(final_df)}개")
        print(f"  ⏱️  소요: {elapsed.seconds // 60}분 {elapsed.seconds % 60}초")
        print(f"  🔑 API 할당량: ~{self.api.quota_used} 단위")
        print("=" * 60 + "\n")

        return final_df

    def _maybe_checkpoint(self, idx: int, total: int, output_path: str):
        """5개마다 체크포인트 저장 후 버퍼 비우기"""
        collected_in_buffer = len(self.unsaved_buffer)
        if collected_in_buffer >= CHECKPOINT_INTERVAL:
            save_checkpoint(
                self.unsaved_buffer,
                output_path,
                tag=f" {idx + 1}/{total}"
            )
            self.unsaved_buffer = []

    def _wait(self, idx: int, total: int):
        """마지막 영상이 아닌 경우 랜덤 대기"""
        if idx < total - 1:
            delay = random.uniform(DELAY_MIN, DELAY_MAX)
            print(f"    ⏳ {delay:.1f}초 대기 중...")
            time.sleep(delay)


# =============================================================================
# 6. 실행
# =============================================================================

if __name__ == "__main__":

    collector = CushionCollector(
        api_key=API_KEY,
        cookie_path=COOKIE_PATH,
    )

    df = collector.run(
        target_path=TARGET_LIST_PATH,
        output_path=OUTPUT_PATH,
        comments_per_video=COMMENTS_PER_VIDEO,
    )

    # 결과 미리보기
    if not df.empty:
        print("\n📋 수집 결과 미리보기:")
        preview_cols = ['title', 'view_count', 'duration_seconds',
                        'transcript_source', 'status']
        available = [c for c in preview_cols if c in df.columns]
        print(df[available].to_string())

        print(f"\n📊 상태 요약:")
        print(df['status'].value_counts().to_string())
