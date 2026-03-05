🛡️ Risk-Guard API
비정형 데이터 구조화를 통한 구매 미스매치 및 반품 리스크 사전 탐지 시스템 

📌 1. Executive Summary본 프로젝트는 유튜브 리뷰와 커머스 데이터를 결합한 비정형 텍스트 구조화 파이프라인을 통해 소비자의 '구매 미스매치' 리스크를 탐지하는 Decision Intelligence 솔루션입니다. 

킬러 데이터인 trigger_sentence를 활용해 구매 전 단계에서 반품 리스크를 예측하고 운영 비용을 최적화합니다. 

🛠️ 2. Tech Stack
구분 (Category),기술 스택 (Tech Stack)
Language,Python 3.13
Framework,Streamlit (Dashboard UI)
AI Engine,Google Gemini 2.0 Flash (NLP & Data Structuring)
Library,"Pandas, Plotly, python-dotenv"
Deployment,Streamlit Community Cloud / Render


🚀 3. Quick Start
로컬 환경에서 대시보드를 실행하는 방법입니다.

Bash
# 1. 환경 변수 설정 (.env 파일 생성)
GEMINI_API_KEY=your_api_key_here

# 2. 필수 패키지 설치
pip install -r requirements.txt

# 3. 대시보드 실행
streamlit run dashboard.py


📂 4. Project StructurePlaintext.
├── dashboard.py                # 메인 대시보드 실행 파일
├── requirements.txt            # 라이브러리 의존성 파일
├── outputs/
│   └── cleaned_product_summary.csv    # 구조화된 제품 데이터셋
├── prototype_results/
│   └── [ProductName].md        # AI 기반 상세 리스크 리포트 (Markdown)
└── README.md                   # 프로젝트 설명서

🎯 5. Core Strategy: Trigger Sentence
본 프로젝트의 핵심은 **trigger_sentence**의 정의와 추출입니다. 
정의: 소비자의 구매 판단에 결정적 혼선을 유발하는 조건부/경고성 문장.
활용: 유저 프로필(피부 타입 등)과 제품 리스크 데이터를 매칭하여 High Risk군 사전 분류. 