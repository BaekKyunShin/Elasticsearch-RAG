# Elasticsearch와 LangChain을 활용한 RAG 챗봇 (RAG Chatbot using Elasticsearch and LangChain)

Python, Elasticsearch, LangChain, 그리고 Google Gemini 모델을 기반으로 구축한, PDF 문서 기반의 질의응답 RAG(Retrieval-Augmented Generation) 챗봇입니다.

## 주요 기능 

* **의미 기반 벡터 검색**: 사용자의 질문 의도를 파악하여 키워드가 아닌 **의미적으로 가장 유사한** 문서 내용을 Elasticsearch에서 정확하게 찾아냅니다.
* **LLM 기반 자연어 답변 생성**: Google의 강력한 LLM(`gemini-1.5-flash`)을 활용하여, 검색된 문서 내용을 기반으로 자연스럽고 논리적인 답변을 생성합니다.
* **PDF 문서 기반 질의응답**: `data` 폴더에 원하는 PDF 파일들을 넣어두기만 하면, 해당 문서의 내용을 학습하여 전문가처럼 답변하는 챗봇을 만들 수 있습니다.
* **대화형 CLI**: 터미널 환경에서 사용자와 자연스럽게 대화를 주고받을 수 있는 직관적인 인터페이스를 제공합니다.
* **클라우드 확장성을 고려한 설계**: 백엔드 데이터베이스로 **Elastic Cloud**를 활용할 수 있으며, 향후 **Streamlit** 등 웹 프레임워크와 결합하여 클라우드에 배포하기 용이한 구조로 설계되었습니다.

## 나만의 데이터로 활용하기 (Customization)

이 프로젝트는 다른 PDF 문서 데이터에도 쉽게 적용할 수 있습니다.

1.  **데이터 준비**: 프로젝트 내 `data/` 폴더에 질의응답의 기반으로 삼고 싶은 PDF 파일들을 넣어주세요.
2.  **환경 변수 설정 (`.env`)**: `.env.example` 파일을 복사하여 `.env` 파일을 생성하고, **Elasticsearch 접속 정보**와 **Google API 키**를 자신의 환경에 맞게 입력합니다.
3.  **데이터 색인 (Indexing)**: `app/main.py` 파일에서 문서 색인 관련 코드의 주석을 잠시 해제한 후, 스크립트를 실행하여 PDF 데이터를 Elasticsearch에 저장합니다. (자세한 방법은 아래 `로컬 환경에서 실행하기` 참고)
4.  **챗봇 실행**: 데이터 색인이 완료되면, `app/main.py`의 색인 코드를 다시 주석 처리하고 스크립트를 실행하여 챗봇과 대화를 시작합니다.

## 시스템 아키텍처 

이 시스템은 **데이터 색인(Indexing)**과 **질의응답(Inference)** 두 단계로 동작합니다.

**1. 데이터 색인 (Indexing Phase)**
```
[PDF Files] -> [LangChain: PyPDFLoader] -> [LangChain: TextSplitter] -> [HuggingFace Embedding Model] -> [Elasticsearch Index]
                    (문서 로딩)                   (의미 단위로 분할)                 (텍스트를 벡터로 변환)             (벡터 데이터 저장)
```
**2. 질의응답 (Inference Phase)**
```
[User Query] -> [HuggingFace Embedding Model] -> [Elasticsearch: Vector Search] -> [Retrieved Docs] --+
 (사용자 질문)          (질문을 벡터로 변환)                    (유사한 문서 벡터 검색)           (관련 문서 내용)    |
                                                                                                      |
                                         +------------------------------------------------------------+
                                         |
                                         V
      [Google Gemini LLM with LangChain Prompt] <--- [User Query]
            (검색된 내용과 질문을 함께 LLM에 전달)
                                         |
                                         V
                                     [Answer]
                                     (최종 답변)
```

## 기술 스택
* **Backend**: Python (`3.11+`), LangChain
* **Database / Vector Search**: Elasticsearch (`8.x`)
* **AI / LLM**: Google Gemini (`gemini-1.5-flash`), HuggingFace Sentence Transformers
* **Interface**: Command Line Interface (CLI)
* **Data Handling**: PyPDF
* **Deployment**: Docker (for local Elasticsearch)
* **Dependency Management**: Poetry

## 로컬 환경에서 실행하기 (Setup & Installation)

### 1. 프로젝트 클론 (Clone Repository)
```bash
git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_ID]/[YOUR_REPOSITORY_NAME].git
cd [YOUR_REPOSITORY_NAME]
```

### 2. Elasticsearch 서버 실행 (Docker)
로컬 환경에서 테스트할 수 있는 Elasticsearch 서버를 Docker로 실행합니다.
(Nori 형태소 분석기가 필수는 아니지만, 추후 키워드 검색 기능 확장을 위해 포함된 Dockerfile 사용을 권장합니다.)
```bash
# 1. (선택) Nori 분석기가 포함된 Docker 이미지 빌드
docker build -t elasticsearch-nori .

# 2. Docker 컨테이너 실행 (보안 기능 비활성화, 로컬 테스트용)
docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" --name es-rag-container elasticsearch-nori
```

### 3. 환경 변수 파일 생성
`.env` 파일을 예시는 아래와 같습니다. 적절히 내용을 수정해서 사용하면 됩니다.
```
# .env

# Google API Key
GOOGLE_API_KEY="YOUR_GOOGLE_AI_STUDIO_API_KEY"

# Elasticsearch Connection Info (로컬 Docker 실행 기준)
ELASTIC_HOST="http://localhost:9200"
ELASTIC_USER="elastic"
ELASTIC_PASSWORD="YOUR_ELASTIC_PASSWORD" # 보안 비활성화 시 비워둬도 무방
```

### 4. 파이썬 라이브러리 설치
```bash
poetry install
```

### 5. 데이터 색인 (최초 1회 실행)
`app/main.py` 파일의 **문서 색인 관련 코드 블록의 주석을 해제**한 후, 아래 명령어를 실행하여 data/ 폴더의 PDF 문서를 Elasticsearch에 저장합니다.
```bash
poetry run python app/main.py
```
색인이 성공적으로 완료되면, 다시 해당 코드 블록을 **원래대로 주석 처리**하여 다음 실행 시 중복 작업을 피하도록 합니다.

### 6. RAG 챗봇 실행
이제 준비가 완료되었습니다. 아래 명령어로 챗봇을 실행하고 터미널에서 질문을 입력하세요.
```bash
poetry run python app/main.py
```
바로 대화를 시작할 수 있습니다.

## License
This project is licensed under the MIT License.










