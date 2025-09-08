import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_community.vectorstores import ElasticsearchStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app import config

class RAGService:
    """
    RAG 시스템의 핵심 비즈니스 로직을 담당하는 서비스 클래스
    """
    def __init__(self):
        """
        RAGService가 생성될 때 임베딩 모델과 Elasticsearch 클라이언트를 초기화합니다.
        """
        # --- 1. 임베딩 모델 초기화 ---
        print("임베딩 모델을 로딩합니다...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("임베딩 모델 로딩 완료.")
        
        # --- 2. Elasticsearch 클라이언트 초기화 ---
        print("Elasticsearch 클라이언트를 초기화합니다...")
        self.es_client = Elasticsearch(
            hosts=[config.ELASTIC_HOST],
            basic_auth=(config.ELASTIC_USER, config.ELASTIC_PASSWORD),
            verify_certs=False
        )
        self.index_name = "rag_documents"
        print("Elasticsearch 클라이언트 초기화 완료.")

        # --- 3. LangChain 벡터 저장소(Vector Store) 초기화 ---
        print("LangChain 벡터 저장소를 초기화합니다...")
        self.vector_store = ElasticsearchStore(
            index_name=self.index_name,
            embedding=self.embedding_model,
            es_connection=self.es_client,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                hybrid=False
            )
        )
        print("LangChain 벡터 저장소 초기화 완료.")

        # --- 4. LLM(생성기) 초기화 ---
        print("LLM(생성기)을 초기화합니다...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=config.GOOGLE_API_KEY
        )
        print("LLM(생성기) 초기화 완료.")
        
        # --- 5. RAG 체인 생성 ---
        self.rag_chain = self.create_rag_chain()

    def create_rag_chain(self):
        """
        검색기, 프롬프트, LLM을 연결하여 RAG 체인을 생성합니다.
        """
        print("RAG 체인을 생성합니다...")

        # 우리가 초기화한 ElasticsearchStore 객체를 LangChain 표준 검색기(Retriever) 인터페이스로 변환합니다.
        # k=5: 사용자 질문과 가장 유사한 문서 조각을 5개 찾아오도록 설정합니다
        retriever = self.vector_store.as_retriever(
            search_kwargs={'k': 5}
        )

        # 프롬프트 템플릿을 정의합니다.
        # {context}: 이 부분은 나중에 검색기가 찾아온 문서 내용으로 채워집니다.
        # {question}: 이 부분은 사용자가 입력한 원본 질문으로 채워집니다.
        prompt_template = """
        당신은 사용자의 질문에 대해 주어진 컨텍스트(Context) 정보를 기반으로 답변하는 AI 어시스턴트입니다.
        컨텍스트 내용을 벗어난 답변은 하지 마세요. 만약 컨텍스트에서 답변을 찾을 수 없다면, "제공된 정보만으로는 답변을 찾을 수 없습니다." 라고 솔직하게 답변하세요.
        
        [Context]
        {context}
        
        [Question]
        {question}
        
        [Answer]
        """
        # 프롬프트 템플릿을 LangChain이 이해할 수 있는 프롬프트 객체로 변환합니다.
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # LCEL(LangChain Expression Language)을 사용하여 파이프라인을 구성하는 코드입니다. 
        # | (파이프) 기호로 각 단계를 연결합니다.
        rag_chain = (
            # "context": retriever: 사용자의 질문이 들어오면, 그 질문을 retriever에게 보내 관련 문서를 찾아오고 그 결과를 context 키에 담습니다.
            # "question": RunnablePassthrough(): RunnablePassthrough는 입력을 아무런 변경 없이 그대로 전달하는 역할을 합니다. 즉, 사용자의 원본 질문을 question 키에 그대로 담습니다.
            {"context": retriever, "question": RunnablePassthrough()} 
            # prompt는 이 값들을 사용해 {context}와 {question} 부분을 채워 완성된 지시서(프롬프트)를 만듭니다.
            | prompt
            # 완성된 프롬프트를 LLM에게 전달하여 최종 답변을 생성하도록 요청합니다.
            | self.llm
            # LLM의 출력은 복잡한 객체 형태이므로, 그중에서 우리가 필요한 답변 텍스트만 깔끔하게 추출해주는 파서(Parser)입니다.
            | StrOutputParser()
        )
        print("RAG 체인 생성이 완료되었습니다.")
        return rag_chain

    def setup_elasticsearch_index(self):
        """
        Elasticsearch 인덱스를 설정하고, 존재하지 않으면 생성합니다.
        벡터 검색을 위한 매핑을 정의합니다.
        """
        if self.es_client.indices.exists(index=self.index_name):
            print(f"인덱스 '{self.index_name}'는 이미 존재합니다.")
            return

        print(f"인덱스 '{self.index_name}'를 생성합니다...")
        
        # 인덱스 매핑 정의
        mapping = {
            "properties": {
                "text": {"type": "text"},
                "vector": {
                    "type": "dense_vector",
                    "dims": 384  # 임베딩 모델(all-MiniLM-L6-v2)의 벡터 차원 수
                },
                "metadata": { # 메타데이터 필드 추가
                    "properties": {
                        "source": {"type": "keyword"},
                        "page": {"type": "integer"}
                    }
                }
            }
        }
        
        self.es_client.indices.create(index=self.index_name, mappings=mapping)
        print(f"인덱스 '{self.index_name}' 생성이 완료되었습니다.")

    def index_documents(self, chunks: list, embeddings: list):
        """
        분할된 문서(Chunk)와 그에 해당하는 벡터를 Elasticsearch에 인덱싱(저장)합니다.

        :param chunks: 분할된 Document 객체 리스트
        :param embeddings: 텍스트 조각들에 대한 벡터 리스트
        """
        if not chunks or not embeddings:
            print("인덱싱할 문서나 임베딩이 없습니다.")
            return
            
        print(f"총 {len(chunks)}개의 문서를 Elasticsearch에 인덱싱합니다...")

        actions = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            action = {
                "_index": self.index_name,
                "_source": { 
                    "text": chunk.page_content,
                    "vector": vector, 
                    "metadata": chunk.metadata
                }
            }
            actions.append(action)

        # bulk API를 사용하여 대량의 문서를 효율적으로 인덱싱
        success, failed = bulk(self.es_client, actions)
        
        print(f"인덱싱 완료. 성공: {success}, 실패: {failed}")
        if len(failed) > 0:
            print("일부 문서 인덱싱에 실패했습니다.")

    def embed_documents(self, chunks: list) -> list:
        """
        분할된 Document Chunk들의 텍스트를 임베딩하여 벡터로 변환합니다.

        :param chunks: 분할된 Document 객체 리스트
        :return: 텍스트 조각들에 대한 벡터 리스트
        """
        print(f"총 {len(chunks)}개의 Chunk를 임베딩합니다...")
        
        # 각 Document 객체에서 텍스트 내용만 추출
        texts = [chunk.page_content for chunk in chunks]
        
        # 임베딩 모델을 사용하여 텍스트 리스트를 벡터 리스트로 변환
        embeddings = self.embedding_model.embed_documents(texts)
        
        print("임베딩 완료.")
        return embeddings

    def split_documents(self, documents: list) -> list:
        """
        로드된 Document 객체들을 의미있는 단위(Chunk)로 분할합니다.

        :param documents: LangChain의 Document 객체 리스트
        :return: 분할된 Document 객체 리스트 (Chunks)
        """
        print(f"총 {len(documents)}개 페이지를 의미있는 단위(Chunk)로 분할합니다...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        print(f"총 {len(chunks)}개의 Chunk로 분할되었습니다.")
        return chunks

    def load_documents_from_directory(self, directory_path: str) -> list:
        """
        지정된 디렉터리에서 모든 PDF 문서를 로드하고 텍스트를 추출합니다.

        :param directory_path: PDF 파일이 있는 디렉터리 경로
        :return: LangChain의 Document 객체 리스트
        """
        print(f"'{directory_path}' 디렉터리에서 PDF 문서를 로딩합니다...")
        
        # 지정된 경로의 모든 .pdf 파일 목록을 가져옵니다.
        pdf_files = glob.glob(f"{directory_path}/*.pdf")
        
        if not pdf_files:
            print("경고: 해당 디렉터리에서 PDF 파일을 찾을 수 없습니다.")
            return []

        documents = []
        for file_path in pdf_files:
            # PyPDFLoader를 사용하여 PDF를 로드합니다.
            loader = PyPDFLoader(file_path)
            # load_and_split()은 PDF를 페이지별로 나누어 Document 객체로 만듭니다.
            documents.extend(loader.load_and_split())
        
        print(f"총 {len(pdf_files)}개의 PDF 파일에서 {len(documents)}개의 페이지를 로드했습니다.")
        return documents