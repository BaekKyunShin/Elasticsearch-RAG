import sys
import os
import locale

# 인코딩 문제 해결을 위한 환경 설정
if sys.platform.startswith('win'):
    # Windows 환경에서의 인코딩 설정
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    sys.stdin = codecs.getreader('utf-8')(sys.stdin.detach())
else:
    # Unix/Linux/macOS 환경에서의 인코딩 설정
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag_service import RAGService

def main():
    """
    메인 실행 함수
    """
    # RAGService 객체를 생성합니다.
    rag_service = RAGService()
    
    # Elasticsearch 인덱스를 설정합니다.
    rag_service.setup_elasticsearch_index()

    # PDF 문서들이 있는 디렉터리 경로를 지정합니다.
    # 'data'는 elasticsearch-RAG 폴더 바로 아래의 폴더입니다.
    data_directory = "data"
    
    # 처음 실행하거나 PDF 문서가 추가되었을 때만 아래 주석을 해제하고 실행하세요.
    """----------------------------------
    # 문서를 로드하는 메서드를 호출합니다.
    documents = rag_service.load_documents_from_directory(data_directory)
    
    if not documents:
        print("처리할 문서가 없어 프로그램을 종료합니다.")
        return

    # 로드된 문서를 Chunk 단위로 분할합니다.
    chunks = rag_service.split_documents(documents)

    # 분할된 Chunk를 임베딩합니다.
    embeddings = rag_service.embed_documents(chunks)

    # Chunk와 Embedding을 Elasticsearch에 인덱싱합니다.
    rag_service.index_documents(chunks, embeddings)

    print("\n--- 모든 작업이 성공적으로 완료되었습니다. ---")
    ----------------------------------"""

    print("\n--- RAG 시스템이 준비되었습니다. 질문을 입력하세요 ('exit' 입력 시 종료) ---")

    while True:
        try:
            query = input("Question: ")
        except UnicodeDecodeError:
            print("입력 인코딩 오류가 발생했습니다. 다시 시도해주세요.")
            continue
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
            
        if query.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break
        
        # RAG 체인을 사용하여 답변 생성
        # invoke() 메서드에 사용자 질문(query)을 전달하면, 우리가 설계한 파이프라인(검색 → 프롬프트 조합 → LLM 답변 생성)이 실행되고 최종 결과가 반환됩니다.
        answer = rag_service.rag_chain.invoke(query)
        
        print("\nAnswer:")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()