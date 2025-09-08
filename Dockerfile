# 1. 엘라스틱서치 8.15.0 공식 이미지를 기반으로 시작합니다.
FROM docker.elastic.co/elasticsearch/elasticsearch:8.15.0

# 2. nori 한글 형태소 분석기 플러그인을 설치하는 명령어를 실행합니다.
RUN /usr/share/elasticsearch/bin/elasticsearch-plugin install analysis-nori