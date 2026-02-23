# 1. Base Image: 파이썬 3.13 슬림 버전
FROM python:3.13-slim

# 2. 작업 디렉토리 설정
WORKDIR /project

# 3. 시스템 필수 패키지 설치 (PyTorch 구동 등에 필요한 기본 C 라이브러리)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 패키지 설치
# (이 부분을 먼저 하는 이유: 소스코드가 바뀌어도 패키지 설치는 캐싱해두어 빌드 속도를 높이기 위함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 실제 구동에 필요한 소스코드만 복사 (app 폴더와 src/models 폴더)
COPY app/ ./app/
COPY src/models/ ./src/models/

# 6. 환경 변수 설정 (우리가 터미널에서 쳤던 PYTHONPATH=. 의 역할)
ENV PYTHONPATH=/project

# 7. 외부와 통신할 포트 뚫어주기
EXPOSE 8000

# 8. 최종 시동 명령어!
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]