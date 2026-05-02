# Lambda 배포용 컨테이너 이미지.
# 로컬 개발은 `uvicorn main:app --reload` 로 직접 실행하면 충분하지만,
# 프로덕션은 AWS Lambda Container Image 로 빌드해 ECR 에 push 한다.
FROM public.ecr.aws/lambda/python:3.11

# 의존성 캐시
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# 코드
COPY main.py ${LAMBDA_TASK_ROOT}/
COPY routes ${LAMBDA_TASK_ROOT}/routes
COPY rag ${LAMBDA_TASK_ROOT}/rag
COPY prompts ${LAMBDA_TASK_ROOT}/prompts
COPY models ${LAMBDA_TASK_ROOT}/models

# Mangum 어댑터를 통해 ASGI → Lambda 변환
CMD ["main.handler"]
