FROM langchain/langgraph-api:3.13-92975bb

RUN apt update && apt upgrade -y
RUN apt install -y build-essential wget curl vim lsb-release gnupg2

ADD . /deps/ai_finance

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -r /deps/ai_finance/requirements.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{ \
    "finance_agent": "/deps/ai_finance/app/finance_agent/graph.py:graph" \
}'

WORKDIR /deps/ai_finance

CMD exec uvicorn langgraph_api.server:app --log-config /api/logging.json --host 0.0.0.0 --port $PORT --no-access-log --reload
