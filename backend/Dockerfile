FROM langchain/langgraph-api:3.13

RUN apt update && apt upgrade -y
RUN apt install -y build-essential wget curl vim lsb-release gnupg2

COPY netskope.pem /usr/local/share/ca-certificates/netskope.crt
RUN update-ca-certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

ADD . /deps/ai_finance

RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -r /deps/ai_finance/requirements.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{ \
    "finance_agent": "/deps/ai_finance/app/finance_agent/graph.py:graph" \
}'

WORKDIR /deps/ai_finance

CMD exec uvicorn langgraph_api.server:app --log-config /api/logging.json --host 0.0.0.0 --port $PORT --no-access-log --reload
