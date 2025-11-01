# AI Finance Assistant

# Local Development

## Quick Start
- Create `.env` file following the `.env.example` file.
- ```shell
  uv python pin 3.1
  uv venv --no-python-downloads
  ```
- ```shell
  docker-compose up -d
  docker logs -f ai_finance-api-1
  docker exec -it ai_finance-api-1 /bin/bash
  docker compose down
  ```
- Open `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:{LANGGRAPH_LOCAL_PORT}` in your browser, e.g. https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8129

Here is a simple flow chart:

```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```
