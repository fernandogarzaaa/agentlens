# agentlens trace viewer + CLI image.
#   docker build -t agentlens .
#   docker run --rm -p 8000:8000 -v "$PWD/traces:/data" -e AGENTLENS_TRACES=/data/traces.jsonl agentlens
#   docker run --rm agentlens agentlens --help
FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AGENTLENS_TRACES=/data/traces.jsonl
COPY --from=build /dist/*.whl /tmp/
RUN WHEEL=$(ls /tmp/*.whl) && pip install --no-cache-dir "${WHEEL}[server]" && rm -f /tmp/*.whl \
    && useradd --create-home --uid 10001 agentlens && mkdir -p /data && chown agentlens /data
WORKDIR /home/agentlens
COPY --chown=agentlens examples ./examples
USER agentlens
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health').status != 200)"
CMD ["uvicorn", "agentlens.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
