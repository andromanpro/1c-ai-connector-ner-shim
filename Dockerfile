# Сервис распознавания имен ИИконы: Natasha + HTTP. Модели зашиты в пакет Natasha, сети контейнер не требует.
FROM python:3.12-slim

# Зависимости ставятся по uv.lock - те же версии, на которых прошли тесты.
RUN pip install --no-cache-dir uv==0.10.3 && useradd -m shim
WORKDIR /app
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project && rm -rf /root/.cache

COPY shim.py /app/shim.py
USER shim

ENV PATH="/app/.venv/bin:$PATH" \
    NER_HOST=0.0.0.0 \
    NER_PORT=9002

EXPOSE 9002
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9002/health')"

CMD ["python", "shim.py"]
