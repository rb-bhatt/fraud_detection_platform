FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -c "from pathlib import Path; lines=Path('requirements.txt').read_text(encoding='utf-16').splitlines(); Path('requirements-linux.txt').write_text('\n'.join(line for line in lines if not line.startswith('pywin32==')), encoding='utf-8')" \
    && pip install --no-cache-dir -r requirements-linux.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY api ./api
COPY src ./src
COPY models ./models
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]