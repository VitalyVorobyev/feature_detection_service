# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /

# Add system deps if needed:
# RUN apt-get update && apt-get install -y libpq5 ...

# Copy and install deps
COPY pyproject.toml poetry.lock* requirements*.txt* ./
# pick your toolchain; here is a plain pip path:
RUN pip install --upgrade pip \
 && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi \
 || pip install .

# Copy code
COPY . .

# Expose service on 8080 (change as needed)
ENV HOST=0.0.0.0 PORT=8080
CMD ["uvicorn", "fds:app", "--host", "0.0.0.0", "--port", "8080"]
