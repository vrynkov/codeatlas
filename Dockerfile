# ---- Stage 1: build the React frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: the actual app (FastAPI serves both the API and the built frontend) ----
FROM python:3.11-slim
WORKDIR /app/backend

# git is needed at runtime: the indexer clones repos with it.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
# Lands at /app/frontend/dist - matches app/main.py's own path.parents[2]/"frontend"/"dist" check.
COPY --from=frontend-build /frontend/dist /app/frontend/dist

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Hugging Face Spaces' Docker SDK expects the app on port 7860.
EXPOSE 7860
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
