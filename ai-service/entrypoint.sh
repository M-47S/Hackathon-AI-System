#!/bin/bash
set -e

echo "🚀 Checking models..."

# Запускаем загрузчик эмбеддера
python RAG/Embedding/model_loader.py

# Запускаем загрузчик реранкера
python RAG/Ranking/model_loader.py

echo "✅ Models ready. Starting FastAPI..."

# Запускаем сервис
exec python -m uvicorn API.main:app --host 0.0.0.0 --port 8001