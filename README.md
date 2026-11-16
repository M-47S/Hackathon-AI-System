
# EnergyTool AI — ИИ-консультант для бизнеса

**Прототип B2B-консультанта для сайта MTS EnergyTool.** Диалоговый ИИ, который помогает клиенту подобрать продукт, собирает профиль его бизнеса и формирует готовую заявку менеджеру.

---

## 📌 Что делает проект

Пользователь пишет в чат свободным текстом («У нас 50 магазинов, собираем показания вручную»). ИИ:

1. **Понимает задачу** — задаёт уточняющие вопросы, если данных недостаточно.
2. **Находит решение** — ищет в базе знаний EnergyTool (RAG) и рекомендует подходящий продукт.
3. **Извлекает профиль** — тип бизнеса, количество объектов, способ учёта, проблему.
4. **Формирует заявку** — при готовности клиента передаёт менеджеру контакты + резюме диалога.
5. **Работает в режиме отладки** — отдаёт `sequence` (цепочка мыслей ИИ) и `used_documents` (какие `.md` использовались).

---

## 🏗 Архитектура

```text
┌──────────────┐
│   Browser    │
│ (React SPA)  │
└──────┬───────┘
       │ HTTP / JSON
       ▼
┌─────────────────────────┐
│   Frontend (nginx)      │  :80
│   React + Vite          │
└──────┬──────────────────┘
       │ /api/* (proxy)
       ▼
┌─────────────────────────┐
│   Backend (Go/Gin)      │  :8080
│   Auth + Orchestration  │
└──────┬──────────┬───────┘
       │          │
       │ /v1/chat │ SQL
       ▼          ▼
┌──────────┐  ┌────────────┐
│AI Service│  │ PostgreSQL │  :5432
│(FastAPI) │  │   (GORM)   │
│  :8001   │  └────────────┘
└────┬─────┘
     │
     ▼
┌──────────────────┐
│ GigaChat / LLM   │
│ (внешний API)    │
└──────────────────┘
```

**Три сервиса, три сети — полная изоляция:**

| Сеть              | Кто видит     | Кто не видит                 |
| --------------------- | --------------------- | -------------------------------------- |
| `frontend-net`      | frontend ↔ backend   | db, ai-service                         |
| `db-net` (internal) | backend ↔ db         | frontend, ai-service, интернет |
| `ai-net`            | backend ↔ ai-service | frontend, db                           |

---

## 📦 Стек

| Слой                               | Технологии                                                                           |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Frontend**                     | React 19, TypeScript, Vite 8, React Router v7, Nginx                                           |
| **Backend**                      | Go 1.26, Gin, GORM, PostgreSQL 16, Resty, bcrypt                                               |
| **AI Service**                   | Python 3.11, FastAPI, Pydantic 2, FAISS, llama_index, sentence-transformers                    |
| **LLM**                          | GigaChat (по умолчанию) / Gemini / OpenAI (провайдер-агностично) |
| **Инфраструктура** | Docker Compose, healthcheck chain, named volumes                                               |

---

## 📁 Структура репозитория

```text
HackAI/                        ← 🔴 КОРЕНЬ РЕПОЗИТОРИЯ (запуск здесь)
├── docker-compose.yml         # Оркестрация 4 сервисов + 3 сетей
├── .env                       # Общие переменные (DB_USER, DB_PASSWORD, DB_NAME)
│
├── ai-service/                # Python AI Service (RAG + LLM)
│   ├── .env.example           # ⚠️ Скопировать в .env перед запуском
│   ├── .env                   # AI_AUTH_KEY, настройки RAG (не коммитится)
│   ├── Dockerfile
│   ├── API/                   # FastAPI, ChatService, Pydantic schemas
│   ├── RAG/                   # Chunker, Embedder, Reranker, FAISS
│   ├── LLM/                   # GigaChat/OpenAI client, auth
│   ├── Storage/               # Source (.md), Catalog (products.json), Index (FAISS)
│   └── Models/                # ML-модели (скачиваются автоматически)
│
├── backend/                   # Go Backend (API + Auth + DB)
│   ├── .env.example           # ⚠️ Скопировать в .env перед запуском
│   ├── Dockerfile
│   ├── main.go, routes.go, handlers.go, functions.go
│   └── tables.go, models.go
│
└── frontend/                  # React SPA
    ├── .env.example           # ⚠️ Скопировать в .env перед запуском
    ├── Dockerfile             # Multi-stage: node build → nginx serve
    ├── nginx.conf             # Проксирование /api/* → backend
    └── src/
```

---

## 🚀 Быстрый старт

### ⚠️ Важно: команды запускаются **из корня репозитория**

`docker-compose.yml` находится **в корне** репозитория (там же, где папки `ai-service/`, `backend/`, `frontend/`). Все команды ниже нужно выполнять **именно из этой директории**:

```bash
# Открой терминал и перейди в корень репозитория (там, где лежит docker-compose.yml)
cd /path/to/HackAI

# Проверь, что видишь docker-compose.yml
ls docker-compose.yml
```

---

### Шаг 1: Скопируй `.env.example` → `.env` в каждом сервисе

В репозитории **три** отдельных `.env.example` — по одному в каждой папке. Перед первым запуском создай `.env` в каждой:

```bash
# 1. AI Service (самый важный — здесь LLM-ключ)
cp ai-service/.env.example ai-service/.env
# → Открой ai-service/.env и впиши AI_AUTH_KEY (ключ GigaChat/Gemini/OpenAI)

# 2. Backend
cp backend/.env.example backend/.env
# → Правь только если меняешь порты/пароли БД (по умолчанию работает)

# 3. Frontend
cp frontend/.env.example frontend/.env
# → VITE_API_URL по умолчанию http://localhost:8080, менять не нужно
```

**Минимум для запуска:** заполнить **только `ai-service/.env`** полем `AI_AUTH_KEY`. Остальное — работает из коробки.

---

### Шаг 2: Запусти проект из корня

```bash
# Убедись, что ты в корне репозитория (где docker-compose.yml)
docker compose -p hackapp up -d
```

### Шаг 3: Дождись, пока все сервисы станут healthy

```bash
docker compose -p hackapp ps
```

**Ожидаемый результат через 5–15 минут:**

```
NAME             STATUS
ai-cont          Up (healthy)     ← AI-сервис (сборка индекса + загрузка моделей)
db-cont          Up (healthy)     ← PostgreSQL
backend-cont     Up (healthy)     ← Go API (ждёт ai-service)
frontend-cont    Up               ← Nginx (ждёт backend)
```

### Шаг 4: Открой приложение

| Что                              | URL                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| **Чат-интерфейс** | **http://localhost:80** ← основной адрес фронтенда |
| **Swagger AI-сервиса** | http://localhost:8001/docs                                                      |
| **Backend health**            | http://localhost:8080/api/health                                                |
| **AI health**                 | http://localhost:8001/health                                                    |

⚠️ **Фронтенд слушает порт `80`** (стандартный HTTP-порт nginx), поэтому открывай именно **`http://localhost:80`**. Браузер может автоматически убрать `:80` в адресной строке, но не удивляйся — это тот же адрес.

Если порт 80 занят (например, локальным nginx/IIS) — поменяй маппинг в `docker-compose.yml`:

```yaml
  frontend:
    ports:
      - "3000:80"   # тогда открывать http://localhost:3000
```

---

## ⏱ ⚠️ Первый запуск — дольше обычного

**Первый запуск занимает 5–15 минут** (вместо ~30 секунд при повторных). Это нормально:

1. **Скачиваются ML-модели** (~2.3 GB):

   - `multilingual-e5-large` (эмбеддинги, ~1.5 GB)
   - `bge-reranker-v2-m3` (реранкер, ~800 MB)
2. **Собирается FAISS-индекс** из `.md`-файлов базы знаний (~1–3 минуты на CPU).
3. **Инициализируется PostgreSQL** и применяются миграции.

**Как отслеживать прогресс:**

```bash
docker compose -p hackapp logs -f ai-service
```

В логах увидишь прогресс:

```
[RAG-BUILD] [embed] batch #1: 32 texts, 28500 chars
[RAG-BUILD] [embed] batch #1 done in 24.3s
...
✅ AI Service ready at http://0.0.0.0:8001
```

**Модели скачиваются один раз** и живут в named volumes (`hackapp_ai-models`, `hackapp_ai-index`). **Повторные запуски — ~30 секунд.**

⚠️ **НЕ используй `docker compose down -v`** — флаг `-v` удаляет named volumes, и модели (2.3 GB) придётся качать заново.

---

## 🛑 Остановка

```bash
# Из корня репозитория
docker compose -p hackapp down
```

Данные (модели, индекс, БД) **сохраняются** в named volumes и восстанавливаются при следующем запуске.

---

## 🔄 Ежедневные команды (из корня репозитория)

```bash
# Запустить (быстро, всё из кэша)
docker compose -p hackapp up -d

# Остановить (данные сохраняются)
docker compose -p hackapp down

# Пересобрать один сервис (после правки кода)
docker compose -p hackapp up -d --build backend

# Логи в реальном времени
docker compose -p hackapp logs -f ai-service
docker compose -p hackapp logs -f backend

# Статус
docker compose -p hackapp ps
```

---

## 🔑 Порты

| Сервис         | Хост | Контейнер | Доступ извне         |
| -------------------- | -------- | ------------------ | ------------------------------- |
| **frontend**   | `80`   | `80`             | **http://localhost:80**   |
| **backend**    | `8080` | `8080`           | http://localhost:8080           |
| **ai-service** | `8001` | `8001`           | http://localhost:8001/docs      |
| **db**         | `5433` | `5432`           | localhost:5433 (для DBeaver) |

---

## 🧪 Smoke-тест после запуска

```bash
# 1. AI-сервис жив?
curl http://localhost:8001/health
# → {"status":"ok"}

# 2. Backend жив?
curl http://localhost:8080/api/health
# → {"status":"ok"}

# 3. Frontend доступен?
curl -I http://localhost:80
# → HTTP/1.1 200 OK

# 4. Полный тест AI-чата
curl -X POST http://localhost:8001/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "request_id": "r1",
    "message": "У нас 50 магазинов, собираем показания вручную",
    "history": [],
    "customer_profile": {},
    "conversation_summary": null
  }'
```

**Ожидание:** `action: "recommend"`, `recommended_product_ids: ["cloud_askue"]`, заполненный `customer_profile`.

**Демо-сценарий в браузере** (http://localhost:80):

1. Зарегистрируйся (поставь галочку «Администратор», чтобы увидеть панель менеджера).
2. Напиши в чат: *«У нас 50 магазинов, собираем показания вручную»*.
3. Дождись ответа с карточкой продукта (5–15 сек на первый ответ).
4. Продолжи диалог — ИИ задаст уточняющий вопрос.
5. При `action: "offer_handoff"` заполни форму заявки.
6. Зайди в `/admin` — заявка должна быть в списке.

---

## 🐛 Troubleshooting

### `backend-cont` unhealthy бесконечно

Backend ждёт `ai-service` (может занимать 5–15 минут на первом запуске). Проверь:

```bash
docker compose -p hackapp logs ai-service --tail=50
```

Если видишь `[embed] batch #N` — идёт сборка индекса, **просто жди**.

Если `ai-service` healthy, а backend нет:

```bash
docker compose -p hackapp logs backend-cont --tail=50
```

Возможные причины:

- **`wget` не находит `/api/health`** — проверь `docker exec backend-cont wget -qO- http://localhost:8080/api/health`
- **Backend упал с panic** — нет подключения к БД (проверь `DATABASE_URL` в логах)

### Frontend показывает `405 Method Not Allowed` на `/api/*`

В `frontend/nginx.conf` **нет проксирования** `/api/` на backend. Проверь:

```bash
docker exec frontend-cont cat /etc/nginx/conf.d/default.conf
```

Должен быть блок:

```nginx
location /api/ {
    proxy_pass http://backend:8080;
    ...
}
```

Если нет — пересобери образ `frontend` с правильным `nginx.conf` или примонтируй файл через `volumes:`.

### Порт 80 занят (Windows: IIS, Skype, локальный nginx)

Проверь:

```powershell
Get-NetTCPConnection -LocalPort 80
```

Если кто-то слушает — либо останови сервис, либо поменяй маппинг в compose:

```yaml
  frontend:
    ports:
      - "3000:80"   # открывать http://localhost:3000
```

### Модели не скачиваются, таймаут

Проверь интернет внутри контейнера:

```bash
docker exec ai-cont curl -I https://huggingface.co
```

Если таймаут — проблема в сети. Попробуй скачать модели предварительно:

```bash
docker run --rm \
  -v hackapp_ai-models:/models \
  python:3.11-slim \
  bash -c "pip install -q huggingface_hub && \
    python -c \"from huggingface_hub import snapshot_download; \
      snapshot_download('intfloat/multilingual-e5-large', local_dir='/models/multilingual-e5-large'); \
      snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='/models/bge-reranker-v2-m3')\""
```

### `docker compose down -v` случайно удалил модели

Восстановление: перезапусти `docker compose -p hackapp up -d` — модели скачаются заново (5–15 минут). В будущем делай бэкап:

```bash
docker run --rm \
  -v hackapp_ai-models:/data:ro \
  -v $(pwd)/backups:/backup \
  alpine tar -czf /backup/ai-models.tgz -C /data .
```

---

## 📚 Документация по сервисам

Подробности по каждому модулю — в `README.md` соответствующих папок:

- **`ai-service/ABOUT_ai-service.md`** — RAG-пайплайн, переменные, архитектура консультанта
- **`backend/README.md`** — API endpoints, схема БД, зависимости
- **`frontend/README.md`** — React-компоненты, роутинг, сборка

---

## ✅ Что реализовано (по ТЗ кейса)

| Требование                               | Реализация                                                                                              |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| RAG по базе знаний                     | FAISS + cross-encoder reranking + Small-to-Big                                                                    |
| Консультант, а не поисковик | Structured Outputs: `answer_question` / `ask_question` / `recommend` / `offer_handoff` / `out_of_scope` |
| Сбор заявки                              | `customer_profile` + `conversation_summary` извлекаются LLM'ом                                   |
| Контекст диалога                    | Полная история + merge профиля (не теряем факты)                                 |
| Скорость                                   | Параллельная обработка, retry на 429, connection pooling                                   |
| Веб-интерфейс с отладкой      | `sequence` (мысли ИИ) + `used_documents` (какие `.md` использовались)             |
| Интеграция с сайтом               | REST API + Swagger, Pydantic-валидация                                                                   |
| Развёртывание                         | Docker Compose, 3 изолированные сети, healthcheck-цепочка                                 |

**Оценка на онлайн-грейдере МТС:** **0.63951** (лучший воспроизводимый результат).

---

## 🎯 Команда запуска (кратко)

```bash
# ⚠️ Все команды — ИЗ КОРНЯ РЕПОЗИТОРИЯ (где docker-compose.yml)
cd /path/to/HackAI

# 1. Один раз — создать .env во всех трёх папках
cp ai-service/.env.example ai-service/.env    # вписать AI_AUTH_KEY
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# 2. Запуск (первый раз 5–15 мин, потом ~30 сек)
docker compose -p hackapp up -d

# 3. Открыть фронтенд
# → http://localhost:80

# 4. Остановка
docker compose -p hackapp down
```
