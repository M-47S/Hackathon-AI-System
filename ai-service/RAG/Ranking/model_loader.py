import os
from huggingface_hub import snapshot_download
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 1. Настройки ---
MODEL_ID = "BAAI/bge-reranker-v2-m3"

MODELS_DIR = os.getenv("MODELS_DIR", f"C:/Users/0/OneDrive/Desktop/HackAI/ai-service/Models")
MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", f"bge-reranker-v2-m3")

MODEL_PATH = MODELS_DIR + '/' + MODEL_NAME

# --- 2. Проверка и скачивание ---
# Проверяем, существует ли папка и есть ли в ней файлы (например, config.json)
model_config_file = os.path.join(MODEL_PATH, "config.json")

if not os.path.exists(model_config_file):
    print(f"Модель не найдена по пути: {MODEL_PATH}")
    print("Начинаем скачивание. Это займет время...")
    
    # Создаем папку, если её нет
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Скачиваем модель прямо в указанную папку
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_PATH
    )
    print("✅ Модель успешно скачана!")
else:
    print("✅ Модель уже существует локально, используем её.")

# --- 3. Инициализация модели ---
print("Загрузка модели в память...")
rank_model = HuggingFaceEmbedding(
    model_name=MODEL_PATH, # Теперь передаем путь к скачанной папке
    device="cpu"           # или "cuda", если есть видеокарта
)

# --- 4. Тест ---
print("Модель готова к работе!")
# Проверим, что она выдает векторы
test_vector = rank_model.get_text_embedding("passage: Тестовый текст")
print(f"Размерность вектора: {len(test_vector)}")