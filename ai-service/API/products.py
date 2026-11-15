from os import getenv
from pathlib import Path
from warnings import warn
from json import load

#Статичный каталог продуктов EnergyTool.
DEFAULT_CATALOG: dict[str, dict[str, str]] = {
    "cloud_askue": {
        "name": "Облачная АСКУЭ",
        "description": (
            "Автоматизированный сбор и обработка показаний приборов учёта "
            "электроэнергии. Работает с 300+ типами счётчиков, позволяет "
            "видеть расход по объектам, арендаторам и цехам в одном кабинете."
        ),
        "when": "клиент собирает показания вручную; нужен учёт по объектам/цехам/арендаторам",
    },
    "solar_station": {
        "name": "Солнечные электростанции",
        "description": (
            "Сетевые (снижение платежа) и гибридные (с накопителями, для "
            "автономии) СЭС мощностью 15–500 кВт."
        ),
        "when": "клиент хочет снизить расходы на электроэнергию, есть подходящая кровля или площадь",
    },
    "orem": {
        "name": "Выход на оптовый рынок электроэнергии (ОРЕМ)",
        "description": (
            "Переход на оптовый рынок для предприятий с присоединённой "
            "мощностью от 750 кВт. Включает аудит существующего учёта."
        ),
        "when": "клиент крупный потребитель (>= 750 кВт) и хочет снизить стоимость электроэнергии",
    },
    "loss_detection": {
        "name": "Поиск коммерческих потерь",
        "description": (
            "ML-сервис приоритизации выездных проверок для электросетевых "
            "компаний. Помогает находить хищения и вмешательства в счётчики."
        ),
        "when": "клиент — сетевая или ресурсоснабжающая организация, ищет потери",
    },
    "photo_meter_reading": {
        "name": "Распознавание показаний по фотографии",
        "description": (
            "Для приборов, не подключённых к автоматике. Электроэнергия и вода. "
            "Распознаёт показания, серийный номер, штрих-код, год выпуска."
        ),
        "when": "у клиента есть счётчики без автоматической передачи данных",
    },
    "audit": {
        "name": "Аудит учёта и расчёт экономического эффекта",
        "description": (
            "Проверка существующей системы учёта и оценка потенциальной "
            "экономии до внедрения решения."
        ),
        "when": "клиент хочет сначала понять, сколько можно сэкономить",
    },
}


def get_catalog(path_to_catalog: str) -> dict[str, dict[str, str]]:
    global DEFAULT_CATALOG
    fpath: Path = Path(path_to_catalog)
    
    if not fpath.exists():
        warn(f"""
             The file of catalog at the specified path({fpath}) was not found.\n 
             The default catalog is used.""")
        
        return DEFAULT_CATALOG
    
    with open(fpath, 'r', encoding="UTF-8") as file:
        catalog = load(file)
        
    return catalog

CATALOG_PATH = getenv("CATALOG_PATH")
PRODUCT_CATALOG: dict[str, dict[str, str]] = get_catalog(CATALOG_PATH)

def products_block() -> str:
    global PRODUCT_CATALOG
    """Форматирует каталог для вставки в SYSTEM_PROMPT."""
    lines = []
    for pid, p in PRODUCT_CATALOG.items():
        lines.append(f"- `{pid}` — {p['name']}: {p['description']}")
        lines.append(f"    Когда уместно: {p['when']}")
    return "\n".join(lines)


def is_valid_product_id(pid: str) -> bool:
    return pid in PRODUCT_CATALOG