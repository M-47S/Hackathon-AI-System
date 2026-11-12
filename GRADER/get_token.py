"""
get_token.py - получает свежий access_token из GigaChat OAuth (срок ~30 мин).

Использование:
    python get_token.py
Печатает токен в stdout (только токен, без лишнего текста).

AUTH_KEY (base64 из кабинета GigaChat) берётся из env GIGACHAT_AUTH_KEY
или из файла .gigachat_auth рядом со скриптом.
"""
import os
import sys
import uuid
import json
import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")


def get_auth_key() -> str:
    key = os.getenv("GIGACHAT_AUTH_KEY")
    if key:
        return key.strip()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gigachat_auth")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    print(
        "ERROR: GIGACHAT_AUTH_KEY not set and .gigachat_auth not found.",
        file=sys.stderr,
    )
    sys.exit(2)


def fetch_token(auth_key: str) -> str:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {auth_key}",
    }
    try:
        r = requests.post(
            OAUTH_URL,
            headers=headers,
            data={"scope": SCOPE},
            timeout=30,
            verify=False,  # сертификат Минцифры; локально это ок
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        print(f"ERROR: OAuth request failed: {e}", file=sys.stderr)
        sys.exit(3)
    token = data.get("access_token")
    if not token:
        print(f"ERROR: no access_token in response: {data}", file=sys.stderr)
        sys.exit(4)
    return token


if __name__ == "__main__":
    print(fetch_token(get_auth_key()))