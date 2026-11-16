const BASE_URL = import.meta.env.VITE_API_URL ?? '';

export class ApiError extends Error {
    status: number;
    code: string;

    constructor (status: number, code: string, message: string) {
        super(message);
        this.name = 'ApiError'
        this.status = status;
        this.code = code;
    }
}

type HttpMethod = 'GET' | 'POST' | 'DELETE'

interface RequestOptions {
    method?: HttpMethod;
    body?: unknown;
    timeoutMs?: number;
}

async function request<T>(
    path: string,
    options: RequestOptions = {}
): Promise<T> {
    const { method = 'GET', body, timeoutMs = 60000 } = options;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

    try {
        const response = await fetch (`${BASE_URL}${path}`, {
            method,
            headers: {
                'Content-Type': 'application/json',
                Accept: 'application/json',
            },
            credentials: 'include',
            body: body !== undefined ? JSON.stringify(body) : undefined,
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (response.status === 204) {
            return undefined as T;
        }

        const data = await response.json().catch(() => null);

        if (!response.ok) {
            console.log(data);
            const code = data?.error_code ?? `HTTP_${response.status}`;
            const message = data?.detail ?? `Запрос завершился с кодом ${response.status}`;
            throw new ApiError(response.status, code, message); 
        }

        return data as T;
    } catch (err) {
        clearTimeout(timeoutId);

        if (err instanceof ApiError) throw err;

        if (err instanceof DOMException && err.name === 'AbortError') {
            throw new ApiError(408, 'TIMEOUT', 'Превышено время ожидания ответа');
        }

        throw new ApiError(0, 'NETWORK_ERROR', 'Не удалось связаться с сервером');
    }
}

export const http = {
    get: <T>(path: string) => request<T>(path, { method: 'GET' }),
    post: <T>(path: string, body?: unknown) => request<T>(path, { method: 'POST', body }),
    delete: <T>(path: string) => request<T>(path, { method: 'DELETE' })
};