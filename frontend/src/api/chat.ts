import { http } from './client';
import type { ChatRequest, ChatResponse } from '../types/api';

export function SendMessage(payload: ChatRequest): Promise<ChatResponse> {
    return http.post<ChatResponse>('api/chat', payload);
}