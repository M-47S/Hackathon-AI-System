import { http } from './client';
import type { ChatRequest, ChatResponse } from '../types/api';

export function sendMessage(payload: ChatRequest): Promise<ChatResponse> {
    return http.post<ChatResponse>('/api/chat', payload);
}