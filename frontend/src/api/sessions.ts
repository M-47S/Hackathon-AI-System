import { http } from './client';
import type { SessionResponse, HistoryResponse } from '../types/api';

export function createSession(): Promise<SessionResponse> {
    return http.post<SessionResponse>('/api/sessions');
}

export function getHistory(sessionId: string): Promise<HistoryResponse> {
    return http.get<HistoryResponse>(`/api/sessions/${sessionId}/messages`);
}