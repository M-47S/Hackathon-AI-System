import { http } from './client';
import type { SessionResponse, HistoryResponse } from '../types/api';

export interface SessionListItem {
    session_id: string;
    title: string;
    created_at: string;
}

export function createSession(): Promise<SessionResponse> {
    return http.post<SessionResponse>('/api/sessions');
}

export function getHistory(sessionId: string): Promise<HistoryResponse> {
    return http.get<HistoryResponse>(`/api/sessions/${sessionId}/messages`);
}

export function getSessionList(): Promise<{ items: SessionListItem[] }> {
    return http.get< { items: SessionListItem[] }>('/api/sessions') 
}

export function deleteSession(sessionId: string): Promise<void> {
    return http.delete<void>(`/api/sessions/${sessionId}`);
}