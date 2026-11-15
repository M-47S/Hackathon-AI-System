import { http } from './client';
import type { SessionListItem } from '../types/api';

export interface AuthCredentials {
    login: string;
    password: string;
    is_admin: boolean;
}

export interface UserProfile {
    login: string;
    is_admin: boolean;
    sessions?: SessionListItem[]; // Бэкенд присылает массив сессий
}

export function login(credentials: AuthCredentials): Promise<UserProfile> {
    return http.post<UserProfile>('/api/login/manual', credentials);
}

export function register(credentials: AuthCredentials): Promise<UserProfile> {
    return http.post<UserProfile>('/api/register', credentials);
}

export function checkAuth(): Promise<UserProfile> {
    return http.get<UserProfile>('/api/login/cookie');
}

export function logout(): Promise<void> {
    return http.post<void>('/api/logout');
}