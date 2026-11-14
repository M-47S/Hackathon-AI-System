import { http } from "./client";


export interface LoginCredentials {
    login: string;
    password: string;
}

export interface RegisterCredentials {
    login: string;
    password: string;
    is_admin: boolean;
}

export interface UserProfile {
    login: string;
    is_admin: boolean;
}

export function login(credentials: LoginCredentials): Promise<UserProfile> {
    return http.post<UserProfile>('/api/login/manual', credentials);
}

export function register(credentials: RegisterCredentials): Promise<UserProfile> {
    return http.post<UserProfile>('/api/register', credentials);
}

export function checkAuth(): Promise<UserProfile> {
    return http.get<UserProfile>('/api/login/cookie');
}