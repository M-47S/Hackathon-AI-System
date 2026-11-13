import { http } from "./client";


export interface AuthCredentials {
    login: string;
    password: string;
    is_admin: boolean;
}

export interface UserProfile {
    login: string;
    is_admin: boolean;
}

export function login(credentials: AuthCredentials): Promise<UserProfile> {
    return http.post<UserProfile>('/api/login', credentials);
}

export function register(credentials: AuthCredentials): Promise<UserProfile> {
    return http.post<UserProfile>('/api/register', credentials);
}