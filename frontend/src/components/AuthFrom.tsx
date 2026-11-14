import { useState } from "react";
import type { SyntheticEvent } from "react";
import type { UserProfile } from "../api/auth";
import { login, register } from "../api/auth";
import './AuthForm.css'


interface AuthFormProps {
    onAuthSuccess: (user: UserProfile) => void;
}

export function AuthForm({ onAuthSuccess }: AuthFormProps) {
    const [isLoginMode, setIsLoginMode] = useState(true);
    const [loginStr, setLoginStr] = useState('');
    const [password, setPassword] = useState('');
    const [isAdmin, setIsAdmin] = useState(false);

    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    async function handleSubmit(e: SyntheticEvent<HTMLFormElement>) {
        e.preventDefault();
        if (!loginStr.trim() || !password.trim()) return;

        setError(null);
        setIsLoading(true);

        try {
            let user: UserProfile;

            if (isLoginMode) {
                user = await login({ login: loginStr, password});
            } else{
                user = await register({ login: loginStr, password, is_admin: isAdmin});
            }

            onAuthSuccess(user);
        } catch (err: any) {
            setError(err.message || 'Ошибка авторизации. Проверьте данные.')
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <div className="auth-container">
            <form className="auth-form" onSubmit={handleSubmit}>
                <h2>{isLoginMode ? 'Вход' : 'Регистрация'} в ENERGYTOOL AI</h2>

                {error && <div className="auth-error">{error}</div>}

                <div className="auth-field">
                    <label>Логин</label>
                    <input 
                        type="text"
                        value={loginStr}
                        onChange={(e) => setLoginStr(e.target.value)}
                        disabled={isLoading}
                        required
                    />
                </div>

                <div className="auth-field">
                    <label>Пароль</label>
                    <input 
                        type="text"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        disabled={isLoading}
                        required
                    />
                </div>

                {!isLoginMode && (
                    <div>
                        <label>
                            <input 
                                type="checkbox"
                                checked={isAdmin}
                                onChange={(e) => setIsAdmin(e.target.checked)}
                                disabled={isLoading}
                            />
                            Зарегестрироваться как администратор
                        </label>
                    </div>
                )}

                <button type="submit" disabled={isLoading || !loginStr || !password}>
                    {isLoading ? 'Подождите' : isLoginMode ? 'Войти' : 'Зарегистрироваться'}
                </button>
                
                <p className="auth-switch" onClick={() => {
                    setIsLoginMode(!isLoginMode);
                    setError(null);
                }}>
                    {isLoginMode ? 'Нет аккаунта? Зарегистрироваться' : 'Уже есть аккаунт? Войти'}
                </p>
            </form>
        </div>
    );
}