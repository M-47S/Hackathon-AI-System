import { useState, useEffect } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { deleteSession } from "../api/sessions";
import type { SessionListItem } from "../api/sessions";
import type { UserProfile } from "../api/auth";
import { mockGetSessionsList } from "../api/mock";

interface SidebarProps {
    user: UserProfile;
    onLogout: () => void;
    isDebugMode: boolean;
    onToggleDebug: () => void;
}

export function Sidebar({ user, onLogout, isDebugMode, onToggleDebug } : SidebarProps) {
    const [sessions, setSessions] = useState<SessionListItem[]>([]);
    const navigate = useNavigate();

    useEffect(() => {
        if (!user.is_admin) {
            mockGetSessionsList().then(data => setSessions(data.items));
        }
    }, [user.is_admin]);

    const handleNewChatClick = () => {
        navigate('/'); 
    };

    const handleDelete = async (e: React.MouseEvent, sessionId: string) => {
        e.preventDefault(); // Чтобы не срабатывал переход по ссылке NavLink
        e.stopPropagation();

        try {
            // Раскомментировать, когда бэк будет готов:
            // await deleteSession(sessionId);
            
            // Заглушка: просто удаляем из стейта
            setSessions(prev => prev.filter(s => s.session_id !== sessionId));
            
            // Если удалили текущий открытый чат, переходим на главную
            if (window.location.pathname.includes(sessionId)) {
                navigate('/');
            }
        } catch (err) {
            console.error('Ошибка удаления', err);
        }
    };

    return (
        <aside className="sidebar">
            <h2 className="sidebar-title">ENERGYTOOL AI</h2>
            
            {/* Интерфейс пользователя */}
            {!user.is_admin && (
                <>
                    <button onClick={handleNewChatClick} className="new-chat-btn">
                        + Новый чат
                    </button>
                    <div className="session-list">
                        {sessions.map(session => (
                            <div key={session.session_id} className="session-item-wrapper">
                                <NavLink 
                                    to={`/chat/${session.session_id}`}
                                    className={({ isActive }) => `session-item ${isActive ? 'active' : ''}`}
                                >
                                    {session.title || 'Новый диалог'}
                                </NavLink>
                                <button 
                                    className="delete-chat-btn" 
                                    onClick={(e) => handleDelete(e, session.session_id)}
                                    title="Удалить чат"
                                >
                                    ✕
                                </button>
                            </div>
                        ))}
                    </div>
                </>
            )}

            {/* Блок внизу */}
            <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                
                {/* Чекбокс отладки ТОЛЬКО для пользователя */}
                {!user.is_admin && (
                    <label style={{ color: '#DCDDDE', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.9rem' }}>
                        <input 
                            type="checkbox" 
                            checked={isDebugMode} 
                            onChange={onToggleDebug} 
                        />
                        Режим отладки
                    </label>
                )}

                <button onClick={onLogout} className="logout-btn" style={{ marginTop: 0 }}>
                    Выйти ({user.login})
                </button>
            </div>
        </aside>
    );
}