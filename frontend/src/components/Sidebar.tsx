import { NavLink, useNavigate } from "react-router-dom";
import { deleteSession } from "../api/sessions";
import type { SessionListItem } from "../types/api";
import type { UserProfile } from "../api/auth";

interface SidebarProps {
    user: UserProfile;
    onLogout: () => void;
    isDebugMode: boolean;
    onToggleDebug: () => void;
    sessions: SessionListItem[];
    setSessions: React.Dispatch<React.SetStateAction<SessionListItem[]>>;
}

export function Sidebar({ user, onLogout, isDebugMode, onToggleDebug, sessions, setSessions }: SidebarProps) {
    const navigate = useNavigate();

    const handleNewChatClick = () => navigate('/');

    const handleDelete = async (e: React.MouseEvent, sessionId: string) => {
        e.preventDefault();
        e.stopPropagation();

        try {
            await deleteSession(sessionId);
            
            // Формируем новый список без удаленного чата
            const remainingSessions = sessions.filter(s => s.session_id !== sessionId);
            setSessions(remainingSessions);

            // Если мы удалили тот чат, в котором сейчас находимся
            if (window.location.pathname.includes(sessionId)) {
                if (remainingSessions.length > 0) {
                    // Кидаем в первый (самый новый) из оставшихся
                    navigate(`/chat/${remainingSessions[0].session_id}`);
                } else {
                    // Если чатов не осталось, идем на главную (Chat создаст новый)
                    navigate('/');
                }
            }
        } catch (err) {
            console.error('Ошибка удаления', err);
        }
    };

    const getSessionTitle = (session: SessionListItem) => {
        if (!session.conversation_summary) return 'Новый диалог';
        return session.conversation_summary.length > 30 
            ? session.conversation_summary.substring(0, 30) + '...' 
            : session.conversation_summary;
    };

    return (
        <aside className="sidebar">
            <h2 className="sidebar-title">ENERGYTOOL AI</h2>
            
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
                                    title={session.conversation_summary || 'Новый диалог'}
                                >
                                    {getSessionTitle(session)}
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

            <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {!user.is_admin && (
                    <label style={{ color: '#DCDDDE', display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.9rem' }}>
                        <input type="checkbox" checked={isDebugMode} onChange={onToggleDebug} />
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