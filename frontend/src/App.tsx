import { useEffect, useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Chat } from "./components/Chat";
import { AuthForm } from "./components/AuthForm"; 
import { AdminPanel } from "./components/AdminPanel";
import { Sidebar } from "./components/Sidebar";
import { checkAuth, logout } from "./api/auth";
import type { UserProfile } from "./api/auth";
import type { SessionListItem } from "./types/api";
import './App.css';

function App() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);
  const [isDebugMode, setIsDebugMode] = useState(false);
  const [sessions, setSessions] = useState<SessionListItem[]>([]);

  useEffect(() => {
    async function initAuth(){
      try {
        const userData = await checkAuth();
        setUser(userData);
        if (userData.sessions) {
            // Разворачиваем массив, чтобы новые (с конца) стали первыми (сверху)
            setSessions([...userData.sessions].reverse()); 
        }
      } catch (err) {
        setUser(null);
      } finally {
        setIsCheckingAuth(false);
      }
    }
    initAuth();
  }, []);

  const handleLoginSuccess = (userData: UserProfile) => {
      setUser(userData);
      if (userData.sessions) {
          setSessions([...userData.sessions].reverse()); 
      }
  };
  const handleLogout = async () => {
      try { await logout(); } catch (e) { console.error(e); } 
      finally { setUser(null); setSessions([]); }
  };

  if (isCheckingAuth) {
    return <div className="auth-container" style={{ color: 'white', fontSize: '1.5rem' }}>Загрузка...</div>;
  }

  if (!user) {
    return <AuthForm onAuthSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="app-layout">
      <Sidebar 
        user={user} 
        onLogout={handleLogout} 
        isDebugMode={isDebugMode} 
        onToggleDebug={() => setIsDebugMode(!isDebugMode)} 
        sessions={sessions}
        setSessions={setSessions}
      />

      <main className="main-content">
        <Routes>
          {user.is_admin && (
            <Route path="/" element={<Navigate to="/admin" replace />} />
          )}
          {!user.is_admin && (
            <Route 
                path="/" 
                element={<Chat 
                    isDebugMode={isDebugMode} 
                    // Добавляем новый чат В НАЧАЛО списка
                    onSessionCreated={(s) => setSessions(prev => [s, ...prev])} 
                />} 
            />
          )}
          {!user.is_admin && (
            <Route 
                path="/chat/:sessionId" 
                element={<Chat 
                    isDebugMode={isDebugMode} 
                    onSessionUpdated={(id, summary) => {
                        setSessions(prev => prev.map(s => 
                            s.session_id === id ? { ...s, conversation_summary: summary } : s
                        ));
                    }} 
                />} 
            />
          )}
          {user.is_admin && (
            <Route path="/admin" element={<AdminPanel />} />
          )}
          <Route path="*" element={<Navigate to={user.is_admin ? "/admin" : "/"} replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;