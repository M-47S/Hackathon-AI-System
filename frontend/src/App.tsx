import { useEffect, useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Chat } from "./components/Chat";
import { AuthForm } from "./components/AuthForm";
import { AdminPanel } from "./components/AdminPanel";
import { Sidebar } from "./components/Sidebar";
import { checkAuth } from "./api/auth";
import type { UserProfile } from "./api/auth";
import './App.css';

function App() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);
  const [isDebugMode, setIsDebugMode] = useState(false);

  useEffect(() => {
    async function initAuth(){
      try {
        const userData = await checkAuth();
        setUser(userData);
      } catch (err) {
        setUser(null);
      } finally {
        setIsCheckingAuth(false);
      }
    }
    initAuth();
  }, []);

  if (isCheckingAuth) {
    return <div className="auth-container" style={{ color: 'white', fontSize: '1.5rem' }}>Загрузка...</div>;
  }

  if (!user) {
    return <AuthForm onAuthSuccess={setUser} />;
  }

  return (
    <div className="app-layout">
      <Sidebar 
        user={user} 
        onLogout={() => setUser(null)} 
        isDebugMode={isDebugMode} 
        onToggleDebug={() => setIsDebugMode(!isDebugMode)} 
      />

      <main className="main-content">
        <Routes>
          {user.is_admin && (
            <Route path="/" element={<Navigate to="/admin" replace />} />
          )}

          {!user.is_admin && (
            <Route path="/" element={<Chat isDebugMode={isDebugMode} />} />
          )}
          
          {!user.is_admin && (
            <Route path="/chat/:sessionId" element={<Chat isDebugMode={isDebugMode} />} />
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