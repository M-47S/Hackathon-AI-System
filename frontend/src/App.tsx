import { useEffect, useState } from "react";
import { Routes, Route, Navigate, NavLink } from "react-router-dom";
import { Chat } from "./components/Chat";
import { AuthForm } from "./components/AuthFrom";
import { AdminPanel } from "./components/AdminPanel";
import { checkAuth, type UserProfile } from "./api/auth";
import './App.css';


function App() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

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
    return (
      <div className="auth-container" style={{ color: 'white', fontSize: '1.5rem' }}>
        Загрузка
      </div>
    );
  }

  if (!user) {
    return <AuthForm onAuthSuccess={setUser} />;
  }

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <h2>EnergyTool AI</h2>
        
        <NavLink to="/" className={({ isActive }) => (isActive ? 'active' : '')}>
          Чат
        </NavLink>
        
        {user.is_admin && (
          <NavLink to="/admin" className={({ isActive }) => (isActive ? 'active' : '')}>
            Панель администратора
          </NavLink>
        )}

        <button onClick={() => setUser(null)} className="logout-btn">
          Выйти ({user.login})
        </button>
      </aside>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Chat />} />
          {user.is_admin && (
            <Route path="/admin" element={<AdminPanel />} />
          )}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;