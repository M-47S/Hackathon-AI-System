import { useState } from "react";
import { Routes, Route, Navigate, Link } from "react-router-dom";
import { Chat } from "./components/Chat";
import { AuthForm } from "./components/AuthFrom";
import { AdminPanel } from "./components/AdminPanel";
import type { UserProfile } from "./api/auth";
import './App.css';


function App() {
  const [user, setUser] = useState<UserProfile | null>(null);

  if (!user) {
    return <AuthForm onAuthSuccess={setUser} />;
  }

  return (
    <div className="app-layout">
      <nav className="top-nav">
        <Link to="/">Чат</Link>
        {user.is_admin && <Link to="/admin">Панель администратора</Link>}
        <button onClick={() => setUser(null)} className="logout-btn">
          Выйти ({user.login})
        </button>
      </nav>
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