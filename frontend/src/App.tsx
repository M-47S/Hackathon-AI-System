import { useState } from "react";
import { Routes, Route, Navigate, Link } from "react-router-dom";
import { Chat } from "./components/Chat";
import { AuthForm } from "./components/AuthFrom";
import type { UserProfile } from "./api/auth";
import './App.css';


function App() {
  const [user, setUser] = useState<UserProfile | null>(null);

  return (
    <div className="app-layout">
      <aside>
        
      </aside>
      <main className="main-content">
        <Chat />
      </main>
    </div>
  );
}

export default App;