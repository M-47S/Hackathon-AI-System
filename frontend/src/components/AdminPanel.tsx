import { useEffect, useState } from "react";
import { mockGetLeads, type MockLead } from "../api/mock";
import { data } from "react-router-dom";
import './AdminPanel.css'

export function AdminPanel() {
  const [leads, setLeads] = useState<MockLead[]>([]);

  useEffect(() => {
    mockGetLeads().then(data => setLeads(data.items));
  }, []);

  return (
    <div className="admin-panel">
      <h2>Заявки от клиентов</h2>
      <div className="leads-grid">
        {leads.map(lead => (
          <div key={lead.id} className="lead-card">
            <h3>{lead.name}</h3>
            <p><strong>Телефон:</strong> {lead.phone}</p>
            <p><strong>Проблема:</strong> {lead.problem}</p>
          </div>
        ))}
      </div>
    </div>
  );
}