import { useState } from 'react'
import LiveFeed from './pages/LiveFeed'
import MerchantInsights from './pages/MerchantInsights'
import RiskAnalytics from './pages/RiskAnalytics'
import ModelInsights from './pages/ModelInsights'
import { Activity, Store, PieChart, Cpu, ShieldAlert, Shield, Search, BellRing } from 'lucide-react'
import './App.css'

const NAV = [
  { id: 'live',     label: 'Live Stream',       icon: <Activity size={18} strokeWidth={2} /> },
  { id: 'merchant', label: 'Merchant Intel',    icon: <Store size={18} strokeWidth={2} /> },
  { id: 'risk',     label: 'Risk Analytics',    icon: <PieChart size={18} strokeWidth={2} style={{ transform: 'translateY(1px)' }} /> },
  { id: 'model',    label: 'Model Insights',    icon: <Cpu size={18} strokeWidth={2} /> },
]

export default function App() {
  const [page, setPage] = useState('live')

  const PAGES = { live: LiveFeed, merchant: MerchantInsights, risk: RiskAnalytics, model: ModelInsights }
  const PageComponent = PAGES[page]

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <img src="/fintech_shield_logo.png" alt="FinShield Logo" style={{ width: 32, height: 32, objectFit: 'contain' }} />
          <div>
            <div className="brand-name">FinShield</div>
            <div className="brand-sub">Provided Intelligence</div>
          </div>
        </div>

        <nav className="nav">
          {NAV.map(n => (
            <button key={n.id} className={`nav-item ${page === n.id ? 'active' : ''}`} onClick={() => setPage(n.id)}>
              <span className="nav-icon">{n.icon}</span>
              <span>{n.label}</span>
            </button>
          ))}
        </nav>

        {/* Custom bespoke footer card resembling a sticky note */}
        <div className="sidebar-footer-card">
          <div className="sfc-icon"><Shield size={16} strokeWidth={2.5} style={{ opacity: 0.8 }} /></div>
          <div className="sfc-title">System Status</div>
          <div className="sfc-desc">
            All pipelines nominal. Fraud models operating at 99.8% availability.
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1 className="welcome-text">{NAV.find(n=>n.id===page)?.label}</h1>
            <p className="welcome-sub">Viewing real-time bespoke metrics</p>
          </div>
          
          <div className="topbar-right">
            <div className="search-bar">
              <Search size={16} strokeWidth={2} />
              <input type="text" placeholder="Search..." />
            </div>
          </div>
        </header>

        <div className="content">
          {PageComponent ? <PageComponent /> : <div className="loading">NOT FOUND</div>}
        </div>
      </main>
    </div>
  )
}
