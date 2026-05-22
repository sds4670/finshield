import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ScatterChart, Scatter, Cell, CartesianGrid } from 'recharts'
import { Store, AlertOctagon } from 'lucide-react'

const COLORS = ['#8B9B88', '#D07B65', '#C48793', '#7288A2', '#A29587', '#93A8AC', '#B1A8A2', '#6D827A']

export default function MerchantInsights() {
  const [data, setData] = useState(null)

  useEffect(() => {
    fetch('http://localhost:8000/api/merchant-insights')
      .then(r => r.json()).then(setData).catch(() => {})
  }, [])

  if (!data) return <div className="loading">AGGREGATING MERCHANT INTEL...</div>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Fraud Rate by Category */}
      <div className="grid g2">
        <div className="card">
          <div className="card-title"><Store size={14} /> Fraud Rate by Category</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.by_category} layout="vertical" margin={{ left: 60, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tickFormatter={v => `${v}%`} domain={[0, 'auto']} />
              <YAxis type="category" dataKey="category" tick={{ fontSize: 11 }} width={80} />
              <Tooltip formatter={v => [`${v}%`, 'Fraud Rate']} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
              <Bar dataKey="fraud_rate" radius={[0,4,4,0]} maxBarSize={30}>
                {data.by_category.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title"><Store size={14} /> Avg Transaction Amount by Category</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.by_category} layout="vertical" margin={{ left: 60, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tickFormatter={v => `₹${(v/1000).toFixed(0)}k`} />
              <YAxis type="category" dataKey="category" tick={{ fontSize: 11 }} width={80} />
              <Tooltip formatter={v => [`₹${Number(v).toLocaleString()}`, 'Avg Amount']} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
              <Bar dataKey="avg_amount" radius={[0,4,4,0]} fill="#A29587" maxBarSize={30} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Volume vs Fraud scatter */}
      <div className="card">
        <div className="card-title"><AlertOctagon size={14} /> Risk Matrix: Volume vs Fraud Rate (by Merchant)</div>
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="txn_count" type="number" name="Volume" label={{ value: 'Transaction Volume', position: 'insideBottom', offset: -10, fill: 'var(--text3)', fontSize: 11 }} />
            <YAxis dataKey="fraud_rate_pct" type="number" name="Fraud Rate" tickFormatter={v=>`${v}%`} label={{ value: 'Fraud Rate %', angle: -90, position: 'insideLeft', fill: 'var(--text3)', fontSize: 11 }} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }}
              formatter={(v, name) => [name === 'fraud_rate_pct' ? `${v}%` : v, name === 'fraud_rate_pct' ? 'Fraud Rate' : 'Volume']} />
            <Scatter data={data.volume_vs_fraud} fill="var(--terracotta)" fillOpacity={0.8} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Top risky merchants table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)' }}>
          <span className="card-title" style={{ margin: 0 }}><AlertOctagon size={14}/> Top High-Risk Merchants</span>
        </div>
        <div className="table-wrap">
          <table className="bespoke-table">
            <thead>
              <tr>
                <th>Merchant</th>
                <th>Category</th>
                <th>City</th>
                <th>Total Txns</th>
                <th>Fraud Count</th>
                <th>Fraud Rate %</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {data.top_risky_merchants.map((m, i) => (
                <tr key={i}>
                  <td style={{ color: 'var(--text-main)', fontWeight: 600 }}>{m.name}</td>
                  <td style={{ color: 'var(--text-muted)' }}>{m.category}</td>
                  <td style={{ color: 'var(--text-muted)' }}>{m.city}</td>
                  <td className="tabular-amt" style={{ color: 'var(--text-main)' }}>{m.total_txns}</td>
                  <td className="tabular-amt" style={{ color: 'var(--dusty-rose)', fontWeight: 600 }}>{m.fraud_count}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <div style={{ width: 80, height: 6, background: 'var(--border)', borderRadius: 3 }}>
                        <div style={{ width: `${Math.min(m.fraud_rate_pct * 3, 100)}%`, height: '100%', background: m.fraud_rate_pct > 15 ? 'var(--dusty-rose)' : m.fraud_rate_pct > 8 ? 'var(--terracotta)' : 'var(--sage-green)', borderRadius: 3 }} />
                      </div>
                      <span className="tabular-amt" style={{ fontSize: 12, fontWeight: 500, color: m.fraud_rate_pct > 15 ? 'var(--dusty-rose)' : m.fraud_rate_pct > 8 ? 'var(--terracotta)' : 'var(--sage-green)' }}>
                        {m.fraud_rate_pct}%
                      </span>
                    </div>
                  </td>
                  <td>
                    {m.is_blacklisted
                      ? <span className="badge-bespoke BLOCKED">BLACKLISTED</span>
                      : <span className="badge-bespoke APPROVED">ACTIVE</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
