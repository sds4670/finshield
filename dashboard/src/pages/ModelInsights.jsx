import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts'
import { Cpu, Target, Eye } from 'lucide-react'

const COLORS = ['#8B9B88', '#D07B65', '#C48793', '#7288A2', '#A29587']

export default function ModelInsights() {
  const [data, setData] = useState(null)

  useEffect(() => {
    fetch('http://localhost:8000/api/model-insights')
      .then(r => r.json()).then(setData).catch(() => {})
  }, [])

  if (!data) return <div className="loading">EXTRACTING ML EXPLANATIONS...</div>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      <div className="grid g2">
        {/* Rule Breakdown */}
        <div className="card">
          <div className="card-title"><Target size={14} /> Rule Trigger Frequency</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.rule_breakdown} layout="vertical" margin={{ left: 80, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" />
              <YAxis type="category" dataKey="rule_triggered" tick={{ fontSize: 10 }} width={100} />
              <Tooltip formatter={v => [v, 'Triggers']} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
              <Bar dataKey="count" radius={[0,4,4,0]} maxBarSize={30}>
                {data.rule_breakdown.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Avg ML Score by Rule */}
        <div className="card">
          <div className="card-title"><Cpu size={14} /> Avg ML Score per Rule</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.score_by_rule} layout="vertical" margin={{ left: 80, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[0, 1]} />
              <YAxis type="category" dataKey="rule" tick={{ fontSize: 10 }} width={100} />
              <Tooltip formatter={v => [v, 'Avg ML Score']} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
              <Bar dataKey="avg_score" radius={[0,4,4,0]} maxBarSize={30}>
                {data.score_by_rule.map((d, i) => (
                  <Cell key={i} fill={d.avg_score > 0.75 ? 'var(--dusty-rose)' : d.avg_score > 0.45 ? 'var(--terracotta)' : 'var(--sage-green)'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* SHAP Explanations Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)' }}>
          <span className="card-title" style={{ margin: 0 }}><Eye size={14}/> ML Model Explainability (Recent Interventions)</span>
        </div>
        <div className="table-wrap">
          <table className="bespoke-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>User</th>
                <th>Merchant</th>
                <th>Amount (₹)</th>
                <th>Rule Triggered</th>
                <th>ML Score</th>
                <th>Decision</th>
                <th>SHAP Explanation</th>
              </tr>
            </thead>
            <tbody>
              {data.recent_flagged.map((r, i) => (
                <tr key={i}>
                  <td style={{ color: 'var(--text3)', fontSize: 12 }}>{r.created_at}</td>
                  <td style={{ color: 'var(--text)', fontWeight: 700 }}>{r.user_name?.split(' ')[0]}</td>
                  <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {r.merchant_name}
                  </td>
                  <td className="tabular-amt" style={{ color: 'var(--text-main)' }}>
                    {Number(r.amount).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </td>
                  <td>
                    <span className="badge-bespoke" style={{ background: 'var(--bg-hover)', color: 'var(--text-muted)', borderRadius: 4 }}>
                      {r.rule_triggered || 'N/A'}
                    </span>
                  </td>
                  <td className="tabular-amt" style={{ color: r.ml_score > 0.75 ? 'var(--dusty-rose)' : r.ml_score > 0.45 ? 'var(--terracotta)' : 'var(--sage-green)' }}>
                    {r.ml_score ? r.ml_score.toFixed(3) : '—'}
                  </td>
                  <td><span className={`badge-bespoke ${r.final_decision}`}>{r.final_decision}</span></td>
                  <td style={{ fontSize: 11, color: 'var(--text-muted)', maxWidth: 220, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                    {r.notes || '—'}
                  </td>
                </tr>
              ))}
              {data.recent_flagged.length === 0 && (
                <tr>
                  <td colSpan="8" style={{ textAlign: 'center', padding: '30px', color: 'var(--text3)' }}>
                    No recent interventions to display.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
