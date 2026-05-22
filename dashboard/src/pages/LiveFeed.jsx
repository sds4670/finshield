import { useState, useEffect, useCallback, useMemo } from 'react'

export default function LiveFeed() {
  const [rows, setRows]   = useState([])
  const [loading, setLoading] = useState(true)
  const [flash, setFlash] = useState(null)
  const [stats, setStats] = useState(null)

  const fetchStats = useCallback(async () => {
    try {
      const res  = await fetch('http://localhost:8000/api/stats')
      const data = await res.json()
      setStats(data)
    } catch {}
  }, [])

  const fetch_ = useCallback(async () => {
    try {
      const res  = await fetch('http://localhost:8000/api/live-feed')
      const data = await res.json()
      setRows(prev => {
        if (data.length > 0 && prev[0]?.transaction_id !== data[0]?.transaction_id) {
          setFlash(data[0]?.transaction_id)
          setTimeout(() => setFlash(null), 800)
        }
        return data
      })
      setLoading(false)
    } catch { setLoading(false) }
  }, [])

  useEffect(() => {
    fetch_()
    fetchStats()
    const id = setInterval(() => { fetch_(); fetchStats(); }, 3000)
    return () => clearInterval(id)
  }, [fetch_, fetchStats])

  if (loading) return <div className="loading">Initializing Stream...</div>

  const approved = rows.filter(r=>r.status==='APPROVED').length
  const flagged  = rows.filter(r=>r.status==='FLAGGED').length
  const blocked  = rows.filter(r=>r.status==='BLOCKED').length

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      
      {/* Asymmetric KPI Bar */}
      <div className="kpi-bar">
        <div className="kpi-left">
          <div className="kpi-card approved">
            <div className="kpi-label">Approved</div>
            <div className="kpi-value">{Number(stats?.approved || approved).toLocaleString()}</div>
          </div>
          <div className="kpi-card flagged">
            <div className="kpi-label">Flagged</div>
            <div className="kpi-value">{Number(stats?.flagged || flagged).toLocaleString()}</div>
          </div>
          <div className="kpi-card blocked">
            <div className="kpi-label">Blocked</div>
            <div className="kpi-value">{Number(stats?.blocked || blocked).toLocaleString()}</div>
          </div>
        </div>
        <div className="kpi-right">
          <div className="kpi-right-label">Last 5 min</div>
          <div className="kpi-right-value">
            {stats?.recent_5min || rows.length} 
            <div className="pulse-dot-subtle" />
          </div>
        </div>
      </div>

      {/* Bespoke Table */}
      <div className="table-wrap">
        <table className="bespoke-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>User</th>
              <th>Merchant</th>
              <th>Amount</th>
              <th>Device</th>
              <th>Status</th>
              <th>Risk Score</th>
              <th>Trigger</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, index) => {
              // Intentional imperfections
              const isCreamRow = index % 7 === 3; 
              // Varying row heights from 56px to 64px
              const rowHeight = 56 + (index % 3) * 4; 

              return (
                <tr key={r.transaction_id} style={{
                  background: r.transaction_id === flash ? '#f1ece1' : (isCreamRow ? '#FFFdfa' : 'transparent'),
                  height: rowHeight,
                  transition: 'background 0.5s ease-out'
                }}>
                  <td style={{ color: 'var(--text-muted)' }}>{r.created_at}</td>
                  <td style={{ fontWeight: 600 }}>{r.user_name?.split(' ')[0]}</td>
                  <td style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {r.merchant_name}
                  </td>
                  <td className="tabular-amt">
                    ₹{Number(r.amount).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </td>
                  <td>{r.device_type}</td>
                  <td>
                    <span className={`badge-bespoke ${r.status}`}>{r.status}</span>
                  </td>
                  <td className="tabular-amt" style={{ color: 'var(--text-muted)' }}>
                    {(r.score || 0).toFixed(3)}
                  </td>
                  <td style={{ fontSize: 12, color: 'var(--text-muted)', maxWidth: 160, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                    {r.rule_triggered || '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

    </div>
  )
}
