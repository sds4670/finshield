import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, CartesianGrid, Area, AreaChart } from 'recharts'
import { Activity, Clock, ShieldAlert, Smartphone } from 'lucide-react'

const PIE_COLORS = ['#8B9B88', '#D07B65', '#C48793', '#7288A2', '#A29587']

export default function RiskAnalytics() {
  const [data, setData] = useState(null)

  useEffect(() => {
    const load = () =>
      fetch('http://localhost:8000/api/risk-analytics')
        .then(r => r.json()).then(setData).catch(() => {})
    load()
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [])

  if (!data) return <div className="loading">ANALYSING RISK VECTORS...</div>

  const hourLabels = Array.from({ length: 24 }, (_, i) => ({
    hour_of_day: i,
    label: i === 0 ? '12a' : i < 12 ? `${i}a` : i === 12 ? '12p' : `${i-12}p`
  }))
  const hourlyWithLabel = hourLabels.map(h => {
    const found = data.hourly_heatmap.find(d => d.hour_of_day === h.hour_of_day)
    return { ...h, fraud_rate: found ? parseFloat(found.fraud_rate) : 0, total: found ? found.total : 0 }
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Fraud trend */}
      <div className="card">
        <div className="card-title"><Activity size={14} /> Global Fraud Rate Trend (Last 30 Minutes)</div>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={data.fraud_trend} margin={{ right: 20, top: 10, left: -20 }}>
            <defs>
              <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--blue)" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="var(--blue)" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorFraud" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--red)" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="var(--red)" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="minute" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v, n) => [v, n === 'fraud_count' ? 'Fraud' : 'Total']} />
            <Area type="monotone" dataKey="total" stroke="var(--primary)" fillOpacity={1} fill="url(#colorTotal)" strokeWidth={2} />
            <Area type="monotone" dataKey="fraud_count" stroke="var(--terracotta)" fillOpacity={1} fill="url(#colorFraud)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', gap: 24, marginTop: 12, justifyContent: 'flex-end' }}>
          <span style={{ fontSize: 11, fontWeight: 700, fontFamily: 'Space Mono', color: 'var(--blue)' }}>── Total Volume</span>
          <span style={{ fontSize: 11, fontWeight: 700, fontFamily: 'Space Mono', color: 'var(--red)', textShadow: 'var(--red-glow)' }}>── Fraud Volume</span>
        </div>
      </div>

      <div className="grid g2">
        {/* Hourly heatmap */}
        <div className="card">
          <div className="card-title"><Clock size={14} /> Temporal Risk Heatmap</div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={hourlyWithLabel} margin={{ left: -15, right: 10, top: 10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 10 }} interval={1} />
              <YAxis tickFormatter={v => `${v}%`} tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => [`${v}%`, 'Fraud Rate']} labelFormatter={l => `Hour: ${l}`} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
              <Bar dataKey="fraud_rate" radius={[4,4,0,0]}>
                {hourlyWithLabel.map((h, i) => (
                  <Cell key={i} fill={
                    h.hour_of_day >= 1 && h.hour_of_day <= 4 ? 'var(--dusty-rose)'
                    : h.fraud_rate > 10 ? 'var(--terracotta)'
                    : 'var(--sage-green)'
                  } />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'Space Mono', marginTop: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ width: 8, height: 8, background: 'var(--dusty-rose)', borderRadius: '50%', display: 'inline-block' }}></span>
            1am-4am high-risk window
          </div>
        </div>

        {/* Score distribution */}
        <div className="card">
          <div className="card-title"><ShieldAlert size={14} /> Risk Score Distribution</div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data.score_distribution} margin={{ left: -15, right: 10, top: 10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => [v, 'Transactions']} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
              <Bar dataKey="count" radius={[4,4,0,0]} maxBarSize={50}>
                {data.score_distribution.map((d, i) => (
                  <Cell key={i} fill={
                    d.bucket === '0.8-1.0' ? 'var(--dusty-rose)'
                    : d.bucket === '0.6-0.8' ? 'var(--terracotta)'
                    : d.bucket === '0.4-0.6' ? 'var(--sage-green)'
                    : 'var(--primary-bg)'
                  } />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Device fraud breakdown */}
      <div className="grid g2">
        <div className="card">
          <div className="card-title"><Smartphone size={14} /> Threat Surface by Device</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 12 }}>
            {data.by_device.map((d, i) => (
              <div key={i}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', fontFamily: 'Space Mono', textTransform: 'uppercase' }}>{d.device_type}</span>
                  <span style={{ fontSize: 13, fontWeight: 700, fontFamily: 'Space Mono', color: d.fraud_rate > 15 ? 'var(--red)' : d.fraud_rate > 8 ? 'var(--amber)' : 'var(--green)' }}>
                    {d.fraud_rate}%
                  </span>
                </div>
                <div style={{ height: 8, background: 'rgba(24, 38, 58, 0.5)', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.min(d.fraud_rate * 3, 100)}%`,
                    height: '100%',
                    borderRadius: 4,
                    background: d.fraud_rate > 15 ? 'var(--red)' : d.fraud_rate > 8 ? 'var(--amber)' : 'var(--green)',
                    boxShadow: d.fraud_rate > 15 ? 'var(--red-glow)' : d.fraud_rate > 8 ? 'var(--amber-glow)' : 'var(--green-glow)',
                    transition: 'width 0.4s cubic-bezier(0.4, 0, 0.2, 1)'
                  }} />
                </div>
                <div style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'Space Mono', marginTop: 6 }}>
                  {d.fraud_count} fraud / {d.total} total
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-title"><ShieldAlert size={14} /> Risk Topology</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={data.score_distribution} dataKey="count" nameKey="bucket"
                cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={4} stroke="none">
                {data.score_distribution.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
              </Pie>
              <Tooltip formatter={(v, n) => [v, n]} />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center', marginTop: 16 }}>
            {data.score_distribution.map((d, i) => (
              <span key={i} style={{ fontSize: 11, fontWeight: 700, fontFamily: 'Space Mono', color: PIE_COLORS[i] }}>
                ■ {d.bucket}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
