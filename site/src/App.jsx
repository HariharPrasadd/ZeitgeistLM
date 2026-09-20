import { useState, useRef, useEffect, useCallback } from 'react'

const FIRST_MONTH = 2008 * 12
const LAST_MONTH = 2024 * 12 + 11
const INITIAL_MONTH = 2020 * 12 + 6
const GENERATE_URL = import.meta.env.VITE_MODAL_GENERATE_URL
const TOTAL_MONTHS = LAST_MONTH - FIRST_MONTH
const MO = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
const MO_FULL = ['January','February','March','April','May','June',
                 'July','August','September','October','November','December']

function monthToDate(m) {
  const year = Math.floor(m / 12)
  const mo = m % 12
  return { year, mo, label: `${MO_FULL[mo]} ${year}`, iso: `${year}-${String(mo+1).padStart(2,'0')}-01` }
}

// ── Dial geometry ────────────────────────────────────────────────────────────
const R = 100, CX = 130, CY = 130, SIZE = 260
const ARC_START = 135, ARC_SWEEP = 270, ARC_END = ARC_START + ARC_SWEEP
const TRAIN_T0 = (2011 * 12 - FIRST_MONTH) / TOTAL_MONTHS
const TRAIN_T1 = (2020 * 12 + 11 - FIRST_MONTH) / TOTAL_MONTHS

const toRad = d => (d * Math.PI) / 180
const pt = (angle, r = R) => [CX + r * Math.cos(toRad(angle)), CY + r * Math.sin(toRad(angle))]
const arcD = (a1, a2) => {
  const [sx, sy] = pt(a1), [ex, ey] = pt(a2)
  const sw = ((a2 - a1) % 360 + 360) % 360
  return `M ${sx.toFixed(2)} ${sy.toFixed(2)} A ${R} ${R} 0 ${sw > 180 ? 1 : 0} 1 ${ex.toFixed(2)} ${ey.toFixed(2)}`
}

function Dial({ month, onChange }) {
  const svgRef = useRef(null)
  const dragging = useRef(false)
  const t = (month - FIRST_MONTH) / TOTAL_MONTHS
  const thumbAngle = ARC_START + t * ARC_SWEEP
  const [tx, ty] = pt(thumbAngle)
  const { year, mo } = monthToDate(month)
  const extrapolated = year < 2011 || year > 2020

  const resolve = useCallback((cx, cy) => {
    const rect = svgRef.current.getBoundingClientRect()
    const mx = cx - rect.left - CX, my = cy - rect.top - CY
    let a = Math.atan2(my, mx) * 180 / Math.PI
    if (a < 0) a += 360
    let o = (a - ARC_START + 360) % 360
    if (o > ARC_SWEEP) o = o - ARC_SWEEP < (360 - ARC_SWEEP) / 2 ? ARC_SWEEP : 0
    onChange(Math.round(FIRST_MONTH + (o / ARC_SWEEP) * TOTAL_MONTHS))
  }, [onChange])

  useEffect(() => {
    const move = e => {
      if (!dragging.current) return
      const p = e.touches?.[0] ?? e
      resolve(p.clientX, p.clientY)
    }
    const up = () => { dragging.current = false }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    window.addEventListener('touchmove', move, { passive: true })
    window.addEventListener('touchend', up)
    return () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
      window.removeEventListener('touchmove', move)
      window.removeEventListener('touchend', up)
    }
  }, [resolve])

  const ticks = Array.from({ length: 17 }, (_, i) => {
    const y = 2008 + i
    const tickT = Math.min(1, Math.max(0, (y * 12 - FIRST_MONTH) / TOTAL_MONTHS))
    const a = ARC_START + tickT * ARC_SWEEP
    const isKey = y === 2011 || y === 2020
    const [x1, y1] = pt(a, R - 6)
    const [x2, y2] = pt(a, R + 6)
    return { y, a, isKey, x1, y1, x2, y2 }
  })

  // Year labels at the four key points
  const yearLabels = [
    { year: 2008, t: 0 },
    { year: 2011, t: TRAIN_T0 },
    { year: 2020, t: TRAIN_T1 },
    { year: 2024, t: 1 },
  ].map(({ year: y, t: lt }) => {
    const a = ARC_START + lt * ARC_SWEEP
    const [lx, ly] = pt(a, R + 22)
    const anchor = lx < CX - 10 ? 'end' : lx > CX + 10 ? 'start' : 'middle'
    return { y, lx, ly, anchor }
  })

  return (
    <div
      className="dial"
      onMouseDown={e => { dragging.current = true; resolve(e.clientX, e.clientY) }}
      onTouchStart={e => { dragging.current = true; const p = e.touches[0]; resolve(p.clientX, p.clientY) }}
    >
      <svg
        ref={svgRef}
        width={SIZE}
        height={SIZE}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        style={{ display: 'block', cursor: 'pointer', touchAction: 'none', userSelect: 'none' }}
        aria-label={`Time dial — ${monthToDate(month).label}`}
        role="slider"
        aria-valuenow={month}
        aria-valuemin={FIRST_MONTH}
        aria-valuemax={LAST_MONTH}
        aria-valuetext={monthToDate(month).label}
      >
        {/* Glow behind filled arc */}
        {t > 0.002 && (
          <path d={arcD(ARC_START, thumbAngle)}
            fill="none" stroke="rgba(0,0,0,.04)" strokeWidth="12" strokeLinecap="round" />
        )}

        {/* Background track */}
        <path d={arcD(ARC_START, ARC_END)}
          fill="none" stroke="rgba(0,0,0,.06)" strokeWidth="4" strokeLinecap="round" />

        {/* Training window band */}
        <path d={arcD(ARC_START + TRAIN_T0 * ARC_SWEEP, ARC_START + TRAIN_T1 * ARC_SWEEP)}
          fill="none" stroke="rgba(0,0,0,.13)" strokeWidth="4" strokeLinecap="round" />

        {/* Filled arc */}
        {t > 0.002 && (
          <path d={arcD(ARC_START, thumbAngle)}
            fill="none" stroke="#111" strokeWidth="4" strokeLinecap="round" />
        )}

        {/* Year ticks */}
        {ticks.map(({ y, isKey, x1, y1, x2, y2 }) => (
          <line key={y}
            x1={x1.toFixed(2)} y1={y1.toFixed(2)}
            x2={x2.toFixed(2)} y2={y2.toFixed(2)}
            stroke={isKey ? 'rgba(0,0,0,.45)' : 'rgba(0,0,0,.13)'}
            strokeWidth={isKey ? '1.5' : '1'}
            strokeLinecap="round"
          />
        ))}

        {/* Year labels */}
        {yearLabels.map(({ y, lx, ly, anchor }) => (
          <text key={y}
            x={lx.toFixed(2)} y={ly.toFixed(2)}
            textAnchor={anchor}
            fontSize="10"
            fontFamily="Inter, sans-serif"
            fill="rgba(0,0,0,.32)"
            dominantBaseline="middle"
          >{y}</text>
        ))}

        {/* Thumb */}
        <circle
          cx={tx.toFixed(2)} cy={ty.toFixed(2)} r="9"
          fill="#fff"
          stroke="#111"
          strokeWidth="2.5"
          style={{ filter: 'drop-shadow(0 2px 6px rgba(0,0,0,.22))' }}
        />
      </svg>

      <div className="dial-info" aria-hidden="true">
        <span className={`dial-year${extrapolated ? ' dial-year--dim' : ''}`}>{year}</span>
        <span className="dial-mo">{MO_FULL[mo]}</span>
        {extrapolated && <span className="dial-extrap">extrap.</span>}
      </div>
    </div>
  )
}

export default function App() {
  const [month, setMonth] = useState(INITIAL_MONTH)
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    const prompt = draft.trim()
    if (!prompt || busy) return
    const id = crypto.randomUUID?.() ?? String(Date.now())
    const date = monthToDate(month)
    setMessages(prev => [{ id, prompt, date, status: 'pending' }, ...prev])
    setDraft('')
    setBusy(true)
    try {
      if (!GENERATE_URL) throw new Error('Model endpoint unavailable.')
      const res = await fetch(GENERATE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, date: date.iso, new_tokens: 64, seed: Math.floor(Math.random() * 1e6) }),
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json.detail || 'Generation failed.')
      setMessages(prev => prev.map(m =>
        m.id === id
          ? { ...m, status: 'done', response: json.continuation?.trim() || 'The model ended this passage.' }
          : m
      ))
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === id ? { ...m, status: 'error', response: err.message } : m
      ))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="page">
      <header className="hd">
        <h1 className="brand">ZEITGEISTLM</h1>
        <p className="byline">Dated text generation · 2.967 B tokens · trained 2011–2020</p>
      </header>

      <div className="card">
        <div className="dial-wrap">
          <Dial month={month} onChange={setMonth} />
        </div>

        <div className="card-rule" />

        <form className="composer" onSubmit={submit}>
          <label className="sr-only" htmlFor="prompt">Prompt</label>
          <textarea
            id="prompt"
            value={draft}
            maxLength={160}
            rows={3}
            placeholder="Begin a sentence and let the model continue it…"
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault()
                e.currentTarget.form.requestSubmit()
              }
            }}
          />
          <div className="composer-foot">
            <span className="char-count">{draft.length}/160</span>
            <button type="submit" disabled={!draft.trim() || busy}>
              {busy ? 'Generating…' : 'Generate'} <span aria-hidden="true">↗</span>
            </button>
          </div>
        </form>
      </div>

      {messages.length > 0 && (
        <section className="log" aria-label="Generations" aria-live="polite">
          {messages.map(item => (
            <article className="entry" key={item.id}>
              <div className="entry-meta">
                <time dateTime={item.date.iso}>{item.date.label}</time>
                {(item.date.year < 2011 || item.date.year > 2020) && (
                  <span className="tag">Extrapolated</span>
                )}
              </div>
              <p className="entry-prompt">{item.prompt}</p>
              <p className={`entry-resp${item.status === 'pending' ? ' pending' : ''}${item.status === 'error' ? ' err' : ''}`}>
                {item.status === 'pending' ? 'Generating…' : item.response}
              </p>
            </article>
          ))}
        </section>
      )}
    </main>
  )
}
