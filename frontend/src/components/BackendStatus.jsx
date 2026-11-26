import { useEffect, useState } from 'react'
import { API_DOCS_URL, HEALTH_PATH, checkApiHealth } from '../services/api.js'

const statusDetails = {
  checking: {
    label: 'Checking…',
    message: 'Contacting the local FastAPI service.',
    color: 'text-amber-200',
  },
  connected: {
    label: 'Connected',
    message: 'The local FastAPI service is responding.',
    color: 'text-signal',
  },
  unavailable: {
    label: 'Unavailable',
    message: 'Start FastAPI, then check the connection again.',
    color: 'text-orange-200',
  },
}

export default function BackendStatus() {
  const [status, setStatus] = useState('checking')
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()

    async function updateStatus() {
      let nextStatus = 'connected'
      try {
        await checkApiHealth(controller.signal)
      } catch {
        nextStatus = 'unavailable'
      }

      if (!controller.signal.aborted) setStatus(nextStatus)
    }

    updateStatus()
    return () => controller.abort()
  }, [attempt])

  function checkAgain() {
    setStatus('checking')
    setAttempt((previous) => previous + 1)
  }

  const { label, message, color } = statusDetails[status]

  return (
    <section
      aria-labelledby="backend-status"
      className="service-card rounded-[1.75rem] border border-white/10 bg-pine p-6 text-white shadow-[0_24px_56px_-36px_#172e2480] sm:p-8"
    >
      <p className="mb-5 inline-flex items-center gap-2 rounded-lg border border-white/15 bg-black/15 px-3 py-2 font-mono text-xs text-white/80">
        <span className="font-semibold text-signal">GET</span>
        <code>{HEALTH_PATH}</code>
      </p>
      <h2 id="backend-status" className="text-2xl font-semibold tracking-tight">
        Backend Status
      </h2>
      <p role="status" className={`mt-5 flex items-center gap-2.5 text-sm font-semibold ${color}`}>
        <span aria-hidden="true" className="size-2 rounded-full bg-current" />
        {label}
      </p>
      <p className="mt-2 text-sm leading-relaxed text-white/75">{message}</p>

      <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-3 border-t border-white/15 pt-5">
        <button
          type="button"
          onClick={checkAgain}
          disabled={status === 'checking'}
          className="cursor-pointer rounded-lg border border-white/20 bg-white/10 px-3.5 py-2 text-sm font-medium transition-colors hover:bg-white/20 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-signal disabled:cursor-wait disabled:opacity-60"
        >
          Check again
        </button>
        <a
          href={API_DOCS_URL}
          className="rounded-sm text-sm font-medium text-signal underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-signal"
        >
          API docs <span aria-hidden="true">↗</span>
        </a>
      </div>
    </section>
  )
}
