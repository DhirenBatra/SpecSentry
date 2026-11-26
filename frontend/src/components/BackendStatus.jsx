import { useEffect, useState } from 'react'
import { checkApiHealth } from '../services/api.js'

const labels = {
  checking: 'Checking backend',
  connected: 'Backend connected',
  unavailable: 'Backend unavailable',
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

  return (
    <div className="connection">
      <p role="status" className="connection-status" data-connection={status}>
        <span className="status-dot" aria-hidden="true" />
        {labels[status]}
      </p>
      {status === 'unavailable' && (
        <button className="text-button" type="button" onClick={checkAgain}>Retry</button>
      )}
    </div>
  )
}
