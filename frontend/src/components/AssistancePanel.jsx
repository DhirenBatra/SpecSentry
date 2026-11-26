import { useEffect, useRef, useState } from 'react'
import { requestAssistance } from '../services/api.js'

export default function AssistancePanel({ payload }) {
  const isRewrite = payload.action === 'rewrite_ambiguity'
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const request = useRef(null)

  useEffect(() => () => request.current?.abort(), [])

  async function getSuggestion() {
    if (request.current) return
    const controller = new AbortController()
    request.current = controller
    setError('')
    setIsLoading(true)
    try {
      const suggestion = await requestAssistance(payload, controller.signal)
      if (!controller.signal.aborted) setResult(suggestion)
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure.message)
    } finally {
      request.current = null
      if (!controller.signal.aborted) setIsLoading(false)
    }
  }

  return (
    <div className="assistance-panel">
      {!result && (
        <button className="button button-secondary button-small" type="button" onClick={getSuggestion} disabled={isLoading}>
          {isLoading ? 'Requesting AI assistance...' : isRewrite ? 'Suggest Rewrite' : 'Explain Conflict'}
        </button>
      )}
      {error && <p className="error-message" role="alert">{error}</p>}
      {result && (
        <div className="ai-result" aria-live="polite">
          <p className="ai-review">AI-generated suggestion — review before use.</p>
          <dl>
            <div>
              <dt>AI Explanation</dt>
              <dd>{result.explanation}</dd>
            </div>
            <div>
              <dt>{isRewrite ? 'Suggested Rewrite' : 'Suggested Resolution'}</dt>
              <dd>{isRewrite ? result.suggested_rewrite : result.suggested_resolution}</dd>
            </div>
          </dl>
        </div>
      )}
    </div>
  )
}
