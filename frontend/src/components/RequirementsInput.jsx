import { useEffect, useRef, useState } from 'react'
import demo from '../../../examples/analyze_demo.json'
import { analyzeRequirements } from '../services/api.js'

const MAX_REQUIREMENTS = 20
const MAX_LINE_LENGTH = 500
const demoText = demo.requirements.map(({ text }) => text).join('\n')

function parseRequirements(text) {
  return text.split(/\r\n?|\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => ({ id: `R${index + 1}`, text: line }))
}

function validate(requirements) {
  if (!requirements.length) return 'Enter at least one requirement to analyze.'
  if (requirements.length > MAX_REQUIREMENTS) {
    return `Analyze up to ${MAX_REQUIREMENTS} requirements at a time. Remove ${requirements.length - MAX_REQUIREMENTS} to continue.`
  }
  const longLine = requirements.find(({ text }) => Array.from(text).length > MAX_LINE_LENGTH)
  return longLine ? `${longLine.id} exceeds ${MAX_LINE_LENGTH} characters. Shorten that requirement to continue.` : ''
}

export default function RequirementsInput({ onAnalyzed }) {
  const [text, setText] = useState('')
  const [error, setError] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const request = useRef(null)
  const requirements = parseRequirements(text)
  const overLimit = requirements.length > MAX_REQUIREMENTS

  useEffect(() => () => request.current?.abort(), [])

  function changeText(value) {
    setText(value)
    setError('')
  }

  async function submit(event) {
    event.preventDefault()
    if (request.current) return

    const validationError = validate(requirements)
    if (validationError) {
      setError(validationError)
      return
    }

    const controller = new AbortController()
    request.current = controller
    setError('')
    setIsAnalyzing(true)
    try {
      const report = await analyzeRequirements(requirements, controller.signal)
      if (!controller.signal.aborted) onAnalyzed(report)
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure.message)
    } finally {
      request.current = null
      if (!controller.signal.aborted) setIsAnalyzing(false)
    }
  }

  return (
    <div>
      <div className="page-heading">
        <h1>Analyze requirements</h1>
        <p>Paste one software requirement per line. SpecSentry assigns each line an ID.</p>
      </div>

      <div className="input-layout">
        <form className="panel editor-panel" onSubmit={submit}>
          <div className="editor-header">
            <label htmlFor="requirements">Requirements</label>
            <p className="requirement-count tabular-nums" data-over-limit={overLimit} aria-live="polite">
              {requirements.length} / {MAX_REQUIREMENTS} requirements
            </p>
          </div>
          <textarea
            id="requirements"
            name="requirements"
            value={text}
            onChange={(event) => changeText(event.target.value)}
            placeholder={'The system shall save a draft when the user selects Save.\nThe dashboard shall return search results within 2 seconds.'}
            aria-describedby={error ? 'input-limit analysis-error' : 'input-limit'}
            aria-invalid={overLimit || undefined}
            readOnly={isAnalyzing}
            spellCheck={false}
          />
          <div className="editor-footer">
            <p id="input-limit">Up to {MAX_LINE_LENGTH} characters per requirement</p>
            <button className="button button-primary" type="submit" disabled={isAnalyzing}>
              Analyze Requirements <span aria-hidden="true">→</span>
            </button>
          </div>
          {isAnalyzing && (
            <div className="analysis-progress" role="status">
              <span className="spinner" aria-hidden="true" />
              <div>
                <p className="font-semibold">Analyzing requirements...</p>
                <p className="progress-note">The first analysis can take tens of seconds while the models load.</p>
              </div>
            </div>
          )}
          {error && <p className="error-message" id="analysis-error" role="alert">{error}</p>}
        </form>

        <aside className="panel demo-panel" aria-labelledby="demo-title">
          <span className="document-mark" aria-hidden="true"><span /></span>
          <h2 id="demo-title">Demo requirements</h2>
          <p>Try the {demo.requirements.length}-requirement sample with ambiguous wording, a paraphrase, and a potential conflict.</p>
          <button className="button button-secondary" disabled={isAnalyzing} type="button" onClick={() => changeText(demoText)}>
            Load Demo Requirements
          </button>
        </aside>
      </div>
    </div>
  )
}
