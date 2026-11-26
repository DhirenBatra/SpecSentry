import { useState } from 'react'
import AnalysisDashboard from './components/AnalysisDashboard.jsx'
import BackendStatus from './components/BackendStatus.jsx'
import RequirementsInput from './components/RequirementsInput.jsx'
import { API_DOCS_URL } from './services/api.js'

export default function App() {
  const [report, setReport] = useState(null)

  function switchView(nextReport) {
    setReport(nextReport)
    if (window.location.hash) {
      window.history.replaceState(window.history.state, '', window.location.pathname + window.location.search)
    }
    window.scrollTo(0, 0)
  }

  return (
    <div className="min-h-svh">
      <header className="app-header">
        <div className="workspace header-inner">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">✓</span>
            <span>SpecSentry</span>
          </div>
          <div className="header-actions">
            <a className="docs-link" href={API_DOCS_URL} target="_blank" rel="noreferrer">API docs ↗</a>
            <BackendStatus />
          </div>
        </div>
      </header>
      <main className="workspace main-content">
        {report
          ? <AnalysisDashboard report={report} onReset={() => switchView(null)} />
          : <RequirementsInput onAnalyzed={switchView} />}
      </main>
    </div>
  )
}
