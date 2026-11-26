import PairFindings from './PairFindings.jsx'
import RequirementExplorer from './RequirementExplorer.jsx'

const number = new Intl.NumberFormat('en', { maximumFractionDigits: 2 })

function Summary({ summary }) {
  const metrics = [
    ['Total Requirements', summary.total_requirements],
    ['Ambiguous Requirements', summary.ambiguous_requirements],
    ['Potential Duplicates', summary.potential_duplicate_pairs],
    ['Potential Contradictions', summary.potential_contradiction_pairs],
  ]
  return (
    <dl className="summary-grid" aria-label="Analysis summary">
      {metrics.map(([label, value]) => (
        <div className="panel summary-card" key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}

function QualityScore({ score, breakdown }) {
  const penalties = [
    ['Ambiguity penalty', breakdown.ambiguity_penalty],
    ['Duplicate penalty', breakdown.duplicate_penalty],
    ['Contradiction penalty', breakdown.contradiction_penalty],
  ]
  return (
    <section className="score-panel" aria-labelledby="quality-score-title">
      <h2 id="quality-score-title">Quality score</h2>
      <p className="score-value"><strong>{number.format(score)}</strong><span> / 100</span></p>
      <p className="score-note">The quality score is a heuristic review score, not a model probability.</p>
      <dl className="score-breakdown" aria-label="Score breakdown">
        {penalties.map(([label, penalty]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{penalty > 0 ? '−' : ''}{number.format(penalty)}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

function CategoryChart({ requirements }) {
  const counts = new Map()
  for (const { category } of requirements) counts.set(category, (counts.get(category) ?? 0) + 1)
  const categories = Array.from(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
  const max = categories[0]?.[1] ?? 1

  return (
    <section className="panel category-panel" aria-labelledby="category-title">
      <h2 id="category-title">Requirement categories</h2>
      <ul className="category-chart" aria-label="Predicted category distribution">
        {categories.map(([label, count]) => (
          <li key={label}>
            <span>{label}</span>
            <span className="chart-track" aria-hidden="true">
              <span className="chart-bar" style={{ width: `${100 * count / max}%` }} />
            </span>
            <strong className="tabular-nums">{count}</strong>
          </li>
        ))}
      </ul>
    </section>
  )
}

export default function AnalysisDashboard({ report, onReset }) {
  const { summary, requirements, duplicate_pairs: duplicates, contradiction_pairs: contradictions, score_breakdown: breakdown } = report
  const requirementsById = new Map(requirements.map((requirement) => [requirement.id, requirement]))

  return (
    <div>
      <div className="page-heading report-heading">
        <h1>Analysis dashboard</h1>
        <button type="button" className="button button-secondary" onClick={onReset}>Analyze New Requirements</button>
      </div>
      <nav className="section-nav" aria-label="Dashboard sections">
        <a href="#overview">Overview</a>
        <a href="#requirements-explorer">Requirements</a>
        <a href="#duplicates">Potential duplicates</a>
        <a href="#contradictions">Potential contradictions</a>
      </nav>
      <section className="report-overview" id="overview" aria-label="Analysis overview">
        <Summary summary={summary} />
        <div className="overview-layout">
          <QualityScore score={summary.quality_score} breakdown={breakdown} />
          <CategoryChart requirements={requirements} />
        </div>
      </section>
      <RequirementExplorer requirements={requirements} />
      <PairFindings kind="duplicates" pairs={duplicates} requirementsById={requirementsById} />
      <PairFindings kind="contradictions" pairs={contradictions} requirementsById={requirementsById} />
    </div>
  )
}
