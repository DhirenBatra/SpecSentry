import { useState } from 'react'
import AssistancePanel from './AssistancePanel.jsx'

const filters = [
  { value: 'all', label: 'All' },
  { value: 'ambiguous', label: 'Ambiguous' },
  { value: 'clear', label: 'Clear' },
]

function RequirementCard({ requirement }) {
  const { id, text, category, binary_category: binaryCategory, ambiguity } = requirement
  return (
    <article className="panel requirement-card" aria-label={`Requirement ${id}`}>
      <div className="requirement-header">
        <div className="requirement-tags">
          <span className="requirement-id">{id}</span>
          <span className="category-tag">{category}</span>
          {binaryCategory && <span className="binary-category">{binaryCategory}</span>}
        </div>
        {ambiguity.ambiguous
          ? <span className="finding-badge finding-review">Ambiguous · {ambiguity.severity}</span>
          : <span className="finding-badge finding-clear">Clear</span>}
      </div>
      <p className="requirement-text">{text}</p>
      {ambiguity.ambiguous && (
        <details className="ambiguity-details">
          <summary>
            <span>Ambiguity evidence</span>
            <span className="ambiguity-score">Score <strong>{ambiguity.score} / 10</strong></span>
          </summary>
          <ul className="ambiguity-issues">
            {ambiguity.issues.map((issue) => (
              <li key={`${issue.category}-${issue.start}-${issue.end}`}>
                <div className="issue-heading">
                  <mark>{issue.text}</mark>
                  <span>{issue.category.replaceAll('_', ' ')}</span>
                </div>
                <p>{issue.explanation}</p>
              </li>
            ))}
          </ul>
          <AssistancePanel payload={{
            action: 'rewrite_ambiguity',
            requirement: text,
            issues: ambiguity.issues.slice(0, 12),
          }} />
        </details>
      )}
    </article>
  )
}

export default function RequirementExplorer({ requirements }) {
  const [filter, setFilter] = useState('all')
  const visible = requirements.filter(({ ambiguity }) => (
    filter === 'all' || ambiguity.ambiguous === (filter === 'ambiguous')
  ))

  return (
    <section className="report-section" id="requirements-explorer" aria-labelledby="explorer-title">
      <div className="section-heading">
        <h2 id="explorer-title">Requirement explorer</h2>
        <div className="filters" role="group" aria-label="Filter requirements">
          {filters.map(({ value, label }) => (
            <button type="button" key={value} aria-pressed={filter === value} onClick={() => setFilter(value)}>
              {label}
            </button>
          ))}
        </div>
      </div>
      {visible.length
        ? <div className="finding-list">{visible.map((requirement) => <RequirementCard key={requirement.id} requirement={requirement} />)}</div>
        : <p className="panel empty-state">{filter === 'ambiguous' ? 'No requirements were flagged as ambiguous.' : 'No clear requirements in this analysis.'}</p>}
    </section>
  )
}
