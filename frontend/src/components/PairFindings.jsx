import { useState } from 'react'
import AssistancePanel from './AssistancePanel.jsx'

const PAGE_SIZE = 4

function PairRequirement({ label, id, text }) {
  return (
    <div className="pair-requirement">
      <div><span>{label}</span><span className="requirement-id">{id}</span></div>
      <p className="requirement-text">{text ?? 'Requirement text is unavailable.'}</p>
    </div>
  )
}

function PairCard({ pair, requirementsById, isContradiction }) {
  const first = requirementsById.get(pair.requirement_a_id)
  const second = requirementsById.get(pair.requirement_b_id)
  return (
    <article className="panel pair-card" aria-label={`${pair.requirement_a_id} and ${pair.requirement_b_id}`}>
      <div className="pair-meta">
        {isContradiction ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-semibold">Potential Contradiction</span>
              <span className="finding-badge finding-review">Review Recommended</span>
            </div>
            <span className="pair-method">NLI · forward A → B</span>
          </>
        ) : (
          <>
            <span className="similarity-value">Cosine similarity <strong>{pair.similarity.toFixed(3)}</strong></span>
            <span className="pair-method">SBERT · MiniLM</span>
          </>
        )}
      </div>
      <div className="pair-requirements">
        <PairRequirement label="Requirement A" id={pair.requirement_a_id} text={first?.text} />
        <PairRequirement label="Requirement B" id={pair.requirement_b_id} text={second?.text} />
      </div>
      {isContradiction && (
        <>
          <dl className="nli-evidence">
            <div><dt>NLI label</dt><dd className="capitalize">{pair.label}</dd></div>
            <div><dt>Contradiction score</dt><dd>{pair.contradiction_score.toFixed(3)}</dd></div>
            <div><dt>Neutral score</dt><dd>{pair.neutral_score.toFixed(3)}</dd></div>
            <div><dt>Entailment score</dt><dd>{pair.entailment_score.toFixed(3)}</dd></div>
          </dl>
          {first && second && (
            <AssistancePanel payload={{
              action: 'explain_contradiction',
              requirement_a: first.text,
              requirement_b: second.text,
              evidence: pair,
            }} />
          )}
        </>
      )}
    </article>
  )
}

export default function PairFindings({ pairs, requirementsById, kind }) {
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE)
  const isContradiction = kind === 'contradictions'
  const title = isContradiction ? 'Potential contradictions' : 'Potential duplicates'
  const remaining = pairs.length - visibleCount

  return (
    <section className="report-section" id={kind} aria-labelledby={`${kind}-title`}>
      <div className="section-heading">
        <h2 id={`${kind}-title`}>{title}</h2>
      </div>
      {isContradiction && pairs.length > 0 && (
        <p className="section-note">Potential contradictions are model-generated review candidates and may include false positives.</p>
      )}
      {pairs.length
        ? <div className="finding-list">
          {pairs.slice(0, visibleCount).map((pair) => (
            <PairCard
              key={`${pair.requirement_a_id}-${pair.requirement_b_id}`}
              pair={pair}
              isContradiction={isContradiction}
              requirementsById={requirementsById}
            />
          ))}
        </div>
        : <p className="panel empty-state">No {title.toLowerCase()} were flagged in this analysis.</p>}
      {remaining > 0 && (
        <div className="pair-pagination">
          <p>Showing {visibleCount} of {pairs.length} pairs</p>
          <button className="button button-secondary button-small" type="button" onClick={() => setVisibleCount((count) => count + PAGE_SIZE)}>
            Show {Math.min(PAGE_SIZE, remaining)} more
          </button>
        </div>
      )}
    </section>
  )
}
