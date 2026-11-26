import BackendStatus from './components/BackendStatus.jsx'

export default function App() {
  return (
    <div className="flex min-h-svh flex-col bg-canvas font-sans text-ink antialiased">
      <header className="border-b border-line">
        <div className="mx-auto flex min-h-19 max-w-7xl items-center justify-between gap-4 px-6 sm:px-10">
          <div className="flex items-center gap-3">
            <span
              aria-hidden="true"
              className="grid size-9 shrink-0 place-items-center rounded-[0.7rem] bg-pine text-lg font-semibold text-signal"
            >
              ✓
            </span>
            <span className="text-lg font-semibold tracking-tight">SpecSentry</span>
          </div>
          <p className="text-right text-xs font-medium text-muted sm:text-sm">
            Phase 1 · Foundation
          </p>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-7xl grow items-center px-6 py-14 sm:px-10 sm:py-20">
        <div className="grid w-full items-center gap-10 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] lg:gap-16">
          <section aria-labelledby="project-title">
            <h1
              id="project-title"
              className="text-[clamp(2.5rem,5.1vw,4.5rem)] leading-[1.04] font-semibold tracking-[-0.06em] text-pretty"
            >
              AI-Powered Software Requirements{' '}
              <span className="inline-block text-accent">Quality Analysis</span>
            </h1>
            <p className="mt-6 max-w-xl text-base leading-relaxed text-muted sm:text-lg">
              Analyze software requirements for ambiguity, duplication, contradictions, and quality
              issues using Natural Language Processing.
            </p>
            <p className="mt-7 border-l-2 border-accent/40 pl-3 text-sm leading-relaxed text-muted">
              Requirement analysis is planned for later phases.
            </p>
          </section>
          <BackendStatus />
        </div>
      </main>

      <footer className="border-t border-line">
        <p className="mx-auto max-w-7xl px-6 py-5 text-sm text-muted sm:px-10">
          Computational Linguistics &amp; NLP Lab
        </p>
      </footer>
    </div>
  )
}
