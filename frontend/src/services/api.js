const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000')
  .replace(/\/+$/, '')

export const API_DOCS_URL = `${baseUrl}/docs`

const analysisMessages = {
  422: 'Check the requirement count and length, then submit again.',
  503: 'Analysis models are unavailable. Check the backend setup and local model files.',
  default: 'Analysis failed. Check the backend terminal and try again.',
  timeout: 'Analysis took longer than expected. Check the backend terminal and try again.',
}

const assistanceMessages = {
  422: 'The selected finding could not be sent for AI assistance. Run the analysis again and retry.',
  502: 'The AI provider could not return a usable suggestion. You can try again.',
  503: 'AI assistance is unavailable. Check the optional OpenAI configuration in the backend.',
  504: 'The AI provider timed out. You can try again.',
  default: 'AI assistance failed. You can try again.',
  timeout: 'The AI request took too long. You can try again.',
}

async function postJson(path, payload, signal, timeout, messages) {
  const deadline = AbortSignal.timeout(timeout)
  const requestSignal = signal ? AbortSignal.any([signal, deadline]) : deadline
  let response
  try {
    response = await fetch(`${baseUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: requestSignal,
    })
  } catch (error) {
    if (signal?.aborted) throw error
    if (deadline.aborted) throw new Error(messages.timeout, { cause: error })
    throw new Error('Could not reach SpecSentry. Make sure the local backend is running.', { cause: error })
  }

  if (!response.ok) throw new Error(messages[response.status] ?? messages.default)
  try {
    return await response.json()
  } catch (error) {
    throw new Error('SpecSentry returned an unreadable response. Try again.', { cause: error })
  }
}

export async function checkApiHealth(signal) {
  const response = await fetch(`${baseUrl}/api/health`, {
    cache: 'no-store',
    signal: AbortSignal.any([signal, AbortSignal.timeout(5000)]),
  })

  if (!response.ok) {
    throw new Error(`Health check failed (HTTP ${response.status}).`)
  }

  const health = await response.json()
  if (health?.status !== 'ok' || health.service !== 'SpecSentry API') {
    throw new Error('Unexpected health response from the API.')
  }
}

export function analyzeRequirements(requirements, signal) {
  return postJson('/api/analyze', { requirements }, signal, 180_000, analysisMessages)
}

export function requestAssistance(payload, signal) {
  return postJson('/api/ai/assist', payload, signal, 30_000, assistanceMessages)
}
