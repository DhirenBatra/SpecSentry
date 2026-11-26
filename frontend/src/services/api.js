const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000')
  .replace(/\/+$/, '')

export const HEALTH_PATH = '/api/health'
export const API_DOCS_URL = `${baseUrl}/docs`

export async function checkApiHealth(signal) {
  const response = await fetch(`${baseUrl}${HEALTH_PATH}`, {
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
