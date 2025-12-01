// Dynamic API detection - prefer updated server on 8787
const COMMON_PORTS = [8787, 8788, 8000, 3000, 5000, 8080, 9000, 8789, 8790]

async function findBackendServer(): Promise<string> {
  // First, try environment variable
  if (import.meta.env.VITE_API_URL) {
    console.log('✅ Using API URL from environment:', import.meta.env.VITE_API_URL)
    return import.meta.env.VITE_API_URL
  }

  // Try common ports - prefer /health endpoint for better detection
  for (const port of COMMON_PORTS) {
    try {
      // Try /health first (more reliable)
      const healthResponse = await fetch(`http://127.0.0.1:${port}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(1000) // 1 second timeout
      })
      if (healthResponse.ok) {
        const healthData = await healthResponse.json()
        console.log(`✅ Found backend server on port ${port} (${healthData.engine || 'unknown'})`)
        return `http://127.0.0.1:${port}`
      }
    } catch (error) {
      // Try /docs as fallback
      try {
        const docsResponse = await fetch(`http://127.0.0.1:${port}/docs`, {
          method: 'HEAD',
          signal: AbortSignal.timeout(1000)
        })
        if (docsResponse.ok) {
          console.log(`✅ Found backend server on port ${port} (via /docs)`)
          return `http://127.0.0.1:${port}`
        }
      } catch (docsError) {
        // Port not available, try next
        continue
      }
    }
  }

  // Fallback to default
  console.warn('⚠️ Could not auto-detect backend server, using fallback')
  return 'http://127.0.0.1:8787'
}

// Cache the API URL once found
let cachedAPI: string | null = null

async function getAPI(): Promise<string> {
  if (cachedAPI) return cachedAPI
  cachedAPI = await findBackendServer()
  return cachedAPI
}

export async function chat(session: any, message: string) {
  const API = await getAPI()
  const r = await fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session, message })
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function getActivity(rowIndex: number, session: any = {}) {
  const API = await getAPI()
  const r = await fetch(`${API}/activities/${rowIndex}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session })
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}


export async function buildDoc(activity: any) {
  const API = await getAPI()
  const r = await fetch(`${API}/documents/build`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ activity })
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function condenseDirections(activity: any) {
  const API = await getAPI()
  const r = await fetch(`${API}/documents/condense`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ activity })
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function modifyActivity(activity: any, instruction: string) {
  const API = await getAPI()
  const r = await fetch(`${API}/documents/modify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ activity, instruction })
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}


