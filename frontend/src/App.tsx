import React, { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { chat, getActivity, buildDoc } from './api/client'
import ActivityList from './components/ActivityList'
import './index.css'

function normalizeBullets(text: string) {
  // Turn lines that start with "•" into Markdown list items ("- ")
  // Handles: start of string or after a newline, optional spaces, then "•", optional spaces.
  return text.replace(/(^|\n)[ \t]*•[ \t]*/g, '$1- ');
}

type Msg = { 
  role: 'user'|'assistant'|'system'; 
  content: string;
  lessonPlan?: {
    docBytes: string; // base64 encoded .docx content
    filename: string;
    activity: any; // Structured activity data for the edit form
    rowIndex: number; // Original index for tracking/feedback
    title: string; // Activity title for tracking/feedback
  }
  activities?: Array<{
    row_index: number
    title: string
    time?: string
    summary?: string
    display_line?: string
    display_id?: string
    score?: number
  }>;
}

type ChatSession = {
  id: string
  name: string
  messages: Msg[]
  lastUpdated: number
  activityTracking: Record<string, ActivityTracking>
  feedbackMeta: FeedbackMeta
  sessionData?: any // Store the backend session data (last_indices, last_results, etc.)
}

type ActivityTracking = {
  activityName: string
  firstSeen: string
  downloadClicked: boolean
  like?: boolean
  reason?: string
  feedbackSentAt?: string
}

type FeedbackMeta = {
  lastPromptAt?: string
  promptCount: number
}

type FeedbackForm = {
  activityId: string
  activityName: string
  like: boolean
  reason: string
  frequency: string
  timeSpent: string
  adaptation: string
}

// Feedback constants
const PROMPT_AFTER_SECONDS = 60
const PROMPT_COOLDOWN_SESSION = 10 * 60 * 1000 // 10 minutes in milliseconds
const PROMPT_MAX_PER_SESSION = 2
const PROMPT_COOLDOWN_PER_ACTIVITY = 7 * 24 * 60 * 60 * 1000 // 7 days in milliseconds

const REASONS_POS = ["Engaging", "Easy to run", "Good length"]
const REASONS_NEG = ["Too long", "Confusing", "Low payoff"]

const FREQUENCY_OPTIONS = [
  "Haven't used yet",
  "Once",
  "2–3 times",
  "4–5 times",
  "6+ times",
]

const TIME_SPENT_OPTIONS = [
  "N/A",
  "<5 min",
  "5–10 min",
  "10–20 min",
  "20–30 min",
  "30–45 min",
  "45–60 min",
  "60+ min",
]

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string>('')
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [session, setSession] = useState<any>({ last_indices: [], last_results: [], overrides: {} })
  const [loading, setLoading] = useState(false)
  const [showSessionList, setShowSessionList] = useState(false)
  const [showFeedbackForm, setShowFeedbackForm] = useState<FeedbackForm | null>(null)
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [editingSessionName, setEditingSessionName] = useState<string>('')
  const listRef = useRef<HTMLDivElement>(null)

  // Initialize sessions from localStorage
  useEffect(() => {
    const savedSessions = localStorage.getItem('chatbot-sessions')
    if (savedSessions) {
      const parsed = JSON.parse(savedSessions)
      setSessions(parsed)
      if (parsed.length > 0) {
        setCurrentSessionId(parsed[0].id)
        setMessages(parsed[0].messages)
        // Restore session data from the first session
        const firstSession = parsed[0]
        if (firstSession.sessionData) {
          setSession(firstSession.sessionData)
        }
      }
    } else {
      // Create initial session
      createNewSession()
    }
  }, [])

  // Save sessions to localStorage whenever sessions change
  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem('chatbot-sessions', JSON.stringify(sessions))
    }
  }, [sessions])

  // Update current session messages and session data when they change
  useEffect(() => {
    if (currentSessionId && messages.length > 0) {
      setSessions(prev => prev.map(s => 
        s.id === currentSessionId 
          ? { ...s, messages, sessionData: session, lastUpdated: Date.now() }
          : s
      ))
    }
  }, [messages, session, currentSessionId])

  // Feedback helper functions
  const trackActivityRendered = (activityId: string, activityName: string) => {
    setSessions(prev => prev.map(s => {
      if (s.id === currentSessionId) {
        const tracking = s.activityTracking[activityId] || {
          activityName,
          firstSeen: new Date().toISOString(),
          downloadClicked: false
        }
        return {
          ...s,
          activityTracking: {
            ...s.activityTracking,
            [activityId]: tracking
          }
        }
      }
      return s
    }))
  }

  const recordDownloadClick = (activityId: string, activityName: string, fileName: string) => {
    setSessions(prev => prev.map(s => {
      if (s.id === currentSessionId) {
        const tracking = s.activityTracking[activityId] || {
          activityName,
          firstSeen: new Date().toISOString(),
          downloadClicked: false
        }
        return {
          ...s,
          activityTracking: {
            ...s.activityTracking,
            [activityId]: {
              ...tracking,
              downloadClicked: true
            }
          }
        }
      }
      return s
    }))
    
    // Check if we should prompt for feedback
    setTimeout(() => maybePromptForFeedback(), 1000)
  }

  const maybePromptForFeedback = () => {
    const currentSession = sessions.find(s => s.id === currentSessionId)
    if (!currentSession) return

    const { feedbackMeta } = currentSession
    const now = Date.now()

    // Check session cooldown
    if (feedbackMeta.lastPromptAt) {
      const lastPrompt = new Date(feedbackMeta.lastPromptAt).getTime()
      if (now - lastPrompt < PROMPT_COOLDOWN_SESSION) return
    }

    // Check max prompts per session
    if (feedbackMeta.promptCount >= PROMPT_MAX_PER_SESSION) return

    // Find activity that needs feedback
    for (const [activityId, tracking] of Object.entries(currentSession.activityTracking)) {
      if (tracking.feedbackSentAt) {
        const lastFeedback = new Date(tracking.feedbackSentAt).getTime()
        if (now - lastFeedback < PROMPT_COOLDOWN_PER_ACTIVITY) continue
      }

      const firstSeen = new Date(tracking.firstSeen).getTime()
      const ageSeconds = (now - firstSeen) / 1000
      const needsPrompt = tracking.downloadClicked || ageSeconds >= PROMPT_AFTER_SECONDS

      if (needsPrompt) {
        setShowFeedbackForm({
          activityId,
          activityName: tracking.activityName,
          like: true,
          reason: REASONS_POS[0],
          frequency: FREQUENCY_OPTIONS[0],
          timeSpent: TIME_SPENT_OPTIONS[0],
          adaptation: ''
        })
        break
      }
    }
  }

  const submitFeedback = (form: FeedbackForm) => {
    setSessions(prev => prev.map(s => {
      if (s.id === currentSessionId) {
        const tracking = s.activityTracking[form.activityId]
        if (tracking) {
          return {
            ...s,
            activityTracking: {
              ...s.activityTracking,
              [form.activityId]: {
                ...tracking,
                like: form.like,
                reason: form.reason,
                feedbackSentAt: new Date().toISOString()
              }
            },
            feedbackMeta: {
              ...s.feedbackMeta,
              lastPromptAt: new Date().toISOString(),
              promptCount: s.feedbackMeta.promptCount + 1
            }
          }
        }
      }
      return s
    }))
    
    setShowFeedbackForm(null)
    
    // Add success message
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: 'Thanks! Your feedback helps improve the suggestions. 🙏'
    }])
  }

  const createNewSession = () => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      name: `Chat ${new Date().toLocaleDateString()}`,
      messages: [
        { role: 'assistant', content: "Hi there! 👋 I'm your AI teaching assistant, and I'm excited to help you create amazing classroom experiences!\n\nHere's what I can do for you:\n\n• **Find Activities:** Search for classroom activities by subject, grade level, or time\n• **Create Lesson Plans:** Generate detailed, ready-to-use lesson plans with materials and directions\n• **Customize Content:** Modify activities to fit your specific needs\n• **Answer Questions:** Help with teaching strategies, classroom management, and more" }
      ],
      lastUpdated: Date.now(),
      activityTracking: {},
      feedbackMeta: { promptCount: 0 },
      sessionData: { last_indices: [], last_results: [], overrides: {} }
    }
    setSessions(prev => [newSession, ...prev])
    setCurrentSessionId(newSession.id)
    setMessages(newSession.messages)
    setSession(newSession.sessionData)
  }

  const switchToSession = (sessionId: string) => {
    const targetSession = sessions.find(s => s.id === sessionId)
    if (targetSession) {
      setCurrentSessionId(sessionId)
      setMessages(targetSession.messages)
      setSession(targetSession.sessionData || { last_indices: [], last_results: [], overrides: {} })
      setShowSessionList(false)
    }
  }

  const deleteSession = (sessionId: string) => {
    setSessions(prev => prev.filter(s => s.id !== sessionId))
    if (sessionId === currentSessionId) {
      if (sessions.length > 1) {
        const remaining = sessions.filter(s => s.id !== sessionId)
        switchToSession(remaining[0].id)
      } else {
        createNewSession()
      }
    }
  }

  const startRenamingSession = (sessionId: string, currentName: string) => {
    setEditingSessionId(sessionId)
    setEditingSessionName(currentName)
  }

  const saveSessionName = () => {
    if (editingSessionId && editingSessionName.trim()) {
      setSessions(prev => prev.map(s => 
        s.id === editingSessionId 
          ? { ...s, name: editingSessionName.trim() }
          : s
      ))
    }
    setEditingSessionId(null)
    setEditingSessionName('')
  }

  const cancelRenaming = () => {
    setEditingSessionId(null)
    setEditingSessionName('')
  }

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight })
  }, [messages])

  function renderBlockFromActivity(activity: any): string {
    if (!activity) return ''
    const title = activity['Activity Name'] || activity['Title'] || 'Lesson Plan'
    const time = (activity['Time'] || '').trim()

    const out: string[] = []
    // H1 with time badge
    out.push(`# ${title}${time ? `  ⏱️ ${time}` : ''}`)

    const addSection = (label: string, value?: string | null) => {
      const v = (value || '').trim()
      if (!v) return
      out.push(`\n**${label}**`)
      // turn multi-line chunks into bullets for readability
      const parts = v.split(/\r?\n/).map(s => s.trim()).filter(Boolean)
      if (['Materials','Student Materials','Examples','Additional Resources'].includes(label)) {
        parts.forEach(p => out.push(`- ${p}`))
      } else if (label === 'Directions') {
        // keep existing numbering; otherwise number the lines
        parts.forEach((p, i) => {
          out.push(/^\d+\./.test(p) ? p : `${i + 1}. ${p}`)
        })
      } else {
        out.push(parts.join(' '))
      }
    }

    addSection('Objective', activity['Objective'])
    addSection('Overview', activity['Introduction'])
    addSection('Advance Preparation', activity['Advance preparation'])
    addSection('Materials', activity['Materials'] || activity['Materials Needed'])
    addSection('Student Materials', activity['Student Materials'])
    addSection('Directions', activity['Directions'])
    addSection('Examples', activity['Examples'])   // 👈 Render examples as bullets in chat
    addSection('Additional Resources', activity['Additional Resources'])
    if ((activity['Source Link'] || '').trim()) {
      out.push(`\n**Source**\n${activity['Source Link'].trim()}`)
    }

    return out.join('\n').trim()
  }

  async function send() {
    const text = input.trim()
    if (!text || loading) return
    setMessages(m => [...m, { role: 'user', content: text }])
    
    // Update session name based on first user message (if it's the first user message)
    if (messages.length === 1) { // Only assistant welcome message exists
      const sessionName = text.length > 30 ? text.substring(0, 30) + '...' : text
      setSessions(prev => prev.map(s => 
        s.id === currentSessionId 
          ? { ...s, name: sessionName }
          : s
      ))
    }
    
    setInput('')
    setLoading(true)
    try {
      // All requests go through the unified /chat endpoint (including modifications)
      const r = await chat(session, text)
      // Merge session data instead of replacing it completely
      setSession(prev => ({ ...prev, ...r.session }))
      const resp = r.response
      
      // Update session state with the latest session data from backend
      setSessions(prev => prev.map(s => 
        s.id === currentSessionId 
          ? { ...s, lastUpdated: Date.now() }
          : s
      ))
      if (resp.type === 'lesson_plan') {
        const b64 = resp.lesson_plan?.doc_bytes
        const fname = resp.lesson_plan?.filename || 'lesson_plan.docx'
        const rowIndex = resp.lesson_plan?.row_index
        const title = resp.lesson_plan?.title || 'Unknown Activity'
        let activityDetail = null;

        // Fetch structured activity for editing
        if (typeof rowIndex === 'number') {
          try {
            const detail = await getActivity(rowIndex, session)
            if (detail?.activity) {
              activityDetail = detail.activity
            }
          } catch (e) {
            console.error('Failed to fetch activity details for editing', e)
          }
        }

        // Use content_md if available, otherwise fall back to chat_markdown/chat_message/content
        const chatText = resp.content_md || resp.chat_markdown || resp.chat_message || resp.content || 'Lesson plan generated'
        setMessages(m => [...m, { 
          role: 'assistant', 
          content: chatText,
          lessonPlan: {
            docBytes: b64 || '',
            filename: fname,
            activity: activityDetail,
            rowIndex: rowIndex || 0,
            title: resp.plan_title || title
          }
        }])

        // Track download click for feedback (if docBytes exist)
        if (b64) {
          recordDownloadClick(rowIndex?.toString() || 'unknown', title, fname)
        }
      } else if (resp.type === 'capabilities') {
        const chatText = resp.content || 'I can help with activities, lesson plans, and customization.'
        setMessages(m => [...m, { role: 'assistant', content: chatText }])

        if (Array.isArray(resp.quick_actions) && resp.quick_actions.length) {
          const qaLines = resp.quick_actions.map((qa: any, i: number) => `- [${qa.label}](#qa-${i})`).join('\n')
          setMessages(m => [...m, { role: 'assistant', content: `Try one:\n\n${qaLines}` }])

          // Bind click handlers to quick action links
          setTimeout(() => {
            document.querySelectorAll('a[href^="#qa-"]').forEach((a, idx) => {
              (a as HTMLAnchorElement).onclick = (ev) => {
                ev.preventDefault()
                const qa = resp.quick_actions[idx]
                if (qa?.payload) {
                  setInput(qa.payload)
                  send()
                }
              }
            })
          }, 0)
        }
      } else {
        // ====================================================================
        // FIX #1: ELIMINATE DUPLICATION
        // Only render activities component if show_shortlist is true
        // Otherwise render content normally
        // ====================================================================
        
        if (resp.show_shortlist && resp.activities?.length) {
          // Render ONLY activities component - NO content duplication
          setMessages(m => [...m, { 
            role: 'assistant', 
            content: 'Here are the activities I found:', 
            activities: resp.activities 
          }])
          
          // Track activities for feedback and context
          resp.activities.forEach((activity: any, index: number) => {
            const activityId = activity.row_index?.toString() || `activity_${index}`
            const activityName = activity.title || 'Unknown Activity'
            trackActivityRendered(activityId, activityName)
          })
          
          // ====================================================================
          // FIX #2: REMOVE VERBOSE FOOTER
          // Teachers know what to do - no need for instructions after every search
          // ====================================================================
          
        } else {
          // For non-search responses (acknowledgments, teaching questions, etc.)
          const chatText = resp.content_md || resp.chat_markdown || resp.chat_message || resp.content || '(no content)'
          setMessages(m => [...m, { role: 'assistant', content: chatText }])
        }
      }
    } catch (e: any) {
      console.error(e)
      setMessages(m => [...m, { role: 'system', content: `Error contacting API: ${e?.message || e}` }])
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="min-h-screen flex bg-white text-[color:var(--text)]">
      {/* Sidebar */}
      <div className={`${showSessionList ? 'w-80' : 'w-0'} transition-all duration-300 overflow-hidden bg-gray-50 border-r flex flex-col`}>
        <div className="p-4 border-b">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-lg">Chat History</h2>
            <button 
              onClick={() => setShowSessionList(false)}
              className="text-gray-500 hover:text-gray-700"
            >
              ✕
            </button>
          </div>
          <button 
            onClick={createNewSession}
            className="w-full bg-[var(--primary)] text-white py-2 px-4 rounded hover:opacity-90 flex items-center justify-center gap-2"
          >
            ➕ New Chat
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4">
          <div className="space-y-2">
                {sessions.map(s => (
                  <div key={s.id} className={`p-3 rounded-lg border transition-colors ${
                    s.id === currentSessionId 
                      ? 'bg-[var(--primary)] text-white border-[var(--primary)]' 
                      : 'bg-white hover:bg-gray-100 border-gray-200'
                  }`}>
                    {editingSessionId === s.id ? (
                      <div className="space-y-2">
                        <input
                          type="text"
                          value={editingSessionName}
                          onChange={(e) => setEditingSessionName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') saveSessionName()
                            if (e.key === 'Escape') cancelRenaming()
                          }}
                          className="w-full px-2 py-1 text-sm border rounded bg-white text-gray-900"
                          autoFocus
                        />
                        <div className="flex gap-2">
                          <button
                            onClick={saveSessionName}
                            className="text-xs bg-green-500 text-white px-2 py-1 rounded hover:bg-green-600"
                          >
                            ✓ Save
                          </button>
                          <button
                            onClick={cancelRenaming}
                            className="text-xs bg-gray-500 text-white px-2 py-1 rounded hover:bg-gray-600"
                          >
                            ✕ Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <button
                          onClick={() => switchToSession(s.id)}
                          className="w-full text-left"
                        >
                          <div className={`text-sm font-medium mb-1 ${
                            s.id === currentSessionId ? 'text-white' : 'text-gray-900'
                          }`}>
                            {s.name}
                          </div>
                          <div className={`text-xs ${
                            s.id === currentSessionId ? 'text-white/80' : 'text-gray-500'
                          }`}>
                            {new Date(s.lastUpdated).toLocaleDateString()} • {s.messages.length} messages
                          </div>
                        </button>
                        <div className="mt-2 flex gap-2">
                          <button
                            onClick={() => startRenamingSession(s.id, s.name)}
                            className={`text-xs ${
                              s.id === currentSessionId 
                                ? 'text-white/80 hover:text-white' 
                                : 'text-blue-500 hover:text-blue-700'
                            }`}
                          >
                            ✏️ Rename
                          </button>
                          {sessions.length > 1 && (
                            <button
                              onClick={() => deleteSession(s.id)}
                              className={`text-xs ${
                                s.id === currentSessionId 
                                  ? 'text-white/80 hover:text-white' 
                                  : 'text-red-500 hover:text-red-700'
                              }`}
                            >
                              🗑️ Delete
                            </button>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <header className="px-4 py-3 bg-[var(--primary)] text-white font-semibold flex items-center">
          <button 
            onClick={() => setShowSessionList(!showSessionList)}
            className="text-sm bg-white/20 hover:bg-white/30 px-3 py-1 rounded flex items-center gap-2 mr-4"
          >
            📋 Sessions ({sessions.length})
          </button>
          <span className="flex-1 text-center">🎓 ClassroomGPT</span>
        </header>
          <div ref={listRef} className="flex-1 overflow-auto p-4 pb-20 space-y-3">
        {messages.map((m, i) => (
          <div key={i} className={m.role==='user' ? 'text-right' : ''}>
            <div className={
              m.role==='user'
                ? 'inline-block rounded px-3 py-2 bg-[var(--accent)] text-black'
                : m.role==='system'
                  ? 'inline-block rounded px-3 py-2 bg-red-100 text-red-800'
                  : 'inline-block rounded px-3 py-2 bg-slate-100'
            }>
                <div className="message-block prose prose-base max-w-none font-sans text-[15px] leading-relaxed">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {normalizeBullets(m.content)}
                  </ReactMarkdown>
                </div>
                {m.activities && (
                  <div className="mt-3">
                    <ActivityList activities={m.activities} />
                  </div>
                )}
                {m.lessonPlan && (
                  <div className="mt-2 flex flex-col gap-2">
                    <a
                      href="#"
                      className="btn-primary inline-block text-center"
                      onClick={async (e) => {
                        e.preventDefault();
                        try {
                          // Build using your Strategic Action template (single click)
                          const res = await buildDoc({
                            ...(m.lessonPlan!.activity || {}),
                            __template: 'strategic_action_v1', // backend can ignore if unsupported
                            __layout: 'table'                  // optional hint
                          });

                          const b64 = res.doc_bytes;
                          const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
                          const blob = new Blob([bytes], { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
                          const url = URL.createObjectURL(blob);
                          const fname = res.filename || (m.lessonPlan!.activity?.['Activity Name'] || 'Lesson Plan') + '.docx';

                          const a = document.createElement('a');
                          a.href = url; a.download = fname; document.body.appendChild(a); a.click(); a.remove();

                          recordDownloadClick(m.lessonPlan!.rowIndex.toString(), m.lessonPlan!.title, fname);
                        } catch (err) {
                          // Fallback: if server already gave us a doc, let them download that
                          if (m.lessonPlan?.docBytes) {
                            const a = document.createElement('a');
                            a.href = `data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,${m.lessonPlan.docBytes}`;
                            a.download = m.lessonPlan.filename;
                            document.body.appendChild(a); a.click(); a.remove();
                            recordDownloadClick(m.lessonPlan.rowIndex.toString(), m.lessonPlan.title, m.lessonPlan.filename);
                          } else {
                            setMessages(prev => [...prev, { role:'system', content:`Error building lesson plan: ${String(err)}` }]);
                          }
                        }
                      }}
                    >
                      📄 Download Lesson Plan
                    </a>

                    {/* Edit Section */}
                    {m.lessonPlan.activity && (
                      <details>
                        <summary className="cursor-pointer underline text-sm">Edit before download</summary>
                        <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-3">
                          <label className="flex flex-col text-sm">Activity Name
                            <input className="border rounded p-2" value={m.lessonPlan.activity['Activity Name']||''} onChange={e=>setMessages(prevMessages => prevMessages.map(msg => msg === m ? {...msg, lessonPlan: {...msg.lessonPlan!, activity: {...msg.lessonPlan!.activity, 'Activity Name': e.target.value}}} : msg))} />
                          </label>
                          <label className="flex flex-col text-sm">Time
                            <input className="border rounded p-2" value={m.lessonPlan.activity['Time']||''} onChange={e=>setMessages(prevMessages => prevMessages.map(msg => msg === m ? {...msg, lessonPlan: {...msg.lessonPlan!, activity: {...msg.lessonPlan!.activity, 'Time': e.target.value}}} : msg))} />
                          </label>
                          <label className="md:col-span-2 flex flex-col text-sm">Objective
                            <textarea className="border rounded p-2" rows={3} value={m.lessonPlan.activity['Objective']||''} onChange={e=>setMessages(prevMessages => prevMessages.map(msg => msg === m ? {...msg, lessonPlan: {...msg.lessonPlan!, activity: {...msg.lessonPlan!.activity, 'Objective': e.target.value}}} : msg))} />
                          </label>
                          <label className="flex flex-col text-sm">Materials
                            <textarea className="border rounded p-2" rows={3} value={m.lessonPlan.activity['Materials']||''} onChange={e=>setMessages(prevMessages => prevMessages.map(msg => msg === m ? {...msg, lessonPlan: {...msg.lessonPlan!, activity: {...msg.lessonPlan!.activity, 'Materials': e.target.value}}} : msg))} />
                          </label>
                          <label className="flex flex-col text-sm">Additional Resources
                            <textarea className="border rounded p-2" rows={3} value={m.lessonPlan.activity['Additional Resources']||''} onChange={e=>setMessages(prevMessages => prevMessages.map(msg => msg === m ? {...msg, lessonPlan: {...msg.lessonPlan!, activity: {...msg.lessonPlan!.activity, 'Additional Resources': e.target.value}}} : msg))} />
                          </label>
                          <label className="md:col-span-2 flex flex-col text-sm">Directions (1. a. b.)
                            <textarea className="border rounded p-2" rows={6} value={m.lessonPlan.activity['Directions']||''} onChange={e=>setMessages(prevMessages => prevMessages.map(msg => msg === m ? {...msg, lessonPlan: {...msg.lessonPlan!, activity: {...msg.lessonPlan!.activity, 'Directions': e.target.value}}} : msg))} />
                          </label>
                        </div>
                      </details>
                    )}
                  </div>
                )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="text-left">
            <div className="inline-block rounded px-3 py-2 bg-slate-100">
              <span className="font-sans text-sm">Thinking…</span>
            </div>
          </div>
        )}
      </div>
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t p-3 flex gap-2 z-10">
        <textarea className="flex-1 border rounded p-2" rows={2} value={input} onChange={e=>setInput(e.target.value)} onKeyDown={onKeyDown} placeholder="Ask me anything…" disabled={loading} />
        <button className="btn-primary" onClick={send} disabled={loading}>{loading ? 'Sending…' : 'Send'}</button>
      </div>
      </div>
      
      {/* Feedback Form Modal */}
      {showFeedbackForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-semibold mb-4">Quick check-in: you looked at <strong>{showFeedbackForm.activityName}</strong>. Was it useful?</h3>
            
            <div className="space-y-4">
              {/* Like/Dislike */}
              <div>
                <label className="block text-sm font-medium mb-2">Was it useful?</label>
                <div className="flex gap-4">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="like"
                      checked={showFeedbackForm.like}
                      onChange={() => setShowFeedbackForm(prev => prev ? {...prev, like: true, reason: REASONS_POS[0]} : null)}
                      className="mr-2"
                    />
                    👍 Yes
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      name="like"
                      checked={!showFeedbackForm.like}
                      onChange={() => setShowFeedbackForm(prev => prev ? {...prev, like: false, reason: REASONS_NEG[0]} : null)}
                      className="mr-2"
                    />
                    👎 No
                  </label>
                </div>
              </div>

              {/* Reason */}
              <div>
                <label className="block text-sm font-medium mb-2">Why?</label>
                <select
                  value={showFeedbackForm.reason}
                  onChange={(e) => setShowFeedbackForm(prev => prev ? {...prev, reason: e.target.value} : null)}
                  className="w-full border rounded p-2"
                >
                  {(showFeedbackForm.like ? REASONS_POS : REASONS_NEG).map(reason => (
                    <option key={reason} value={reason}>{reason}</option>
                  ))}
                </select>
              </div>

              {/* Frequency */}
              <div>
                <label className="block text-sm font-medium mb-2">How many times have you used this activity?</label>
                <select
                  value={showFeedbackForm.frequency}
                  onChange={(e) => setShowFeedbackForm(prev => prev ? {...prev, frequency: e.target.value} : null)}
                  className="w-full border rounded p-2"
                >
                  {FREQUENCY_OPTIONS.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              {/* Time Spent */}
              <div>
                <label className="block text-sm font-medium mb-2">About how much time did you spend on it (per run)?</label>
                <select
                  value={showFeedbackForm.timeSpent}
                  onChange={(e) => setShowFeedbackForm(prev => prev ? {...prev, timeSpent: e.target.value} : null)}
                  className="w-full border rounded p-2"
                >
                  {TIME_SPENT_OPTIONS.map(option => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>

              {/* Adaptation */}
              <div>
                <label className="block text-sm font-medium mb-2">How did you adapt or integrate this activity?</label>
                <textarea
                  value={showFeedbackForm.adaptation}
                  onChange={(e) => setShowFeedbackForm(prev => prev ? {...prev, adaptation: e.target.value} : null)}
                  placeholder="e.g., paired students, trimmed to 10 minutes, used sticky notes instead of handouts…"
                  className="w-full border rounded p-2 h-20 resize-none"
                />
              </div>

              {/* Buttons */}
              <div className="flex gap-2 pt-4">
                <button
                  onClick={() => setShowFeedbackForm(null)}
                  className="flex-1 border border-gray-300 rounded px-4 py-2 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={() => submitFeedback(showFeedbackForm)}
                  className="flex-1 bg-[var(--primary)] text-white rounded px-4 py-2 hover:opacity-90"
                >
                  Send feedback
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}