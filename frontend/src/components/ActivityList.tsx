// components/ActivityList.tsx
import React from 'react'

interface Activity {
  row_index: number
  title: string
  time?: string
  summary?: string
  display_line?: string
  display_id?: string
  score?: number
}

export default function ActivityList({ activities }: { activities: Activity[] }) {
  return (
    <div className="space-y-2">
      {activities.map((a, i) => {
        // Prefer explicit fields, otherwise try to parse display_line as "Title – summary"
        let header = a.title + (a.time ? ` ⏱️ ${a.time}` : '')
        let body = a.summary || ''

        if (!a.summary && a.display_line) {
          const parts = a.display_line.split(' – ')
          header = parts[0] || header
          body = parts.slice(1).join(' – ')
        }

        return (
          <div key={a.row_index || i} className="border-b pb-2">
            <p className="activity-header">{header}</p>
            {body ? <p className="activity-body">{body}</p> : null}
          </div>
        )
      })}
    </div>
  )
}
