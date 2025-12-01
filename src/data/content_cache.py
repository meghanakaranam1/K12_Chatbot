# src/data/content_cache.py
"""
SQLite-based content cache for pre-scraped activity content
"""
import sqlite3
import os
from typing import Optional, Tuple
from datetime import datetime
import hashlib

class ContentCache:
    """Manages pre-scraped content in SQLite database"""
    
    def __init__(self, db_path: str = "data/content_cache.db"):
        self.db_path = db_path
        self._ensure_db_dir()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _ensure_db_dir(self):
        """Create data directory if it doesn't exist"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _create_tables(self):
        """Create cache tables if they don't exist"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scraped_content (
                row_id INTEGER PRIMARY KEY,
                source_url TEXT,
                url_hash TEXT,
                scraped_text TEXT,
                scrape_status TEXT DEFAULT 'success',
                error_message TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content_length INTEGER,
                source_type TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_url_hash 
            ON scraped_content(url_hash)
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scrape_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add new columns for pre-generated content
        # Use try/except because columns might already exist
        try:
            self.conn.execute("""
                ALTER TABLE scraped_content 
                ADD COLUMN generated_lesson_plan TEXT
            """)
            print("✅ Added generated_lesson_plan column")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            self.conn.execute("""
                ALTER TABLE scraped_content 
                ADD COLUMN generated_summary TEXT
            """)
            print("✅ Added generated_summary column")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            self.conn.execute("""
                ALTER TABLE scraped_content 
                ADD COLUMN generated_at TIMESTAMP
            """)
            print("✅ Added generated_at column")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        self.conn.commit()
    
    def _hash_url(self, url: str) -> str:
        """Create hash of URL for fast lookup"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def get(self, row_id: int) -> Optional[Tuple[str, str]]:
        """
        Get cached content by row ID
        Returns: (scraped_text, scrape_status) or (None, None)
        """
        cursor = self.conn.execute(
            "SELECT scraped_text, scrape_status FROM scraped_content WHERE row_id = ?",
            (row_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return row[0], row[1]
        return None, None
    
    def get_by_url(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Get cached content by URL
        Returns: (scraped_text, scrape_status) or (None, None)
        """
        url_hash = self._hash_url(url)
        cursor = self.conn.execute(
            "SELECT scraped_text, scrape_status FROM scraped_content WHERE url_hash = ?",
            (url_hash,)
        )
        row = cursor.fetchone()
        
        if row:
            return row[0], row[1]
        return None, None
    
    def set(self, row_id: int, url: str, text: str, 
            status: str = 'success', error: str = None, source_type: str = None):
        """Store scraped content in cache"""
        url_hash = self._hash_url(url) if url else None
        content_length = len(text) if text else 0
        
        self.conn.execute("""
            INSERT OR REPLACE INTO scraped_content 
            (row_id, source_url, url_hash, scraped_text, scrape_status, 
             error_message, scraped_at, content_length, source_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row_id, url, url_hash, text, status, 
            error, datetime.now(), content_length, source_type
        ))
        self.conn.commit()
    
    def has_content(self, row_id: int) -> bool:
        """Check if row has cached content"""
        cursor = self.conn.execute(
            "SELECT 1 FROM scraped_content WHERE row_id = ? AND scrape_status = 'success'",
            (row_id,)
        )
        return cursor.fetchone() is not None
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN scrape_status = 'success' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN scrape_status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN scrape_status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(content_length) as total_bytes,
                AVG(content_length) as avg_bytes
            FROM scraped_content
        """)
        row = cursor.fetchone()
        
        return {
            'total': row[0] or 0,
            'success': row[1] or 0,
            'failed': row[2] or 0,
            'pending': row[3] or 0,
            'total_bytes': row[4] or 0,
            'avg_bytes': row[5] or 0
        }
    
    def clear_all(self):
        """Clear all cached content (use with caution!)"""
        self.conn.execute("DELETE FROM scraped_content")
        self.conn.commit()
    
    def set_generated(self, row_id: int, lesson_plan: str, summary: str):
        """Store pre-generated lesson plan and summary"""
        
        # Check if row exists
        cursor = self.conn.execute(
            "SELECT row_id FROM scraped_content WHERE row_id = ?",
            (row_id,)
        )
        exists = cursor.fetchone()
        
        if exists:
            # Update existing row
            self.conn.execute("""
                UPDATE scraped_content 
                SET generated_lesson_plan = ?,
                    generated_summary = ?,
                    generated_at = CURRENT_TIMESTAMP
                WHERE row_id = ?
            """, (lesson_plan, summary, row_id))
        else:
            # Insert new row with just the generated content
            self.conn.execute("""
                INSERT INTO scraped_content 
                (row_id, generated_lesson_plan, generated_summary, generated_at, scrape_status)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'generated_only')
            """, (row_id, lesson_plan, summary))
        
        self.conn.commit()
        
        print(f"✅ Cached pre-generated content for row {row_id}")
    
    def get_generated(self, row_id: int):
        """Get pre-generated lesson plan and summary if available"""
        
        cursor = self.conn.execute("""
            SELECT generated_lesson_plan, generated_summary, generated_at
            FROM scraped_content
            WHERE row_id = ?
        """, (row_id,))
        
        result = cursor.fetchone()
        
        if result and result[0]:  # Only require lesson_plan, summary is optional
            lesson_plan, summary, generated_at = result
            summary = summary if summary else None  # Ensure summary is None if empty
            print(f"✅ [PRE-GENERATED] Using cached content for row {row_id} (generated {generated_at})")
            return lesson_plan, summary
        
        print(f"ℹ️  [NO PRE-GENERATED] Row {row_id} needs fresh generation")
        return None, None
    
    def get_generation_stats(self):
        """Get statistics about pre-generated content"""
        
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN generated_lesson_plan IS NOT NULL THEN 1 ELSE 0 END) as pre_generated,
                SUM(LENGTH(generated_lesson_plan)) as total_lesson_bytes,
                SUM(LENGTH(generated_summary)) as total_summary_bytes
            FROM scraped_content
        """)
        
        result = cursor.fetchone()
        
        return {
            'total': result[0] or 0,
            'pre_generated': result[1] or 0,
            'pending': (result[0] or 0) - (result[1] or 0),
            'lesson_bytes': result[2] or 0,
            'summary_bytes': result[3] or 0,
            'total_bytes': (result[2] or 0) + (result[3] or 0)
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Global instance
content_cache = ContentCache()

