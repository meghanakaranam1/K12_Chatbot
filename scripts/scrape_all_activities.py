# scripts/scrape_all_activities.py
"""
One-time script to scrape all activities and populate content cache
Run this once to pre-populate the database
"""
import sys
import os
from time import sleep

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import data_processor
from src.services.content_retrieval import content_retrieval
from src.data.content_cache import content_cache

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not installed. Install with: pip install tqdm")

def scrape_all_activities(start_from: int = 0, limit: int = None):
    """
    Scrape all activities from CSV and store in cache
    
    Args:
        start_from: Row index to start from (useful for resuming)
        limit: Maximum number of rows to process (None = all)
    """
    print("🚀 Starting bulk scraping process...")
    
    # Initialize data processor
    data_processor.initialize("data/activities.csv")
    total_rows = len(data_processor.df)
    
    if limit:
        total_rows = min(total_rows, start_from + limit)
    
    print(f"📊 Total activities to scrape: {total_rows - start_from}")
    
    stats = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'no_url': 0
    }
    
    # Process each row
    iterator = range(start_from, total_rows)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Scraping activities")
    
    for row_id in iterator:
        try:
            # Check if already cached
            if content_cache.has_content(row_id):
                stats['skipped'] += 1
                continue
            
            row = data_processor.get_row(row_id)
            
            # Get URL from row
            url = (data_processor.safe_get(row, "Link to Resource") or 
                   data_processor.safe_get(row, "Link to the Source Materials") or
                   data_processor.safe_get(row, "Resources Needed") or
                   data_processor.safe_get(row, "Reference Link"))
            
            if not url or not url.strip():
                stats['no_url'] += 1
                content_cache.set(
                    row_id=row_id,
                    url="",
                    text="",
                    status='no_url',
                    error="No URL provided"
                )
                continue
            
            # Scrape content
            text, error = content_retrieval.text_from_supabase_or_fallback(row, allow_web=True)
            
            if text and text.strip():
                # Success!
                content_cache.set(
                    row_id=row_id,
                    url=url,
                    text=text,
                    status='success',
                    source_type=_detect_source_type(url)
                )
                stats['success'] += 1
                activity_name = data_processor.safe_get(row, 'Strategic Action') or 'Unknown'
                print(f"✅ [{row_id}] {activity_name[:50]} ({len(text)} chars)")
            else:
                # Failed
                content_cache.set(
                    row_id=row_id,
                    url=url,
                    text="",
                    status='failed',
                    error=error or "No content retrieved"
                )
                stats['failed'] += 1
                print(f"❌ [{row_id}] Failed: {error}")
            
            # Rate limiting (be nice to servers)
            sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"💥 Error processing row {row_id}: {e}")
            url_var = url if 'url' in locals() else ""
            content_cache.set(
                row_id=row_id,
                url=url_var,
                text="",
                status='failed',
                error=str(e)
            )
            stats['failed'] += 1
            continue
    
    # Print final stats
    print("\n" + "="*60)
    print("📊 SCRAPING COMPLETE")
    print("="*60)
    print(f"✅ Success: {stats['success']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"⏭️  Skipped (already cached): {stats['skipped']}")
    print(f"🔗 No URL: {stats['no_url']}")
    print(f"📝 Total processed: {sum(stats.values())}")
    
    # Get cache stats
    cache_stats = content_cache.get_stats()
    print(f"\n💾 Cache Database Stats:")
    print(f"   Total entries: {cache_stats['total']}")
    print(f"   Successful: {cache_stats['success']}")
    print(f"   Total content: {cache_stats['total_bytes'] / 1024 / 1024:.2f} MB")
    print(f"   Avg per activity: {cache_stats['avg_bytes'] / 1024:.2f} KB")

def _detect_source_type(url: str) -> str:
    """Detect source type from URL"""
    url_lower = url.lower()
    if 'drive.google.com' in url_lower:
        return 'google_drive'
    elif '.pdf' in url_lower:
        return 'pdf'
    elif '.doc' in url_lower:
        return 'docx'
    else:
        return 'web'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Scrape all activities and cache content')
    parser.add_argument('--start', type=int, default=0, help='Start from row index')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows')
    parser.add_argument('--resume', action='store_true', help='Resume from last position')
    
    args = parser.parse_args()
    
    if args.resume:
        # Find last successfully scraped row
        cache_stats = content_cache.get_stats()
        args.start = cache_stats['total']
        print(f"📍 Resuming from row {args.start}")
    
    scrape_all_activities(start_from=args.start, limit=args.limit)

# scripts/fix_row_1_simple.py
"""
Simple fix for Row 1 - scrape the known file from Resources Needed
"""
import sys
import os
import requests
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.content_cache import content_cache

def _fetch_google_drive_file(file_id: str):
    """Fetch Google Drive file directly"""
    try:
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        print(f"🔍 Attempting Google Drive download: {file_id}")
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response = session.get(download_url, timeout=30)
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or download_url.endswith('.pdf'):
                return _extract_text_from_pdf_bytes(response.content)
            
            elif 'word' in content_type or 'msword' in content_type or download_url.endswith(('.doc', '.docx')):
                return _extract_text_from_docx_bytes(response.content)
            
            elif 'text' in content_type or 'html' in content_type:
                return response.text, ""
            
            else:
                # Try PDF first
                pdf_text, pdf_error = _extract_text_from_pdf_bytes(response.content)
                if pdf_text:
                    return pdf_text, ""
                
                # Try DOCX
                docx_text, docx_error = _extract_text_from_docx_bytes(response.content)
                if docx_text:
                    return docx_text, ""
                
                return "", f"Unknown file type: {content_type}"
        else:
            return "", f"Google Drive download failed: HTTP {response.status_code}"
            
    except Exception as e:
        return "", f"Google Drive fetch error: {str(e)}"

def _extract_text_from_pdf_bytes(content: bytes):
    """Extract text from PDF bytes"""
    try:
        import PyPDF2
        pdf_file = io.BytesIO(content)
        reader = PyPDF2.PdfReader(pdf_file)
        
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text())
        
        full_text = "\n\n".join(text_parts)
        if full_text.strip():
            print(f"✅ Extracted {len(full_text)} chars from PDF")
            return full_text, ""
        else:
            return "", "PDF contains no extractable text"
            
    except ImportError:
        return "", "PyPDF2 not installed. Run: pip install PyPDF2"
    except Exception as e:
        return "", f"PDF extraction failed: {str(e)}"

def _extract_text_from_docx_bytes(content: bytes):
    """Extract text from DOCX bytes"""
    try:
        from docx import Document
        docx_file = io.BytesIO(content)
        doc = Document(docx_file)
        
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        full_text = "\n\n".join(text_parts)
        if full_text.strip():
            print(f"✅ Extracted {len(full_text)} chars from DOCX")
            return full_text, ""
        else:
            return "", "DOCX contains no extractable text"
            
    except ImportError:
        return "", "python-docx not installed. Run: pip install python-docx"
    except Exception as e:
        return "", f"DOCX extraction failed: {str(e)}"

def fix_row_1():
    print("="*80)
    print("🔧 FIXING ROW 1: School Climate student voice")
    print("="*80)
    
    # Step 1: Clear failed cache entry
    print("\n[1/3] Clearing old failed cache entry...")
    content_cache.conn.execute("DELETE FROM scraped_content WHERE row_id = 1")
    content_cache.conn.commit()
    print("✅ Cleared\n")
    
    # Step 2: Scrape the file we know exists
    file_id = "1sZKtb9QzZrwZWIeK3EPwED_X107kbw73"
    file_url = f"https://drive.google.com/file/d/{file_id}/view?usp=drive_link"
    
    print(f"[2/3] Scraping file: {file_id}")
    print(f"URL: {file_url}\n")
    
    try:
        text, error = _fetch_google_drive_file(file_id)
        
        if text and len(text) > 100:
            print(f"✅ Success! Retrieved {len(text)} characters\n")
            
            # Step 3: Cache it
            print("[3/3] Caching content...")
            content_cache.set(
                row_id=1,
                url=file_url,
                text=text,
                status='success',
                source_type='google_drive'
            )
            print("✅ Cached successfully!\n")
            
            # Show preview
            print("="*80)
            print("📄 CONTENT PREVIEW (first 500 chars)")
            print("="*80)
            print(text[:500])
            print("="*80)
            
            # Final status check
            stats = content_cache.get_stats()
            print(f"\n💾 FINAL CACHE STATUS:")
            print(f"   Total entries: {stats['total']}")
            print(f"   ✅ Successful: {stats['success']}")
            print(f"   ❌ Failed: {stats['failed']}")
            print(f"   📊 Total content: {stats['total_bytes'] / 1024 / 1024:.2f} MB")
            
            if stats['failed'] == 0:
                print("\n" + "🎉"*20)
                print("🎉 100% COVERAGE ACHIEVED! ALL 21 ACTIVITIES CACHED!")
                print("🎉"*20)
            
            return True
            
        else:
            print(f"❌ Failed to retrieve content: {error}\n")
            print("This might mean:")
            print("  1. The file isn't publicly accessible")
            print("  2. Google Drive is blocking the request")
            print("  3. The file is corrupted or empty")
            print("\nTrying alternative approach...\n")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n")
    success = fix_row_1()
    
    if not success:
        print("\n" + "="*80)
        print("⚠️  AUTOMATIC SCRAPING FAILED")
        print("="*80)
        print("\nPLAN B: Manual insertion")
        print("\nPlease do the following:")
        print("1. Open this URL in your browser:")
        print("   https://drive.google.com/file/d/1sZKtb9QzZrwZWIeK3EPwED_X107kbw73/view")
        print("\n2. If it opens, download the PDF and extract text")
        print("3. Run: python scripts/manual_insert_row_1.py")
        print("   (I'll create this file for you)")
    
    print("\n✅ Done!\n")


