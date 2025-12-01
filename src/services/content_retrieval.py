"""
Content retrieval from various sources (NO SUPABASE REQUIRED)
"""
import re
import requests
from typing import Tuple
from urllib.parse import urlparse, parse_qs
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.ai_services import openai_service
from data.processor import data_processor
from data.content_cache import content_cache

class ContentRetrieval:
    """Handles content retrieval from multiple sources without Supabase"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _is_google_drive_url(self, url: str) -> bool:
        return bool(re.search(r'drive\.google\.com', url or ''))
    
    def _extract_google_drive_id(self, url: str) -> str | None:
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
            r'/folders/([a-zA-Z0-9_-]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _is_valid_url(self, url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        try:
            result = urlparse(url.strip())
            return all([result.scheme in ['http', 'https'], result.netloc])
        except Exception:
            return False
    
    def _fetch_google_drive_file(self, file_id: str) -> Tuple[str, str]:
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            print(f"🔍 Attempting Google Drive download: {file_id}")
            response = self.session.get(download_url, timeout=30)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                
                if 'pdf' in content_type or download_url.endswith('.pdf'):
                    return self._extract_text_from_pdf_bytes(response.content)
                
                elif 'word' in content_type or 'msword' in content_type or download_url.endswith(('.doc', '.docx')):
                    return self._extract_text_from_docx_bytes(response.content)
                
                elif 'text' in content_type or 'html' in content_type:
                    return response.text, ""
                
                else:
                    pdf_text, pdf_error = self._extract_text_from_pdf_bytes(response.content)
                    if pdf_text:
                        return pdf_text, ""
                    
                    docx_text, docx_error = self._extract_text_from_docx_bytes(response.content)
                    if docx_text:
                        return docx_text, ""
                    
                    return "", f"Unknown file type: {content_type}"
            else:
                return "", f"Google Drive download failed: HTTP {response.status_code}"
                
        except Exception as e:
            return "", f"Google Drive fetch error: {str(e)}"
    
    def _extract_text_from_pdf_bytes(self, content: bytes) -> Tuple[str, str]:
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
    
    def _extract_text_from_docx_bytes(self, content: bytes) -> Tuple[str, str]:
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
    
    def _fetch_direct_file(self, url: str) -> Tuple[str, str]:
        try:
            print(f"🔍 Fetching direct file: {url[:100]}")
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return "", f"HTTP {response.status_code}"
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                return self._extract_text_from_pdf_bytes(response.content)
            
            elif 'word' in content_type or 'msword' in content_type or url.lower().endswith(('.doc', '.docx')):
                return self._extract_text_from_docx_bytes(response.content)
            
            elif 'text' in content_type or 'html' in content_type:
                if 'html' in content_type:
                    # fetch_url_content returns only string, not tuple
                    text = openai_service.fetch_url_content(url)
                    return text, "" if text else "Failed to fetch HTML"
                else:
                    return response.text, ""
            
            else:
                pdf_text, _ = self._extract_text_from_pdf_bytes(response.content)
                if pdf_text:
                    return pdf_text, ""
                
                docx_text, _ = self._extract_text_from_docx_bytes(response.content)
                if docx_text:
                    return docx_text, ""
                
                return "", f"Cannot parse file type: {content_type}"
                
        except Exception as e:
            return "", f"File fetch failed: {str(e)}"
    
    def text_from_supabase_or_fallback(self, row, allow_web: bool = True) -> Tuple[str, str]:
        """
        Main method: Get text with CACHE-FIRST strategy
        
        Priority:
        1. Check cache (instant)
        2. If not cached and allow_web=True, scrape and cache
        3. Otherwise return empty
        """
        # Get row ID for cache lookup
        try:
            row_id = int(row.name)  # pandas row index
        except:
            row_id = None
        
        # ====================================================================
        # STEP 1: Try cache first (instant!)
        # ====================================================================
        if row_id is not None:
            cached_text, status = content_cache.get(row_id)
            
            if status == 'success' and cached_text:
                print(f"✅ [CACHE HIT] Row {row_id}: {len(cached_text)} chars")
                return cached_text, ""
            
            if status == 'failed':
                print(f"⚠️  [CACHE] Row {row_id} previously failed to scrape")
                return "", "Previously failed to scrape (cached)"
            
            if status == 'no_url':
                return "", "No URL provided"
        
        # ====================================================================
        # STEP 2: Not in cache - scrape if allowed
        # ====================================================================
        if not allow_web:
            return "", "Content not cached and web fetching disabled"
        
        print(f"🔍 [CACHE MISS] Row {row_id} - scraping...")
        
        # Get URL
        link_to_resource = (data_processor.safe_get(row, "Link to Resource") or 
                          data_processor.safe_get(row, "Link to the Source Materials") or
                          data_processor.safe_get(row, "Resources Needed"))
        
        if link_to_resource and link_to_resource.strip():
            link = link_to_resource.strip()
            print(f"🔍 [1] Trying Link to Resource: {link[:100]}")
            
            if not self._is_valid_url(link):
                print(f"⚠️  Invalid URL format: {link}")
            else:
                # Try scraping based on URL type
                if self._is_google_drive_url(link):
                    file_id = self._extract_google_drive_id(link)
                    if file_id:
                        text, error = self._fetch_google_drive_file(file_id)
                        if text.strip():
                            # Cache the result!
                            if row_id is not None:
                                content_cache.set(row_id, link, text, 'success', source_type='google_drive')
                            return text, ""
                        print(f"⚠️  Google Drive fetch failed: {error}")
                
                elif any(ext in link.lower() for ext in ['.pdf', '.doc', '.docx']):
                    text, error = self._fetch_direct_file(link)
                    if text.strip():
                        # Cache the result!
                        if row_id is not None:
                            source_type = 'pdf' if '.pdf' in link.lower() else 'docx'
                            content_cache.set(row_id, link, text, 'success', source_type=source_type)
                        return text, ""
                    print(f"⚠️  Direct file fetch failed: {error}")
                
                else:
                    # fetch_url_content returns only string, not tuple
                    text = openai_service.fetch_url_content(link)
                    if text and text.strip():
                        # Cache the result!
                        if row_id is not None:
                            content_cache.set(row_id, link, text, 'success', source_type='web')
                        return text, ""
                    print(f"⚠️  Web scraping failed")
        
        # Try reference link as fallback
        reference_link = data_processor.safe_get(row, "Reference Link")
        if reference_link and reference_link.strip() and self._is_valid_url(reference_link):
            link = reference_link.strip()
            print(f"🔍 [2] Trying Reference Link: {link[:100]}")
            
            if self._is_google_drive_url(link):
                file_id = self._extract_google_drive_id(link)
                if file_id:
                    text, error = self._fetch_google_drive_file(file_id)
                    if text.strip():
                        # Cache the result!
                        if row_id is not None:
                            content_cache.set(row_id, link, text, 'success', source_type='google_drive')
                        return text, ""
            elif any(ext in link.lower() for ext in ['.pdf', '.doc', '.docx']):
                text, error = self._fetch_direct_file(link)
                if text.strip():
                    # Cache the result!
                    if row_id is not None:
                        source_type = 'pdf' if '.pdf' in link.lower() else 'docx'
                        content_cache.set(row_id, link, text, 'success', source_type=source_type)
                    return text, ""
            else:
                # fetch_url_content returns only string, not tuple
                text = openai_service.fetch_url_content(link)
                if text and text.strip():
                    # Cache the result!
                    if row_id is not None:
                        content_cache.set(row_id, link, text, 'success', source_type='web')
                    return text, ""
        
        # Failed to get content
        print(f"⚠️  No accessible content found")
        if row_id is not None:
            final_url = link_to_resource or reference_link or ""
            content_cache.set(row_id, final_url, "", 'failed', error="No accessible content found")
        
        return "", "No accessible content found"
    
    def get_source_content_for_activity(self, row_id: int) -> str:
        """Get source content for a specific activity by row_id"""
        try:
            row = data_processor.get_row(row_id)
            text, error = self.text_from_supabase_or_fallback(row, allow_web=True)
            if text and text.strip():
                return text
            return ""
        except Exception as e:
            print(f"⚠️  Error getting source content for row {row_id}: {e}")
            return ""


# Global instance
content_retrieval = ContentRetrieval()
