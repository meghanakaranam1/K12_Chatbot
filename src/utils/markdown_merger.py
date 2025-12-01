"""
Generic markdown merger for lesson plans and activity content.
Inserts enrichment blocks into sensible anchors or appends new sections.
"""
import re


def merge_block_into_markdown(md: str, preferred_heading: str, block_md: str) -> str:
    """
    Merge a block of markdown content into an existing markdown document.
    
    Args:
        md: The existing markdown content
        preferred_heading: The heading to look for (e.g., "Encourage Emotional Regulation")
        block_md: The new content to insert
    
    Returns:
        Updated markdown with the block merged in
    """
    md = md or ""
    
    # 1) Try to find preferred heading (e.g., "Encourage Emotional Regulation:")
    heading_pattern = re.compile(rf"^###?\s*{re.escape(preferred_heading)}\s*$", re.IGNORECASE | re.MULTILINE)
    if heading_pattern.search(md):
        # insert right after the heading line
        lines = md.splitlines()
        out = []
        inserted = False
        i = 0
        while i < len(lines):
            out.append(lines[i])
            if re.match(rf"^###?\s*{re.escape(preferred_heading)}\s*$", lines[i], re.IGNORECASE):
                # ensure a blank line and then inject
                out.append("")
                out.append(block_md.strip())
                out.append("")
                inserted = True
            i += 1
        return "\n".join(out)

    # 2) If not found, try to append under a logical step (e.g., Step 5: Reflection/Regulation)
    step5 = re.search(r"^\s*5\.\s.*$", md, re.MULTILINE)
    if step5:
        insert_at = step5.end()
        return md[:insert_at] + "\n\n" + block_md.strip() + "\n\n" + md[insert_at:]

    # 3) Else append to end under a new heading
    appendix = f"\n\n### {preferred_heading}\n\n{block_md.strip()}\n"
    return md + appendix


def find_insertion_point(md: str, preferred_heading: str) -> int:
    """
    Find the best insertion point for new content.
    
    Args:
        md: The markdown content
        preferred_heading: The heading to look for
    
    Returns:
        Index where content should be inserted, or -1 if not found
    """
    # Look for the preferred heading
    heading_pattern = re.compile(rf"^###?\s*{re.escape(preferred_heading)}\s*$", re.IGNORECASE | re.MULTILINE)
    match = heading_pattern.search(md)
    if match:
        return match.end()
    
    # Look for Step 5 (Reflection/Regulation)
    step5 = re.search(r"^\s*5\.\s.*$", md, re.MULTILINE)
    if step5:
        return step5.end()
    
    # Default to end of document
    return len(md)


