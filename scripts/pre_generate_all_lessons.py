"""
Pre-generate all lesson plans and summaries for instant loading
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.content_cache import content_cache
from src.services.ai_services import openai_service
from src.data.processor import data_processor
from src.services.content_retrieval import content_retrieval

def pre_generate_all(skip_existing=True):
    """
    Pre-generate lesson plans and summaries for all 21 activities
    
    Args:
        skip_existing: If True, skip activities that already have pre-generated content
    """
    
    print("="*80)
    print("🚀 PRE-GENERATION SCRIPT")
    print("="*80)
    print("\nThis will generate lesson plans and summaries for all activities.")
    print("Estimated time: ~8-10 minutes (21 activities × ~25s each)")
    print("="*80)
    
    # Initialize data processor
    print("\n[SETUP] Initializing data processor...")
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "activities.csv")
    data_processor.initialize(csv_path)
    print("✅ Ready\n")
    
    # Check current status
    stats = content_cache.get_generation_stats()
    print(f"📊 Current Status:")
    print(f"   Total activities: {stats['total']}")
    print(f"   Pre-generated: {stats['pre_generated']}")
    print(f"   Pending: {stats['pending']}")
    
    if skip_existing and stats['pre_generated'] > 0:
        print(f"\n⚠️  Will skip {stats['pre_generated']} already generated activities")
    
    input("\nPress Enter to start generation...")
    
    # Generate for each activity
    total_time = 0
    successful = 0
    skipped = 0
    failed = 0
    
    for row_id in range(21):
        print("\n" + "="*80)
        print(f"[{row_id + 1}/21] Processing Row {row_id}")
        print("="*80)
        
        # Check if already generated
        if skip_existing:
            existing_lesson, existing_summary = content_cache.get_generated(row_id)
            if existing_lesson and existing_summary:
                print(f"⏭️  Skipping - already generated")
                skipped += 1
                continue
        
        # Get activity metadata
        try:
            row = data_processor.get_row(row_id)
            title = (data_processor.safe_get(row, "Strategic Action") or 
                    data_processor.safe_get(row, "Activity Name") or 
                    "Activity")
            
            # Get source content from row data
            description = data_processor.safe_get(row, "Short Description") or ""
            notes = data_processor.safe_get(row, "Notes") or ""
            time_impl = data_processor.safe_get(row, "Time to implement") or ""
            materials = data_processor.safe_get(row, "Materials") or data_processor.safe_get(row, "Resources Needed") or ""
            
            print(f"📝 Title: {title}")
            
        except Exception as e:
            print(f"⚠️  Could not get row {row_id}: {e}")
            failed += 1
            continue
        
        # Try to get cached source first
        raw_text, status = content_cache.get(row_id)
        
        # If no cached source, try to fetch from content_retrieval
        if not raw_text or status != 'success':
            print(f"🌐 Fetching source content...")
            try:
                # Use content_retrieval to get source
                source_text = content_retrieval.get_source_content_for_activity(row_id)
                if source_text and len(source_text) > 100:
                    raw_text = source_text
                    print(f"✅ Fetched source: {len(raw_text)} chars")
                else:
                    print(f"⚠️  No web source - using CSV data only")
                    # Build source from CSV fields
                    raw_text = f"""Activity: {title}

Description: {description}

Time to implement: {time_impl}

Materials needed: {materials}

Additional notes: {notes}
"""
            except Exception as e:
                print(f"⚠️  Could not fetch source: {e}")
                # Fallback to CSV data
                raw_text = f"""Activity: {title}

Description: {description}

Time to implement: {time_impl}

Materials needed: {materials}

Additional notes: {notes}
"""
        
        if not raw_text or len(raw_text) < 50:
            print(f"❌ Skipping - insufficient content")
            failed += 1
            continue
        
        print(f"✅ Source content: {len(raw_text)} characters")
        
        start_time = time.time()
        
        try:
            # Generate lesson plan
            print(f"\n[1/2] Generating lesson plan...")
            
            lesson_plan = openai_service.chat(
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert teaching assistant. Convert the source material into a well-structured lesson plan.

Format as markdown with these sections:
# [Title] ⏱️ [Duration]

## Objective
Clear learning objective

## Materials Needed
- Bulleted list of materials

## Directions
CRITICAL: Directions MUST use hierarchical numbering format:

1. Main step title
   a. First detailed sub-step with full explanation
   b. Second detailed sub-step with full explanation
   c. Third detailed sub-step with full explanation

2. Next main step title
   a. Another detailed sub-step
   b. Another detailed sub-step
   c. Another detailed sub-step

Rules:
- Use numbers (1. 2. 3.) for main steps
- Use lowercase letters (a. b. c.) for sub-steps
- Indent sub-steps with exactly 3 spaces: "   a. "
- Minimum 3-5 main steps
- Each main step MUST have at least 2-3 sub-steps
- Be detailed and specific
- Full sentences, no ellipses (...)

## Reflection Questions
2-3 questions for students

## Modification Suggestions
2-3 ways to adapt the activity

CRITICAL: The title in the markdown MUST match exactly the activity title provided by the user.
Keep it clear, actionable, and well-formatted."""
                    },
                    {
                        "role": "user",
                        "content": f"""Source material:

{raw_text[:4000]}

Activity title: {title}

Create a structured lesson plan with detailed hierarchical directions (1. a. b. c. format).
IMPORTANT: Use the exact title "{title}" in the markdown header."""
                    }
                ],
                max_tokens=4000,
                temperature=0.3
            )
            
            # Ensure title is correct in generated markdown
            import re
            # Replace any title in the markdown with the correct one
            lesson_plan = re.sub(r'^#\s+.*?(?=\s+⏱️|$)', f'# {title}', lesson_plan, flags=re.MULTILINE)
            
            print(f"   ✅ Generated lesson plan ({len(lesson_plan)} chars)")
            
            # Process the generated markdown to fix formatting
            print(f"   🔧 Processing markdown formatting...")
            from src.processing.document_processor import document_processor
            
            # Parse the markdown back to dict
            parsed = document_processor.markdown_to_dict(lesson_plan)
            
            # ✅ CRITICAL: Merge CSV metadata into parsed dict
            # This ensures Source Link, Time, and other CSV fields are preserved
            parsed["Link to Resource"] = data_processor.safe_get(row, "Link to Resource") or ""
            parsed["Resources Needed"] = data_processor.safe_get(row, "Resources Needed") or ""
            parsed["Time"] = data_processor.safe_get(row, "Time to implement") or time_impl
            
            # Re-format using the template (this will apply _format_directions_smart)
            lesson_plan = document_processor.format_activity_template(parsed)
            
            print(f"   ✅ Formatted lesson plan ({len(lesson_plan)} chars)")
            
            # Generate summary
            print(f"\n[2/2] Generating summary...")
            
            summary = openai_service.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly teaching assistant. Write a brief, conversational 2-3 sentence introduction to this activity."
                    },
                    {
                        "role": "user",
                        "content": f"""Activity:
{lesson_plan[:800]}

Write a warm, encouraging introduction (2-3 sentences max)."""
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            print(f"   ✅ Generated summary ({len(summary)} chars)")
            
            # Save to cache
            print(f"\n[3/3] Saving to cache...")
            content_cache.set_generated(row_id, lesson_plan, summary)
            
            duration = time.time() - start_time
            total_time += duration
            successful += 1
            
            print(f"\n✅ Complete! ({duration:.1f}s)")
            
            # Small delay between requests
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"✅ Successfully generated: {successful}")
    print(f"⏭️  Skipped (already done): {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    
    if successful > 0:
        print(f"⚡ Average time per activity: {total_time/successful:.1f}s")
    
    # Show cache stats
    stats = content_cache.get_generation_stats()
    cache_size_kb = stats['total_bytes'] / 1024
    
    print(f"\n💾 Cache Statistics:")
    print(f"   Pre-generated activities: {stats['pre_generated']}/{stats['total']}")
    print(f"   Total pre-generated size: {cache_size_kb:.1f} KB")
    
    # Show database file size
    db_path = content_cache.db_path
    if os.path.exists(db_path):
        db_size_kb = os.path.getsize(db_path) / 1024
        print(f"   Total database size: {db_size_kb:.1f} KB")
    
    if stats['pre_generated'] == 21:
        print("\n🎉🎉🎉 ALL ACTIVITIES PRE-GENERATED! 🎉🎉🎉")
        print("Your chatbot will now open activities INSTANTLY!")
    
    return successful, skipped, failed

if __name__ == "__main__":
    pre_generate_all(skip_existing=True)