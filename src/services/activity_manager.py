"""
Activity management - Add, delete, modify activities in the database
"""
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from data.processor import data_processor

class ActivityManager:
    """Manages activities in the database"""
    
    def __init__(self):
        self.data_file = Path(config.data_path)
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def delete_activity(self, row_index: int) -> bool:
        """Delete an activity by row index"""
        try:
            # Create backup before deletion
            self._create_backup()
            
            # Load current data
            df = pd.read_csv(self.data_file, dtype=str)
            
            if row_index >= len(df):
                return False
            
            # Remove the row
            df = df.drop(df.index[row_index])
            
            # Save back to file
            df.to_csv(self.data_file, index=False)
            
            # Reload the data processor
            data_processor.initialize(config.data_path)
            
            return True
            
        except Exception as e:
            print(f"Error deleting activity: {e}")
            return False
    
    def add_activity(self, activity_data: Dict[str, str]) -> bool:
        """Add a new activity to the database"""
        try:
            # Create backup before addition
            self._create_backup()
            
            # Load current data
            df = pd.read_csv(self.data_file, dtype=str)
            
            # Create new row
            new_row = {}
            for col in df.columns:
                new_row[col] = activity_data.get(col, "")
            
            # Add the new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save back to file
            df.to_csv(self.data_file, index=False)
            
            # Reload the data processor
            data_processor.initialize(config.data_path)
            
            return True
            
        except Exception as e:
            print(f"Error adding activity: {e}")
            return False
    
    def modify_activity(self, row_index: int, modifications: Dict[str, str]) -> bool:
        """Modify an existing activity"""
        try:
            # Create backup before modification
            self._create_backup()
            
            # Load current data
            df = pd.read_csv(self.data_file, dtype=str)
            
            if row_index >= len(df):
                return False
            
            # Apply modifications
            for key, value in modifications.items():
                if key in df.columns:
                    df.at[row_index, key] = value
            
            # Save back to file
            df.to_csv(self.data_file, index=False)
            
            # Reload the data processor
            data_processor.initialize(config.data_path)
            
            return True
            
        except Exception as e:
            print(f"Error modifying activity: {e}")
            return False
    
    def _create_backup(self):
        """Create a backup of the current data file"""
        try:
            if self.data_file.exists():
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = self.backup_dir / f"activities_backup_{timestamp}.csv"
                
                # Copy current file to backup
                import shutil
                shutil.copy2(self.data_file, backup_file)
                
                # Keep only last 10 backups
                backups = list(self.backup_dir.glob("activities_backup_*.csv"))
                if len(backups) > 10:
                    backups.sort()
                    for old_backup in backups[:-10]:
                        old_backup.unlink()
                        
        except Exception as e:
            print(f"Error creating backup: {e}")
    
    def get_activity_by_index(self, row_index: int) -> Optional[Dict[str, str]]:
        """Get activity data by row index"""
        try:
            df = pd.read_csv(self.data_file, dtype=str)
            if row_index >= len(df):
                return None
            
            row = df.iloc[row_index]
            return row.to_dict()
            
        except Exception as e:
            print(f"Error getting activity: {e}")
            return None
    
    def list_activities(self) -> List[Dict[str, Any]]:
        """List all activities with their indices"""
        try:
            df = pd.read_csv(self.data_file, dtype=str)
            activities = []
            
            for i, row in df.iterrows():
                activities.append({
                    "index": i,
                    "title": row.get("Strategic Action", "Untitled Activity"),
                    "time": row.get("Time", ""),
                    "objective": row.get("Objective", ""),
                    "data": row.to_dict()
                })
            
            return activities
            
        except Exception as e:
            print(f"Error listing activities: {e}")
            return []

# Global activity manager instance
activity_manager = ActivityManager()
