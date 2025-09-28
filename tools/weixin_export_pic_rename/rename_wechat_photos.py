import os
import re
import shutil
import sys
from datetime import datetime

def rename_wechat_photos(directory='.', keep_original=True):
    """
    Rename WeChat exported photos from mmexport1643713016351.jpg format 
    to WX_20220201_163221.jpg format
    
    Args:
        directory (str): Directory to process files in. Default is current directory.
        keep_original (bool): Whether to keep original files. Default is True.
    """
    # Patterns to match WeChat export files with timestamp
    patterns = [
        re.compile(r'^mmexport(\d{13})\.(jpg|jpeg|png|gif|bmp|webp)$'),
        re.compile(r'^wx_camera_(\d{13})\.(jpg|jpeg|png|gif|bmp|webp)$')
    ]
    
    # Counter for processed files
    processed_count = 0
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Ensure we only get the file name, not any path components
        file_basename = os.path.basename(filename)
        for pattern in patterns:
            match = pattern.match(file_basename)
            if match:
                timestamp = match.group(1)
                # Always use group(2) for extension as both patterns have 2 groups
                extension = match.group(2)
                
                # Convert timestamp to datetime
                # WeChat uses milliseconds, so divide by 1000
                try:
                    dt = datetime.fromtimestamp(int(timestamp) / 1000)
                    new_filename = f"WX_{dt.strftime('%Y%m%d_%H%M%S')}.{extension}"
                    
                    # Full paths
                    old_path = os.path.join(directory, filename)
                    new_path = os.path.join(directory, new_filename)
                    
                    # Copy or move file
                    if keep_original:
                        shutil.copy2(old_path, new_path)
                        print(f"Copied: {filename} -> {new_filename}")
                    else:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {filename} -> {new_filename}")
                    
                    processed_count += 1
                    break  # Break inner loop once a pattern matches
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    print(f"Processed {processed_count} files.")

if __name__ == "__main__":
    # usage $ python rename_wechat_photos.py "./target/" -false
    # Default values
    directory = '.'
    keep_original = True
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    
    if len(sys.argv) > 2:
        # Handle various forms of false values, including with or without dash
        arg2 = sys.argv[2].lower().strip()
        # Remove leading dash if present
        if arg2.startswith('-'):
            arg2 = arg2[1:]
        keep_original = arg2 not in ['false', '0', 'no', 'n']
    
    # Normalize path for Windows
    directory = os.path.normpath(directory)
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        sys.exit(1)
    
    print(f"Processing directory: {directory}")
    print(f"Keep original files: {keep_original}")
    print("-" * 40)
    
    rename_wechat_photos(directory, keep_original)