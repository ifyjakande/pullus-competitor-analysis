# Smart Google Sheets Workflow Optimization Guide

This guide documents how to implement content hash-based change detection for Google Sheets workflows to reduce unnecessary execution by ~80%.

## 🎯 Problem Statement

Standard workflows that run on a schedule (e.g., every 30 minutes) execute regardless of whether the source data has changed, wasting resources and GitHub Actions minutes.

**Additional Problem**: When workflows write back to the same spreadsheet (dashboard updates), timestamp-based detection creates infinite loops - every dashboard update triggers the next workflow run.

## 💡 Solution Overview

Implement a two-stage workflow with content hash-based change detection:
1. **Stage 1**: Lightweight source data hash comparison (~15 seconds)
2. **Stage 2**: Heavy processing (only when source data actually changes)

**Key Innovation**: Hash only the source worksheet content, ignoring automated dashboard updates to prevent infinite loops.

## 🏗️ Implementation Steps

### 1. Create Change Detection Script (`check_changes.py`)

```python
#!/usr/bin/env python3
"""
Check if source data worksheet has been modified since last update.
Uses content hash of the source worksheet instead of Drive API timestamp.
"""

import json
import os
import sys
import hashlib
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import gspread

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"\'')
                    os.environ[key] = value

def get_credentials():
    """Get Google API credentials from environment."""
    load_env_file()

    service_account_info = os.getenv('GOOGLE_SERVICE_ACCOUNT')
    if not service_account_info:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT environment variable not set")

    try:
        # Check if it's a file path or JSON content
        if service_account_info.startswith('/') or service_account_info.endswith('.json'):
            # It's a file path - for local development
            credentials = Credentials.from_service_account_file(
                service_account_info,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets.readonly',
                    'https://www.googleapis.com/auth/drive.metadata.readonly'
                ]
            )
            print(f"🔑 Using service account file: {service_account_info}")
        else:
            # It's JSON content - for GitHub Actions
            credentials_dict = json.loads(service_account_info)
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets.readonly',
                    'https://www.googleapis.com/auth/drive.metadata.readonly'
                ]
            )
            print("🔑 Using service account from environment variable")

        return credentials

    except json.JSONDecodeError as e:
        print(f"❌ Error parsing service account JSON: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Service account file not found: {service_account_info}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error creating credentials: {e}")
        sys.exit(1)

def get_source_data_hash(spreadsheet_id, credentials, source_worksheet_name="Logistics Data"):
    """Get content hash of the source data worksheet."""
    try:
        # Use gspread for easier worksheet access
        gc = gspread.authorize(credentials)
        spreadsheet = gc.open_by_key(spreadsheet_id)

        # Get the source worksheet
        source_worksheet = spreadsheet.worksheet(source_worksheet_name)

        # Get all values from the source worksheet
        all_values = source_worksheet.get_all_values()

        # Create hash of the content
        content_str = str(all_values)
        content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()

        print(f"📊 Source worksheet: {source_worksheet_name}")
        print(f"📝 Rows with data: {len([row for row in all_values if any(cell.strip() for cell in row)])}")
        print(f"🔗 Content hash: {content_hash}")

        return content_hash

    except gspread.WorksheetNotFound:
        print(f"❌ Source worksheet '{source_worksheet_name}' not found")
        print("Available worksheets:")
        try:
            gc = gspread.authorize(credentials)
            spreadsheet = gc.open_by_key(spreadsheet_id)
            for ws in spreadsheet.worksheets():
                print(f"  - {ws.title}")
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error getting source data hash: {e}")
        sys.exit(1)

def load_last_hash():
    """Load the last processed content hash from file."""
    hash_file = 'last_source_hash.json'
    try:
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                data = json.load(f)
                return data.get('content_hash')
        return None
    except Exception as e:
        print(f"⚠️  Warning: Could not load last hash: {e}")
        return None

def save_hash(content_hash):
    """Save the current content hash to file."""
    hash_file = 'last_source_hash.json'
    try:
        data = {
            'content_hash': content_hash,
            'updated_at': hashlib.md5(str(content_hash).encode()).hexdigest()
        }
        with open(hash_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved new content hash: {content_hash}")
    except Exception as e:
        print(f"❌ Error saving hash: {e}")
        sys.exit(1)

def main():
    """Main function to check for changes in source data."""
    try:
        # Load .env file first
        load_env_file()

        # Get environment variables
        spreadsheet_id = os.getenv('SPREADSHEET_ID')
        if not spreadsheet_id:
            print("❌ SPREADSHEET_ID environment variable not set")
            sys.exit(1)

        # Allow customization of source worksheet name
        source_worksheet_name = os.getenv('SOURCE_WORKSHEET_NAME', 'Logistics Data')

        print(f"🔍 Checking for changes in source worksheet '{source_worksheet_name}'...")

        # Get credentials
        credentials = get_credentials()

        # Get current source data hash
        current_hash = get_source_data_hash(spreadsheet_id, credentials, source_worksheet_name)

        # Load last processed hash
        last_hash = load_last_hash()
        print(f"📅 Last processed hash: {last_hash or 'Never'}")

        # Compare hashes
        if current_hash != last_hash:
            print("✅ Source data changes detected! Update needed.")
            save_hash(current_hash)
            print("NEEDS_UPDATE=true")
            return True
        else:
            print("⏭️  No changes in source data detected. Skipping update.")
            print("NEEDS_UPDATE=false")
            return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 2. Update Main Script for Dual Credential Support

Add this to your main processing script:

```python
if __name__ == "__main__":
    # Try GOOGLE_SERVICE_ACCOUNT (JSON content) first, then fall back to file path
    service_account_json = os.getenv('GOOGLE_SERVICE_ACCOUNT')
    credentials_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE')
    spreadsheet_id = os.getenv('SPREADSHEET_ID')

    if not service_account_json and not credentials_path:
        raise ValueError("Please set either GOOGLE_SERVICE_ACCOUNT or GOOGLE_SERVICE_ACCOUNT_FILE environment variable")

    if not spreadsheet_id:
        raise ValueError("Please set SPREADSHEET_ID environment variable")

    # Handle JSON content vs file path
    if service_account_json:
        # Write JSON content to temporary file for gspread
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(json.loads(service_account_json), temp_file, indent=2)
            temp_credentials_path = temp_file.name

        try:
            updater = YourUpdaterClass(temp_credentials_path, spreadsheet_id)
            success = updater.run_update()
        finally:
            os.unlink(temp_credentials_path)
    else:
        updater = YourUpdaterClass(credentials_path, spreadsheet_id)
        success = updater.run_update()
```

### 3. Configure GitHub Actions Workflow

```yaml
name: Update Dashboard

on:
  schedule:
    - cron: '*/5 * * * *'  # Every 5 minutes
  workflow_dispatch:

permissions:
  contents: write

jobs:
  check-changes:
    runs-on: ubuntu-latest
    outputs:
      needs_update: ${{ steps.check.outputs.needs_update }}

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install google-auth google-api-python-client

    - name: Check for changes
      id: check
      env:
        GOOGLE_SERVICE_ACCOUNT: ${{ secrets.GOOGLE_SERVICE_ACCOUNT }}
        SPREADSHEET_ID: ${{ secrets.SPREADSHEET_ID }}
        CI: "true"
      run: |
        python check_changes.py > check_output.txt 2>&1
        if grep -q "NEEDS_UPDATE=true" check_output.txt; then
          echo "needs_update=true" >> $GITHUB_OUTPUT
        else
          echo "needs_update=false" >> $GITHUB_OUTPUT
        fi
        cat check_output.txt

    - name: Commit timestamp file
      if: steps.check.outputs.needs_update == 'true'
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add -f last_source_hash.json
        git diff --staged --quiet || git commit -m "Update last processed timestamp [skip ci]"
        git push

  update-dashboard:
    needs: check-changes
    if: always() && needs.check-changes.result == 'success' && needs.check-changes.outputs.needs_update == 'true'
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install google-auth google-api-python-client gspread

    - name: Update dashboard
      env:
        GOOGLE_SERVICE_ACCOUNT: ${{ secrets.GOOGLE_SERVICE_ACCOUNT }}
        SPREADSHEET_ID: ${{ secrets.SPREADSHEET_ID }}
      run: |
        python your_main_script.py
```

### 4. Update .gitignore

```gitignore
# Service account credentials - never commit these!
*.json
your-service-account-*.json

# Exception: Allow hash tracking files
!last_source_hash.json

# Python cache
__pycache__/
*.py[cod]
*$py.class

# Environment files
.env
.env.local
.env.*.local

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Logs
*.log
```

## 🔍 How Content Hashing Works

### The Magic of MD5 Hashing for Change Detection

**Content Hashing Approach:**
1. **Read entire source worksheet**: `get_all_values()` returns 2D array of all cells
2. **Convert to string**: `str(all_values)` creates string representation
3. **Generate hash**: `hashlib.md5()` creates 32-character fingerprint
4. **Compare fingerprints**: Different content = different hash = trigger update

**Example:**
```python
# 1,000 rows of logistics data
source_data = [['Date', 'From', 'To'], ['2024-01-01', 'Lagos', 'Abuja'], ...]
content_string = str(source_data)  # Convert to string
hash_value = hashlib.md5(content_string.encode('utf-8')).hexdigest()
# Result: "7f160e04a3b921ce990a8a9ca8b49cad" (always 32 chars)
```

**Why This Works Brilliantly:**
- **ANY change** to ANY cell = different hash (avalanche effect)
- **Same content** = same hash = no unnecessary updates
- **Fixed size**: 1 row or 1 million rows = same 32-character hash
- **Fast comparison**: String comparison vs complex timestamp logic
- **Infinite scalability**: MD5 can handle unlimited data size

**Prevents Infinite Loops:**
- Only hashes the source worksheet (e.g., "Logistics Data")
- Dashboard updates don't affect the source hash
- Workflow only triggers on actual data changes

## 🔧 Required GitHub Secrets

Set these in your repository settings → Secrets and variables → Actions:

1. **GOOGLE_SERVICE_ACCOUNT**: Full service account JSON content
2. **SPREADSHEET_ID**: Your Google Sheet ID from the URL
3. **SOURCE_WORKSHEET_NAME** (optional): Name of source worksheet (defaults to "Logistics Data")

## 🚨 Critical Implementation Notes

### Conditional Execution Fix
The most critical fix for GitHub Actions conditional execution:

```yaml
if: always() && needs.check-changes.result == 'success' && needs.check-changes.outputs.needs_update == 'true'
```

Without `always()`, the job won't run even when conditions are met.

### Gitignore Exception
Essential for hash tracking:

```gitignore
# Block all JSON files but allow the hash file
*.json
!last_source_hash.json
```

### Forced Git Add
Use `git add -f` to add the hash file despite gitignore:

```bash
git add -f last_source_hash.json
```

### Google API Scopes
Required for both scripts:

```python
scopes=[
    'https://www.googleapis.com/auth/spreadsheets.readonly',  # For reading sheet data
    'https://www.googleapis.com/auth/drive.metadata.readonly'  # For checking modification time
]
```

## 📊 Expected Results

- **No Changes**: Workflow completes in ~15 seconds (check-changes job only)
- **With Changes**: Full workflow runs (check-changes + update-dashboard)
- **Resource Savings**: ~80% reduction in unnecessary workflow executions
- **Cost Savings**: Significant reduction in GitHub Actions minutes usage
- **Infinite Loop Prevention**: Dashboard updates no longer trigger new workflow runs
- **Precision**: Only actual source data changes trigger updates

## 🔄 Testing the Implementation

1. **Initial Setup**: Run workflow manually to create initial content hash
2. **No Changes**: Run again - should skip update stage with "No changes in source data detected"
3. **With Changes**: Modify source worksheet, run workflow - should execute both stages
4. **Dashboard Updates**: Verify that dashboard updates don't trigger new workflow runs
5. **Verify Logs**: Check GitHub Actions logs for hash comparison and conditional execution

## 🎯 Key Success Factors

1. **Proper Conditional Logic**: Use `always()` function correctly
2. **Dual Credential Support**: Handle both local dev and CI/CD environments
3. **Hash Persistence**: Ensure `last_source_hash.json` can be committed
4. **Source Worksheet Targeting**: Hash only the source data, not generated sheets
5. **Error Handling**: Robust error handling in change detection
6. **API Permissions**: Correct Google API scopes for both Drive and Sheets

## 🔗 Adaptable to Other Projects

This pattern works for any Google Sheets-based automation:

1. **Replace** `your_main_script.py` with your processing script
2. **Update** dependencies in workflow YAML
3. **Modify** credential handling if using different libraries
4. **Adjust** schedule frequency as needed
5. **Customize** source worksheet name via `SOURCE_WORKSHEET_NAME` environment variable
6. **Adapt** hash file location if required

## 📈 Performance Impact

- **Before**: 288 workflow runs per day (every 5 min) = ~576 minutes daily
- **After**: ~288 lightweight hash checks + ~10 actual updates = ~92 minutes daily
- **Savings**: ~484 minutes daily, ~14,520 minutes monthly
- **Bonus**: Eliminates infinite loops from dashboard updates

This implementation provides massive resource savings while maintaining the same functionality and responsiveness to changes.