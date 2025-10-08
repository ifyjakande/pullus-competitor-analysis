#!/usr/bin/env python3
"""
Check if source data worksheets have been modified since last update.
Uses content hash of source worksheets (Abuja_Entry, Kaduna_Entry, Kano_Entry)
instead of Drive API timestamp to detect actual data changes.
"""

import base64
import binascii
import json
import os
import sys
import hashlib
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print("🔐 Loaded credentials from .env file")

def get_credentials():
    """Get Google API credentials from environment."""
    # Try CI environment first, then local .env
    if os.getenv('CI') != 'true':
        load_env_file()

    credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH')
    if not credentials_path:
        raise ValueError("GOOGLE_CREDENTIALS_PATH environment variable not set")

    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ]

        credentials_value = credentials_path.strip()
        if (credentials_value.startswith('"') and credentials_value.endswith('"')) or \
           (credentials_value.startswith("'") and credentials_value.endswith("'")):
            credentials_value = credentials_value[1:-1].strip()

        def _parse_credentials(raw_value):
            try:
                return json.loads(raw_value)
            except json.JSONDecodeError:
                return None

        credentials_info = _parse_credentials(credentials_value)
        credential_source = None

        if credentials_info is None:
            cleaned = ''.join(credentials_value.split())
            if cleaned:
                padding = len(cleaned) % 4
                if padding:
                    cleaned += '=' * (4 - padding)
                for decoder in (base64.b64decode, base64.urlsafe_b64decode):
                    try:
                        decoded_bytes = decoder(cleaned)
                        decoded_str = decoded_bytes.decode('utf-8').strip()
                        credentials_info = _parse_credentials(decoded_str)
                        if credentials_info is not None:
                            credential_source = 'base64'
                            break
                    except (binascii.Error, UnicodeDecodeError):
                        continue

        else:
            credential_source = 'embedded'

        if credentials_info is not None:
            source_label = 'embedded JSON' if credential_source == 'embedded' else 'base64 JSON'
            print(f"🔑 Using {source_label} service account credentials")
            return Credentials.from_service_account_info(credentials_info, scopes=scopes)

        print("🔑 Using service account credentials from file path")
        return Credentials.from_service_account_file(credentials_value, scopes=scopes)

    except FileNotFoundError:
        print(f"❌ Credentials file not found: {credentials_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        sys.exit(1)

def get_source_data_hash(spreadsheet_id, credentials, source_worksheets):
    """Get combined content hash of all source worksheets."""
    try:
        gc = gspread.authorize(credentials)
        spreadsheet = gc.open_by_key(spreadsheet_id)

        combined_content = []

        for worksheet_name in source_worksheets:
            try:
                worksheet = spreadsheet.worksheet(worksheet_name)
                all_values = worksheet.get_all_values()

                # Add worksheet content to combined content
                combined_content.append({
                    'worksheet': worksheet_name,
                    'data': all_values
                })

                rows_with_data = len([row for row in all_values if any(cell.strip() for cell in row)])
                print(f"📊 {worksheet_name}: {rows_with_data} rows with data")

            except gspread.WorksheetNotFound:
                print(f"⚠️ Worksheet '{worksheet_name}' not found, skipping")
                continue

        # Create hash of combined content
        content_str = str(combined_content)
        content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()

        print(f"🔗 Combined content hash: {content_hash}")
        return content_hash

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
        print(f"⚠️ Warning: Could not load last hash: {e}")
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
        print(f"💾 Saved new content hash")
    except Exception as e:
        print(f"❌ Error saving hash: {e}")
        sys.exit(1)

def main():
    """Main function to check for changes in source data."""
    try:
        # Get environment variables
        spreadsheet_id = os.getenv('GOOGLE_SHEET_ID')
        if not spreadsheet_id:
            print("❌ GOOGLE_SHEET_ID environment variable not set")
            sys.exit(1)

        # Source worksheets to monitor
        source_worksheets = ['Abuja_Entry', 'Kaduna_Entry', 'Kano_Entry']

        print(f"🔍 Checking for changes in source worksheets: {', '.join(source_worksheets)}...")

        # Get credentials
        credentials = get_credentials()

        # Get current source data hash
        current_hash = get_source_data_hash(spreadsheet_id, credentials, source_worksheets)

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
            print("⏭️ No changes in source data detected. Skipping update.")
            print("NEEDS_UPDATE=false")
            return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
