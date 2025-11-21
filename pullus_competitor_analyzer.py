#!/usr/bin/env python3
"""
Pullus Competitor Analysis Engine
Provides clean market intelligence for management decision-making
"""

import base64
import binascii
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time
import random
import os
import yaml
import json
import argparse
from functools import wraps
from dotenv import load_dotenv
from email_alerts import EmailAlerts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"⚠️ Config file not found at {config_path}, using defaults")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to load config: {sanitize_error(e)}")
        return None

def sanitize_error(error):
    """Remove sensitive information from error messages"""
    error_str = str(error)

    # Remove file paths
    import re
    error_str = re.sub(r'/[^\s]+\.json', '[CREDENTIALS_PATH]', error_str)
    error_str = re.sub(r'/Users/[^\s]+', '[PATH]', error_str)
    error_str = re.sub(r'C:\\Users\\[^\s]+', '[PATH]', error_str)

    # Remove potential API keys or tokens (alphanumeric strings > 20 chars)
    error_str = re.sub(r'\b[A-Za-z0-9]{20,}\b', '[REDACTED]', error_str)

    # Remove sheet IDs
    error_str = re.sub(r'\b[A-Za-z0-9_-]{30,}\b', '[SHEET_ID]', error_str)

    return error_str

def rate_limit_handler(max_retries=5, base_delay=1):
    """
    Decorator to handle Google Sheets API rate limits with exponential backoff
    
    Google Sheets API Limits:
    - Write requests: 60 per minute per user per project
    - Write requests: 300 per minute per project
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if "429" in str(e) or "Quota exceeded" in str(e):
                        if attempt < max_retries - 1:
                            # Exponential backoff with jitter
                            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), 64)
                            logger.warning(f"⚠️ Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                        else:
                            logger.error(f"❌ Max retries exceeded for rate limit")
                            raise e
                    else:
                        # Non-rate-limit error, don't retry
                        raise e
            
            return None
        return wrapper
    return decorator

class RateLimitedGoogleSheets:
    """Rate-limited wrapper for Google Sheets operations"""
    
    def __init__(self, spreadsheet):
        self.spreadsheet = spreadsheet
        self.request_count = 0
        self.minute_start = time.time()
        self.max_requests_per_minute = 50  # Conservative limit (below 60)
    
    def _check_rate_limit(self):
        """Check if we're approaching rate limits and pause if needed"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.minute_start >= 60:
            self.request_count = 0
            self.minute_start = current_time
        
        # If approaching limit, wait until next minute
        if self.request_count >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.minute_start) + 1  # +1 for safety
            if wait_time > 0:
                logger.info(f"🔄 Rate limit protection: waiting {wait_time:.1f}s for quota reset")
                time.sleep(wait_time)
                self.request_count = 0
                self.minute_start = time.time()
    
    @rate_limit_handler()
    def batch_update(self, requests):
        """Rate-limited batch update"""
        self._check_rate_limit()
        self.request_count += 1
        return self.spreadsheet.batch_update(requests)
    
    @rate_limit_handler()
    def worksheet(self, name):
        """Rate-limited worksheet access"""
        self._check_rate_limit()
        self.request_count += 1
        ws = self.spreadsheet.worksheet(name)
        return RateLimitedWorksheet(ws, self)
    
    @rate_limit_handler()
    def add_worksheet(self, **kwargs):
        """Rate-limited worksheet creation"""
        self._check_rate_limit()
        self.request_count += 1
        ws = self.spreadsheet.add_worksheet(**kwargs)
        return RateLimitedWorksheet(ws, self)

class RateLimitedWorksheet:
    """Rate-limited wrapper for worksheet operations"""
    
    def __init__(self, worksheet, rate_limiter):
        self.worksheet = worksheet
        self.rate_limiter = rate_limiter
        self.id = worksheet.id
    
    @rate_limit_handler()
    def update(self, **kwargs):
        """Rate-limited update"""
        self.rate_limiter._check_rate_limit()
        self.rate_limiter.request_count += 1
        return self.worksheet.update(**kwargs)
    
    @rate_limit_handler()
    def clear(self):
        """Rate-limited clear"""
        self.rate_limiter._check_rate_limit()
        self.rate_limiter.request_count += 1
        return self.worksheet.clear()
    
    @rate_limit_handler()
    def get_all_records(self):
        """Rate-limited get all records"""
        self.rate_limiter._check_rate_limit()
        self.rate_limiter.request_count += 1
        return self.worksheet.get_all_records()

class PullusCompetitorAnalyzer:
    def __init__(self, credentials_path: str, sheet_id: str, config: dict = None):
        self.credentials_path = credentials_path
        self.sheet_id = sheet_id
        self.client = None
        self.spreadsheet = None
        self.rate_limited_sheets = None
        self.data = {}
        self.config = config or {}

        # Product categories for analysis (from config or defaults)
        self.product_columns = self.config.get('products', [
            'Whole Chicken', 'Mini Chicken', 'Gizzard', 'Laps',
            'Fillet - Breast', 'Wings', 'Breast', 'Bone', 'Fillet - Thigh',
            'Liver', 'Neck', 'Head/Leg', 'Carcas', 'Cut-4', 'Eggs'
        ])

        # Analysis settings from config
        self.competitive_threshold = self.config.get('analysis', {}).get('competitive_threshold', 200)
        self.max_data_age_days = self.config.get('analysis', {}).get('max_data_age_days', 7)
        self.urgent_review_threshold = self.config.get('analysis', {}).get('urgent_review_threshold', 300)

        # Analysis timestamp
        self.analysis_date = datetime.now().strftime("%d-%b-%Y %H:%M")

        # Color palette from config or defaults
        default_colors = {
            'header': {'red': 0.05, 'green': 0.43, 'blue': 0.99},
            'winning': {'red': 0.83, 'green': 0.96, 'blue': 0.88},
            'competitive': {'red': 0.87, 'green': 0.95, 'blue': 1.0},
            'losing': {'red': 1.0, 'green': 0.93, 'blue': 0.93},
            'no_data': {'red': 0.97, 'green': 0.98, 'blue': 0.98},
            'metric': {'red': 0.95, 'green': 0.97, 'blue': 1.0},
            'data_warning': {'red': 1.0, 'green': 0.96, 'blue': 0.87},
            'urgent': {'red': 0.93, 'green': 0.89, 'blue': 0.89},
            'heatmap_winning': {'red': 0.7, 'green': 0.9, 'blue': 0.75},
            'heatmap_losing': {'red': 0.95, 'green': 0.85, 'blue': 0.85}
        }
        self.colors = self.config.get('colors', default_colors)

    def load_processed_dates(self):
        """Load dates that have already been alerted about"""
        try:
            with open('last_processed.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {'Abuja': [], 'Kaduna': [], 'Kano': []}

    def save_processed_dates(self, processed_dates):
        """Save processed dates to file"""
        try:
            with open('last_processed.json', 'w') as f:
                json.dump(processed_dates, f, indent=2)
            logger.info("✅ Saved processed dates")
        except Exception as e:
            logger.error(f"❌ Error saving processed dates: {sanitize_error(e)}")

    def detect_fresh_dates(self, daily_analysis):
        """
        Detect dates that haven't been alerted about yet.
        Returns dict of {location: [new_dates]} or None if no new dates.
        """
        processed_dates = self.load_processed_dates()
        new_dates = {}
        pruned_processed = {}
        processed_changed = False

        for location, location_data in daily_analysis.items():
            sheet_dates = set(location_data.keys())
            processed_set = set(processed_dates.get(location, []))
            valid_processed = processed_set & sheet_dates

            if valid_processed != processed_set:
                processed_changed = True

            pruned_processed[location] = sorted(list(valid_processed))[-30:]

            unprocessed = sheet_dates - valid_processed

            if unprocessed:
                new_dates[location] = sorted(list(unprocessed), reverse=True)
                logger.info(f"📅 {location}: Found {len(unprocessed)} new dates")

        # Carry over any locations missing from current analysis
        for location, dates in processed_dates.items():
            if location not in pruned_processed:
                pruned_processed[location] = sorted(list(set(dates)))[-30:]
                if dates:
                    processed_changed = True

        if processed_changed:
            self.save_processed_dates(pruned_processed)

        return new_dates if new_dates else None

    def mark_dates_as_processed(self, new_dates):
        """Mark new dates as processed"""
        processed_dates = self.load_processed_dates()

        for location, dates in new_dates.items():
            if location not in processed_dates:
                processed_dates[location] = []
            processed_dates[location].extend(dates)
            # Keep only last 30 days to prevent file bloat
            processed_dates[location] = sorted(list(set(processed_dates[location])))[-30:]

        self.save_processed_dates(processed_dates)

    def connect_to_sheets(self):
        """Connect to Google Sheets"""
        try:
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]

            credentials_source = (self.credentials_path or "").strip()
            if (credentials_source.startswith('"') and credentials_source.endswith('"')) or \
               (credentials_source.startswith("'") and credentials_source.endswith("'")):
                credentials_source = credentials_source[1:-1].strip()

            def _parse_credentials(raw_value):
                try:
                    return json.loads(raw_value)
                except json.JSONDecodeError:
                    return None

            credentials_info = _parse_credentials(credentials_source)
            credential_source = None

            if credentials_info is None:
                cleaned = ''.join(credentials_source.split())
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
                label = 'embedded JSON' if credential_source == 'embedded' else 'base64 JSON'
                logger.info("🔑 Loaded Google credentials from %s", label)
                credentials = Credentials.from_service_account_info(credentials_info, scopes=scopes)
            elif os.path.isfile(credentials_source):
                logger.info("🔑 Loaded Google credentials from file path")
                credentials = Credentials.from_service_account_file(credentials_source, scopes=scopes)
            else:
                raise ValueError(
                    "GOOGLE_CREDENTIALS_PATH must contain either service account JSON "
                    "(raw or base64-encoded) or a valid filesystem path to the JSON file"
                )
            self.client = gspread.authorize(credentials)
            self.spreadsheet = self.client.open_by_key(self.sheet_id)
            self.rate_limited_sheets = RateLimitedGoogleSheets(self.spreadsheet)
            logger.info("✅ Connected to Google Sheets with rate limiting")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Google Sheets: {sanitize_error(e)}")
            return False
    
    def load_data(self):
        """Load competitor data from all locations"""
        if not self.connect_to_sheets():
            return False

        try:
            locations = self.config.get('locations', ['Abuja', 'Kaduna', 'Kano'])
            
            for location in locations:
                logger.info(f"Loading {location} data...")
                ws = self.rate_limited_sheets.worksheet(f"{location}_Entry")
                data = ws.get_all_records()
                
                if data:
                    df = pd.DataFrame(data)
                    # Keep product columns as strings to preserve formats like "3600/3700"
                    # Numeric conversion will be handled during price comparison logic
                    
                    df['Location'] = location
                    self.data[location] = df
                    logger.info(f"✅ Loaded {len(df)} entries for {location}")
                else:
                    logger.warning(f"⚠️ No data found for {location}")
            
            if self.data:
                # Combine all data
                self.combined_data = pd.concat(list(self.data.values()), ignore_index=True)
                logger.info(f"📊 Total entries loaded: {len(self.combined_data)}")
                return True
            else:
                logger.error("❌ No data loaded from any location")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to load data: {sanitize_error(e)}")
            return False
    
    def auto_resize_columns(self, worksheet):
        """Auto-resize all columns for better readability"""
        try:
            requests = [{
                'autoResizeDimensions': {
                    'dimensions': {
                        'sheetId': worksheet.id,
                        'dimension': 'COLUMNS',
                        'startIndex': 0,
                        'endIndex': 20  # Resize up to column T
                    }
                }
            }]

            self.rate_limited_sheets.batch_update({'requests': requests})
        except Exception as e:
            logger.warning(f"⚠️ Auto-resize failed: {sanitize_error(e)}")

    def clear_sheet_formatting(self, worksheet, max_rows=1000, max_cols=20):
        """Clear all formatting from a worksheet (background colors, text formatting, borders, etc.)"""
        try:
            requests = [{
                'repeatCell': {
                    'range': {
                        'sheetId': worksheet.id,
                        'startRowIndex': 0,
                        'endRowIndex': max_rows,
                        'startColumnIndex': 0,
                        'endColumnIndex': max_cols
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'backgroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0},
                            'textFormat': {
                                'foregroundColor': {'red': 0.0, 'green': 0.0, 'blue': 0.0},
                                'fontSize': 10,
                                'bold': False
                            },
                            'horizontalAlignment': 'LEFT',
                            'verticalAlignment': 'BOTTOM'
                        }
                    },
                    'fields': 'userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)'
                }
            }]

            self.rate_limited_sheets.batch_update({'requests': requests})
            logger.info(f"✅ Cleared formatting for worksheet")
        except Exception as e:
            logger.warning(f"⚠️ Clear formatting failed: {sanitize_error(e)}")

    def set_heatmap_column_widths(self, worksheet):
        """Set optimal column widths specifically for heatmap dashboard"""
        try:
            # Define optimal widths for heatmap columns
            column_widths = [
                150,  # A: Product names (wider for "Fillet - Breast", etc.)
                80,   # B: Abuja emoji indicators
                100,  # C: Abuja % values
                80,   # D: Kaduna emoji indicators
                100,  # E: Kaduna % values
                80,   # F: Kano emoji indicators
                100,  # G: Kano % values
                120,  # H: Overall Win Rate
                140,  # I: Trend ("📉 Weak", "📈 Strong")
                120   # J: Priority ("🔴 Review", "🟢 Maintain")
            ]

            requests = []
            for i, width in enumerate(column_widths):
                requests.append({
                    'updateDimensionProperties': {
                        'range': {
                            'sheetId': worksheet.id,
                            'dimension': 'COLUMNS',
                            'startIndex': i,
                            'endIndex': i + 1
                        },
                        'properties': {
                            'pixelSize': width
                        },
                        'fields': 'pixelSize'
                    }
                })

            self.rate_limited_sheets.batch_update({'requests': requests})

        except Exception as e:
            logger.warning(f"⚠️ Heatmap column width setting failed: {sanitize_error(e)}")
    
    def get_daily_competitive_analysis(self):
        """Analyze same-day competitive position with simple 3-color system"""
        try:
            daily_analysis = {}
            
            for location in self.data.keys():
                df = self.data[location]
                df['Date'] = df['Date'].astype(str)
                location_results = {}
                
                # Get unique dates in this location
                unique_dates = sorted(df['Date'].unique(), reverse=True)  # Most recent first
                
                for date in unique_dates:
                    date_data = df[df['Date'] == date]
                    date_results = {}
                    
                    for product in self.product_columns:
                        if product in df.columns:
                            # Get Pullus price for this date
                            pullus_data = date_data[
                                (date_data['Brand'].str.strip().str.lower() == 'pullus') &
                                (date_data[product].notna()) & 
                                (date_data[product] != '') & 
                                (date_data[product] != 0)
                            ]
                            
                            if not pullus_data.empty:
                                # Get Pullus price
                                pullus_price_str = str(pullus_data[product].iloc[0]).strip()
                                try:
                                    if '/' in pullus_price_str:
                                        prices = pullus_price_str.split('/')
                                        pullus_price = sum(float(p) for p in prices) / len(prices)  # Use average
                                    else:
                                        pullus_price = float(pullus_price_str)
                                except (ValueError, TypeError):
                                    continue
                                
                                # Get competitor prices for SAME DATE ONLY
                                competitor_data = date_data[
                                    (date_data['Brand'].str.strip().str.lower() != 'pullus') &
                                    (date_data[product].notna()) & 
                                    (date_data[product] != '') & 
                                    (date_data[product] != 0)
                                ]
                                
                                if not competitor_data.empty:
                                    # Get all competitor prices for same date
                                    competitor_prices = []
                                    competitor_details = []
                                    
                                    for _, row in competitor_data.iterrows():
                                        price_str = str(row[product]).strip()
                                        try:
                                            if '/' in price_str:
                                                prices = price_str.split('/')
                                                price = sum(float(p) for p in prices) / len(prices)  # Use average
                                            else:
                                                price = float(price_str)
                                            competitor_prices.append(price)
                                            competitor_details.append({
                                                'brand': row['Brand'],
                                                'price': price
                                            })
                                        except (ValueError, TypeError):
                                            continue
                                    
                                    if competitor_prices:
                                        min_competitor_price = min(competitor_prices)

                                        # Get ALL competitors tied at minimum price
                                        cheapest_competitors = [
                                            comp['brand'] for comp in competitor_details
                                            if comp['price'] == min_competitor_price
                                        ]

                                        # Format competitor display based on count
                                        if len(cheapest_competitors) == 1:
                                            cheapest_competitor = cheapest_competitors[0]
                                        elif len(cheapest_competitors) <= 3:
                                            cheapest_competitor = ", ".join(cheapest_competitors)
                                        else:
                                            # 4+ competitors: show count with sample
                                            sample = ", ".join(cheapest_competitors[:2])
                                            others_count = len(cheapest_competitors) - 2
                                            cheapest_competitor = f"{len(cheapest_competitors)} tied ({sample} + {others_count} others)"
                                        
                                        # Simple 3-color classification
                                        price_difference = pullus_price - min_competitor_price

                                        if pullus_price <= min_competitor_price:
                                            status = "🟢 BEST PRICE"
                                            status_code = "winning"
                                            message = f"Cheapest at ₦{pullus_price:,.0f}"
                                        elif price_difference <= self.competitive_threshold:
                                            status = "🟡 COMPETITIVE"
                                            status_code = "competitive"
                                            message = f"₦{price_difference:,.0f} above cheapest"
                                        else:
                                            status = "🔴 ABOVE MARKET"
                                            status_code = "losing"
                                            message = f"₦{price_difference:,.0f} above cheapest"
                                        
                                        date_results[product] = {
                                            'pullus_price': pullus_price,
                                            'cheapest_competitor': cheapest_competitor,
                                            'cheapest_price': min_competitor_price,
                                            'price_difference': price_difference,
                                            'status': status,
                                            'status_code': status_code,
                                            'message': message,
                                            'all_competitors': len(competitor_prices) + 1,  # Include Pullus
                                            'competitor_details': competitor_details
                                        }
                                else:
                                    # No competitors on same date
                                    date_results[product] = {
                                        'pullus_price': pullus_price,
                                        'cheapest_competitor': 'No competitors',
                                        'cheapest_price': 0,
                                        'price_difference': 0,
                                        'status': '⚪ NO DATA',
                                        'status_code': 'no_data',
                                        'message': 'No competitor data for this date',
                                        'all_competitors': 1,
                                        'competitor_details': []
                                    }
                    
                    if date_results:
                        location_results[date] = date_results
                
                daily_analysis[location] = location_results
            
            logger.info("✅ Daily competitive analysis completed with same-day comparisons")
            return daily_analysis
            
        except Exception as e:
            logger.error(f"❌ Daily competitive analysis failed: {sanitize_error(e)}")
            return {}
    
    def get_historical_trends(self, daily_analysis):
        """Generate historical win/loss patterns for heatmap"""
        try:
            trends = {}
            
            for location in daily_analysis.keys():
                location_trends = {}
                location_data = daily_analysis[location]
                
                # Get all products across all dates for this location
                all_products = set()
                for date_data in location_data.values():
                    all_products.update(date_data.keys())
                
                for product in all_products:
                    product_history = []
                    win_count = 0
                    total_checks = 0
                    
                    # Get chronological data for this product
                    for date in sorted(location_data.keys(), reverse=True):  # Most recent first
                        if product in location_data[date]:
                            data = location_data[date][product]
                            product_history.append({
                                'date': date,
                                'status': data['status_code'],
                                'status_emoji': data['status'].split()[0],  # Get just the emoji
                                'pullus_price': data['pullus_price'],
                                'price_difference': data['price_difference']
                            })
                            
                            if data['status_code'] == 'winning':
                                win_count += 1
                            
                            if data['status_code'] != 'no_data':
                                total_checks += 1
                    
                    if product_history:
                        win_rate = (win_count / total_checks * 100) if total_checks > 0 else 0
                        
                        # Determine overall trend
                        if len(product_history) >= 2:
                            recent_statuses = [h['status'] for h in product_history[:2]]  # Last 2 checks
                            if recent_statuses.count('winning') == 2:
                                trend = '📈 Improving'
                            elif recent_statuses.count('losing') == 2:
                                trend = '📉 Declining'
                            else:
                                trend = '➡️ Mixed'
                        else:
                            trend = '➡️ Stable'
                        
                        location_trends[product] = {
                            'win_rate': win_rate,
                            'total_checks': total_checks,
                            'win_count': win_count,
                            'trend': trend,
                            'recent_status': product_history[0]['status'] if product_history else 'no_data',
                            'recent_emoji': product_history[0]['status_emoji'] if product_history else '⚪',
                            'history': product_history[:5]  # Last 5 checks
                        }
                
                trends[location] = location_trends
            
            logger.info("✅ Historical trends analysis completed")
            return trends
            
        except Exception as e:
            logger.error(f"❌ Historical trends analysis failed: {sanitize_error(e)}")
            return {}

    def create_management_sheets(self, daily_analysis, trends, confidence_scores=None, stale_data=None):
        """Create simplified management insight sheets"""
        try:
            logger.info("📊 Creating simplified management insight sheets...")

            # Sheet 1: Daily Competitive Position (by location)
            for location in daily_analysis.keys():
                self.create_daily_position_sheet(location, daily_analysis[location], confidence_scores)

            # Sheet 2: Historical Heatmap Dashboard
            self.create_heatmap_dashboard(trends)

            # Sheet 3: Executive Scorecard
            self.create_executive_scorecard(daily_analysis, trends, stale_data)

            # Sheet 4: Simple Metrics Explainer
            self.create_simple_metrics_explainer()

            logger.info("✅ Simplified management sheets created successfully")
            
        except Exception as e:
            logger.error(f"❌ Management sheet creation failed: {sanitize_error(e)}")
    
    def create_daily_position_sheet(self, location, location_data, confidence_data=None):
        """Create simple daily position sheet for each location"""
        try:
            # Create or get worksheet
            sheet_name = f"{location}_Daily"
            try:
                ws = self.rate_limited_sheets.worksheet(sheet_name)
                ws.clear()
                self.clear_sheet_formatting(ws)
            except:
                ws = self.rate_limited_sheets.add_worksheet(title=sheet_name, rows=100, cols=7)

            # Enhanced headers with confidence
            headers = ["Date", "Product", "Pullus Price", "Cheapest Competitor", "Their Price", "Status", "Message", "Confidence"]

            # Prepare data rows (most recent dates first)
            data_rows = []
            color_mapping = []  # Track which rows need which colors

            # Sort dates chronologically (most recent first)
            sorted_dates = sorted(location_data.keys(),
                                key=lambda d: pd.to_datetime(d, errors='coerce'),
                                reverse=True)

            for date in sorted_dates:
                date_data = location_data[date]
                for product, product_data in date_data.items():
                    # Get confidence info
                    confidence_info = "N/A"
                    if confidence_data and location in confidence_data and date in confidence_data[location] and product in confidence_data[location][date]:
                        conf_data = confidence_data[location][date][product]
                        competitor_count = conf_data['competitor_count']
                        competitor_word = "competitor" if competitor_count == 1 else "competitors"
                        confidence_info = f"{conf_data['confidence_level']} ({competitor_count} {competitor_word})"
                    else:
                        # Calculate basic confidence from existing data
                        competitor_count = product_data.get('all_competitors', 1) - 1
                        competitor_word = "competitor" if competitor_count == 1 else "competitors"
                        if competitor_count >= 3:
                            confidence_info = f"MEDIUM ({competitor_count} {competitor_word})"
                        elif competitor_count >= 2:
                            confidence_info = f"LOW ({competitor_count} {competitor_word})"
                        else:
                            confidence_info = f"VERY LOW ({competitor_count} {competitor_word})"

                    row = [
                        date,
                        product,
                        f"₦{product_data['pullus_price']:,.0f}",
                        product_data['cheapest_competitor'],
                        f"₦{product_data['cheapest_price']:,.0f}" if product_data['cheapest_price'] > 0 else "N/A",
                        product_data['status'],
                        product_data['message'],
                        confidence_info
                    ]
                    data_rows.append(row)
                    color_mapping.append(product_data['status_code'])

            # Combine headers and data in single update to reduce API calls
            all_data = [headers]
            if data_rows:
                all_data.extend(data_rows)
            ws.update(range_name=f'A1:H{len(all_data)}', values=all_data)

            # Prepare batch formatting requests
            format_requests = []

            # Header formatting
            format_requests.append(self.create_row_format_request(ws, 0, 'header', 8))

            # Data row formatting
            for i, status_code in enumerate(color_mapping):
                row_index = i + 1  # +1 for header row
                if status_code in self.colors:
                    format_requests.append(self.create_row_format_request(ws, row_index, status_code, 8))

            # Apply all formatting in one batch
            self.batch_format_sheet(ws, format_requests)
            
            # Auto-resize columns
            self.auto_resize_columns(ws)
            
            logger.info(f"✅ {location} daily position sheet created")
            
        except Exception as e:
            logger.error(f"❌ {location} daily position sheet creation failed: {sanitize_error(e)}")
    
    def batch_format_sheet(self, worksheet, formatting_data):
        """Efficiently format entire sheet with single batch request"""
        try:
            requests = []
            
            # Add all formatting requests to batch
            for format_request in formatting_data:
                requests.append(format_request)
            
            # Execute all formatting in one batch request
            if requests:
                self.rate_limited_sheets.batch_update({'requests': requests})
                logger.info(f"✅ Applied {len(requests)} formatting operations in batch")
            
        except Exception as e:
            logger.warning(f"⚠️ Batch formatting failed: {sanitize_error(e)}")
    
    def create_row_format_request(self, worksheet, row_index, color_type, num_cols=10):
        """Create a single row formatting request with enhanced text formatting"""
        cell_format = {
            'backgroundColor': self.colors[color_type]
        }

        # Professional text formatting for light theme
        if color_type == 'header':
            cell_format['textFormat'] = {
                'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0},  # White text on blue
                'bold': True,
                'fontSize': 11
            }
        elif color_type in ['urgent', 'losing']:
            cell_format['textFormat'] = {
                'foregroundColor': {'red': 0.7, 'green': 0.2, 'blue': 0.2},  # Dark red text
                'bold': True
            }
        elif color_type == 'data_warning':
            cell_format['textFormat'] = {
                'foregroundColor': {'red': 0.8, 'green': 0.4, 'blue': 0.0},  # Dark orange text
                'bold': True
            }
        elif color_type == 'winning':
            cell_format['textFormat'] = {
                'foregroundColor': {'red': 0.1, 'green': 0.4, 'blue': 0.2},  # Dark green text
                'bold': True
            }
        elif color_type == 'competitive':
            cell_format['textFormat'] = {
                'foregroundColor': {'red': 0.1, 'green': 0.3, 'blue': 0.6},  # Dark blue text
                'bold': True
            }

        fields = 'userEnteredFormat(backgroundColor'
        if 'textFormat' in cell_format:
            fields += ',textFormat'
        fields += ')'

        return {
            'repeatCell': {
                'range': {
                    'sheetId': worksheet.id,
                    'startRowIndex': row_index,
                    'endRowIndex': row_index + 1,
                    'startColumnIndex': 0,
                    'endColumnIndex': num_cols
                },
                'cell': {
                    'userEnteredFormat': cell_format
                },
                'fields': fields
            }
        }

    def create_heatmap_dashboard(self, trends):
        """Create visual heatmap dashboard showing win/loss patterns"""
        try:
            # Create or get worksheet
            try:
                ws = self.rate_limited_sheets.worksheet("Heatmap_Dashboard")
                ws.clear()
                self.clear_sheet_formatting(ws)
            except:
                ws = self.rate_limited_sheets.add_worksheet(title="Heatmap_Dashboard", rows=50, cols=10)
            
            # Build heatmap data
            all_products = set()
            for location_trends in trends.values():
                all_products.update(location_trends.keys())

            data_rows = []

            for product in sorted(all_products):
                row = [product]

                # Get data for each location
                total_wins = 0
                total_checks = 0
                location_results = []

                for location in ['Abuja', 'Kaduna', 'Kano']:
                    if location in trends and product in trends[location]:
                        trend_data = trends[location][product]
                        emoji = trend_data['recent_emoji']
                        win_rate = trend_data['win_rate']

                        row.extend([emoji, f"{win_rate:.0f}%"])

                        total_wins += trend_data['win_count']
                        total_checks += trend_data['total_checks']
                        location_results.append(trend_data['recent_status'])
                    else:
                        row.extend(["⚪", "N/A"])
                        location_results.append("no_data")

                # Overall metrics
                overall_win_rate = (total_wins / total_checks * 100) if total_checks > 0 else 0

                # Determine trend across locations
                winning_count = location_results.count('winning')
                losing_count = location_results.count('losing')

                if winning_count >= 2:
                    trend_arrow = "📈 Strong"
                    priority = "🟢 Maintain"
                elif losing_count >= 2:
                    trend_arrow = "📉 Weak"
                    priority = "🔴 Review"
                else:
                    trend_arrow = "➡️ Mixed"
                    priority = "🟡 Monitor"

                row.extend([f"{overall_win_rate:.0f}%", trend_arrow, priority])
                data_rows.append(row)

            # Combine title, headers, and data in single update to reduce API calls
            all_data = [
                ["🔥 PULLUS COMPETITIVE HEATMAP - WIN/LOSS PATTERNS", "", "", "", "", "", "", "", "", ""],
                ["Product", "Abuja", "Abuja %", "Kaduna", "Kaduna %", "Kano", "Kano %", "Overall Win Rate", "Trend", "Priority"]
            ]
            if data_rows:
                all_data.extend(data_rows)
            ws.update(range_name=f'A1:J{len(all_data)}', values=all_data)

            # Initial auto-resize after data population
            self.auto_resize_columns(ws)

            # Prepare batch formatting
            format_requests = []

            # Title and header formatting
            format_requests.append(self.create_row_format_request(ws, 0, 'header', 10))  # Title
            format_requests.append(self.create_row_format_request(ws, 1, 'metric', 10))  # Column headers

            # Data rows formatting based on performance
            for i, row in enumerate(data_rows):
                row_idx = i + 2  # +2 for title and header rows
                priority = row[-1]  # Last column is priority

                if "🔴" in priority:
                    format_requests.append(self.create_row_format_request(ws, row_idx, 'losing', 10))
                elif "🟢" in priority:
                    format_requests.append(self.create_row_format_request(ws, row_idx, 'winning', 10))
                else:
                    format_requests.append(self.create_row_format_request(ws, row_idx, 'competitive', 10))

            # Apply all formatting in one batch
            self.batch_format_sheet(ws, format_requests)

            # Set specific optimal column widths for heatmap dashboard
            self.set_heatmap_column_widths(ws)
            
            logger.info("✅ Heatmap dashboard created with visual formatting")
            
        except Exception as e:
            logger.error(f"❌ Heatmap dashboard creation failed: {sanitize_error(e)}")
    
    def create_executive_scorecard(self, daily_analysis, trends, stale_data=None):
        """Create simple executive scorecard with key metrics"""
        try:
            # Create or get worksheet
            try:
                ws = self.rate_limited_sheets.worksheet("Executive_Scorecard")
                ws.clear()
                self.clear_sheet_formatting(ws)
            except:
                ws = self.rate_limited_sheets.add_worksheet(title="Executive_Scorecard", rows=50, cols=6)
            
            
            # Calculate overall metrics
            total_winning = 0
            total_competitive = 0
            total_losing = 0
            total_products = 0
            hotspots = []
            
            # Analyze current performance
            for location, location_data in daily_analysis.items():
                for date, date_data in location_data.items():
                    for product, product_data in date_data.items():
                        status = product_data['status_code']
                        if status == 'winning':
                            total_winning += 1
                        elif status == 'competitive':
                            total_competitive += 1
                        elif status == 'losing':
                            total_losing += 1
                            # Track products needing attention
                            if product_data['price_difference'] > self.urgent_review_threshold:
                                hotspots.append({
                                    'location': location,
                                    'product': product,
                                    'difference': product_data['price_difference'],
                                    'date': date
                                })
                        
                        if status != 'no_data':
                            total_products += 1
            
            # Sort hotspots by price difference
            hotspots.sort(key=lambda x: x['difference'], reverse=True)
            
            # Build scorecard data with simplified language
            scorecard_data = [
                ["", "", "", "", "", ""],
                ["📊 TODAY'S PRICE PERFORMANCE", "", "", "", "", ""],
                ["", "", "", "", "", ""],
                ["What these numbers mean:", "We checked prices for products", "across all locations and dates", "", "", ""],
                ["", "", "", "", "", ""],
                [f"✅ BEST PRICE", f"{total_winning} times", f"({total_winning/total_products*100:.1f}% of checks)" if total_products > 0 else "", "Customers choose us", "", ""],
                [f"🟡 COMPETITIVE", f"{total_competitive} times", f"({total_competitive/total_products*100:.1f}% of checks)" if total_products > 0 else "", "We're close to winning", "", ""],
                [f"🔴 ABOVE MARKET", f"{total_losing} times", f"({total_losing/total_products*100:.1f}% of checks)" if total_products > 0 else "", "Customers may go elsewhere", "", ""],
                ["", "", "", "", "", ""],
                ["🔥 URGENT PRICE REVIEWS NEEDED", "", "", "", "", ""],
                ["", "", "", "", "", ""]
            ]
            
            # Add top 5 hotspots
            for i, hotspot in enumerate(hotspots[:5]):
                scorecard_data.append([
                    f"{i+1}. {hotspot['location']} - {hotspot['product']}",
                    f"₦{hotspot['difference']:,.0f} above cheapest",
                    f"Date: {hotspot['date']}",
                    "😨 URGENT REVIEW",
                    "", ""
                ])
            
            if not hotspots:
                scorecard_data.append(["🎉 No urgent pricing issues found!", "", "", "", "", ""])
            
            scorecard_data.extend([
                ["", "", "", "", "", ""],
                ["⚠️ DATA QUALITY ALERTS", "", "", "", "", ""],
                ["", "", "", "", "", ""],
            ])

            # Add data freshness warnings
            if stale_data:
                for location, info in stale_data.items():
                    scorecard_data.append([
                        f"📅 {location} Data Age",
                        f"{info['count']} old entries",
                        f"Oldest: {info['oldest_date']}",
                        "🔄 UPDATE NEEDED",
                        "", ""
                    ])
            else:
                scorecard_data.append(["✅ All data is fresh (within 7 days)", "", "", "", "", ""])

            scorecard_data.extend([
                ["", "", "", "", "", ""],
                ["💭 MANAGEMENT SUMMARY", "", "", "", "", ""],
                ["", "", "", "", "", ""],
            ])
            
            # Add location performance summary
            for location in daily_analysis.keys():
                location_wins = 0
                location_total = 0
                
                for date_data in daily_analysis[location].values():
                    for product_data in date_data.values():
                        if product_data['status_code'] != 'no_data':
                            location_total += 1
                            if product_data['status_code'] == 'winning':
                                location_wins += 1
                
                win_rate = (location_wins / location_total * 100) if location_total > 0 else 0
                status_emoji = "🟢" if win_rate >= 50 else "🟡" if win_rate >= 30 else "🔴"
                
                scorecard_data.append([
                    f"{status_emoji} {location}",
                    f"Winning: {location_wins}/{location_total}",
                    f"Win Rate: {win_rate:.1f}%",
                    "", "", ""
                ])
            
            # Combine title and scorecard data in single update to reduce API calls
            all_data = [["🏆 PULLUS COMPETITIVE SCORECARD - TODAY'S POSITION", "", "", "", "", ""]]
            all_data.extend(scorecard_data)

            # Update the sheet
            ws.update(range_name=f'A1:F{len(all_data)}', values=all_data)
            
            # Prepare batch formatting
            format_requests = []
            
            # Title and section headers
            format_requests.append(self.create_row_format_request(ws, 0, 'header', 6))  # Main title
            format_requests.append(self.create_row_format_request(ws, 1, 'metric', 6))  # "TODAY'S COMPETITIVE PERFORMANCE"
            format_requests.append(self.create_row_format_request(ws, 7, 'metric', 6))  # "IMMEDIATE ACTION NEEDED"
            format_requests.append(self.create_row_format_request(ws, 7 + len(hotspots[:5]) + 3, 'metric', 6))  # "MANAGEMENT SUMMARY"
            
            # Performance metrics formatting with improved colors
            format_requests.append(self.create_row_format_request(ws, 5, 'winning', 6))  # WINNING row
            format_requests.append(self.create_row_format_request(ws, 6, 'competitive', 6))  # COMPETITIVE row
            format_requests.append(self.create_row_format_request(ws, 7, 'losing', 6))  # LOSING row

            # Data quality alerts section header
            data_quality_start = 9 + len(hotspots[:5])
            format_requests.append(self.create_row_format_request(ws, data_quality_start, 'data_warning', 6))  # "DATA QUALITY ALERTS"

            # Data freshness warning rows
            if stale_data:
                for i, _ in enumerate(stale_data.items()):
                    format_requests.append(self.create_row_format_request(ws, data_quality_start + 2 + i, 'data_warning', 6))

            # Urgent items formatting with stronger red
            for i in range(len(hotspots[:5])):
                format_requests.append(self.create_row_format_request(ws, 11 + i, 'urgent', 6))
            
            # Apply all formatting in one batch
            self.batch_format_sheet(ws, format_requests)
            
            # Auto-resize columns
            self.auto_resize_columns(ws)
            
            logger.info("✅ Executive scorecard created with key metrics")
            
        except Exception as e:
            logger.error(f"❌ Executive scorecard creation failed: {sanitize_error(e)}")
    
    def create_simple_metrics_explainer(self):
        """Create simple metrics explainer for management understanding"""
        try:
            # Create or get worksheet
            try:
                ws = self.rate_limited_sheets.worksheet("How_To_Read_Results")
                ws.clear()
                self.clear_sheet_formatting(ws)
            except:
                ws = self.rate_limited_sheets.add_worksheet(title="How_To_Read_Results", rows=30, cols=4)
            
            # Simple explanations with clearer language
            explanations = [
                ["", "", "", ""],
                ["SYMBOL", "MEANING", "WHAT IT MEANS FOR BUSINESS", "ACTION TO TAKE"],
                ["", "", "", ""],
                ["🟢 BEST PRICE", "Pullus has the cheapest price", "Customers will choose us", "Keep this advantage!"],
                ["🟡 COMPETITIVE", "Pullus within ₦200 of cheapest", "We're competitive but not leading", "Monitor closely"],
                ["🔴 ABOVE MARKET", "Pullus more than ₦200 expensive", "Customers may choose competitors", "Review pricing urgently"],
                ["⚪ NO DATA", "No competitors checked same day", "Can't compare pricing", "Get competitor data"],
                ["", "", "", ""],
                ["📈 Improving", "Last 2 checks were better", "Pricing strategy working", "Continue current approach"],
                ["📉 Declining", "Last 2 checks were worse", "Losing competitive position", "Review pricing strategy"],
                ["➡️ Mixed", "Win some, lose some", "Inconsistent performance", "Analyze by product"],
                ["", "", "", ""],
                ["UNDERSTANDING PERCENTAGES", "", "", ""],
                ["", "", "", ""],
                ["% of checks", "How often we win/lose", "NOT % of product types", "Each check = product+location+date"],
                ["Example: 64% above market", "64 times out of 100 checks", "We were too expensive", "Most pricing needs review"],
                ["10% best price", "10 times out of 100 checks", "We had the best price", "Protect these advantages"],
                ["", "", "", ""],
                ["HOW TO USE DAILY SHEETS", "", "", ""],
                ["", "", "", ""],
                ["Check the Status column", "Look for 🔴 ABOVE MARKET products", "These need immediate attention", "Review pricing for red items"],
                ["Look at Message column", "Shows exact price difference", "How much we're over/under", "Adjust by this amount"],
                ["Check Date column", "All comparisons same-day only", "Fair competitive comparison", "Focus on recent dates"],
                ["Confidence column", "How reliable is this data", "HIGH = trust it, LOW = get more data", "Act on HIGH confidence first"],
                ["", "", "", ""],
                ["HOW TO USE HEATMAP", "", "", ""],
                ["", "", "", ""],
                ["Green emojis = Good", "We're winning in that location", "Strong market position", "Maintain advantage"],
                ["Red emojis = Problem", "We're losing in that location", "Weak market position", "Urgent pricing review"],
                ["Win Rate %", "Percentage of times we win", "Overall competitiveness", "Aim for 60%+ win rate"],
                ["", "", "", ""],
                ["MANAGEMENT ACTIONS", "", "", ""],
                ["", "", "", ""],
                ["Daily: Fix all 🔴 items", "Address above-market products", "Protects market share", "Weekly pricing review"],
                ["Weekly: Check heatmap", "See overall patterns", "Strategic positioning", "Adjust pricing strategy"],
                ["Monthly: Review win rates", "Track improvement", "Measure pricing success", "Set new pricing targets"]
            ]

            # Combine title and explanations in single update to reduce API calls
            all_data = [["📚 HOW TO READ PULLUS COMPETITIVE ANALYSIS", "", "", ""]]
            all_data.extend(explanations)

            # Add all data to sheet
            ws.update(range_name=f'A1:D{len(all_data)}', values=all_data)
            
            # Prepare batch formatting
            format_requests = []
            
            # Title and section headers
            format_requests.append(self.create_row_format_request(ws, 0, 'header', 4))  # Main title
            format_requests.append(self.create_row_format_request(ws, 1, 'metric', 4))  # Column headers
            format_requests.append(self.create_row_format_request(ws, 12, 'metric', 4))  # "HOW TO USE DAILY SHEETS"
            format_requests.append(self.create_row_format_request(ws, 17, 'metric', 4))  # "HOW TO USE HEATMAP"
            format_requests.append(self.create_row_format_request(ws, 22, 'metric', 4))  # "MANAGEMENT ACTIONS"
            
            # Alternating row colors for better readability
            for i in range(3, len(explanations), 2):
                format_requests.append(self.create_row_format_request(ws, i, 'no_data', 4))
            
            # Apply all formatting in one batch
            self.batch_format_sheet(ws, format_requests)
            
            # Auto-resize columns
            self.auto_resize_columns(ws)
            
            logger.info("✅ Simple metrics explainer created")
            
        except Exception as e:
            logger.error(f"❌ Simple metrics explainer creation failed: {sanitize_error(e)}")

    def validate_data_freshness(self, max_days_old=7):
        """Flag data older than max_days_old days"""
        try:
            from datetime import datetime, timedelta

            cutoff_date = datetime.now() - timedelta(days=max_days_old)
            stale_data = {}

            for location, df in self.data.items():
                if not df.empty:
                    # Create a copy to avoid modifying original data
                    df_copy = df.copy()
                    df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
                    old_entries = df_copy[df_copy['Date'] < cutoff_date]

                    if not old_entries.empty:
                        stale_data[location] = {
                            'count': len(old_entries),
                            'oldest_date': old_entries['Date'].min().strftime('%Y-%m-%d') if not old_entries['Date'].isna().all() else 'Unknown',
                            'brands': old_entries['Brand'].unique().tolist()
                        }

            if stale_data:
                logger.warning(f"⚠️ Stale data detected (older than {max_days_old} days):")
                for location, info in stale_data.items():
                    logger.warning(f"  {location}: {info['count']} entries, oldest: {info['oldest_date']}")
            else:
                logger.info(f"✅ All data is fresh (within {max_days_old} days)")

            return stale_data

        except Exception as e:
            logger.error(f"❌ Data freshness validation failed: {sanitize_error(e)}")
            return {}

    def add_confidence_scores(self, analysis_data):
        """Add confidence levels based on competitor count and data quality"""
        try:
            confidence_data = {}

            for location, location_data in analysis_data.items():
                location_confidence = {}

                for date, date_data in location_data.items():
                    date_confidence = {}

                    for product, product_data in date_data.items():
                        competitor_count = product_data.get('all_competitors', 1) - 1  # Exclude Pullus

                        # Calculate confidence score based on competitor count
                        if competitor_count >= 5:
                            confidence = "HIGH"
                            confidence_score = 95
                        elif competitor_count >= 3:
                            confidence = "MEDIUM"
                            confidence_score = 80
                        elif competitor_count >= 2:
                            confidence = "LOW"
                            confidence_score = 60
                        else:
                            confidence = "VERY LOW"
                            confidence_score = 30

                        # Adjust for no competitor data
                        if product_data['status_code'] == 'no_data':
                            confidence = "NO DATA"
                            confidence_score = 0

                        competitor_word = "competitor" if competitor_count == 1 else "competitors"
                        date_confidence[product] = {
                            'confidence_level': confidence,
                            'confidence_score': confidence_score,
                            'competitor_count': competitor_count,
                            'status': product_data['status'],
                            'message': f"{confidence} confidence ({competitor_count} {competitor_word})"
                        }

                    location_confidence[date] = date_confidence
                confidence_data[location] = location_confidence

            logger.info("✅ Confidence scores calculated based on competitor sample sizes")
            return confidence_data

        except Exception as e:
            logger.error(f"❌ Confidence score calculation failed: {sanitize_error(e)}")
            return {}


    def run_analysis(self):
        """Run simplified competitive analysis with same-day comparisons"""
        try:
            logger.info("🚀 Starting Simplified Pullus Competitive Analysis...")
            
            if not self.load_data():
                return False

            # Validate data freshness
            stale_data = self.validate_data_freshness(max_days_old=self.max_data_age_days)

            # Generate new simplified analysis
            daily_analysis = self.get_daily_competitive_analysis()
            trends = self.get_historical_trends(daily_analysis)

            # Add confidence scores
            confidence_scores = self.add_confidence_scores(daily_analysis)
            
            if daily_analysis:
                # Calculate summary metrics
                total_winning = 0
                total_competitive = 0
                total_losing = 0
                total_products = 0
                
                for location_data in daily_analysis.values():
                    for date_data in location_data.values():
                        for product_data in date_data.values():
                            status = product_data['status_code']
                            if status == 'winning':
                                total_winning += 1
                            elif status == 'competitive':
                                total_competitive += 1
                            elif status == 'losing':
                                total_losing += 1
                            
                            if status != 'no_data':
                                total_products += 1

                # Create management sheets (detailed results are in the sheets, not console)
                self.create_management_sheets(daily_analysis, trends, confidence_scores, stale_data)
                
                print("\n✅ Analysis Complete!")
                print("📊 Simplified management sheets created:")
                print("   • Daily sheets for each location (Abuja_Daily, Kaduna_Daily, Kano_Daily)")
                print("   • Heatmap_Dashboard - Visual win/loss patterns")
                print("   • Executive_Scorecard - Key metrics and urgent items")
                print("   • How_To_Read_Results - Simple explanations")
                
                return True
            else:
                logger.error("❌ Failed to generate daily analysis")
                return False
                
        except Exception as e:
            logger.error(f"❌ Analysis failed: {sanitize_error(e)}")
            return False

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pullus Competitor Analysis')
    parser.add_argument('--update-only', action='store_true',
                       help='Update Google Sheets only (no email)')
    parser.add_argument('--check-and-alert', action='store_true',
                       help='Check for new dates and send email if found')
    args = parser.parse_args()

    # Load environment variables (CI or local .env)
    if os.getenv('CI') != 'true':
        load_dotenv()

    # Get credentials from environment variables
    CREDENTIALS_PATH = os.getenv('GOOGLE_CREDENTIALS_PATH')
    SHEET_ID = os.getenv('GOOGLE_SHEET_ID')

    if not CREDENTIALS_PATH or not SHEET_ID:
        logger.error("❌ Missing required environment variables: GOOGLE_CREDENTIALS_PATH and/or GOOGLE_SHEET_ID")
        logger.error("   Please ensure .env file exists with these variables")
        return

    # Load configuration
    config = load_config('config.yaml')

    # Initialize analyzer with config
    analyzer = PullusCompetitorAnalyzer(CREDENTIALS_PATH, SHEET_ID, config)

    if args.check_and_alert:
        # Daily workflow: Check for new dates and send email
        logger.info("🔔 Running daily alert workflow...")

        if not analyzer.load_data():
            logger.error("❌ Failed to load data")
            return

        # Get daily analysis
        daily_analysis = analyzer.get_daily_competitive_analysis()
        if not daily_analysis:
            logger.error("❌ Failed to generate analysis")
            return

        # Check for new dates
        new_dates = analyzer.detect_fresh_dates(daily_analysis)

        if new_dates:
            logger.info(f"✅ Found new dates to alert about")

            # Send email
            try:
                email_sender = EmailAlerts(config)
                if email_sender.send_email(daily_analysis, new_dates):
                    logger.info("📧 Email sent successfully!")

                    # Mark dates as processed
                    analyzer.mark_dates_as_processed(new_dates)

                    # Also update sheets
                    trends = analyzer.get_historical_trends(daily_analysis)
                    stale_data = analyzer.validate_data_freshness(analyzer.max_data_age_days)
                    confidence_scores = analyzer.add_confidence_scores(daily_analysis)
                    analyzer.create_management_sheets(daily_analysis, trends, confidence_scores, stale_data)

                    logger.info("🎉 Daily alert workflow completed successfully!")
                else:
                    logger.error("❌ Failed to send email")
            except Exception as e:
                logger.error(f"❌ Email workflow failed: {sanitize_error(e)}")
        else:
            logger.info("ℹ️ No new dates found. No email sent.")

    elif args.update_only:
        # Continuous workflow: Just update sheets
        logger.info("📊 Running sheet update only...")

        if analyzer.run_analysis():
            logger.info("✅ Sheets updated successfully!")
        else:
            logger.error("❌ Sheet update failed")

    else:
        # Default: Local development - just update sheets
        logger.info("🔧 Running local analysis...")

        if analyzer.run_analysis():
            logger.info("🎉 Analysis completed successfully!")
        else:
            logger.error("❌ Analysis failed")

if __name__ == "__main__":
    main()
