# -*- coding: utf-8 -*-
"""
Bot de Telegram para verificar tarjetas - VERSIÓN CON CLASIFICACIÓN MEJORADA
Basado en reglas precisas: CHARGED, 3DS, CAPTCHA, DECLINED, UNKNOWN
"""

import os
import json
import logging
import asyncio
import time
import random
import sqlite3
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import signal
import sys

from telegram import Update, Document, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import aiohttp

# ================== MANEJO DE SEÑALES ==================
def handle_shutdown(signum, frame):
    logger.info(f"🛑 Recibida señal {signum}, cerrando gracefulmente...")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# ================== CONFIGURACIÓN ==================
class Settings:
    """Configuración global del bot"""
    TOKEN = os.environ.get("BOT_TOKEN")
    if not TOKEN:
        raise ValueError("❌ ERROR: BOT_TOKEN no está configurado")

    API_ENDPOINTS = [
        os.environ.get("API_URL", "https://auto-shopify-api-production.up.railway.app/index.php"),
        os.environ.get("API_URL2", "https://auto-shopify-api-production.up.railway.app/index.php"),
        os.environ.get("API_URL3", "https://auto-shopify-api-production.up.railway.app/index.php"),
    ]

    DB_FILE = os.environ.get("DB_FILE", "bot_database.db")
    MAX_WORKERS_PER_USER = int(os.environ.get("MAX_WORKERS", 8))
    RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT", 2))
    DAILY_LIMIT_CHECKS = int(os.environ.get("DAILY_LIMIT", 1000))
    MASS_LIMIT_PER_HOUR = int(os.environ.get("MASS_LIMIT", 3))
    MASS_COOLDOWN_MINUTES = 0  # 👈 SIN COOLDOWN
    ADMIN_IDS = [int(id) for id in os.environ.get("ADMIN_IDS", "").split(",") if id]

    # Configuración de timeouts
    TIMEOUT_CONFIG = {
        "connect": 5,           # Proxy malo muere rápido
        "sock_connect": 5,
        "sock_read": 45,        # Shopify puede tardar
        "total": None,           # NO matar procesos válidos
        "response_body": 45,    
        "bin_lookup": 2,
    }

# ================== LOGGING ==================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

INSTANCE_ID = os.environ.get("RAILWAY_DEPLOYMENT_ID", str(time.time()))

# ================== ENUMS MEJORADOS ==================
class CheckStatus(Enum):
    # Éxito real
    CHARGED = "charged"
    
    # Bloqueos / Autenticación
    CAPTCHA_REQUIRED = "captcha_required"
    THREE_DS_REQUIRED = "3ds_required"
    WAF_BLOCK = "waf_block"
    RATE_LIMIT = "rate_limit"
    BLOCKED = "blocked"
    
    # Declinados
    DECLINED = "declined"                    # Confirmado
    DECLINED_LIKELY = "declined_likely"      # Probable
    
    # Errores de red (NO son decline)
    CONNECT_TIMEOUT = "connect_timeout"      # Proxy murió
    READ_TIMEOUT = "read_timeout"             # Body lento
    SITE_DOWN = "site_down"
    
    # Respuesta ambigua
    UNKNOWN = "unknown"

class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    CONFIRMED = "CONFIRMED"

# ================== DATACLASSES ==================
@dataclass
class Job:
    site: str
    proxy: str
    card_data: Dict
    job_id: int
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class JobResult:
    job: Job
    status: CheckStatus
    confidence: Confidence
    reason: str
    response_time: float
    http_code: Optional[int]
    response_text: str
    success: bool
    bin_info: Dict = field(default_factory=dict)
    price: str = "N/A"
    patterns_detected: List[str] = field(default_factory=list)

# ================== CLASIFICADOR MEJORADO ==================
class ResponseClassifier:
    """
    Clasifica respuestas según reglas estrictas:
    1️⃣ ERRORES DE RED
    2️⃣ CAPTCHA / WAF
    3️⃣ 3DS / AUTENTICACIÓN
    4️⃣ CHARGED (solo si es real)
    5️⃣ DECLINED CONFIRMADO
    6️⃣ DECLINED PROBABLE
    7️⃣ UNKNOWN
    """
    
    # Patrones de CHARGED (REAL)
    CHARGED_PATTERNS = {
        "thank_you": re.compile(r'thank\s*you', re.I),
        "order_confirmed": re.compile(r'order\s*confirmed|order\s*#\d+', re.I),
        "receipt": re.compile(r'receipt|invoice', re.I),
        "payment_complete": re.compile(r'payment\s*complete|transaction\s*complete', re.I),
        "success_page": re.compile(r'success|successful', re.I),
    }
    
    # Patrones de CAPTCHA / WAF
    CAPTCHA_PATTERNS = {
        "captcha": re.compile(r'captcha|recaptcha', re.I),
        "challenge": re.compile(r'challenge.*js|js.*challenge', re.I),
        "cloudflare": re.compile(r'cloudflare|cf-ray', re.I),
        "access_denied": re.compile(r'access\s*denied|access\s*blocked', re.I),
        "ddos": re.compile(r'ddos|protection', re.I),
    }
    
    # Patrones de 3DS
    THREE_DS_PATTERNS = {
        "3ds": re.compile(r'3ds|3d\s*secure', re.I),
        "verified_by_visa": re.compile(r'verified\s*by\s*visa', re.I),
        "mastercard_secure": re.compile(r'mastercard.*secure|securecode', re.I),
        "amex_safe": re.compile(r'american\s*express.*safe|safe.*amex', re.I),
        "acs": re.compile(r'acs|access\s*control\s*server', re.I),
        "redirect_3ds": re.compile(r'redirect.*3ds|3ds.*redirect', re.I),
    }
    
    # Patrones de DECLINED CONFIRMADO
    DECLINED_CONFIRMED = {
        "insufficient_funds": re.compile(r'insufficient\s*funds|insufficient.*balance', re.I),
        "no_balance": re.compile(r'not\s*enough\s*balance|low\s*balance', re.I),
        "cvv_error": re.compile(r'cvv.*incorrect|invalid.*cvv', re.I),
        "expired": re.compile(r'expired\s*card|card.*expired', re.I),
        "do_not_honor": re.compile(r'do\s*not\s*honor', re.I),
        "payment_declined": re.compile(r'payment.*declined|transaction.*declined', re.I),
    }
    
    # Patrones de DECLINED PROBABLE (rápido/genérico)
    DECLINED_LIKELY = {
        "http_402": re.compile(r'402', re.I),
        "fast_reject": re.compile(r'declined|rejected|denied', re.I),
        "generic_error": re.compile(r'error.*payment|payment.*error', re.I),
    }
    
    @classmethod
    def classify(cls, context) -> Tuple[CheckStatus, Confidence, str, List[str]]:
        patterns_detected = []
        
        # ===== 1️⃣ ERRORES DE RED =====
        if context.http_code is None:
            if context.response_time < Settings.TIMEOUT_CONFIG["connect"]:
                return CheckStatus.CONNECT_TIMEOUT, Confidence.CONFIRMED, "proxy_failed", ["connect_timeout"]
            else:
                return CheckStatus.READ_TIMEOUT, Confidence.CONFIRMED, "slow_response_no_body", ["read_timeout"]
        
        if context.http_code >= 500:
            return CheckStatus.SITE_DOWN, Confidence.HIGH, f"server_error_{context.http_code}", [f"http_{context.http_code}"]
        
        # ===== 2️⃣ CAPTCHA / WAF =====
        response_lower = context.response_text.lower()
        
        for block_type, pattern in cls.CAPTCHA_PATTERNS.items():
            if pattern.search(response_lower):
                patterns_detected.append(f"captcha:{block_type}")
                return CheckStatus.CAPTCHA_REQUIRED, Confidence.HIGH, "captcha_challenge_detected", patterns_detected
        
        # ===== 3️⃣ 3DS / AUTENTICACIÓN =====
        for auth_type, pattern in cls.THREE_DS_PATTERNS.items():
            if pattern.search(response_lower):
                patterns_detected.append(f"3ds:{auth_type}")
                return CheckStatus.THREE_DS_REQUIRED, Confidence.HIGH, "authentication_challenge", patterns_detected
        
        # ===== 4️⃣ CHARGED (solo si es real) =====
        charged_matches = []
        price_matches = extract_price(context.response_text)
        
        for charged_type, pattern in cls.CHARGED_PATTERNS.items():
            if pattern.search(response_lower):
                charged_matches.append(charged_type)
                patterns_detected.append(f"charged:{charged_type}")
        
        # Condiciones para CHARGED real:
        # - Tiene al menos un patrón de éxito
        # - Tiempo > 3s (no instantáneo sospechoso)
        # - HTTP 200
        if charged_matches and context.http_code == 200:
            if context.response_time > 3.0:
                if price_matches != "N/A":
                    return CheckStatus.CHARGED, Confidence.HIGH, "confirmed_checkout", patterns_detected + [f"price_{price_matches}"]
                else:
                    return CheckStatus.CHARGED, Confidence.HIGH, "confirmed_checkout", patterns_detected
            else:
                # Respuesta muy rápida con patrones de éxito - sospechoso
                patterns_detected.append("fast_success_suspicious")
        
        # ===== 5️⃣ DECLINED CONFIRMADO =====
        for decline_type, pattern in cls.DECLINED_CONFIRMED.items():
            if pattern.search(response_lower):
                patterns_detected.append(f"declined:{decline_type}")
                return CheckStatus.DECLINED, Confidence.HIGH, f"{decline_type.replace('_', ' ')}", patterns_detected
        
        # ===== 6️⃣ DECLINED PROBABLE =====
        # HTTP 402 Payment Required
        if context.http_code == 402:
            return CheckStatus.DECLINED_LIKELY, Confidence.MEDIUM, "http_402_payment_required", ["http_402"]
        
        # Respuesta rápida (<1s) con mensaje genérico
        if context.response_time < 1.0:
            for likely_type, pattern in cls.DECLINED_LIKELY.items():
                if pattern.search(response_lower):
                    patterns_detected.append(f"likely:{likely_type}")
                    return CheckStatus.DECLINED_LIKELY, Confidence.MEDIUM, "fast_rejection", patterns_detected
        
        # ===== 7️⃣ UNKNOWN =====
        # Si hay respuesta pero no coincide con nada
        if context.response_text and len(context.response_text) > 100:
            return CheckStatus.UNKNOWN, Confidence.LOW, "unrecognized_response", ["has_response_no_pattern"]
        
        return CheckStatus.UNKNOWN, Confidence.LOW, "no_clear_pattern", ["ambiguous"]

# ================== FUNCIONES AUXILIARES ==================
def get_status_emoji(status: CheckStatus) -> str:
    emoji_map = {
        CheckStatus.CHARGED: "✅",
        CheckStatus.CAPTCHA_REQUIRED: "🤖",
        CheckStatus.THREE_DS_REQUIRED: "🔒",
        CheckStatus.WAF_BLOCK: "🛡️",
        CheckStatus.RATE_LIMIT: "⏳",
        CheckStatus.BLOCKED: "🚫",
        CheckStatus.DECLINED: "❌",
        CheckStatus.DECLINED_LIKELY: "⚠️",
        CheckStatus.CONNECT_TIMEOUT: "⏱️",
        CheckStatus.READ_TIMEOUT: "⏱️",
        CheckStatus.SITE_DOWN: "🌐",
        CheckStatus.UNKNOWN: "❓",
    }
    return emoji_map.get(status, "❓")

def get_confidence_icon(confidence: Confidence) -> str:
    icon_map = {
        Confidence.HIGH: "🔴",
        Confidence.MEDIUM: "🟡",
        Confidence.LOW: "🟢",
        Confidence.CONFIRMED: "🔴",
    }
    return icon_map.get(confidence, "⚪")

def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

def extract_price(text: str) -> str:
    try:
        patterns = [
            r'\$?(\d+\.\d{2})',
            r'Price["\s:]+(\d+\.\d{2})',
            r'amount["\s:]+(\d+\.\d{2})',
            r'total["\s:]+(\d+\.\d{2})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return f"${match.group(1)}"
    except:
        pass
    return "N/A"

def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    if total == 0:
        return "[" + "░" * width + "]"
    filled = int((current / total) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

# ================== CACHÉ DE BIN ==================
BIN_CACHE = {}
BIN_CACHE_LOCK = asyncio.Lock()
BIN_SESSION = None
BIN_SESSION_LOCK = asyncio.Lock()

async def get_bin_session() -> aiohttp.ClientSession:
    global BIN_SESSION
    async with BIN_SESSION_LOCK:
        if BIN_SESSION is None or BIN_SESSION.closed:
            timeout = aiohttp.ClientTimeout(total=Settings.TIMEOUT_CONFIG["bin_lookup"])
            BIN_SESSION = aiohttp.ClientSession(timeout=timeout)
        return BIN_SESSION

async def get_bin_info(bin_code: str) -> Dict:
    async with BIN_CACHE_LOCK:
        if bin_code in BIN_CACHE:
            cache_time, data = BIN_CACHE[bin_code]
            if (datetime.now() - cache_time).total_seconds() < 86400:
                return data
    
    try:
        session = await get_bin_session()
        async with session.get(f"https://lookup.binlist.net/{bin_code}") as resp:
            if resp.status == 200:
                try:
                    data = await asyncio.wait_for(resp.json(), timeout=Settings.TIMEOUT_CONFIG["bin_lookup"])
                    result = {
                        "bank": data.get("bank", {}).get("name", "Unknown"),
                        "brand": data.get("scheme", "Unknown").upper(),
                        "country": data.get("country", {}).get("alpha2", "UN"),
                        "type": data.get("type", "Unknown"),
                    }
                    
                    async with BIN_CACHE_LOCK:
                        BIN_CACHE[bin_code] = (datetime.now(), result)
                    
                    return result
                except asyncio.TimeoutError:
                    logger.debug(f"Timeout leyendo BIN {bin_code}")
    except Exception as e:
        logger.debug(f"Error consultando BIN {bin_code}: {e}")
    
    default = {"bank": "Unknown", "brand": "UNKNOWN", "country": "UN", "type": "Unknown"}
    async with BIN_CACHE_LOCK:
        BIN_CACHE[bin_code] = (datetime.now(), default)
    return default

# ================== VALIDACIÓN DE TARJETAS ==================
class CardValidator:
    @staticmethod
    def luhn_check(card_number: str) -> bool:
        def digits_of(n):
            return [int(d) for d in str(n)]
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10 == 0

    @staticmethod
    def validate_expiry(month: str, year: str) -> bool:
        try:
            exp_month = int(month)
            exp_year = int(year)
            if len(year) == 2:
                exp_year += 2000
            now = datetime.now()
            if exp_year < now.year:
                return False
            if exp_year == now.year and exp_month < now.month:
                return False
            return True
        except:
            return False

    @staticmethod
    def validate_cvv(cvv: str) -> bool:
        return cvv.isdigit() and len(cvv) in (3, 4)

    @staticmethod
    def parse_card(card_str: str) -> Optional[Dict]:
        parts = card_str.split('|')
        if len(parts) != 4:
            return None
        number, month, year, cvv = parts
        if not number.isdigit() or len(number) < 13 or len(number) > 19:
            return None
        if not CardValidator.luhn_check(number):
            return None
        if not CardValidator.validate_expiry(month, year):
            return None
        if not CardValidator.validate_cvv(cvv):
            return None
        return {
            "number": number,
            "month": month,
            "year": year,
            "cvv": cvv,
            "bin": number[:6],
            "last4": number[-4:]
        }

# ================== DETECCIÓN INTELIGENTE ==================
def detect_line_type(line: str) -> Tuple[str, Optional[str]]:
    line = line.strip()
    if not line:
        return None, None

    if line.startswith(('http://', 'https://')):
        rest = line.split('://')[1]
        if '.' in rest and not rest.startswith('.') and ' ' not in rest:
            return 'site', line

    if not line.startswith(('http://', 'https://')):
        parts = line.split(':')
        
        if len(parts) == 2:
            host, port = parts
            if port.isdigit() and 1 <= int(port) <= 65535:
                if re.match(r'^[a-zA-Z0-9\.\-_]+$', host):
                    return 'proxy', line
        
        elif len(parts) == 4:
            host, port, user, password = parts
            if port.isdigit() and 1 <= int(port) <= 65535:
                if re.match(r'^[a-zA-Z0-9\.\-_]+$', host):
                    return 'proxy', line
        
        elif len(parts) == 3 and parts[2] == '':
            host, port, _ = parts
            if port.isdigit() and 1 <= int(port) <= 65535:
                if re.match(r'^[a-zA-Z0-9\.\-_]+$', host):
                    return 'proxy', line

    if '|' in line:
        parts = line.split('|')
        if len(parts) == 4:
            numero, mes, año, cvv = parts
            if (numero.isdigit() and len(numero) >= 13 and len(numero) <= 19 and
                mes.isdigit() and 1 <= int(mes) <= 12 and
                año.isdigit() and len(año) in (2, 4) and
                cvv.isdigit() and len(cvv) in (3, 4)):
                if CardValidator.luhn_check(numero):
                    return 'card', line

    return None, None

# ================== BASE DE DATOS SIMPLIFICADA ==================
class Database:
    def __init__(self, db_path=Settings.DB_FILE):
        self.db_path = db_path
        self._write_lock = asyncio.Lock()
        
    async def initialize(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    sites TEXT DEFAULT '[]',
                    proxies TEXT DEFAULT '[]',
                    cards TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    card_bin TEXT,
                    card_last4 TEXT,
                    site TEXT,
                    proxy TEXT,
                    status TEXT,
                    confidence TEXT,
                    reason TEXT,
                    response_time REAL,
                    http_code INTEGER,
                    price TEXT,
                    bin_info TEXT,
                    patterns TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rate_limits (
                    user_id INTEGER PRIMARY KEY,
                    last_check TIMESTAMP,
                    checks_today INTEGER DEFAULT 0,
                    last_mass TIMESTAMP,
                    mass_count_hour INTEGER DEFAULT 0,
                    last_reset DATE DEFAULT CURRENT_DATE
                )
            ''')
            
            conn.commit()
        logger.info("✅ Base de datos inicializada")

    async def save_result(self, user_id: int, result: JobResult):
        async with self._write_lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO results 
                    (user_id, card_bin, card_last4, site, proxy, status, confidence,
                     reason, response_time, http_code, price, bin_info, patterns)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    result.job.card_data['bin'],
                    result.job.card_data['last4'],
                    result.job.site,
                    result.job.proxy,
                    result.status.value,
                    result.confidence.value,
                    result.reason,
                    result.response_time,
                    result.http_code,
                    result.price,
                    json.dumps(result.bin_info),
                    json.dumps(result.patterns_detected)
                ))
                conn.commit()

    async def update_rate_limit(self, user_id: int, command: str):
        now = datetime.now()
        today = now.date()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if command == "mass":
                cursor.execute('''
                    INSERT INTO rate_limits (user_id, last_mass, mass_count_hour, last_reset)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        last_mass = excluded.last_mass,
                        mass_count_hour = mass_count_hour + 1,
                        last_reset = excluded.last_reset
                ''', (user_id, now, today))
            else:
                cursor.execute('''
                    INSERT INTO rate_limits (user_id, last_check, checks_today, last_reset)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        last_check = excluded.last_check,
                        checks_today = checks_today + 1,
                        last_reset = excluded.last_reset
                ''', (user_id, now, today))
            
            conn.commit()

    async def check_rate_limit(self, user_id: int, command: str) -> Tuple[bool, str]:
        today = datetime.now().date()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rate_limits WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            
            if not row:
                return True, ""
            
            # Reset diario
            last_reset = datetime.fromisoformat(row[5]).date() if row[5] else today
            if last_reset < today:
                return True, ""
            
            checks_today = row[2] if row[2] else 0
            mass_count = row[4] if row[4] else 0
            
            if command == "mass":
                if mass_count >= Settings.MASS_LIMIT_PER_HOUR:
                    return False, f"⚠️ Máximo {Settings.MASS_LIMIT_PER_HOUR} mass/hora"
                # 👇 COOLDOWN ELIMINADO - no hay espera entre mass
            else:
                if checks_today >= Settings.DAILY_LIMIT_CHECKS:
                    return False, f"📅 Límite diario ({Settings.DAILY_LIMIT_CHECKS}) alcanzado"
                
                if row[1]:  # last_check
                    last_check = datetime.fromisoformat(row[1])
                    elapsed = (datetime.now() - last_check).seconds
                    if elapsed < Settings.RATE_LIMIT_SECONDS:
                        wait = Settings.RATE_LIMIT_SECONDS - elapsed
                        return False, f"⏳ Espera {wait}s"
            
            return True, ""

    async def get_user_data(self, user_id: int) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sites, proxies, cards FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            
            if not row:
                cursor.execute("INSERT INTO users (user_id, sites, proxies, cards) VALUES (?, ?, ?, ?)",
                             (user_id, '[]', '[]', '[]'))
                conn.commit()
                return {"sites": [], "proxies": [], "cards": []}
            
            return {
                "sites": json.loads(row[0]),
                "proxies": json.loads(row[1]),
                "cards": json.loads(row[2])
            }

    async def update_user_data(self, user_id: int, sites=None, proxies=None, cards=None):
        current = await self.get_user_data(user_id)
        
        if sites is not None:
            current["sites"] = sites
        if proxies is not None:
            current["proxies"] = proxies
        if cards is not None:
            current["cards"] = cards
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET sites = ?, proxies = ?, cards = ? WHERE user_id = ?",
                (json.dumps(current["sites"]), json.dumps(current["proxies"]), 
                 json.dumps(current["cards"]), user_id)
            )
            conn.commit()

    async def get_stats(self, user_id: int) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM results WHERE user_id = ?", (user_id,))
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM results WHERE user_id = ? AND status = 'charged'", (user_id,))
            charged = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM results WHERE user_id = ? AND status = 'declined'", (user_id,))
            declined = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM results WHERE user_id = ? AND status = 'declined_likely'", (user_id,))
            declined_likely = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM results WHERE user_id = ? AND status LIKE '%timeout%'", (user_id,))
            timeout = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM results WHERE user_id = ? AND status = 'captcha_required'", (user_id,))
            captcha = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM results WHERE user_id = ? AND status = '3ds_required'", (user_id,))
            three_ds = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM results WHERE user_id = ? AND status = 'unknown'", (user_id,))
            unknown = cursor.fetchone()[0]
            
            return {
                "total": total,
                "charged": charged,
                "declined": declined,
                "declined_likely": declined_likely,
                "timeout": timeout,
                "captcha": captcha,
                "three_ds": three_ds,
                "unknown": unknown
            }

    async def shutdown(self):
        global BIN_SESSION
        if BIN_SESSION and not BIN_SESSION.closed:
            await BIN_SESSION.close()

# ================== PROXY HEALTH CHECKER ==================
class ProxyHealthChecker:
    def __init__(self, db: Database, user_id: int):
        self.db = db
        self.user_id = user_id
        self.test_url = "https://httpbin.org/ip"
        self.timeout = aiohttp.ClientTimeout(
            total=None,
            connect=Settings.TIMEOUT_CONFIG["connect"],
            sock_read=10
        )
        
    async def check_proxy(self, proxy: str) -> Dict:
        start_time = time.time()
        result = {
            "proxy": proxy,
            "alive": False,
            "response_time": 0,
            "error": None,
            "ip": None,
        }
        
        try:
            proxy_parts = proxy.split(':')
            if len(proxy_parts) == 4:
                proxy_url = f"http://{proxy_parts[2]}:{proxy_parts[3]}@{proxy_parts[0]}:{proxy_parts[1]}"
            else:
                proxy_url = f"http://{proxy_parts[0]}:{proxy_parts[1]}"
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(self.test_url, proxy=proxy_url) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - start_time
                        result["alive"] = True
                        result["response_time"] = elapsed
                        try:
                            data = await resp.json()
                            result["ip"] = data.get("origin", "Unknown")
                        except:
                            pass
        except Exception as e:
            result["error"] = str(e)[:50]
        
        return result
    
    async def check_all_proxies(self, proxies: List[str], max_concurrent: int = 25) -> List[Dict]:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def check_with_semaphore(proxy):
            async with semaphore:
                return await self.check_proxy(proxy)
        
        tasks = [check_with_semaphore(p) for p in proxies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                final_results.append({
                    "proxy": proxies[i],
                    "alive": False,
                    "response_time": 0,
                    "error": str(res),
                    "ip": None
                })
            else:
                final_results.append(res)
        
        return final_results

# ================== EJECUTOR DE JOBS ==================
class JobExecutor:
    @staticmethod
    async def execute(job: Job, session: aiohttp.ClientSession) -> JobResult:
        card_data = job.card_data
        card_str = f"{card_data['number']}|{card_data['month']}|{card_data['year']}|{card_data['cvv']}"
        params = {"site": job.site, "cc": card_str, "proxy": job.proxy}
        
        start_time = time.time()
        
        # Obtener info de BIN
        bin_info = await get_bin_info(card_data['bin'])
        
        try:
            api_endpoint = random.choice(Settings.API_ENDPOINTS)
            
            # Configurar proxy
            proxy_parts = job.proxy.split(':')
            if len(proxy_parts) == 4:
                proxy_url = f"http://{proxy_parts[2]}:{proxy_parts[3]}@{proxy_parts[0]}:{proxy_parts[1]}"
            elif len(proxy_parts) == 3 and proxy_parts[2] == '':
                proxy_url = f"http://{proxy_parts[0]}:{proxy_parts[1]}"
            elif len(proxy_parts) == 2:
                proxy_url = f"http://{job.proxy}"
            else:
                proxy_url = None
            
            try:
                if proxy_url:
                    async with session.get(api_endpoint, params=params, proxy=proxy_url) as resp:
                        elapsed = time.time() - start_time
                        try:
                            response_text = await asyncio.wait_for(resp.text(), timeout=Settings.TIMEOUT_CONFIG["response_body"])
                        except asyncio.TimeoutError:
                            response_text = ""
                else:
                    async with session.get(api_endpoint, params=params) as resp:
                        elapsed = time.time() - start_time
                        try:
                            response_text = await asyncio.wait_for(resp.text(), timeout=Settings.TIMEOUT_CONFIG["response_body"])
                        except asyncio.TimeoutError:
                            response_text = ""
                            
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                if elapsed < Settings.TIMEOUT_CONFIG["connect"]:
                    return JobResult(
                        job=job,
                        status=CheckStatus.CONNECT_TIMEOUT,
                        confidence=Confidence.CONFIRMED,
                        reason="proxy_failed",
                        response_time=elapsed,
                        http_code=None,
                        response_text="",
                        success=False,
                        bin_info=bin_info,
                        price="N/A"
                    )
                else:
                    return JobResult(
                        job=job,
                        status=CheckStatus.READ_TIMEOUT,
                        confidence=Confidence.CONFIRMED,
                        reason="slow_response_no_body",
                        response_time=elapsed,
                        http_code=None,
                        response_text="",
                        success=False,
                        bin_info=bin_info,
                        price="N/A"
                    )
            
            # Crear contexto para clasificador
            class Context:
                pass
            
            context = Context()
            context.http_code = resp.status if 'resp' in locals() else None
            context.response_time = elapsed
            context.response_text = response_text if 'response_text' in locals() else ""
            context.response_size = len(response_text) if 'response_text' in locals() else 0
            
            # Clasificar respuesta
            status, confidence, reason, patterns = ResponseClassifier.classify(context)
            price = extract_price(response_text) if 'response_text' in locals() else "N/A"
            success = (status == CheckStatus.CHARGED)
            
            return JobResult(
                job=job,
                status=status,
                confidence=confidence,
                reason=reason,
                response_time=elapsed,
                http_code=resp.status if 'resp' in locals() else None,
                response_text=response_text if 'response_text' in locals() else "",
                success=success,
                bin_info=bin_info,
                price=price,
                patterns_detected=patterns
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error en job {job.job_id}: {e}")
            return JobResult(
                job=job,
                status=CheckStatus.UNKNOWN,
                confidence=Confidence.LOW,
                reason=f"error: {str(e)[:50]}",
                response_time=elapsed,
                http_code=None,
                response_text=str(e),
                success=False,
                bin_info=bin_info,
                price="N/A"
            )

# ================== USER MANAGER ==================
class UserManager:
    def __init__(self, db: Database):
        self.db = db

    async def get_user_data(self, user_id: int) -> Dict:
        return await self.db.get_user_data(user_id)

    async def update_user_data(self, user_id: int, sites=None, proxies=None, cards=None):
        await self.db.update_user_data(user_id, sites, proxies, cards)

    async def check_rate_limit(self, user_id: int, command: str) -> Tuple[bool, str]:
        return await self.db.check_rate_limit(user_id, command)

    async def increment_checks(self, user_id: int, command: str):
        await self.db.update_rate_limit(user_id, command)

# ================== CARD CHECK SERVICE ==================
class CardCheckService:
    def __init__(self, db: Database, user_manager: UserManager):
        self.db = db
        self.user_manager = user_manager

    async def check_single(self, user_id: int, card_data: Dict, site: str, proxy: str) -> JobResult:
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=Settings.TIMEOUT_CONFIG["connect"],
            sock_read=Settings.TIMEOUT_CONFIG["sock_read"]
        )
        
        job = Job(
            site=site,
            proxy=proxy,
            card_data=card_data,
            job_id=0
        )
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            result = await JobExecutor.execute(job, session)
        
        await self.db.save_result(user_id, result)
        
        return result

    async def check_mass(
        self, 
        user_id: int,
        cards: List[Dict],
        sites: List[str],
        proxies: List[str],
        progress_callback=None
    ) -> Tuple[List[JobResult], Dict[str, int], float]:
        
        total_cards = len(cards)
        
        # Crear jobs
        jobs = []
        for i, card in enumerate(cards):
            site = sites[i % len(sites)]
            proxy = proxies[i % len(proxies)]
            
            job = Job(
                site=site,
                proxy=proxy,
                card_data=card,
                job_id=i
            )
            jobs.append(job)
        
        queue = asyncio.Queue()
        for job in jobs:
            await queue.put(job)
        
        results = [None] * len(jobs)
        
        processed = 0
        counts = {
            "charged": 0,
            "declined": 0,
            "declined_likely": 0,
            "captcha": 0,
            "three_ds": 0,
            "timeout": 0,
            "unknown": 0
        }
        start_time = time.time()
        
        counter_lock = asyncio.Lock()
        running = True
        
        num_workers = min(len(proxies), Settings.MAX_WORKERS_PER_USER, total_cards)
        
        timeout_cfg = aiohttp.ClientTimeout(
            total=None,
            connect=Settings.TIMEOUT_CONFIG["connect"],
            sock_read=Settings.TIMEOUT_CONFIG["sock_read"]
        )
        
        async def progress_updater():
            nonlocal processed
            while running or processed < total_cards:
                await asyncio.sleep(0.5)
                
                if progress_callback:
                    async with counter_lock:
                        await progress_callback(processed, counts, total_cards)
        
        async def worker(worker_id: int):
            nonlocal processed, counts
            
            async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                while True:
                    try:
                        job = await queue.get()
                    except asyncio.QueueEmpty:
                        break
                    
                    result = await JobExecutor.execute(job, session)
                    
                    results[job.job_id] = result
                    
                    async with counter_lock:
                        if result.status == CheckStatus.CHARGED:
                            counts["charged"] += 1
                        elif result.status == CheckStatus.DECLINED:
                            counts["declined"] += 1
                        elif result.status == CheckStatus.DECLINED_LIKELY:
                            counts["declined_likely"] += 1
                        elif result.status == CheckStatus.CAPTCHA_REQUIRED:
                            counts["captcha"] += 1
                        elif result.status == CheckStatus.THREE_DS_REQUIRED:
                            counts["three_ds"] += 1
                        elif "timeout" in result.status.value:
                            counts["timeout"] += 1
                        else:
                            counts["unknown"] += 1
                        
                        processed += 1
                    
                    await self.db.save_result(user_id, result)
                    
                    queue.task_done()
        
        updater_task = asyncio.create_task(progress_updater())
        
        worker_tasks = []
        for i in range(num_workers):
            task = asyncio.create_task(worker(i))
            worker_tasks.append(task)
        
        await queue.join()
        
        running = False
        await updater_task
        
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        
        return results, counts, elapsed

# ================== VARIABLES GLOBALES ==================
db = None
user_manager = None
card_service = None
cancel_mass = {}
user_state = {}
active_mass = set()

# ================== MANEJO DE MENSAJES ==================
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    
    if user_id in active_mass:
        await update.message.reply_text("❌ Mass check in progress. Use /stop to cancel.")
        return
    
    # Verificar si es una tarjeta
    card_data = CardValidator.parse_card(text)
    if card_data:
        user_data = await user_manager.get_user_data(user_id)
        
        if not user_data["sites"] or not user_data["proxies"]:
            await update.message.reply_text("❌ Necesitas al menos 1 site y 1 proxy")
            return
        
        # Rate limit
        allowed, msg = await user_manager.check_rate_limit(user_id, "check")
        if not allowed:
            await update.message.reply_text(msg)
            return
        
        msg = await update.message.reply_text("🔄 Verificando...")
        
        site = user_data["sites"][0]
        proxy = user_data["proxies"][0]
        
        result = await card_service.check_single(user_id, card_data, site, proxy)
        await user_manager.increment_checks(user_id, "check")
        
        emoji = get_status_emoji(result.status)
        confidence_icon = get_confidence_icon(result.confidence)
        
        response = (
            f"{emoji} *RESULTADO*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"💳 Tarjeta: `{card_data['bin']}xxxxxx{card_data['last4']}`\n"
            f"📊 Estado: {result.status.value.upper()}\n"
            f"{confidence_icon} Confianza: {result.confidence.value}\n"
            f"📝 Razón: {result.reason}\n"
            f"⏱️ Tiempo: {result.response_time:.1f}s\n"
        )
        
        if result.price != "N/A":
            response += f"💰 Precio: {result.price}\n"
        
        response += f"\n🏦 Banco: {result.bin_info.get('bank', 'Unknown')}\n"
        response += f"🌍 País: {result.bin_info.get('country', 'UN')}"
        
        await msg.edit_text(response, parse_mode="Markdown")
    else:
        await update.message.reply_text(
            "❌ Formato inválido. Usa:\n"
            "`NUMBER|MES|AÑO|CVV`\n"
            "Ejemplo: `4377110010309114|08|2026|501`",
            parse_mode="Markdown"
        )

# ================== MANEJO DE DOCUMENTOS ==================
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    document = update.message.document
    
    if user_id in active_mass:
        await update.message.reply_text("❌ Mass check in progress. Use /stop to cancel.")
        return
    
    if not document.file_name.endswith('.txt'):
        await update.message.reply_text("❌ Solo archivos .txt")
        return
    
    file = await context.bot.get_file(document.file_id)
    content = await file.download_as_bytearray()
    text = content.decode('utf-8', errors='ignore')
    
    cards = []
    invalid = 0
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        
        line_type, normalized = detect_line_type(line)
        if line_type == 'card' and CardValidator.parse_card(normalized):
            cards.append(normalized)
        else:
            invalid += 1
    
    if cards:
        user_data = await user_manager.get_user_data(user_id)
        user_data["cards"].extend(cards)
        await user_manager.update_user_data(user_id, cards=user_data["cards"])
        
        await update.message.reply_text(
            f"✅ {len(cards)} tarjetas válidas guardadas\n"
            f"❌ {invalid} líneas inválidas ignoradas"
        )
    else:
        await update.message.reply_text("❌ No se encontraron tarjetas válidas")

# ================== COMANDOS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id in active_mass:
        active_mass.discard(user_id)
    
    await update.message.reply_text(
        "🤖 *BOT DE VERIFICACIÓN*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Envía una tarjeta en formato:\n"
        "`NUMBER|MES|AÑO|CVV`\n\n"
        "Ejemplo:\n"
        "`4377110010309114|08|2026|501`\n\n"
        "Comandos:\n"
        "/stats - Estadísticas\n"
        "/stop - Detener mass check",
        parse_mode="Markdown"
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    stats = await db.get_stats(user_id)
    
    text = (
        f"📊 *ESTADÍSTICAS*\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"Total checks: {stats['total']}\n"
        f"✅ Charged: {stats['charged']}\n"
        f"❌ Declined: {stats['declined']}\n"
        f"⚠️ Declined (likely): {stats['declined_likely']}\n"
        f"🤖 Captcha: {stats['captcha']}\n"
        f"🔒 3DS: {stats['three_ds']}\n"
        f"⏱️ Timeout: {stats['timeout']}\n"
        f"❓ Unknown: {stats['unknown']}"
    )
    
    await update.message.reply_text(text, parse_mode="Markdown")

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in active_mass:
        cancel_mass[user_id] = True
        await update.message.reply_text("⏹ Deteniendo mass check...")
    else:
        await update.message.reply_text("No hay mass check activo.")

# ================== MAIN ==================
async def shutdown(application: Application):
    logger.info("🛑 Cerrando...")
    if db:
        await db.shutdown()
    logger.info("✅ Cerrado")

async def post_init(application: Application):
    global db, user_manager, card_service
    
    db = Database()
    await db.initialize()
    
    user_manager = UserManager(db)
    card_service = CardCheckService(db, user_manager)
    
    logger.info("✅ Bot inicializado con clasificación mejorada")

def main():
    app = Application.builder().token(Settings.TOKEN).post_init(post_init).build()
    app.post_shutdown = shutdown

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("stop", stop_command))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    app.add_handler(MessageHandler(filters.Document.FileExtension("txt"), document_handler))

    logger.info("🚀 Bot iniciado - SIN COOLDOWN - CLASIFICACIÓN MEJORADA")
    app.run_polling()

if __name__ == "__main__":
    main()
