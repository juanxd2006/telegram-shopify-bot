# -*- coding: utf-8 -*-
"""
Bot de Telegram para verificar tarjetas - VERSIÓN INTELIGENTE
Con clasificación por patrones, niveles de confianza, aprendizaje por BIN.
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
ADMIN_IDS = [int(id) for id in os.environ.get("ADMIN_IDS", "").split(",") if id]

# Configuración de timeouts
TIMEOUT_CONFIG = {
    "total": 8,
    "connect": 3,
    "sock_read": 5,
}

# Configuración de confianza
CONFIDENCE_CONFIG = {
    "fast_charged_threshold": 1.5,  # Menos de 1.5s es sospechoso
    "normal_charged_min": 2.0,       # Mínimo normal para charged
    "normal_charged_max": 7.0,       # Máximo normal para charged
    "high_confidence_time": 4.0,     # Tiempo ideal para alta confianza
}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

INSTANCE_ID = os.environ.get("RAILWAY_DEPLOYMENT_ID", str(time.time()))

# ================== ENUMS AVANZADOS ==================
class CheckStatus(Enum):
    # Estados de éxito
    CHARGED = "charged"
    
    # Estados de decline
    DECLINED = "declined"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_ERROR = "card_error"
    
    # Estados de bloqueo
    RATE_LIMIT = "rate_limit"        # HTTP 429
    BLOCKED = "blocked"               # HTTP 403
    SITE_DOWN = "site_down"           # HTTP 5xx
    WAF_BLOCK = "waf_block"           # Bloqueo por firewall
    
    # Estados de verificación
    THREE_DS = "3ds"
    CAPTCHA = "captcha"
    PENDING = "pending"               # Pendiente de confirmación
    
    # Estados de timeout
    SOFT_TIMEOUT = "soft_timeout"     # Timeout de lectura
    HARD_TIMEOUT = "hard_timeout"      # Timeout de conexión
    
    # Estados de error
    SITE_ERROR = "site_error"          # Error del sitio
    UNKNOWN = "unknown"                 # No clasificado
    UNKNOWN_SUCCESS = "unknown_success" # Posible éxito sin confirmación

class ConfidenceLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SUSPICIOUS = "SUSPICIOUS"

class TimeoutType(Enum):
    CONNECT = "connect"
    READ = "read"
    TOTAL = "total"

@dataclass
class CheckResult:
    card_bin: str
    card_last4: str
    site: str
    proxy: str
    status: CheckStatus
    confidence: ConfidenceLevel
    response_time: float
    http_code: Optional[int]
    response_text: str
    success: bool = False
    bin_info: Dict = field(default_factory=dict)
    price: str = "N/A"
    timeout_type: Optional[TimeoutType] = None
    patterns_detected: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

# ================== CLASIFICADOR DE RESPUESTAS ==================
class ResponseClassifier:
    """Clasifica respuestas de la API usando patrones y contexto"""
    
    # Patrones de éxito
    SUCCESS_PATTERNS = [
        r'thank\s*you',
        r'order\s*confirmed',
        r'payment\s*accepted',
        r'transaction\s*approved',
        r'receipt',
        r'charged',
        r'success',
        r'complete',
    ]
    
    # Patrones de decline
    DECLINE_PATTERNS = [
        r'decline',
        r'insufficient\s*funds',
        r'card\s*error',
        r'invalid\s*card',
        r'do\s*not\s*honor',
        r'rejected',
        r'failed',
    ]
    
    # Patrones de verificación
    VERIFICATION_PATTERNS = {
        '3ds': [r'3ds', r'3d\s*secure', r'authentication', r'verified\s*by\s*visa'],
        'captcha': [r'captcha', r'recaptcha', r'challenge', r'robot'],
        'pending': [r'pending', r'processing', r'in\s*review'],
    }
    
    # Patrones de bloqueo
    BLOCK_PATTERNS = {
        'rate_limit': [r'rate\s*limit', r'too\s*many\s*requests', r'429'],
        'blocked': [r'blocked', r'forbidden', r'access\s*denied', r'403'],
        'waf': [r'waf', r'firewall', r'security\s*check'],
    }
    
    @classmethod
    def classify(cls, http_code: Optional[int], response_text: str, response_time: float) -> Tuple[CheckStatus, ConfidenceLevel, List[str]]:
        """Clasifica la respuesta y determina nivel de confianza"""
        
        response_lower = response_text.lower()
        patterns_detected = []
        
        # ===== 1. Clasificar por código HTTP =====
        if http_code:
            if http_code == 429:
                return CheckStatus.RATE_LIMIT, ConfidenceLevel.HIGH, ["http_429"]
            elif http_code == 403:
                return CheckStatus.BLOCKED, ConfidenceLevel.HIGH, ["http_403"]
            elif 500 <= http_code < 600:
                return CheckStatus.SITE_DOWN, ConfidenceLevel.HIGH, [f"http_{http_code}"]
            elif http_code == 408:
                return CheckStatus.SOFT_TIMEOUT, ConfidenceLevel.HIGH, ["http_408"]
        
        # ===== 2. Buscar patrones de éxito =====
        for pattern in cls.SUCCESS_PATTERNS:
            if re.search(pattern, response_lower):
                patterns_detected.append(f"success:{pattern}")
                
                # Determinar confianza basada en tiempo y patrones
                if response_time < CONFIDENCE_CONFIG["fast_charged_threshold"]:
                    # Muy rápido -> sospechoso
                    confidence = ConfidenceLevel.SUSPICIOUS
                elif CONFIDENCE_CONFIG["normal_charged_min"] <= response_time <= CONFIDENCE_CONFIG["normal_charged_max"]:
                    # Tiempo normal -> alta confianza
                    confidence = ConfidenceLevel.HIGH
                else:
                    # Tiempo fuera de rango -> confianza media
                    confidence = ConfidenceLevel.MEDIUM
                
                return CheckStatus.CHARGED, confidence, patterns_detected
        
        # ===== 3. Buscar patrones de decline =====
        for pattern in cls.DECLINE_PATTERNS:
            if re.search(pattern, response_lower):
                patterns_detected.append(f"decline:{pattern}")
                
                # Clasificar tipo específico de decline
                if 'insufficient' in response_lower:
                    return CheckStatus.INSUFFICIENT_FUNDS, ConfidenceLevel.HIGH, patterns_detected
                elif 'card error' in response_lower:
                    return CheckStatus.CARD_ERROR, ConfidenceLevel.HIGH, patterns_detected
                else:
                    return CheckStatus.DECLINED, ConfidenceLevel.HIGH, patterns_detected
        
        # ===== 4. Buscar patrones de verificación =====
        for status, patterns in cls.VERIFICATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, response_lower):
                    patterns_detected.append(f"verification:{pattern}")
                    if status == '3ds':
                        return CheckStatus.THREE_DS, ConfidenceLevel.HIGH, patterns_detected
                    elif status == 'captcha':
                        return CheckStatus.CAPTCHA, ConfidenceLevel.HIGH, patterns_detected
                    elif status == 'pending':
                        return CheckStatus.PENDING, ConfidenceLevel.MEDIUM, patterns_detected
        
        # ===== 5. Buscar patrones de bloqueo =====
        for status, patterns in cls.BLOCK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, response_lower):
                    patterns_detected.append(f"block:{pattern}")
                    if status == 'rate_limit':
                        return CheckStatus.RATE_LIMIT, ConfidenceLevel.HIGH, patterns_detected
                    elif status == 'blocked':
                        return CheckStatus.BLOCKED, ConfidenceLevel.HIGH, patterns_detected
                    elif status == 'waf':
                        return CheckStatus.WAF_BLOCK, ConfidenceLevel.HIGH, patterns_detected
        
        # ===== 6. Timeouts =====
        if response_time >= TIMEOUT_CONFIG["total"]:
            return CheckStatus.SOFT_TIMEOUT, ConfidenceLevel.HIGH, ["timeout"]
        
        # ===== 7. Si no se clasificó, es UNKNOWN =====
        # Verificar si parece éxito (pero sin confirmación)
        if any(word in response_lower for word in ['accept', 'complete', 'done']):
            return CheckStatus.UNKNOWN_SUCCESS, ConfidenceLevel.LOW, patterns_detected
        
        return CheckStatus.UNKNOWN, ConfidenceLevel.LOW, patterns_detected

# ================== FUNCIONES AUXILIARES ==================
def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    if total == 0:
        return "[" + "░" * width + "]"
    filled = int((current / total) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

def format_price(price_str: str) -> str:
    try:
        patterns = [
            r'\$?(\d+\.\d{2})',
            r'Price["\s:]+(\d+\.\d{2})',
            r'amount["\s:]+(\d+\.\d{2})',
        ]
        for pattern in patterns:
            match = re.search(pattern, price_str, re.IGNORECASE)
            if match:
                return f"${match.group(1)}"
        match = re.search(r'(\d+\.\d{2})', price_str)
        if match:
            return f"${match.group(1)}"
    except:
        pass
    return "N/A"

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
                        "prepaid": data.get("prepaid", False),
                    }
    except:
        pass
    return {
        "bank": "Unknown",
        "brand": "UNKNOWN",
        "country": "UN",
        "type": "Unknown",
        "prepaid": False,
    }

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

# ================== BASE DE DATOS CON BIN LEARNING ==================
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
            conn.execute("PRAGMA synchronous=NORMAL")
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
            
            # Aprendizaje por sitio + proxy
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    site TEXT,
                    proxy TEXT,
                    attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    declines INTEGER DEFAULT 0,
                    timeouts INTEGER DEFAULT 0,
                    captchas INTEGER DEFAULT 0,
                    three_ds INTEGER DEFAULT 0,
                    total_time REAL DEFAULT 0,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, site, proxy)
                )
            ''')
            
            # Aprendizaje por BIN (nuevo)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bin_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    bin TEXT,
                    attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    declines INTEGER DEFAULT 0,
                    three_ds INTEGER DEFAULT 0,
                    captchas INTEGER DEFAULT 0,
                    avg_time REAL DEFAULT 0,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, bin)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bin_user ON bin_learning(user_id)')
            
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
                    response_time REAL,
                    http_code INTEGER,
                    price TEXT,
                    bin_info TEXT,
                    response_text TEXT,
                    patterns TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_user ON results(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_date ON results(created_at)')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    card_bin TEXT,
                    site TEXT,
                    proxy TEXT,
                    status TEXT,
                    response_time REAL,
                    response_text TEXT,
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
                    response_time, http_code, price, bin_info, response_text, patterns)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                batch
            )
            conn.commit()

    async def save_result(self, user_id: int, result: CheckResult):
        patterns_json = json.dumps(result.patterns_detected)
        async with self._batch_lock:
            self._batch_queue.append((
                user_id, result.card_bin, result.card_last4,
                result.site, result.proxy, result.status.value,
                result.confidence.value, result.response_time,
                result.http_code, result.price,
                json.dumps(result.bin_info), result.response_text[:500],
                patterns_json
            ))

    async def save_anomaly(self, user_id: int, result: CheckResult):
        """Guarda casos raros para análisis"""
        await self.execute(
            """INSERT INTO anomalies 
               (user_id, card_bin, site, proxy, status, response_time, response_text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, result.card_bin, result.site, result.proxy,
             result.status.value, result.response_time, result.response_text[:500])
        )

    async def get_bin_stats(self, user_id: int, bin_code: str) -> Dict:
        """Obtiene estadísticas de un BIN específico"""
        row = await self.fetch_one(
            "SELECT * FROM bin_learning WHERE user_id = ? AND bin = ?",
            (user_id, bin_code)
        )
        if not row:
            return {"attempts": 0, "success_rate": 0.5, "three_ds_rate": 0}
        
        attempts = row["attempts"]
        success_rate = row["successes"] / attempts if attempts > 0 else 0.5
        three_ds_rate = row["three_ds"] / attempts if attempts > 0 else 0
        
        return {
            "attempts": attempts,
            "success_rate": success_rate,
            "three_ds_rate": three_ds_rate,
        }

    async def update_bin_stats(self, user_id: int, bin_code: str, result: CheckResult):
        """Actualiza estadísticas de BIN"""
        existing = await self.fetch_one(
            "SELECT * FROM bin_learning WHERE user_id = ? AND bin = ?",
            (user_id, bin_code)
        )
        
        if existing:
            attempts = existing["attempts"] + 1
            successes = existing["successes"] + (1 if result.success else 0)
            declines = existing["declines"] + (1 if result.status == CheckStatus.DECLINED else 0)
            three_ds = existing["three_ds"] + (1 if result.status == CheckStatus.THREE_DS else 0)
            captchas = existing["captchas"] + (1 if result.status == CheckStatus.CAPTCHA else 0)
            
            # Actualizar tiempo promedio
            total_time = (existing["avg_time"] * existing["attempts"]) + result.response_time
            avg_time = total_time / attempts
            
            await self.execute(
                """UPDATE bin_learning SET 
                   attempts = ?, successes = ?, declines = ?, three_ds = ?,
                   captchas = ?, avg_time = ?, last_seen = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (attempts, successes, declines, three_ds, captchas, avg_time, existing["id"])
            )
        else:
            await self.execute(
                """INSERT INTO bin_learning 
                   (user_id, bin, attempts, successes, declines, three_ds, captchas, avg_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, bin_code, 1, 1 if result.success else 0,
                 1 if result.status == CheckStatus.DECLINED else 0,
                 1 if result.status == CheckStatus.THREE_DS else 0,
                 1 if result.status == CheckStatus.CAPTCHA else 0,
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

# ================== SISTEMA DE APRENDIZAJE AVANZADO ==================
class LearningSystem:
    def __init__(self, db: Database, user_id: int):
        self.db = db
        self.user_id = user_id
        self.BASE_EPSILON = 0.15
        
        # Pesos para cada tipo de resultado
        self.WEIGHTS = {
            CheckStatus.CHARGED: 1.0,
            CheckStatus.UNKNOWN_SUCCESS: 0.5,
            CheckStatus.DECLINED: -0.3,
            CheckStatus.INSUFFICIENT_FUNDS: -0.2,
            CheckStatus.CARD_ERROR: -0.4,
            CheckStatus.THREE_DS: -0.5,
            CheckStatus.CAPTCHA: -0.6,
            CheckStatus.RATE_LIMIT: -0.7,
            CheckStatus.BLOCKED: -0.8,
            CheckStatus.SOFT_TIMEOUT: -0.9,
            CheckStatus.HARD_TIMEOUT: -1.0,
            CheckStatus.UNKNOWN: -0.1,
        }

    async def update(self, result: CheckResult):
        """Actualiza estadísticas con pesos diferenciados"""
        
        # Actualizar aprendizaje de BIN
        await self.db.update_bin_stats(self.user_id, result.card_bin, result)
        
        # Actualizar aprendizaje sitio+proxy
        existing = await self.db.fetch_one(
            "SELECT * FROM learning WHERE user_id = ? AND site = ? AND proxy = ?",
            (self.user_id, result.site, result.proxy)
        )
        
        if existing:
            attempts = existing["attempts"] + 1
            successes = existing["successes"] + (1 if result.success else 0)
            declines = existing["declines"] + (1 if result.status == CheckStatus.DECLINED else 0)
            timeouts = existing["timeouts"] + (1 if "timeout" in result.status.value else 0)
            captchas = existing["captchas"] + (1 if result.status == CheckStatus.CAPTCHA else 0)
            three_ds = existing["three_ds"] + (1 if result.status == CheckStatus.THREE_DS else 0)
            total_time = existing["total_time"] + result.response_time
            
            await self.db.execute(
                """UPDATE learning SET 
                   attempts = ?, successes = ?, declines = ?, timeouts = ?,
                   captchas = ?, three_ds = ?, total_time = ?, last_seen = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (attempts, successes, declines, timeouts, captchas, three_ds, total_time, existing["id"])
            )
        else:
            await self.db.execute(
                """INSERT INTO learning 
                   (user_id, site, proxy, attempts, successes, declines, timeouts, captchas, three_ds, total_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.user_id, result.site, result.proxy, 1, 1 if result.success else 0,
                 1 if result.status == CheckStatus.DECLINED else 0,
                 1 if "timeout" in result.status.value else 0,
                 1 if result.status == CheckStatus.CAPTCHA else 0,
                 1 if result.status == CheckStatus.THREE_DS else 0,
                 result.response_time)
            )

    async def get_score(self, site: str, proxy: str, bin_stats: Dict) -> float:
        """Calcula score combinando sitio+proxy y BIN"""
        
        # Score de sitio+proxy
        row = await self.db.fetch_one(
            "SELECT * FROM learning WHERE user_id = ? AND site = ? AND proxy = ?",
            (self.user_id, site, proxy)
        )
        
        base_score = 0.5
        if row and row["attempts"] >= 3:
            attempts = row["attempts"]
            weighted_sum = (
                row["successes"] * self.WEIGHTS[CheckStatus.CHARGED] +
                row["declines"] * self.WEIGHTS[CheckStatus.DECLINED] +
                row["timeouts"] * self.WEIGHTS[CheckStatus.SOFT_TIMEOUT] +
                row["captchas"] * self.WEIGHTS[CheckStatus.CAPTCHA] +
                row["three_ds"] * self.WEIGHTS[CheckStatus.THREE_DS]
            )
            base_score = max(0.1, min(2.0, 0.5 + weighted_sum / attempts))
        
        # Penalizar si el BIN suele dar 3DS
        three_ds_penalty = bin_stats.get("three_ds_rate", 0) * 2.0
        
        # Bonus si el BIN tiene alta tasa de éxito
        success_bonus = bin_stats.get("success_rate", 0.5) * 1.5
        
        final_score = base_score - three_ds_penalty + success_bonus
        return max(0.1, min(2.0, final_score))

    async def choose_combination(self, sites: List[str], proxies: List[str], bin_code: str) -> Tuple[str, str]:
        """Elige combinación considerando BIN"""
        
        # Obtener estadísticas del BIN
        bin_stats = await self.db.get_bin_stats(self.user_id, bin_code)
        
        # Epsilon-greedy con exploración
        if random.random() < self.BASE_EPSILON:
            return random.choice(sites), random.choice(proxies)
        
        # Explotación: mejor score
        scores = []
        for site in sites:
            for proxy in proxies:
                score = await self.get_score(site, proxy, bin_stats)
                scores.append((score, site, proxy))
        
        scores.sort(reverse=True)
        return scores[0][1], scores[0][2]

# ================== CHECKER CON CLASIFICACIÓN INTELIGENTE ==================
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
        
        try:
            async with session.get(API_ENDPOINTS[0], params=params) as resp:
                elapsed = time.time() - start_time
                response_text = await resp.text()
                
                # Clasificar respuesta
                status, confidence, patterns = ResponseClassifier.classify(
                    resp.status, response_text, elapsed
                )
                
                price = format_price(response_text)
                success = status in [CheckStatus.CHARGED, CheckStatus.UNKNOWN_SUCCESS]
                
                result = CheckResult(
                    card_bin=card_data['bin'],
                    card_last4=card_data['last4'],
                    site=site,
                    proxy=proxy,
                    status=status,
                    confidence=confidence,
                    response_time=elapsed,
                    http_code=resp.status,
                    response_text=response_text,
                    success=success,
                    bin_info=bin_info,
                    price=price,
                    patterns_detected=patterns
                )
                
                return result
                
        except asyncio.TimeoutError as e:
            elapsed = time.time() - start_time
            error_str = str(e).lower()
            
            if "connect" in error_str:
                status = CheckStatus.HARD_TIMEOUT
            else:
                status = CheckStatus.SOFT_TIMEOUT
            
            return CheckResult(
                card_bin=card_data['bin'],
                card_last4=card_data['last4'],
                site=site,
                proxy=proxy,
                status=status,
                confidence=ConfidenceLevel.HIGH,
                response_time=elapsed,
                http_code=None,
                response_text="",
                success=False,
                bin_info=bin_info,
                price="N/A",
                patterns_detected=["timeout"]
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return CheckResult(
                card_bin=card_data['bin'],
                card_last4=card_data['last4'],
                site=site,
                proxy=proxy,
                status=CheckStatus.UNKNOWN,
                confidence=ConfidenceLevel.LOW,
                response_time=elapsed,
                http_code=None,
                response_text=str(e),
                success=False,
                bin_info=bin_info,
                price="N/A",
                patterns_detected=["error"]
            )
        finally:
            await self.return_session(session)

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
                if mass_count_hour >= MASS_LIMIT_PER_HOUR:
                    return False, f"⚠️ Máximo {MASS_LIMIT_PER_HOUR} mass/hora"
                
                if row.get("last_mass"):
                    last_mass = datetime.fromisoformat(row["last_mass"])
                    elapsed = (datetime.now() - last_mass).seconds
                    if elapsed < 60:
                        wait = 60 - elapsed
                        return False, f"⏳ Espera {wait}s para otro mass"
            
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
        
        # Guardar anomalías para análisis
        if result.status in [CheckStatus.UNKNOWN, CheckStatus.UNKNOWN_SUCCESS] or \
           (result.status == CheckStatus.CHARGED and result.response_time < 1.5):
            await self.db.save_anomaly(user_id, result)
        
        learning = LearningSystem(self.db, user_id)
        await learning.update(result)
        
        return result

    async def check_mass(
        self, 
        user_id: int,
        cards: List[Dict],
        sites: List[str],
        proxies: List[str],
        num_workers: int,
        progress_callback=None
    ) -> Tuple[List[CheckResult], int, float]:
        
        queue = asyncio.Queue()
        for card in cards:
            await queue.put(card)
        
        result_queue = asyncio.Queue()
        processed = 0
        success_count = 0
        start_time = time.time()
        learning = LearningSystem(self.db, user_id)
        
        async def worker(worker_id: int):
            worker_processed = 0
            worker_success = 0
            
            while not queue.empty() and not cancel_mass.get(user_id, False):
                try:
                    card_data = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    break
                
                # Elegir combinación considerando BIN
                site, proxy = await learning.choose_combination(
                    sites, proxies, card_data['bin']
                )
                
                result = await self.checker.check_card(site, proxy, card_data)
                
                worker_processed += 1
                if result.success:
                    worker_success += 1
                
                await result_queue.put(result)
                
                if progress_callback and worker_processed % 5 == 0:
                    current_processed = processed + worker_processed
                    current_success = success_count + worker_success
                    await progress_callback(current_processed, current_success, len(cards))
            
            return worker_processed, worker_success
        
        tasks = [asyncio.create_task(worker(i)) for i in range(num_workers)]
        
        # Recolectar resultados
        results = []
        while len(results) < len(cards) and not all(t.done() for t in tasks):
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                results.append(result)
                processed += 1
                if result.success:
                    success_count += 1
                
                # Actualizar aprendizaje
                await learning.update(result)
                await self.db.save_result(user_id, result)
                
            except asyncio.TimeoutError:
                continue
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        while not result_queue.empty():
            result = await result_queue.get()
            results.append(result)
            processed += 1
            if result.success:
                success_count += 1
            await learning.update(result)
            await self.db.save_result(user_id, result)
        
        elapsed = time.time() - start_time
        return results, success_count, elapsed

# ================== VARIABLES GLOBALES ==================
db = None
user_manager = None
checker = None
card_service = None
cancel_mass = {}

# ================== HANDLERS ==================

def get_status_emoji(status: CheckStatus, confidence: ConfidenceLevel) -> str:
    """Obtiene emoji según estado y confianza"""
    if status == CheckStatus.CHARGED:
        if confidence == ConfidenceLevel.HIGH:
            return "✅"
        elif confidence == ConfidenceLevel.MEDIUM:
            return "🟡"
        else:
            return "⚠️"
    elif status == CheckStatus.UNKNOWN_SUCCESS:
        return "🟡"
    elif status in [CheckStatus.DECLINED, CheckStatus.INSUFFICIENT_FUNDS, CheckStatus.CARD_ERROR]:
        return "❌"
    elif status in [CheckStatus.THREE_DS, CheckStatus.CAPTCHA]:
        return "🔒"
    elif status in [CheckStatus.RATE_LIMIT, CheckStatus.BLOCKED, CheckStatus.WAF_BLOCK]:
        return "🚫"
    elif "timeout" in status.value:
        return "⏱️"
    else:
        return "❓"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    texto = (
        "❖ *SHOPIFY CHECKER BOT* ❖\n"
        "━━━━━━━━━━━━━━━━━━━━━\n\n"
        "*GATES*  \n"
        "• `/sh [card]` · SHOPIFY CUSTOM GATE\n"
        "• `/msh [workers]` · MASS SHOPIFY\n"
        "• `/proxyhealth` · PROXY CHECKER\n\n"
        "*SITES*  \n"
        "• `/addsite [url]` · ADD SHOPIFY STORE\n"
        "• `/listsites` · LIST SITES\n"
        "• `/removesite [n]` · REMOVE SITE\n\n"
        "*PROXIES*  \n"
        "• `/addproxy [ip:port]` · ADD PROXY\n"
        "• `/listproxies` · LIST PROXIES\n"
        "• `/removeproxy [n]` · REMOVE PROXY\n"
        "• `/cleanproxies` · REMOVE DEAD PROXIES\n\n"
        "*CARDS*  \n"
        "• `/addcards` · ADD CARDS (via .txt)\n"
        "• `/listcards` · LIST CARDS\n"
        "• `/removecard [n]` · REMOVE CARD\n\n"
        "*STATS*  \n"
        "• `/stats` · BOT STATISTICS\n"
        "• `/stop` · STOP CURRENT PROCESS\n\n"
        "❖ *PROTECTION ACTIVE* ❖"
    )
    await update.message.reply_text(texto, parse_mode="Markdown")

# ===== COMANDOS DE SITIOS =====
async def addsite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "❖ *USAGE*: `/addsite [url]`\n"
            "EXAMPLE: `/addsite mystore.myshopify.com`",
            parse_mode="Markdown"
        )
        return
    
    url = context.args[0].strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    user_data["sites"].append(url)
    await user_manager.update_user_data(user_id, sites=user_data["sites"])
    
    await update.message.reply_text(
        f"✅ *SITIO GUARDADO*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"• Sitio: `{url}`\n"
        f"• Total: `{len(user_data['sites'])}` sitios",
        parse_mode="Markdown"
    )

async def listsites(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    
    if not sites:
        await update.message.reply_text(
            "📭 *NO HAY SITIOS*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Usa `/addsite` para agregar tiendas.",
            parse_mode="Markdown"
        )
        return
    
    lines = ["📌 *SITIOS GUARDADOS*", "━━━━━━━━━━━━━━━━━━━━━", ""]
    for i, site in enumerate(sites, 1):
        lines.append(f"`{i}.` {site}")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def removesite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Uso: /removesite [número]")
        return
    
    try:
        index = int(context.args[0]) - 1
    except:
        await update.message.reply_text("❌ Número inválido.")
        return
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    
    if index < 0 or index >= len(sites):
        await update.message.reply_text(f"❌ Índice inválido. Tienes {len(sites)} sitios.")
        return
    
    removed = sites.pop(index)
    await user_manager.update_user_data(user_id, sites=sites)
    await update.message.reply_text(f"🗑️ Sitio eliminado: {removed}")

# ===== COMANDOS DE PROXIES =====
async def addproxy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "❖ *USAGE*: `/addproxy [ip:port]` or `/addproxy [ip:port:user:pass]`",
            parse_mode="Markdown"
        )
        return
    
    proxy_input = context.args[0].strip()
    colon_count = proxy_input.count(':')
    
    if colon_count == 1:
        proxy = f"{proxy_input}::"
    elif colon_count == 3:
        proxy = proxy_input
    else:
        await update.message.reply_text("❌ Formato inválido.")
        return
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    user_data["proxies"].append(proxy)
    await user_manager.update_user_data(user_id, proxies=user_data["proxies"])
    
    display = proxy.split(':')[0] + ':' + proxy.split(':')[1]
    await update.message.reply_text(
        f"✅ *PROXY GUARDADO*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"• Proxy: `{display}`\n"
        f"• Total: `{len(user_data['proxies'])}` proxies",
        parse_mode="Markdown"
    )

async def listproxies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        await update.message.reply_text(
            "📭 *NO HAY PROXIES*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Usa `/addproxy` para agregar proxies.",
            parse_mode="Markdown"
        )
        return
    
    lines = ["📌 *PROXIES GUARDADOS*", "━━━━━━━━━━━━━━━━━━━━━", ""]
    for i, p in enumerate(proxies, 1):
        display = p.split(':')[0] + ':' + p.split(':')[1]
        lines.append(f"`{i}.` {display}")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def removeproxy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Uso: /removeproxy [número]")
        return
    
    try:
        index = int(context.args[0]) - 1
    except:
        await update.message.reply_text("❌ Número inválido.")
        return
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if index < 0 or index >= len(proxies):
        await update.message.reply_text(f"❌ Índice inválido. Tienes {len(proxies)} proxies.")
        return
    
    removed = proxies.pop(index)
    await user_manager.update_user_data(user_id, proxies=proxies)
    
    display = removed.split(':')[0] + ':' + removed.split(':')[1]
    await update.message.reply_text(f"🗑️ Proxy eliminado: {display}")

async def proxyhealth_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        await update.message.reply_text("📭 No hay proxies para verificar.")
        return
    
    msg = await update.message.reply_text("🔄 Verificando proxies...")
    
    health_checker = ProxyHealthChecker(db, user_id)
    results = await health_checker.check_all_proxies(proxies)
    
    alive = [r for r in results if r["alive"]]
    dead = [r for r in results if not r["alive"]]
    
    keyboard = []
    if dead:
        keyboard.append([InlineKeyboardButton("🗑️ ELIMINAR MUERTOS", callback_data=f"clean_{user_id}")])
    
    lines = [
        f"📊 *RESULTADO HEALTH CHECK*",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"• ✅ VIVOS: {len(alive)}",
        f"• ❌ MUERTOS: {len(dead)}",
    ]
    
    if alive:
        lines.append(f"\n✅ *TOP 5 MÁS RÁPIDOS:*")
        for i, r in enumerate(sorted(alive, key=lambda x: x["response_time"])[:5]):
            display = r['proxy'].split(':')[0] + ':' + r['proxy'].split(':')[1]
            lines.append(f"  {i+1}. `{display}` · ⚡ {r['response_time']:.2f}s")
    
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    await msg.edit_text("\n".join(lines), parse_mode="Markdown", reply_markup=reply_markup)

async def cleanproxies_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        await update.message.reply_text("📭 No hay proxies.")
        return
    
    msg = await update.message.reply_text("🔄 Limpiando proxies...")
    
    health_checker = ProxyHealthChecker(db, user_id)
    results = await health_checker.check_all_proxies(proxies)
    
    alive_proxies = [r["proxy"] for r in results if r["alive"]]
    dead_count = len([r for r in results if not r["alive"]])
    
    await user_manager.update_user_data(user_id, proxies=alive_proxies)
    
    await msg.edit_text(
        f"🗑️ *LIMPIEZA COMPLETADA*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"• ✅ CONSERVADOS: {len(alive_proxies)}\n"
        f"• ❌ ELIMINADOS: {dead_count}",
        parse_mode="Markdown"
    )

# ===== COMANDOS DE TARJETAS =====
async def addcards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Envía un archivo `.txt` con tarjetas en formato:\n"
        "`NÚMERO|MES|AÑO|CVV`\n\n"
        "Ejemplo:\n"
        "`4377110010309114|08|2026|501`"
    )

async def listcards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    
    if not cards:
        await update.message.reply_text(
            "📭 *NO HAY TARJETAS*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Envía un archivo `.txt` para agregar tarjetas.",
            parse_mode="Markdown"
        )
        return
    
    lines = ["📌 *TARJETAS GUARDADAS*", "━━━━━━━━━━━━━━━━━━━━━", ""]
    for i, card in enumerate(cards, 1):
        bin_code = card.split('|')[0][:6]
        last4 = card.split('|')[0][-4:]
        lines.append(f"`{i}.` {bin_code}xxxxxx{last4}")
    
    if len(cards) > 10:
        lines.append(f"\n... y {len(cards)-10} más.")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def removecard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Uso: /removecard [número]")
        return
    
    try:
        index = int(context.args[0]) - 1
    except:
        await update.message.reply_text("❌ Número inválido.")
        return
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    
    if index < 0 or index >= len(cards):
        await update.message.reply_text(f"❌ Índice inválido. Tienes {len(cards)} tarjetas.")
        return
    
    removed = cards.pop(index)
    await user_manager.update_user_data(user_id, cards=cards)
    
    bin_code = removed.split('|')[0][:6]
    last4 = removed.split('|')[0][-4:]
    await update.message.reply_text(f"🗑️ Tarjeta eliminada: {bin_code}xxxxxx{last4}")

# ===== COMANDOS DE VERIFICACIÓN =====
async def sh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 1:
        await update.message.reply_text(
            "❖ *SHOPIFY CUSTOM GATE* ❖\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "*USO*: `/sh [card]`\n"
            "*EJEMPLO*: `/sh 4377110010309114|08|2026|501`",
            parse_mode="Markdown"
        )
        return
    
    card_str = context.args[0]
    card_data = CardValidator.parse_card(card_str)
    if not card_data:
        await update.message.reply_text(
            "❌ *FORMATO INVÁLIDO*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Formato correcto: `NÚMERO|MES|AÑO|CVV`",
            parse_mode="Markdown"
        )
        return
    
    user_id = update.effective_user.id
    
    allowed, msg = await user_manager.check_rate_limit(user_id, "check")
    if not allowed:
        await update.message.reply_text(msg)
        return
    
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    proxies = user_data["proxies"]
    
    if not sites:
        await update.message.reply_text("❌ No hay sitios guardados. Usa /addsite")
        return
    
    if not proxies:
        await update.message.reply_text("❌ No hay proxies guardados. Usa /addproxy")
        return
    
    learning = LearningSystem(db, user_id)
    site, proxy = await learning.choose_combination(sites, proxies, card_data['bin'])
    
    msg = await update.message.reply_text(
        f"🔍 *VERIFICANDO...*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💳 Tarjeta: `{card_data['bin']}xxxxxx{card_data['last4']}`\n"
        f"🌐 Sitio: `{site}`\n"
        f"🔒 Proxy: `{proxy.split(':')[0]}:{proxy.split(':')[1]}`",
        parse_mode="Markdown"
    )
    
    result = await card_service.check_single(user_id, card_data, site, proxy)
    await user_manager.increment_checks(user_id, "check")
    
    emoji = get_status_emoji(result.status, result.confidence)
    
    # Formatear respuesta según confianza
    confidence_text = f"({result.confidence.value} CONFIDENCE)"
    
    response = (
        f"{emoji} *RESULTADO DEL CHK* {emoji}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💳 *Tarjeta:* `{result.card_bin}xxxxxx{result.card_last4}`\n"
        f"🌐 *Sitio:* `{result.site}`\n"
        f"🔒 *Proxy:* `{result.proxy.split(':')[0]}:{result.proxy.split(':')[1]}`\n"
        f"📊 *Estado:* {emoji} `{result.status.value.upper()}` {confidence_text}\n"
        f"🏦 *Banco:* `{result.bin_info.get('bank', 'Unknown')}`\n"
        f"💳 *Marca:* `{result.bin_info.get('brand', 'Unknown')}`\n"
        f"🌍 *País:* `{result.bin_info.get('country', 'UN')}`\n"
        f"💰 *Precio:* `{result.price}`\n"
        f"⚡ *Tiempo:* `{result.response_time:.2f}s`\n"
        f"📟 *HTTP:* `{result.http_code or 'N/A'}`"
    )
    
    if result.warnings:
        response += f"\n\n⚠️ *Advertencias:* {', '.join(result.warnings)}"
    
    await msg.edit_text(response, parse_mode="Markdown")

async def msh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    allowed, msg = await user_manager.check_rate_limit(user_id, "mass")
    if not allowed:
        await update.message.reply_text(msg)
        return
    
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    sites = user_data["sites"]
    proxies = user_data["proxies"]
    
    if not cards:
        await update.message.reply_text("❌ No hay tarjetas. Envía un archivo .txt")
        return
    
    if not sites:
        await update.message.reply_text("❌ No hay sitios. Usa /addsite")
        return
    
    if not proxies:
        await update.message.reply_text("❌ No hay proxies. Usa /addproxy")
        return
    
    valid_cards = []
    for card_str in cards:
        card_data = CardValidator.parse_card(card_str)
        if card_data:
            valid_cards.append(card_data)
    
    num_workers = min(MAX_WORKERS_PER_USER, len(valid_cards))
    if context.args:
        try: num_workers = min(int(context.args[0]), MAX_WORKERS_PER_USER)
        except: pass
    
    progress_msg = await update.message.reply_text(
        f"🚀 *MASS CHECK INICIADO*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 Tarjetas: {len(valid_cards)}\n"
        f"⚙️ Workers: {num_workers}\n"
        f"🔄 Procesando..."
    )
    
    async def progress_callback(proc: int, succ: int, total: int):
        bar = create_progress_bar(proc, total)
        elapsed = time.time() - start_time
        speed = proc / elapsed if elapsed > 0 else 0
        
        await progress_msg.edit_text(
            f"🚀 *MASS CHECK*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📊 Progreso: {bar} {proc}/{total}\n"
            f"✅ Aprobadas: {succ}\n"
            f"⚡ Velocidad: {speed:.1f} cards/s\n"
            f"⏱️ Tiempo: {elapsed:.1f}s\n"
            f"🔄 Procesando..."
        )
    
    start_time = time.time()
    results, success_count, elapsed = await card_service.check_mass(
        user_id=user_id,
        cards=valid_cards,
        sites=sites,
        proxies=proxies,
        num_workers=num_workers,
        progress_callback=progress_callback
    )
    
    await user_manager.increment_checks(user_id, "mass")
    
    # Resumen
    summary = (
        f"✅ *MASS CHECK COMPLETADO*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 Procesadas: {len(valid_cards)}\n"
        f"✅ Aprobadas: {success_count}\n"
        f"❌ Rechazadas: {len(valid_cards) - success_count}\n"
        f"⏱️ Tiempo: {elapsed:.1f}s"
    )
    
    await progress_msg.edit_text(summary, parse_mode="Markdown")
    
    # Generar archivo TXT con detalles
    filename = f"mass_{user_id}_{int(time.time())}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"RESULTADOS MASS CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {len(valid_cards)} | Aprobadas: {success_count}\n")
        f.write("="*80 + "\n\n")
        
        for i, r in enumerate(results, 1):
            emoji = get_status_emoji(r.status, r.confidence)
            f.write(f"[{i}] {emoji} Tarjeta: {r.card_bin}xxxxxx{r.card_last4}\n")
            f.write(f"    Estado: {r.status.value.upper()} ({r.confidence.value})\n")
            f.write(f"    Sitio: {r.site}\n")
            f.write(f"    Proxy: {r.proxy}\n")
            f.write(f"    Precio: {r.price}\n")
            f.write(f"    Tiempo: {r.response_time:.2f}s\n")
            f.write(f"    Banco: {r.bin_info.get('bank', 'Unknown')}\n")
            f.write(f"    HTTP: {r.http_code or 'N/A'}\n")
            f.write("-"*40 + "\n")
    
    with open(filename, "rb") as f:
        await update.message.reply_document(
            document=f,
            filename=filename,
            caption=f"📊 Resultados completos - {len(valid_cards)} tarjetas"
        )
    
    os.remove(filename)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    total = await db.fetch_one(
        "SELECT COUNT(*) as count FROM results WHERE user_id = ?",
        (user_id,)
    )
    total_count = total["count"] if total else 0
    
    charged = await db.fetch_one(
        "SELECT COUNT(*) as count FROM results WHERE user_id = ? AND status = 'charged'",
        (user_id,)
    )
    charged_count = charged["count"] if charged else 0
    
    # Obtener estadísticas de BIN
    bin_stats = await db.fetch_all(
        "SELECT bin, attempts, successes, three_ds FROM bin_learning WHERE user_id = ? ORDER BY attempts DESC LIMIT 5",
        (user_id,)
    )
    
    response = (
        f"📊 *ESTADÍSTICAS*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📦 Total checks: {total_count}\n"
        f"✅ Aprobadas: {charged_count}\n"
        f"❌ Rechazadas: {total_count - charged_count}\n\n"
    )
    
    if bin_stats:
        response += "*TOP 5 BINs:*\n"
        for b in bin_stats:
            success_rate = (b["successes"] / b["attempts"]) * 100
            response += f"• `{b['bin']}`: {b['attempts']} tries, {success_rate:.1f}% success\n"
    
    await update.message.reply_text(response, parse_mode="Markdown")

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cancel_mass[user_id] = True
    await update.message.reply_text("⏹️ Proceso detenido.")

# ===== MANEJO DE ARCHIVOS =====
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if not document.file_name.endswith('.txt'):
        await update.message.reply_text("❌ Solo archivos .txt")
        return
    
    file = await context.bot.get_file(document.file_id)
    file_content = await file.download_as_bytearray()
    text = file_content.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    
    sites = []
    proxies = []
    cards = []
    invalid = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        line_type, _ = detect_line_type(line)
        if line_type == 'site':
            sites.append(line)
        elif line_type == 'proxy':
            proxies.append(line)
        elif line_type == 'card':
            if CardValidator.parse_card(line):
                cards.append(line)
            else:
                invalid.append(line)
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    updated = False
    
    if sites:
        user_data["sites"].extend(sites)
        updated = True
    if proxies:
        normalized = []
        for p in proxies:
            if p.count(':') == 1:
                normalized.append(f"{p}::")
            else:
                normalized.append(p)
        user_data["proxies"].extend(normalized)
        updated = True
    if cards:
        user_data["cards"].extend(cards)
        updated = True
    
    if updated:
        await user_manager.update_user_data(
            user_id, 
            sites=user_data["sites"], 
            proxies=user_data["proxies"], 
            cards=user_data["cards"]
        )
    
    parts = []
    if sites: parts.append(f"✅ {len(sites)} sitios")
    if proxies: parts.append(f"✅ {len(proxies)} proxies")
    if cards: parts.append(f"✅ {len(cards)} tarjetas")
    if invalid: parts.append(f"⚠️ {len(invalid)} inválidas")
    
    await update.message.reply_text("\n".join(parts) if parts else "❌ No se encontraron datos válidos.")

# ===== CALLBACKS =====
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    if data.startswith("clean_"):
        user_id = int(data.split("_")[1])
        
        if query.from_user.id != user_id:
            await query.edit_message_text("❌ Acción no autorizada.")
            return
        
        user_data = await user_manager.get_user_data(user_id)
        proxies = user_data["proxies"]
        
        health_checker = ProxyHealthChecker(db, user_id)
        results = await health_checker.check_all_proxies(proxies)
        
        alive = [r["proxy"] for r in results if r["alive"]]
        dead = len([r for r in results if not r["alive"]])
        
        await user_manager.update_user_data(user_id, proxies=alive)
        
        await query.edit_message_text(
            f"🗑️ *LIMPIEZA COMPLETADA*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"• ✅ CONSERVADOS: {len(alive)}\n"
            f"• ❌ ELIMINADOS: {dead}",
            parse_mode="Markdown"
        )

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
    
    logger.info("✅ Bot inicializado con clasificación inteligente")

def main():
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.post_shutdown = shutdown

    # Comandos públicos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("sh", sh_command))
    app.add_handler(CommandHandler("msh", msh_command))
    app.add_handler(CommandHandler("proxyhealth", proxyhealth_command))
    app.add_handler(CommandHandler("addsite", addsite))
    app.add_handler(CommandHandler("addproxy", addproxy))
    app.add_handler(CommandHandler("addcards", addcards))
    app.add_handler(CommandHandler("listsites", listsites))
    app.add_handler(CommandHandler("listproxies", listproxies))
    app.add_handler(CommandHandler("listcards", listcards))
    app.add_handler(CommandHandler("removesite", removesite))
    app.add_handler(CommandHandler("removeproxy", removeproxy))
    app.add_handler(CommandHandler("removecard", removecard))
    app.add_handler(CommandHandler("cleanproxies", cleanproxies_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("stop", stop_command))
    
    # Archivos y callbacks
    app.add_handler(MessageHandler(filters.Document.FileExtension("txt"), handle_document))
    app.add_handler(CallbackQueryHandler(button_callback))

    logger.info("🚀 Bot iniciado con clasificación inteligente")
    app.run_polling()

if __name__ == "__main__":
    main()
