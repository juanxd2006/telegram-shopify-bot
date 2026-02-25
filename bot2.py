# -*- coding: utf-8 -*-
"""
Bot de Telegram para verificar tarjetas - VERSIÓN SIMPLE
Solo comandos: /addsite, /addproxy, /chk
"""

import os
import json
import logging
import asyncio
import time
import random
import sqlite3
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import signal
import sys

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import aiohttp

# ================== CONFIGURACIÓN ==================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Manejo de señales
def handle_shutdown(signum, frame):
    logger.info("🛑 Cerrando...")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# ================== CONFIGURACIÓN ==================
class Settings:
    """Configuración global del bot"""
    TOKEN = os.environ.get("BOT_TOKEN")
    if not TOKEN:
        raise ValueError("❌ ERROR: BOT_TOKEN no está configurado")

    # API endpoints (la misma que usa tu otro bot)
    API_ENDPOINTS = [
        os.environ.get("API_URL", "https://auto-shopify-api-production.up.railway.app/index.php"),
        os.environ.get("API_URL2", "https://auto-shopify-api-production.up.railway.app/index.php"),
        os.environ.get("API_URL3", "https://auto-shopify-api-production.up.railway.app/index.php"),
    ]

    DB_FILE = "bot_simple.db"
    
    # Timeouts
    TIMEOUT_CONFIG = {
        "connect": 10,
        "sock_read": 45,
        "total": None,
        "response_body": 45,
    }

# ================== BASE DE DATOS SIMPLE ==================
class Database:
    """Base de datos solo para sitios y proxies"""
    
    def __init__(self):
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(Settings.DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    sites TEXT DEFAULT '[]',
                    proxies TEXT DEFAULT '[]'
                )
            ''')
            conn.commit()
        logger.info("✅ Base de datos inicializada")
    
    def get_user_data(self, user_id: int) -> Dict:
        with sqlite3.connect(Settings.DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sites, proxies FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            
            if not row:
                cursor.execute("INSERT INTO users (user_id, sites, proxies) VALUES (?, ?, ?)",
                             (user_id, '[]', '[]'))
                conn.commit()
                return {"sites": [], "proxies": []}
            
            return {
                "sites": json.loads(row[0]),
                "proxies": json.loads(row[1])
            }
    
    def add_site(self, user_id: int, site: str):
        data = self.get_user_data(user_id)
        if site not in data["sites"]:
            data["sites"].append(site)
            self._save(user_id, data)
            return True
        return False
    
    def add_proxy(self, user_id: int, proxy: str):
        data = self.get_user_data(user_id)
        if proxy not in data["proxies"]:
            data["proxies"].append(proxy)
            self._save(user_id, data)
            return True
        return False
    
    def _save(self, user_id: int, data: Dict):
        with sqlite3.connect(Settings.DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET sites = ?, proxies = ? WHERE user_id = ?",
                (json.dumps(data["sites"]), json.dumps(data["proxies"]), user_id)
            )
            conn.commit()

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

# ================== VERIFICADOR SIMPLE ==================
class SimpleChecker:
    @staticmethod
    async def check_card(card_data: Dict, site: str, proxy: str) -> Dict:
        """Verifica una tarjeta usando la API"""
        
        card_str = f"{card_data['number']}|{card_data['month']}|{card_data['year']}|{card_data['cvv']}"
        params = {"site": site, "cc": card_str, "proxy": proxy}
        
        # Elegir endpoint aleatorio
        api_endpoint = random.choice(Settings.API_ENDPOINTS)
        
        # Configurar proxy
        proxy_parts = proxy.split(':')
        if len(proxy_parts) == 4:
            proxy_url = f"http://{proxy_parts[2]}:{proxy_parts[3]}@{proxy_parts[0]}:{proxy_parts[1]}"
        elif len(proxy_parts) == 2:
            proxy_url = f"http://{proxy}"
        else:
            proxy_url = None
        
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=Settings.TIMEOUT_CONFIG["connect"],
            sock_read=Settings.TIMEOUT_CONFIG["sock_read"]
        )
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if proxy_url:
                    async with session.get(api_endpoint, params=params, proxy=proxy_url) as resp:
                        response_text = await resp.text()
                        elapsed = time.time() - start_time
                else:
                    async with session.get(api_endpoint, params=params) as resp:
                        response_text = await resp.text()
                        elapsed = time.time() - start_time
                
                return {
                    "success": True,
                    "status_code": resp.status,
                    "response": response_text[:500],  # Limitar tamaño
                    "time": round(elapsed, 2),
                    "site": site,
                    "proxy": proxy.split(':')[0] + ':' + proxy.split(':')[1] if proxy else "directo"
                }
                
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "error": "timeout",
                "time": round(elapsed, 2)
            }
        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "error": str(e)[:100],
                "time": round(elapsed, 2)
            }

# ================== VARIABLES GLOBALES ==================
db = Database()

# ================== COMANDOS ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start"""
    await update.message.reply_text(
        "🤖 *BOT SIMPLE SHOPIFY*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "Comandos:\n"
        "• `/addsite <url>` - Agregar sitio\n"
        "• `/addproxy <ip:puerto>` - Agregar proxy\n"
        "• `/chk <tarjeta>` - Verificar tarjeta\n\n"
        "Ejemplo:\n"
        "`/chk 4377110010309114|08|2026|501`",
        parse_mode="Markdown"
    )

async def addsite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Agrega un sitio"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("❌ Uso: /addsite <url>")
        return
    
    url = context.args[0]
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    if db.add_site(user_id, url):
        await update.message.reply_text(f"✅ Sitio agregado: {url}")
    else:
        await update.message.reply_text("⚠️ El sitio ya existe")

async def addproxy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Agrega un proxy"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "❌ Uso: /addproxy ip:puerto\n"
            "Ejemplo: /addproxy 23.26.53.37:6003"
        )
        return
    
    proxy = context.args[0]
    parts = proxy.split(':')
    
    if len(parts) != 2 or not parts[1].isdigit():
        await update.message.reply_text("❌ Formato inválido. Usa ip:puerto")
        return
    
    if db.add_proxy(user_id, proxy):
        await update.message.reply_text(f"✅ Proxy agregado: {proxy}")
    else:
        await update.message.reply_text("⚠️ El proxy ya existe")

async def chk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Verifica una tarjeta"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "❌ Uso: /chk numero|mes|año|cvv\n"
            "Ejemplo: /chk 4377110010309114|08|2026|501"
        )
        return
    
    card_str = context.args[0]
    card_data = CardValidator.parse_card(card_str)
    
    if not card_data:
        await update.message.reply_text("❌ Tarjeta inválida")
        return
    
    # Obtener datos del usuario
    user_data = db.get_user_data(user_id)
    
    if not user_data["sites"]:
        await update.message.reply_text("❌ Primero agrega un sitio con /addsite")
        return
    
    if not user_data["proxies"]:
        await update.message.reply_text("❌ Primero agrega un proxy con /addproxy")
        return
    
    # Usar primer sitio y primer proxy
    site = user_data["sites"][0]
    proxy = user_data["proxies"][0]
    
    msg = await update.message.reply_text(
        f"🔄 Verificando...\n"
        f"📍 Sitio: {site}\n"
        f"🔌 Proxy: {proxy}"
    )
    
    # Verificar
    result = await SimpleChecker.check_card(card_data, site, proxy)
    
    if result["success"]:
        # Mostrar respuesta de la API
        response_text = (
            f"📊 *RESULTADO*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"💳 Tarjeta: `{card_data['bin']}xxxxxx{card_data['last4']}`\n"
            f"🌐 Sitio: {result['site']}\n"
            f"🔌 Proxy: {result['proxy']}\n"
            f"⏱️ Tiempo: {result['time']}s\n"
            f"📋 HTTP: {result['status_code']}\n\n"
            f"📝 *Respuesta API:*\n"
            f"```\n{result['response']}\n```"
        )
    else:
        response_text = (
            f"❌ *ERROR*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"💳 Tarjeta: `{card_data['bin']}xxxxxx{card_data['last4']}`\n"
            f"⏱️ Tiempo: {result['time']}s\n"
            f"⚠️ Error: {result.get('error', 'desconocido')}"
        )
    
    await msg.edit_text(response_text, parse_mode="Markdown")

async def list_sites(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lista los sitios guardados"""
    user_id = update.effective_user.id
    user_data = db.get_user_data(user_id)
    
    if not user_data["sites"]:
        await update.message.reply_text("📭 No tienes sitios guardados")
        return
    
    sites = "\n".join([f"{i+1}. {s}" for i, s in enumerate(user_data["sites"])])
    await update.message.reply_text(f"📋 *SITIOS GUARDADOS*\n{sites}", parse_mode="Markdown")

async def list_proxies(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lista los proxies guardados"""
    user_id = update.effective_user.id
    user_data = db.get_user_data(user_id)
    
    if not user_data["proxies"]:
        await update.message.reply_text("📭 No tienes proxies guardados")
        return
    
    proxies = "\n".join([f"{i+1}. `{p}`" for i, p in enumerate(user_data["proxies"])])
    await update.message.reply_text(f"📋 *PROXIES GUARDADOS*\n{proxies}", parse_mode="Markdown")

# ================== MANEJO DE ERRORES ==================
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja errores"""
    logger.error(f"Error: {context.error}")

# ================== MAIN ==================
def main():
    """Función principal"""
    app = Application.builder().token(Settings.TOKEN).build()
    
    # Handlers de comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("addsite", addsite))
    app.add_handler(CommandHandler("addproxy", addproxy))
    app.add_handler(CommandHandler("chk", chk))
    app.add_handler(CommandHandler("sites", list_sites))
    app.add_handler(CommandHandler("proxies", list_proxies))
    
    # Error handler
    app.add_error_handler(error_handler)
    
    logger.info("🚀 Bot simple iniciado")
    app.run_polling()

if __name__ == "__main__":
    main()
