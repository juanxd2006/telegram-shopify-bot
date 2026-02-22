# database.py
import sqlite3
from contextlib import contextmanager
from config import DATABASE_FILE

@contextmanager
def get_db():
    """Context manager para conexiones a base de datos"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Inicializa la base de datos"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Tabla de usuarios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                approved BOOLEAN DEFAULT 0,
                registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP,
                role TEXT DEFAULT 'user'
            )
        ''')
        
        # Tabla de proxies
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proxies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proxy TEXT UNIQUE,
                added_by INTEGER,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_checked TIMESTAMP,
                is_alive BOOLEAN DEFAULT 1,
                response_time REAL,
                times_used INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                last_error TEXT,
                FOREIGN KEY (added_by) REFERENCES users(user_id)
            )
        ''')
        
        # Tabla de sitios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                site TEXT UNIQUE,
                added_by INTEGER,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_checked TIMESTAMP,
                last_response TEXT,
                response_time REAL,
                is_valid BOOLEAN DEFAULT 1,
                times_used INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                fail_count INTEGER DEFAULT 0,
                consecutive_failures INTEGER DEFAULT 0,
                last_error TEXT,
                products_found INTEGER DEFAULT 0,
                FOREIGN KEY (added_by) REFERENCES users(user_id)
            )
        ''')
        
        # Tabla de resultados
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cc TEXT,
                site TEXT,
                proxy TEXT,
                status_code INTEGER,
                response_type TEXT,
                response_preview TEXT,
                elapsed REAL,
                user_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                price REAL,
                product_found BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        conn.commit()

# === FUNCIONES DE USUARIOS ===
def is_approved(user_id):
    """Verifica si un usuario está aprobado"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT approved FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        return result is not None and result[0] == 1

def register_user(user_id, username, first_name):
    """Registra un nuevo usuario"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users (user_id, username, first_name, approved, registered_date)
            VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
        ''', (user_id, username, first_name))
        conn.commit()

def update_last_active(user_id):
    """Actualiza la última actividad del usuario"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?', (user_id,))
        conn.commit()

def get_user_count():
    """Obtiene el número total de usuarios aprobados"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users WHERE approved = 1')
        return cursor.fetchone()[0]

# === FUNCIONES DE PROXIES ===
def add_proxy(proxy, user_id):
    """Agrega un proxy a la base de datos"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO proxies (proxy, added_by)
                VALUES (?, ?)
            ''', (proxy, user_id))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def get_all_proxies(only_alive=True):
    """Obtiene todos los proxies"""
    with get_db() as conn:
        cursor = conn.cursor()
        if only_alive:
            cursor.execute('''
                SELECT proxy, added_date, times_used, success_count, fail_count, is_alive, last_error, response_time
                FROM proxies 
                WHERE is_alive = 1 
                ORDER BY success_count DESC, times_used ASC
            ''')
        else:
            cursor.execute('''
                SELECT proxy, added_date, times_used, success_count, fail_count, is_alive, last_error, response_time
                FROM proxies 
                ORDER BY is_alive DESC, success_count DESC
            ''')
        return cursor.fetchall()

def get_proxy_count():
    """Obtiene el conteo de proxies vivos y muertos"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM proxies WHERE is_alive = 1')
        alive = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM proxies')
        total = cursor.fetchone()[0]
        return alive, total - alive

def update_proxy_stats(proxy, success, response_time=None, error_msg=None):
    """Actualiza las estadísticas de un proxy"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE proxies 
            SET times_used = times_used + 1,
                success_count = success_count + ?,
                fail_count = fail_count + ?,
                last_checked = CURRENT_TIMESTAMP,
                response_time = COALESCE(?, response_time),
                last_error = COALESCE(?, last_error),
                is_alive = CASE WHEN ? = 1 THEN 1 ELSE is_alive END
            WHERE proxy = ?
        ''', (1 if success else 0, 0 if success else 1, response_time, error_msg, 1 if success else 0, proxy))
        conn.commit()

def mark_proxy_dead(proxy, error_msg=None):
    """Marca un proxy como muerto"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE proxies 
            SET is_alive = 0,
                last_checked = CURRENT_TIMESTAMP,
                last_error = COALESCE(?, last_error)
            WHERE proxy = ?
        ''', (error_msg, proxy))
        conn.commit()

# === FUNCIONES DE SITIOS ===
def add_site(site, user_id):
    """Agrega un sitio a la base de datos"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            if not site.startswith(('http://', 'https://')):
                site = 'https://' + site
            cursor.execute('''
                INSERT INTO sites (site, added_by)
                VALUES (?, ?)
            ''', (site, user_id))
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
    """Obtiene todos los sitios"""
    with get_db() as conn:
        cursor = conn.cursor()
        if only_valid:
            cursor.execute('''
                SELECT site, added_date, times_used, success_count, fail_count, is_valid, response_time, products_found
                FROM sites 
                WHERE is_valid = 1
                ORDER BY products_found DESC, success_count DESC
            ''')
        else:
            cursor.execute('''
                SELECT site, added_date, times_used, success_count, fail_count, is_valid, response_time, products_found
                FROM sites 
                ORDER BY is_valid DESC, products_found DESC
            ''')
        return cursor.fetchall()

def get_site_count():
    """Obtiene el conteo de sitios válidos e inválidos"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM sites WHERE is_valid = 1')
        valid = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM sites')
        total = cursor.fetchone()[0]
        return valid, total - valid

def get_products_ready_count():
    """Obtiene el número de sitios con productos detectados"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM sites WHERE is_valid = 1 AND products_found > 0')
        return cursor.fetchone()[0]

def update_site_stats(site, success, response_time=None, response_text=None, has_product=False):
    """Actualiza las estadísticas de un sitio"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        if success:
            consecutive_failures = 0
        else:
            cursor.execute('SELECT consecutive_failures FROM sites WHERE site = ?', (site,))
            row = cursor.fetchone()
            consecutive_failures = (row[0] if row else 0) + 1
        
        cursor.execute('''
            UPDATE sites 
            SET times_used = times_used + 1,
                success_count = success_count + ?,
                fail_count = fail_count + ?,
                last_checked = CURRENT_TIMESTAMP,
                response_time = COALESCE(?, response_time),
                last_response = ?,
                consecutive_failures = ?,
                products_found = products_found + ?,
                is_valid = CASE WHEN ? = 1 THEN 1 ELSE is_valid END
            WHERE site = ?
        ''', (
            1 if success else 0, 
            0 if success else 1, 
            response_time, 
            (response_text[:200] if response_text else None),
            consecutive_failures,
            1 if has_product else 0,
            1 if success else 0,
            site
        ))
        conn.commit()

def mark_site_invalid(site, error_msg=None):
    """Marca un sitio como inválido"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sites 
            SET is_valid = 0,
                last_checked = CURRENT_TIMESTAMP,
                last_error = COALESCE(?, last_error)
            WHERE site = ?
        ''', (error_msg, site))
        conn.commit()

# === FUNCIONES DE RESULTADOS ===
def add_result(cc, site, proxy, status_code, response_text, elapsed, user_id, price=None, product_found=False):
    """Agrega un resultado a la base de datos"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO results (cc, site, proxy, status_code, response_type, response_preview, elapsed, user_id, price, product_found)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            cc[:6] + '...' + cc[-4:],
            site,
            proxy,
            status_code,
            'PRODUCT' if product_found else 'RESPONSE',
            response_text[:200],
            elapsed,
            user_id,
            price,
            1 if product_found else 0
        ))
        conn.commit()
