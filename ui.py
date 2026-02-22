# ui.py
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def get_main_menu():
    """Menú principal"""
    keyboard = [
        [InlineKeyboardButton("🌐 SITIOS SHOPIFY", callback_data='menu_sites')],
        [InlineKeyboardButton("🔒 PROXIES", callback_data='menu_proxies')],
        [InlineKeyboardButton("💳 CHECKS", callback_data='menu_checks')],
        [InlineKeyboardButton("📊 ESTADÍSTICAS", callback_data='menu_stats')],
        [InlineKeyboardButton("📁 ARCHIVOS", callback_data='menu_files')],
        [InlineKeyboardButton("❓ AYUDA", callback_data='menu_help')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_sites_menu():
    """Submenú de sitios Shopify"""
    keyboard = [
        [InlineKeyboardButton("➕ AGREGAR SITIO", callback_data='site_add')],
        [InlineKeyboardButton("📋 VER SITIOS", callback_data='site_list')],
        [InlineKeyboardButton("✅ VALIDAR SITIOS", callback_data='site_validate')],
        [InlineKeyboardButton("⬅️ VOLVER", callback_data='back_main')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_proxies_menu():
    """Submenú de proxies"""
    keyboard = [
        [InlineKeyboardButton("➕ AGREGAR PROXY", callback_data='proxy_add')],
        [InlineKeyboardButton("📋 VER PROXIES", callback_data='proxy_list')],
        [InlineKeyboardButton("🔄 VERIFICAR PROXIES", callback_data='proxy_check')],
        [InlineKeyboardButton("⬅️ VOLVER", callback_data='back_main')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_checks_menu():
    """Submenú de checks"""
    keyboard = [
        [InlineKeyboardButton("▶️ INICIAR SHOPIFY CHECK", callback_data='check_start')],
        [InlineKeyboardButton("⏹️ DETENER CHECK", callback_data='check_stop')],
        [InlineKeyboardButton("⬅️ VOLVER", callback_data='back_main')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_stats_menu():
    """Submenú de estadísticas"""
    keyboard = [
        [InlineKeyboardButton("📊 MIS ESTADÍSTICAS", callback_data='stats_my')],
        [InlineKeyboardButton("🌐 ESTADÍSTICAS GLOBALES", callback_data='stats_global')],
        [InlineKeyboardButton("⬅️ VOLVER", callback_data='back_main')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_files_menu():
    """Submenú de archivos"""
    keyboard = [
        [InlineKeyboardButton("📋 VER INSTRUCCIONES", callback_data='file_help')],
        [InlineKeyboardButton("⬅️ VOLVER", callback_data='back_main')]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_back_button(menu):
    """Botón para volver a un menú específico"""
    keyboard = [[InlineKeyboardButton("⬅️ VOLVER", callback_data=menu)]]
    return InlineKeyboardMarkup(keyboard)
