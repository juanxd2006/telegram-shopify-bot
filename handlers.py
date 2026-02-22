# handlers.py
from telegram import Update
from telegram.ext import ContextTypes
from database import *
from ui import *
from config import BOT_NAME
import requests

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start - Muestra el menú principal"""
    user = update.effective_user
    
    if not is_approved(user.id):
        await update.message.reply_text(
            f"# {BOT_NAME}\n\n"
            f"❌ ACCESO DENEGADO\n\n"
            f"• Usa /register para registrarte",
            parse_mode='Markdown'
        )
        return
    
    # Obtener estadísticas
    proxies_alive, proxies_dead = get_proxy_count()
    sites_valid, sites_invalid = get_site_count()
    products_ready = get_products_ready_count()
    user_count = get_user_count()
    
    # Obtener todos los proxies para contar rotating/static
    proxies = get_all_proxies(only_alive=True)
    rotating = sum(1 for p in proxies if 'rotate' in p[0].lower()) if proxies else 0
    static = proxies_alive - rotating
    
    # Crear mensaje con UI
    message = (
        f"# {BOT_NAME}\n"
        f"👥 {user_count} usuarios\n\n"
        f"📊 **RESUMEN**\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Proxies vivos: {proxies_alive}\n"
        f"🔄 Rotating: {rotating} | 🖥️ Static: {static}\n"
        f"🌐 Sitios válidos: {sites_valid}\n"
        f"📦 Productos listos: {products_ready}/{sites_valid}\n\n"
        f"Selecciona una opción:"
    )
    
    await update.message.reply_text(
        message,
        parse_mode='Markdown',
        reply_markup=get_main_menu()
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja todos los botones de los menús"""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'back_main':
        # Volver al menú principal
        proxies_alive, proxies_dead = get_proxy_count()
        sites_valid, sites_invalid = get_site_count()
        products_ready = get_products_ready_count()
        user_count = get_user_count()
        proxies = get_all_proxies(only_alive=True)
        rotating = sum(1 for p in proxies if 'rotate' in p[0].lower()) if proxies else 0
        static = proxies_alive - rotating
        
        message = (
            f"# {BOT_NAME}\n"
            f"👥 {user_count} usuarios\n\n"
            f"📊 **RESUMEN**\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🟢 Proxies vivos: {proxies_alive}\n"
            f"🔄 Rotating: {rotating} | 🖥️ Static: {static}\n"
            f"🌐 Sitios válidos: {sites_valid}\n"
            f"📦 Productos listos: {products_ready}/{sites_valid}\n\n"
            f"Selecciona una opción:"
        )
        
        await query.edit_message_text(
            message,
            parse_mode='Markdown',
            reply_markup=get_main_menu()
        )
    
    elif query.data == 'menu_sites':
        await query.edit_message_text(
            "🌐 **MENÚ DE SITIOS SHOPIFY**\n\n"
            "Selecciona una opción:",
            parse_mode='Markdown',
            reply_markup=get_sites_menu()
        )
    
    elif query.data == 'site_add':
        await query.edit_message_text(
            "➕ **AGREGAR SITIO SHOPIFY**\n\n"
            "**Uso:** /addsh <url>\n\n"
            "**Ejemplo:**\n"
            "`/addsh store.myshopify.com`\n\n"
            "También puedes subir un archivo `.txt` con varios sitios\n\n"
            "Presiona 'VOLVER' para regresar al menú anterior",
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_sites')
        )
    
    elif query.data == 'site_list':
        sites = get_all_sites(only_valid=False)
        if not sites:
            await query.edit_message_text(
                "📭 No hay sitios guardados",
                parse_mode='Markdown',
                reply_markup=get_back_button('menu_sites')
            )
            return
        
        msg = "📋 **SITIOS GUARDADOS**\n\n"
        for site_data in sites[:10]:
            site = site_data[0]
            is_valid = site_data[5]
            products = site_data[7]
            status = "✅" if is_valid else "❌"
            product_icon = "📦" if products > 0 else "⏳"
            display_site = site if len(site) < 40 else site[:37] + "..."
            msg += f"{status} {product_icon} `{display_site}`\n"
        
        if len(sites) > 10:
            msg += f"\n... y {len(sites) - 10} más"
        
        await query.edit_message_text(
            msg,
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_sites')
        )
    
    elif query.data == 'site_validate':
        await query.edit_message_text(
            "🔄 **VALIDANDO SITIOS**\n\n"
            "Usa el comando /validatesh para comenzar la validación.\n\n"
            "Este proceso puede tomar varios minutos dependiendo de la cantidad de sitios.",
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_sites')
        )
    
    elif query.data == 'menu_proxies':
        await query.edit_message_text(
            "🔒 **MENÚ DE PROXIES**\n\n"
            "Selecciona una opción:",
            parse_mode='Markdown',
            reply_markup=get_proxies_menu()
        )
    
    elif query.data == 'proxy_add':
        await query.edit_message_text(
            "➕ **AGREGAR PROXY**\n\n"
            "**Uso:** /addrproxy <proxy>\n\n"
            "**Formatos aceptados:**\n"
            "• `host:port:user:pass`\n"
            "• `user:pass@host:port`\n"
            "• `host:port`\n\n"
            "**Ejemplos:**\n"
            "`/addrproxy 45.155.88.66:7497:user:pass`\n"
            "`/addrproxy user:pass@45.155.88.66:7497`\n\n"
            "También puedes subir un archivo `.txt` con varios proxies",
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_proxies')
        )
    
    elif query.data == 'proxy_list':
        proxies = get_all_proxies(only_alive=False)
        if not proxies:
            await query.edit_message_text(
                "📭 No hay proxies guardados",
                parse_mode='Markdown',
                reply_markup=get_back_button('menu_proxies')
            )
            return
        
        alive, dead = get_proxy_count()
        msg = f"🔒 **PROXIES GUARDADOS**\n"
        msg += f"🟢 Vivos: {alive} | 🔴 Muertos: {dead}\n\n"
        
        for proxy_data in proxies[:10]:
            proxy = proxy_data[0]
            is_alive = proxy_data[5]
            response_time = proxy_data[7]
            status = "🟢" if is_alive else "🔴"
            time_str = f" ⚡ {response_time}s" if response_time and is_alive else ""
            display_proxy = proxy if len(proxy) < 40 else proxy[:37] + "..."
            msg += f"{status} `{display_proxy}`{time_str}\n"
        
        if len(proxies) > 10:
            msg += f"\n... y {len(proxies) - 10} más"
        
        await query.edit_message_text(
            msg,
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_proxies')
        )
    
    elif query.data == 'proxy_check':
        await query.edit_message_text(
            "🔄 **VERIFICANDO PROXIES**\n\n"
            "Usa el comando /healthcheck para verificar todos los proxies.\n\n"
            "Los proxies muertos serán marcados automáticamente.",
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_proxies')
        )
    
    elif query.data == 'menu_checks':
        await query.edit_message_text(
            "💳 **MENÚ DE CHECKS**\n\n"
            "Selecciona una opción:",
            parse_mode='Markdown',
            reply_markup=get_checks_menu()
        )
    
    elif query.data == 'check_start':
        await query.edit_message_text(
            "▶️ **INICIAR SHOPIFY CHECK**\n\n"
            "**Requisitos:**\n"
            "• Tener sitios válidos guardados\n"
            "• Tener proxies vivos\n"
            "• Subir un archivo con tarjetas\n\n"
            "**Pasos:**\n"
            "1. Sube un archivo `.txt` con tarjetas\n"
            "2. Usa el comando /shcheck\n\n"
            "El proceso mostrará el progreso en tiempo real.",
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_checks')
        )
    
    elif query.data == 'check_stop':
        await query.edit_message_text(
            "⏹️ **DETENER CHECK**\n\n"
            "Usa el comando /shstop para detener cualquier proceso en ejecución.",
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_checks')
        )
    
    elif query.data == 'menu_stats':
        await query.edit_message_text(
            "📊 **MENÚ DE ESTADÍSTICAS**\n\n"
            "Selecciona una opción:",
            parse_mode='Markdown',
            reply_markup=get_stats_menu()
        )
    
    elif query.data == 'stats_my':
        user_id = update.effective_user.id
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM results WHERE user_id = ?', (user_id,))
            total = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM results WHERE user_id = ? AND product_found = 1', (user_id,))
            products = cursor.fetchone()[0]
        
        await query.edit_message_text(
            f"📊 **TUS ESTADÍSTICAS**\n\n"
            f"• Total CHKs: {total}\n"
            f"• Productos encontrados: {products}\n"
            f"• Tasa de éxito: {(products/total*100) if total > 0 else 0:.1f}%",
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_stats')
        )
    
    elif query.data == 'stats_global':
        sites_valid, sites_invalid = get_site_count()
        proxies_alive, proxies_dead = get_proxy_count()
        products_ready = get_products_ready_count()
        user_count = get_user_count()
        
        await query.edit_message_text(
            f"📊 **ESTADÍSTICAS GLOBALES**\n\n"
            f"👥 Usuarios: {user_count}\n"
            f"🟢 Proxies vivos: {proxies_alive}\n"
            f"🔴 Proxies muertos: {proxies_dead}\n"
            f"✅ Sitios válidos: {sites_valid}\n"
            f"❌ Sitios inválidos: {sites_invalid}\n"
            f"📦 Productos listos: {products_ready}",
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_stats')
        )
    
    elif query.data == 'menu_files':
        await query.edit_message_text(
            "📁 **MENÚ DE ARCHIVOS**\n\n"
            "Selecciona una opción:",
            parse_mode='Markdown',
            reply_markup=get_files_menu()
        )
    
    elif query.data == 'file_help':
        await query.edit_message_text(
            "📁 **INSTRUCCIONES PARA ARCHIVOS**\n\n"
            "**Formatos aceptados:**\n\n"
            "**Sitios (.txt):**\n"
            "• Una URL por línea\n"
            "• Ejemplo: `store.myshopify.com`\n\n"
            "**Proxies (.txt):**\n"
            "• Formatos: host:port o user:pass@host:port\n"
            "• Ejemplo: `45.155.88.66:7497:user:pass`\n\n"
            "**Tarjetas (.txt):**\n"
            "• Formato: número|mes|año|cvv\n"
            "• Ejemplo: `4111111111111111|12|2025|123`",
            parse_mode='Markdown',
            reply_markup=get_back_button('menu_files')
        )
    
    elif query.data == 'menu_help':
        await query.edit_message_text(
            "❓ **AYUDA**\n\n"
            "**Comandos disponibles:**\n"
            "• /start - Menú principal\n"
            "• /register - Registrarse\n"
            "• /addsh - Agregar sitio\n"
            "• /addrproxy - Agregar proxy\n"
            "• /mysh - Ver sitios\n"
            "• /myproxy - Ver proxies\n"
            "• /validatesh - Validar sitios\n"
            "• /shcheck - Iniciar check\n"
            "• /shstop - Detener check\n"
            "• /stats - Ver estadísticas\n\n"
            "También puedes usar los botones del menú para navegar.",
            parse_mode='Markdown',
            reply_markup=get_back_button('main')
        )
