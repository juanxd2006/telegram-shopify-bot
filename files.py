# files.py
import os
from telegram import Update
from telegram.ext import ContextTypes
from database import add_sites_bulk, add_proxy, get_site_count, get_proxy_count
from utils import parse_cc_line  # Ahora importa del archivo, no de la carpeta

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Procesa archivos TXT"""
    document = update.message.document
    
    if not document.file_name.endswith('.txt'):
        await update.message.reply_text("❌ Solo archivos .txt")
        return
    
    processing_msg = await update.message.reply_text("🔄 Procesando archivo...")
    
    file = await context.bot.get_file(document.file_id)
    await file.download_to_drive('temp.txt')
    
    with open('temp.txt', 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        await processing_msg.edit_text("❌ Archivo vacío")
        os.remove('temp.txt')
        return
    
    file_name = document.file_name.lower()
    
    if 'site' in file_name or any('shop' in line or '.com' in line for line in lines[:5]):
        # Archivo de sitios
        added, dup = add_sites_bulk(lines, update.effective_user.id)
        sites_valid, sites_invalid = get_site_count()
        await processing_msg.edit_text(
            f"✅ **SITIOS AGREGADOS**\n\n"
            f"• Total: {len(lines)}\n"
            f"• Agregados: {added}\n"
            f"• Duplicados: {dup}\n"
            f"• Total sitios: ✅ {sites_valid} | ❌ {sites_invalid}"
        )
    
    elif 'proxy' in file_name or any(':' in line for line in lines[:5]):
        # Archivo de proxies
        added = 0
        for proxy in lines:
            if add_proxy(proxy, update.effective_user.id):
                added += 1
        alive, dead = get_proxy_count()
        await processing_msg.edit_text(
            f"✅ **PROXIES AGREGADOS**\n\n"
            f"• Total: {len(lines)}\n"
            f"• Agregados: {added}\n"
            f"• Total proxies: 🟢 {alive} | 🔴 {dead}"
        )
    
    else:
        # Archivo de CCs
        valid_cards = []
        for line in lines:
            cc = parse_cc_line(line)
            if cc:
                valid_cards.append(cc)
        
        if valid_cards:
            context.user_data['pending_cards'] = valid_cards[:1000]
            await processing_msg.edit_text(
                f"✅ **TARJETAS CARGADAS**\n\n"
                f"• Total líneas: {len(lines)}\n"
                f"• Tarjetas válidas: {len(valid_cards[:1000])}\n"
                f"• En cola: {min(1000, len(valid_cards))}\n\n"
                f"Usa /shcheck para comenzar"
            )
        else:
            await processing_msg.edit_text("❌ No se encontraron tarjetas válidas")
    
    os.remove('temp.txt')
