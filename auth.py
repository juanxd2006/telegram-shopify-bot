# auth.py
from telegram import Update
from telegram.ext import ContextTypes
from database import register_user, is_approved

async def register(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Registro de usuarios"""
    user = update.effective_user
    
    if is_approved(user.id):
        await update.message.reply_text(
            "✅ Ya estás registrado",
            parse_mode='Markdown'
        )
        return
    
    register_user(user.id, user.username, user.first_name)
    
    await update.message.reply_text(
        f"✅ **REGISTRO EXITOSO**\n\n"
        f"¡Bienvenido {user.first_name}!\n"
        f"Usa /start para comenzar",
        parse_mode='Markdown'
    )
