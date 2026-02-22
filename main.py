# main.py
import logging
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters
from config import TOKEN, BOT_NAME
from database import init_database
from handlers import start, button_handler
from auth import register
from files import handle_document

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def main():
    """Punto de entrada principal"""
    print(f"╔════════════════════════════╗")
    print(f"║     🤖 {BOT_NAME}          ║")
    print(f"║    🚀 INICIANDO...         ║")
    print(f"╚════════════════════════════╝")
    
    # Inicializar BD
    init_database()
    
    # Crear aplicación
    app = Application.builder().token(TOKEN).build()
    
    # Comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("register", register))
    
    # Manejador de botones
    app.add_handler(CallbackQueryHandler(button_handler))
    
    # Manejador de archivos
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    print(f"✅ {BOT_NAME} listo para usar!")
    app.run_polling()

if __name__ == "__main__":
    main()
