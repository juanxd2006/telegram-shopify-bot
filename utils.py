# utils.py
import re

def parse_cc_line(line: str) -> str:
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
            if parts[1].isdigit() and parts[2].isdigit():
                return f"{parts[0]}|{parts[1]}|{parts[2]}|{parts[3]}"
    return None

def analyze_shopify_response(response_text: str) -> dict:
    """Analiza la respuesta de Shopify"""
    result = {
        'has_product': False,
        'price': None,
        'product_found': False,
        'message': None
    }
    
    text = response_text.lower()
    
    # Detectar precio
    price_match = re.search(r'"price":\s*"?\$?([0-9.]+)"?', response_text, re.IGNORECASE)
    if price_match:
        result['has_product'] = True
        result['product_found'] = True
        result['price'] = price_match.group(1)
        result['message'] = f"💰 Producto encontrado - ${result['price']}"
    
    # Detectar otros patrones de Shopify
    elif 'product id is empty' in text:
        result['message'] = "📦 Product ID empty"
    elif 'captcha_required' in text:
        result['message'] = "🛡️ CAPTCHA requerido"
    elif 'del ammount empty' in text:
        result['message'] = "📦 Del amount empty"
    elif 'generic_error' in text:
        result['message'] = "⚠️ Generic error"
    
    return result

def format_card(card: str) -> str:
    """Formatea una tarjeta para mostrar (oculta dígitos medios)"""
    if '|' in card:
        parts = card.split('|')
        if len(parts[0]) >= 15:
            return f"{parts[0][:6]}xxxxxx{parts[0][-4:]}|{parts[1]}|{parts[2]}|{parts[3]}"
    return card

# ===== FUNCIONES DE GENERACIÓN DE CCS =====
def luhn_checksum(card_number):
    """Calcula el dígito de verificación Luhn"""
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
    """Calcula el dígito Luhn y devuelve el número completo"""
    checksum = luhn_checksum(int(card_number) * 10)
    return int(card_number) * 10 + ((10 - checksum) % 10)

def generate_cc(bin_pattern: str, month: str = None, year: str = None, cvv: str = None, count: int = 10) -> list:
    """
    Genera CCs basadas en un patrón
    Ejemplo: "451993217159xxxx|02|2030|rnd"
    """
    results = []
    
    parts = bin_pattern.split('|')
    if len(parts) >= 1:
        card_pattern = parts[0]
    else:
        card_pattern = bin_pattern
    
    # Determinar mes
    if month:
        use_month = month
    elif len(parts) >= 2:
        use_month = parts[1]
    else:
        use_month = str(random.randint(1, 12)).zfill(2)
    
    # Determinar año
    if year:
        use_year = year
    elif len(parts) >= 3:
        use_year = parts[2]
    else:
        use_year = str(random.randint(2025, 2030))
    
    # Determinar CVV
    if cvv:
        use_cvv = cvv
    elif len(parts) >= 4:
        use_cvv = parts[3]
    else:
        use_cvv = "rnd"
    
    for _ in range(count):
        # Generar número de tarjeta
        if 'x' in card_pattern.lower():
            card_temp = card_pattern.lower()
            x_count = card_temp.count('x')
            
            # Generar dígitos aleatorios para las x
            for _ in range(x_count):
                card_temp = card_temp.replace('x', str(random.randint(0, 9)), 1)
            
            # Calcular Luhn si tiene 15 dígitos (necesita el último)
            if len(card_temp) == 15:
                base = card_temp
                card_number = calculate_luhn(base)
                card_temp = str(card_number)
        else:
            card_temp = card_pattern
        
        # Generar CVV
        if use_cvv.lower() == 'rnd':
            cvv_gen = str(random.randint(0, 999)).zfill(3)
        else:
            cvv_gen = use_cvv
        
        results.append({
            'cc': f"{card_temp}|{use_month}|{use_year}|{cvv_gen}",
            'month': use_month,
            'year': use_year,
            'cvv': cvv_gen,
            'number': card_temp
        })
    
    return results
