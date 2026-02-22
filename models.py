# models.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Site:
    site: str
    added_by: int
    added_date: datetime = None
    is_valid: bool = True
    times_used: int = 0
    success_count: int = 0
    products_found: int = 0

@dataclass
class Proxy:
    proxy: str
    added_by: int
    added_date: datetime = None
    is_alive: bool = True
    times_used: int = 0
    success_count: int = 0

@dataclass
class Result:
    cc: str
    site: str
    proxy: str
    status_code: int
    elapsed: float
    user_id: int
    product_found: bool = False
    price: float = None
    timestamp: datetime = None
