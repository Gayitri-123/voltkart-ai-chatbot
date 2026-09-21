"""Sample store data. Replace these with calls to your real catalog, OMS and refund systems."""

PRODUCTS = [
    {
        "sku": "VK-TV-1001", "name": "Samsung 55\" Crystal 4K UHD Smart TV", "category": "Televisions",
        "brand": "Samsung", "price": 47990, "mrp": 64900, "in_stock": True,
        "specs": {"screen": "55 inch", "resolution": "3840x2160", "hdr": "HDR10+", "os": "Tizen", "warranty": "1 year"},
        "offers": ["10% instant discount on HDFC credit cards", "No-cost EMI up to 12 months"],
    },
    {
        "sku": "VK-MOB-2001", "name": "Apple iPhone 16 (128 GB, Black)", "category": "Mobiles",
        "brand": "Apple", "price": 74900, "mrp": 79900, "in_stock": True,
        "specs": {"display": "6.1 inch OLED", "chip": "A18", "camera": "48MP + 12MP", "storage": "128 GB", "warranty": "1 year"},
        "offers": ["Exchange bonus up to Rs 6,000"],
    },
    {
        "sku": "VK-MOB-2002", "name": "Redmi Note 14 5G (8 GB / 256 GB)", "category": "Mobiles",
        "brand": "Xiaomi", "price": 19999, "mrp": 24999, "in_stock": True,
        "specs": {"display": "6.67 inch AMOLED 120Hz", "battery": "5110 mAh", "camera": "50MP", "warranty": "1 year"},
        "offers": ["Rs 1,000 off on UPI payments"],
    },
    {
        "sku": "VK-LAP-3001", "name": "HP Pavilion 15 Laptop (Intel Core i5, 16 GB, 512 GB SSD)", "category": "Laptops",
        "brand": "HP", "price": 62990, "mrp": 78000, "in_stock": False,
        "specs": {"cpu": "Intel Core i5-1335U", "ram": "16 GB", "storage": "512 GB SSD", "display": "15.6 inch FHD", "warranty": "1 year"},
        "offers": [],
    },
    {
        "sku": "VK-AC-4001", "name": "LG 1.5 Ton 5 Star Dual Inverter Split AC", "category": "Air Conditioners",
        "brand": "LG", "price": 44990, "mrp": 75990, "in_stock": True,
        "specs": {"capacity": "1.5 ton", "energy_rating": "5 star", "compressor": "Dual Inverter", "warranty": "1 year product, 10 years compressor"},
        "offers": ["Free standard installation"],
    },
    {
        "sku": "VK-AUD-5001", "name": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones", "category": "Audio",
        "brand": "Sony", "price": 29990, "mrp": 34990, "in_stock": True,
        "specs": {"battery": "30 hours", "anc": "Yes", "connectivity": "Bluetooth 5.2", "warranty": "1 year"},
        "offers": [],
    },
]

ORDERS = {
    "ORD10001": {
        "phone": "9876543210", "status": "Delivered", "placed_on": "2026-09-02", "delivered_on": "2026-09-05",
        "items": [{"sku": "VK-AUD-5001", "qty": 1, "price": 29990}], "payment": "UPI",
    },
    "ORD10002": {
        "phone": "9876543210", "status": "Out for delivery", "placed_on": "2026-09-14", "expected_delivery": "2026-09-17",
        "items": [{"sku": "VK-MOB-2002", "qty": 1, "price": 19999}], "payment": "Credit Card",
    },
    "ORD10003": {
        "phone": "9123456780", "status": "Return requested", "placed_on": "2026-08-20", "delivered_on": "2026-08-24",
        "items": [{"sku": "VK-TV-1001", "qty": 1, "price": 47990}], "payment": "Debit Card",
        "refund": {"status": "Pickup scheduled", "amount": 47990, "method": "Original payment method",
                   "note": "Refund is initiated after the product passes quality check at pickup."},
    },
}

POLICIES = {
    "returns": "Most products can be returned within 7 days of delivery if damaged, defective, or different from what was ordered. "
               "Mobiles, laptops and tablets are replacement-only within 7 days for manufacturing defects. "
               "The product must be unused, with original box, accessories, and invoice.",
    "refunds": "Refunds start after the returned product passes quality check. UPI and wallets: 1-3 business days. "
               "Credit/debit cards and net banking: 5-7 business days. Cash on delivery: refunded to a bank account in 5-7 business days.",
    "cancellation": "Orders can be cancelled free of charge before they are shipped. Once shipped, refuse delivery or request a return.",
    "delivery": "Standard delivery takes 2-5 business days in most cities. Large appliances include free delivery; installation depends on the product.",
    "warranty": "All products carry the manufacturer's warranty. Extended warranty plans can be bought at checkout or within 30 days of purchase.",
    "payment": "Accepted: UPI, credit/debit cards, net banking, EMI (including no-cost EMI on selected products), and cash on delivery for orders under Rs 50,000.",
}
