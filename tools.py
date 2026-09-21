"""Tools the model can call. Each handler returns a JSON-serialisable dict."""

from data import ORDERS, POLICIES, PRODUCTS

TOOLS = [
    {
        "name": "search_products",
        "description": "Search the store catalog by keyword, category, brand, and/or maximum price. "
                       "Use this whenever the customer asks about products, prices, availability or recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text keywords, e.g. 'noise cancelling headphones'"},
                "category": {"type": "string", "description": "e.g. Televisions, Mobiles, Laptops, Air Conditioners, Audio"},
                "brand": {"type": "string"},
                "max_price": {"type": "number", "description": "Maximum price in rupees"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_product_details",
        "description": "Get full specs, price, offers and stock for one product by SKU.",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_order_status",
        "description": "Look up an order's status, items, delivery and refund details. "
                       "Requires the order ID and the phone number used for the order, for verification.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "e.g. ORD10001"},
                "phone": {"type": "string", "description": "10-digit registered mobile number"},
            },
            "required": ["order_id", "phone"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_policy",
        "description": "Get store policy text for returns, refunds, cancellation, delivery, warranty or payment.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string", "enum": sorted(POLICIES)}},
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
]


def _product_summary(p):
    return {k: p[k] for k in ("sku", "name", "brand", "category", "price", "mrp", "in_stock")}


def search_products(query="", category=None, brand=None, max_price=None):
    words = [w for w in query.lower().split() if len(w) > 2]
    results = []
    for p in PRODUCTS:
        if category and category.lower() not in p["category"].lower():
            continue
        if brand and brand.lower() != p["brand"].lower():
            continue
        if max_price is not None and p["price"] > max_price:
            continue
        haystack = " ".join([p["name"], p["category"], p["brand"], *map(str, p["specs"].values())]).lower()
        if words and not any(w in haystack for w in words):
            continue
        results.append(_product_summary(p))
    return {"count": len(results), "products": results}


def get_product_details(sku):
    for p in PRODUCTS:
        if p["sku"].lower() == sku.lower():
            return p
    return {"error": f"No product with SKU {sku}"}


def get_order_status(order_id, phone):
    order = ORDERS.get(order_id.strip().upper())
    if not order or order["phone"] != "".join(ch for ch in phone if ch.isdigit())[-10:]:
        return {"error": "No order found for that order ID and phone number."}
    names = {p["sku"]: p["name"] for p in PRODUCTS}
    result = {k: v for k, v in order.items() if k != "phone"}
    result["items"] = [{**i, "name": names.get(i["sku"], i["sku"])} for i in order["items"]]
    return {"order_id": order_id.upper(), **result}


def get_policy(topic):
    return {"topic": topic, "policy": POLICIES.get(topic, "Unknown topic")}


HANDLERS = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "get_order_status": get_order_status,
    "get_policy": get_policy,
}


def run_tool(name, tool_input):
    handler = HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool {name}"}, True
    try:
        return handler(**tool_input), False
    except TypeError as e:
        return {"error": f"Invalid input: {e}"}, True
