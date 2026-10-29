from data import ORDERS, POLICIES, PRODUCTS


def test_product_fields_and_unique_skus():
    required = {"sku", "name", "brand", "category", "price", "mrp", "in_stock", "specs", "offers"}
    for p in PRODUCTS:
        assert required <= p.keys(), p["sku"]
        assert 0 < p["price"] <= p["mrp"], p["sku"]
    skus = [p["sku"] for p in PRODUCTS]
    assert len(skus) == len(set(skus))


def test_order_items_reference_real_products():
    skus = {p["sku"] for p in PRODUCTS}
    for oid, order in ORDERS.items():
        assert len(order["phone"]) == 10 and order["phone"].isdigit(), oid
        for item in order["items"]:
            assert item["sku"] in skus, oid


def test_policies_cover_all_topics():
    assert set(POLICIES) == {"returns", "refunds", "cancellation", "delivery", "warranty", "payment"}
    assert all(text.strip() for text in POLICIES.values())
