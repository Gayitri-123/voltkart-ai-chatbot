import pytest

from tools import HANDLERS, TOOLS, get_order_status, get_policy, get_product_details, run_tool, search_products


def test_every_tool_has_a_handler():
    assert {t["name"] for t in TOOLS} == set(HANDLERS)


@pytest.mark.parametrize("kwargs, expected_skus", [
    ({"query": "noise cancelling headphones"}, ["VK-AUD-5001"]),
    ({"category": "mobiles"}, ["VK-MOB-2001", "VK-MOB-2002"]),
    ({"brand": "SAMSUNG"}, ["VK-TV-1001"]),
    ({"category": "Mobiles", "max_price": 20000}, ["VK-MOB-2002"]),
    ({"query": "AMOLED"}, ["VK-MOB-2002"]),  # matches spec values
    ({"query": "toaster"}, []),
])
def test_search_products(kwargs, expected_skus):
    result = search_products(**kwargs)
    assert [p["sku"] for p in result["products"]] == expected_skus
    assert result["count"] == len(expected_skus)


def test_search_returns_summary_only():
    product = search_products(query="iphone")["products"][0]
    assert set(product) == {"sku", "name", "brand", "category", "price", "mrp", "in_stock"}


def test_search_with_no_filters_returns_everything():
    assert search_products()["count"] == 6


def test_get_product_details_case_insensitive():
    assert get_product_details("vk-lap-3001")["in_stock"] is False
    assert "error" in get_product_details("NOPE")


@pytest.mark.parametrize("order_id, phone", [
    ("ORD10001", "9876543210"),
    (" ord10001 ", "+91 98765-43210"),  # normalised id and phone
])
def test_get_order_status_ok(order_id, phone):
    result = get_order_status(order_id, phone)
    assert result["status"] == "Delivered"
    assert "phone" not in result
    assert result["items"][0]["name"].startswith("Sony")


def test_get_order_status_order_id_is_normalised_in_result():
    assert get_order_status(" ord10001 ", "9876543210")["order_id"] == "ORD10001"


@pytest.mark.parametrize("order_id, phone", [("ORD10001", "9000000000"), ("ORD99999", "9876543210")])
def test_get_order_status_rejects_bad_lookup(order_id, phone):
    assert get_order_status(order_id, phone) == {"error": "No order found for that order ID and phone number."}


def test_get_order_status_refund_details():
    assert get_order_status("ORD10003", "9123456780")["refund"]["amount"] == 47990


def test_get_policy():
    assert "7 days" in get_policy("returns")["policy"]
    assert get_policy("nope")["policy"] == "Unknown topic"


def test_run_tool_errors():
    assert run_tool("missing", {}) == ({"error": "Unknown tool missing"}, True)
    output, is_error = run_tool("get_order_status", {"order_id": "ORD10001"})
    assert is_error and "Invalid input" in output["error"]
    output, is_error = run_tool("get_policy", {"topic": "returns"})
    assert not is_error and output["topic"] == "returns"
