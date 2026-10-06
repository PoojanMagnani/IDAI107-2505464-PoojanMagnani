import pytest
from stocksense.logic import stock_status, summarize


@pytest.mark.parametrize("count,expected",[(0,"Out of stock"),(1,"Low stock"),(3,"Low stock"),(4,"In stock")])
def test_threshold_boundaries(count, expected):
    assert stock_status(count) == expected


def test_absent_expected_products_are_not_dropped():
    rows=summarize([dict(category="milk",confidence=.8)], ["milk","water"])
    assert rows[0]["category"] == "water"
    assert rows[0]["count"] == 0
    assert rows[0]["suggested_top_up"] == 6
    assert rows[1]["count"] == 1


def test_unexpected_categories_do_not_generate_stock_alerts():
    assert len(summarize([], ["coffee"])) == 1
    assert summarize([dict(category="water",confidence=.9)], ["coffee"])[0]["count"] == 0


def test_invalid_target_rejected():
    with pytest.raises(ValueError):
        summarize([], ["milk"],3,3)
