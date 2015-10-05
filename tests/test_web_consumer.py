import json
import unittest
from meerkat.web_service import web_consumer
from tests.fixture import web_consumer_fixture


class WebConsumerTest(unittest.TestCase):
    """Our UnitTest class."""

    def setUp(self):
        self.consumer = web_consumer.Web_Consumer()
        return

    def tearDown(self):
        return

    def test_static_bank_category_map(self):
        transactions = web_consumer_fixture.get_transaction_bank_fallback_classifiable()
        self.consumer._Web_Consumer__apply_missing_categories(transactions)
        self.assertEqual(len(transactions[0]["category_labels"]), 1)
        self.assertEqual(transactions[0]["category_labels"][0], "Utilities")

    def test_static_card_category_map(self):
        transactions = web_consumer_fixture.get_transaction_card_fallback_classifiable()
        self.consumer._Web_Consumer__apply_missing_categories(transactions)
        self.assertEqual(len(transactions[0]["category_labels"]), 1)
        self.assertEqual(transactions[0]["category_labels"][0], "Restaurants/Dining")

    def test_static_subtype_category_map(self):
        transactions = web_consumer_fixture.get_transaction_subtype_fallback()
        self.consumer._Web_Consumer__apply_missing_categories(transactions)
        self.assertEqual(len(transactions[0]["category_labels"]), 1)
        self.assertEqual(transactions[0]["category_labels"][0], "Joseph Rules")

if __name__ == "__main__":
    unittest.main()
