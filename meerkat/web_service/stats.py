class Stats():
    __stats = {
        "total": 0,
        "factual_searches": 0,
        "factual_matches": 0,
        "bloom_locations_found": 0,
        "subtype_found": 0,
        "subtype_only": 0,
        "bank": {
            "total": 0,
            "merchant_found": 0,
            "credit": {
                "total": 0,
                "subtype_found": 0,
            },
            "debit": {
                "total": 0,
                "subtype_found": 0
            }
        },
        "card": {
            "total": 0,
            "merchant_found": 0,
            "credit": {
                "total": 0,
                "subtype_found": 0
            },
            "debit": {
                "total": 0,
                "subtype_found": 0
            }
        }
    }

    __types = ["credit", "debit"]

    def add_stats(self, transactions, container):
        if not transactions or len(transactions) == 0:
            return

        container = container.lower()
        event = {
            container: {}
        }
        for t_type in self.__types:
            event[container][t_type] = {}

        event["total"] = len(transactions)
        event[container]["total"] = event["total"]
        self.__stats["total"] += event["total"]
        self.__stats[container]["total"] += event["total"]

        event["factual_searches"] = len([t for t in transactions if t.get("is_physical_merchant")])
        self.__stats["factual_searches"] += event["factual_searches"]

        event["factual_matches"] = len([t for t in transactions if t.get("match_found")])
        self.__stats["factual_matches"] += event["factual_matches"]

        event["bloom_locations_found"] = len([t for t in transactions if t.get("locale_bloom")])
        self.__stats["bloom_locations_found"] += event["bloom_locations_found"]

        event["subtype_found"] = len([t for t in transactions if t.get("txn_sub_type")])
        self.__stats["subtype_found"] += event["subtype_found"]

        event["subtype_only"] = len([t for t in transactions if not t.get("is_physical_merchant") and not t.get("match_found") and not t.get("locale_bloom")])
        self.__stats["subtype_only"] += event["subtype_only"]

        event[container]["merchant_found"] = len([t for t in transactions if t.get("CNN")])
        self.__stats[container]["merchant_found"] += event[container]["merchant_found"]

        for t_type in self.__types:
            event[container][t_type]["total"] = len([t for t in transactions if t["ledger_entry"].lower() == t_type])
            self.__stats[container][t_type]["total"] += event[container][t_type]["total"]
            event[container][t_type]["subtype_found"] = len([t for t in transactions if t["ledger_entry"].lower() == t_type and t.get("cnn_type_found")])
            self.__stats[container][t_type]["subtype_found"] += event[container][t_type]["subtype_found"]

    def get_stats(self):
        return self.__stats
