import unittest
import time
from http.client import HTTPConnection
from threading import Thread
from http.server import ThreadingHTTPServer

from app import AutoSearch, SchoolModel, create_handler, validate_schools


class SchoolModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SchoolModel()

    def test_baseline_matches_saved_blocksnet_results(self):
        result = self.model.calculate([])
        self.assertEqual(result["summary"]["base_deficit"], 521)
        self.assertEqual(result["summary"]["deficit"], 521)
        self.assertEqual(result["summary"]["base_capacity"], 1176)

    def test_new_capacity_is_added_to_the_selected_block_only(self):
        block_id = 293
        before = self.model.inputs["capacity"].copy(deep=True)
        baseline = self.model.calculate([])
        scenario = self.model.calculate([{"block_id": block_id, "capacity": 600}])
        rows = {row["block_id"]: row for row in scenario["rows"]}
        self.assertEqual(rows[block_id]["capacity"], rows[block_id]["base_capacity"] + 600)
        self.assertEqual(scenario["summary"]["added_capacity"], 600)
        self.assertEqual(baseline["summary"]["capacity"], 1176)
        self.assertTrue(self.model.inputs["capacity"].equals(before))

    def test_invalid_capacity_is_rejected(self):
        for value in (0, -1, 1.5, True, "600"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_schools([{"block_id": 0, "capacity": value}], self.model.block_ids)

    def test_two_school_scenario_adds_both_capacities(self):
        result = self.model.calculate(
            [{"block_id": 10, "capacity": 300}, {"block_id": 20, "capacity": 300}]
        )
        self.assertEqual(result["summary"]["capacity"], 1776)
        self.assertEqual(result["summary"]["added_capacity"], 600)

    def test_unknown_block_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Неизвестный квартал"):
            self.model.calculate([{"block_id": -1, "capacity": 600}])

    def test_api_allows_pages_preflight_and_rejects_other_origins(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(self.model, AutoSearch(self.model)))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request(
                "OPTIONS",
                "/api/calculate",
                headers={
                    "Origin": "https://annamironets.github.io",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            allowed = connection.getresponse()
            self.assertEqual(allowed.status, 204)
            self.assertEqual(allowed.getheader("Access-Control-Allow-Origin"), "https://annamironets.github.io")
            allowed.read()

            connection.request(
                "OPTIONS",
                "/api/calculate",
                headers={"Origin": "https://example.invalid", "Access-Control-Request-Method": "POST"},
            )
            rejected = connection.getresponse()
            self.assertEqual(rejected.status, 403)
            self.assertIsNone(rejected.getheader("Access-Control-Allow-Origin"))
            rejected.read()
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_auto_search_checks_each_block_independently_and_keeps_ties(self):
        class FakeModel:
            block_ids = {1, 2, 3}

            def __init__(self):
                self.calls = []

            def calculate(self, schools):
                self.calls.append(schools)
                block_id = schools[0]["block_id"]
                reduction = {1: 5, 2: 5, 3: 2}[block_id]
                return {
                    "summary": {
                        "base_deficit": 10,
                        "deficit": 10 - reduction,
                        "deficit_reduction": reduction,
                    }
                }

        model = FakeModel()
        search = AutoSearch(model)
        job_id = search.start(600)["job_id"]
        while not search.status(job_id)["done"]:
            time.sleep(0.01)

        result = search.status(job_id)
        self.assertEqual(result["checked"], 3)
        self.assertEqual([call[0]["block_id"] for call in model.calls], [1, 2, 3])
        self.assertTrue(all(call[0]["capacity"] == 600 for call in model.calls))
        self.assertEqual(result["best"], [1, 2])
        self.assertEqual(result["error_count"], 0)

    def test_auto_search_reports_individual_errors_and_incomplete_best(self):
        class FakeModel:
            block_ids = {1, 2}

            def calculate(self, schools):
                if schools[0]["block_id"] == 2:
                    raise RuntimeError("искусственная ошибка проверки")
                return {
                    "summary": {
                        "base_deficit": 10,
                        "deficit": 5,
                        "deficit_reduction": 5,
                    }
                }

        search = AutoSearch(FakeModel())
        job_id = search.start(600)["job_id"]
        while not search.status(job_id)["done"]:
            time.sleep(0.01)

        result = search.status(job_id)
        self.assertEqual(result["checked"], 2)
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["errors"][0]["block_id"], 2)
        self.assertEqual(result["best"], [1])


if __name__ == "__main__":
    unittest.main()
