from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
from blocksnet.analysis.provision import competitive_provision
from blocksnet.config import service_types_config
from importlib.metadata import PackageNotFoundError, version


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEB_DIR = ROOT / "web"
RESULT_COLUMNS = [
    "demand",
    "capacity",
    "demand_left",
    "demand_within",
    "demand_without",
    "capacity_left",
    "capacity_within",
    "capacity_without",
    "provision_strong",
    "provision_weak",
]
TIE_TOLERANCE = 1e-6
DEFAULT_ALLOWED_ORIGINS = {"https://annamironets.github.io", "http://127.0.0.1:8765", "http://localhost:8765"}


def allowed_origins() -> set[str]:
    configured = os.environ.get("SCHOOL_PLACEMENT_ALLOWED_ORIGINS")
    if configured is None:
        return DEFAULT_ALLOWED_ORIGINS
    return {origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()}


def _number(value: Any) -> float | int | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    if numeric.is_integer():
        return int(numeric)
    return numeric


def validate_schools(schools: Any, block_ids: set[int]) -> list[dict[str, int]]:
    if not isinstance(schools, list):
        raise ValueError("Список новых школ задан некорректно.")

    validated = []
    for school in schools:
        if not isinstance(school, dict):
            raise ValueError("Каждая школа должна содержать квартал и вместимость.")
        block_id = school.get("block_id")
        capacity = school.get("capacity")
        if isinstance(block_id, bool) or not isinstance(block_id, int) or block_id not in block_ids:
            raise ValueError(f"Неизвестный квартал: {block_id!r}.")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Вместимость школы должна быть положительным целым числом.")
        validated.append({"block_id": block_id, "capacity": capacity})
    return validated


class SchoolModel:
    def __init__(self) -> None:
        try:
            blocksnet_version = version("blocksnet")
        except PackageNotFoundError as exc:
            raise RuntimeError("BlocksNet не установлен. Установите BlocksNet 1.0.0a9 по инструкции README.md.") from exc
        if blocksnet_version != "1.0.0a9":
            raise RuntimeError(
                f"Обнаружен BlocksNet {blocksnet_version}; для воспроизводимости требуется версия 1.0.0a9."
            )

        results_path = DATA_DIR / "sovetsky_pipeline_results.csv"
        matrix_path = DATA_DIR / "accessibility_walk_minutes.csv"
        blocks_path = DATA_DIR / "blocks.geojson"
        provision_path = DATA_DIR / "blocks_school_provision.geojson"
        for path in (results_path, matrix_path, blocks_path, provision_path):
            if not path.is_file():
                raise FileNotFoundError(f"Не найден обязательный файл данных: {path}")

        results = pd.read_csv(results_path).set_index("block_id")
        results.index = pd.to_numeric(results.index).astype(int)
        if results.index.has_duplicates:
            raise ValueError("В таблице результатов найдены повторяющиеся ID кварталов.")
        self.block_ids = set(results.index)

        matrix = pd.read_csv(matrix_path, index_col=0)
        matrix.index = pd.to_numeric(matrix.index).astype(int)
        matrix.columns = pd.to_numeric(matrix.columns).astype(int)
        if set(matrix.index) != self.block_ids or set(matrix.columns) != self.block_ids:
            raise ValueError("ID матрицы времени не совпадают с ID кварталов в исходных результатах.")
        self.matrix = matrix.reindex(index=results.index, columns=results.index)

        collection = json.loads(blocks_path.read_text(encoding="utf-8"))
        self.features = collection.get("features", [])
        feature_ids = {
            int(feature["properties"]["block_id"])
            for feature in self.features
            if feature.get("properties", {}).get("block_id") is not None
        }
        if feature_ids != self.block_ids or len(self.features) != len(self.block_ids):
            raise ValueError("Набор геометрий и набор ID кварталов в таблице результатов различаются.")

        self.inputs = results[["building_population_estimate", "capacity"]].rename(
            columns={"building_population_estimate": "population"}
        )
        self.inputs["capacity"] = pd.to_numeric(self.inputs["capacity"], errors="raise").astype(int)
        self.inputs["population"] = pd.to_numeric(self.inputs["population"], errors="raise")
        service_config = service_types_config["school"]
        self.accessibility = int(service_config["accessibility"])
        self.demand_norm = int(service_config["demand"])
        self.baseline = self._run(self.inputs)
        self._verify_saved_baseline(provision_path)

    def _run(self, inputs: pd.DataFrame) -> pd.DataFrame:
        provision, _ = competitive_provision(
            inputs.copy(),
            self.matrix,
            self.accessibility,
            self.demand_norm,
        )
        return provision

    def _verify_saved_baseline(self, provision_path: Path) -> None:
        saved = gpd.read_file(provision_path).set_index("index")
        saved.index = pd.to_numeric(saved.index).astype(int)
        if set(saved.index) != self.block_ids:
            raise ValueError("ID сохранённых результатов BlocksNet не совпадают с исходными кварталами.")
        for column in RESULT_COLUMNS:
            if column not in saved.columns or column not in self.baseline.columns:
                raise ValueError(f"В сохранённом исходном расчёте отсутствует поле {column!r}.")
            actual = self.baseline[column].sort_index().astype(float)
            expected = saved[column].reindex(actual.index).astype(float)
            if not (actual.sub(expected).abs().fillna(0) <= 1e-9).all():
                raise ValueError(f"Повторный расчёт BlocksNet расходится с сохранённым полем {column!r}.")

    def calculate(self, schools: list[dict[str, int]]) -> dict[str, Any]:
        schools = validate_schools(schools, self.block_ids)
        inputs = self.inputs.copy(deep=True)
        added_by_block: dict[int, int] = {}
        for school in schools:
            block_id = school["block_id"]
            added_by_block[block_id] = added_by_block.get(block_id, 0) + school["capacity"]
        for block_id, added in added_by_block.items():
            inputs.loc[block_id, "capacity"] += added

        provision = self._run(inputs)
        base = self.baseline
        rows = []
        for block_id, row in provision.iterrows():
            old = base.loc[block_id]
            rows.append(
                {
                    "block_id": int(block_id),
                    "population": _number(self.inputs.loc[block_id, "population"]),
                    "demand": _number(row["demand"]),
                    "base_capacity": _number(old["capacity"]),
                    "capacity": _number(row["capacity"]),
                    "added_capacity": added_by_block.get(int(block_id), 0),
                    "base_deficit": _number(old["demand_left"]),
                    "deficit": _number(row["demand_left"]),
                    "deficit_change": _number(row["demand_left"] - old["demand_left"]),
                    "provision_strong": _number(row["provision_strong"]),
                    "provision_weak": _number(row["provision_weak"]),
                }
            )

        return {
            "schools": schools,
            "rows": rows,
            "summary": {
                "population": int(round(float(self.inputs["population"].sum()))),
                "demand": int(self.baseline["demand"].sum()),
                "base_capacity": int(self.baseline["capacity"].sum()),
                "capacity": int(provision["capacity"].sum()),
                "added_capacity": sum(school["capacity"] for school in schools),
                "base_deficit": int(self.baseline["demand_left"].sum()),
                "deficit": int(provision["demand_left"].sum()),
                "deficit_reduction": int(self.baseline["demand_left"].sum() - provision["demand_left"].sum()),
                "accessibility": self.accessibility,
                "demand_norm": self.demand_norm,
            },
        }

    def baseline_payload(self) -> dict[str, Any]:
        result = self.calculate([])
        return {
            "features": self.features,
            "baseline": result,
            "settings": {
                "blocks": len(self.block_ids),
                "accessibility": self.accessibility,
                "demand_norm": self.demand_norm,
                "blocksnet": "1.0.0a9",
            },
        }


class AutoSearch:
    def __init__(self, model: SchoolModel) -> None:
        self.model = model
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.active_job: str | None = None

    def start(self, capacity: Any) -> dict[str, str]:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Вместимость новой школы должна быть положительным целым числом.")
        with self.lock:
            if self.active_job is not None:
                job = self.jobs[self.active_job]
                if not job["done"]:
                    raise ValueError("Автоматический поиск уже выполняется.")
            job_id = uuid.uuid4().hex
            job = {
                "done": False,
                "checked": 0,
                "total": len(self.model.block_ids),
                "capacity": capacity,
                "results": [],
                "errors": [],
                "best": [],
                "error_count": 0,
            }
            self.jobs[job_id] = job
            self.active_job = job_id
        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
        return {"job_id": job_id}

    def _run(self, job_id: str) -> None:
        job = self.jobs[job_id]
        for block_id in sorted(self.model.block_ids):
            try:
                result = self.model.calculate([{"block_id": block_id, "capacity": job["capacity"]}])
                summary = result["summary"]
                entry = {
                    "block_id": block_id,
                    "base_deficit": summary["base_deficit"],
                    "deficit": summary["deficit"],
                    "deficit_reduction": summary["deficit_reduction"],
                }
                job["results"].append(entry)
            except Exception as exc:
                job["errors"].append({"block_id": block_id, "message": f"{type(exc).__name__}: {exc}"})
            finally:
                job["checked"] += 1

        job["results"].sort(key=lambda row: (-row["deficit_reduction"], row["block_id"]))
        job["error_count"] = len(job["errors"])
        if job["results"]:
            best_reduction = job["results"][0]["deficit_reduction"]
            job["best"] = [
                row["block_id"]
                for row in job["results"]
                if abs(row["deficit_reduction"] - best_reduction) <= TIE_TOLERANCE
            ]
        job["done"] = True

    def status(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise ValueError("Задача автоматического поиска не найдена.")
            return dict(job)


def create_handler(model: SchoolModel, search: AutoSearch) -> type[BaseHTTPRequestHandler]:
    class RequestHandler(BaseHTTPRequestHandler):
        def _add_cors_headers(self) -> None:
            origin = self.headers.get("Origin", "").rstrip("/")
            if origin in allowed_origins():
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._add_cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Any:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Не удалось прочитать данные запроса.") from exc

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/data":
                self._send_json(200, model.baseline_payload())
            elif path.startswith("/api/auto/"):
                try:
                    self._send_json(200, search.status(path.rsplit("/", 1)[-1]))
                except ValueError as exc:
                    self._send_json(404, {"error": str(exc)})
            elif path == "/":
                self._serve_index()
            else:
                self._send_json(404, {"error": "Страница не найдена."})

        def do_OPTIONS(self) -> None:
            origin = self.headers.get("Origin", "").rstrip("/")
            if origin not in allowed_origins():
                self._send_json(403, {"error": "Источник запроса не разрешён."})
                return
            self.send_response(204)
            self._add_cors_headers()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
                if not isinstance(payload, dict):
                    raise ValueError("Ожидался JSON-объект.")
                if path == "/api/calculate":
                    self._send_json(200, model.calculate(payload.get("schools")))
                elif path == "/api/auto/start":
                    self._send_json(202, search.start(payload.get("capacity")))
                else:
                    self._send_json(404, {"error": "Метод не найден."})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": f"Ошибка расчёта: {type(exc).__name__}: {exc}"})

        def _serve_index(self) -> None:
            index_path = WEB_DIR / "index.html"
            if not index_path.is_file():
                self._send_json(500, {"error": "Не найден интерфейс приложения: web/index.html"})
                return
            body = index_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._add_cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

    return RequestHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальное приложение для сценариев размещения школ.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    model = SchoolModel()
    search = AutoSearch(model)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(model, search))
    print(f"Приложение доступно: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nРабота сервера остановлена.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
