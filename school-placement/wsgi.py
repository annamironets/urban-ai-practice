import json

from flask import Flask, jsonify, make_response, request

from app import AutoSearch, SchoolModel, allowed_origins


def create_app(model=None):
    app = Flask(__name__)
    model = model if model is not None else SchoolModel()
    search = AutoSearch(model)

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin", "").rstrip("/")
        if origin in allowed_origins():
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.before_request
    def handle_preflight():
        if request.method != "OPTIONS":
            return None
        origin = request.headers.get("Origin", "").rstrip("/")
        if origin not in allowed_origins():
            return jsonify({"error": "Источник запроса не разрешён."}), 403
        response = make_response("", 204)
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "600"
        return response

    @app.get("/api/data")
    def get_data():
        return jsonify(model.baseline_payload())

    @app.post("/api/calculate")
    def calculate():
        try:
            payload = _read_json_object()
            return jsonify(model.calculate(payload.get("schools")))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            app.logger.exception("School placement calculation failed")
            return jsonify({"error": f"Ошибка расчёта: {type(exc).__name__}: {exc}"}), 500

    @app.post("/api/auto/start")
    def start_auto_search():
        try:
            payload = _read_json_object()
            return jsonify(search.start(payload.get("capacity"))), 202
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            app.logger.exception("Automatic school placement search failed")
            return jsonify({"error": f"Ошибка расчёта: {type(exc).__name__}: {exc}"}), 500

    @app.get("/api/auto/<job_id>")
    def get_auto_search(job_id):
        try:
            return jsonify(search.status(job_id))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Страница не найдена."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return jsonify({"error": "Метод не найден."}), 405

    return app


def _read_json_object():
    try:
        payload = json.loads(request.get_data(cache=False).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Не удалось прочитать данные запроса.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Ожидался JSON-объект.")
    return payload


application = create_app()
