.PHONY: test lint build-web smoke validate-artifacts audit-predictions reconcile-predictions docker-up docker-down

test:
	python3 -m pytest -q \
		tests/test_api_integration.py \
		tests/test_backend_services.py \
		tests/test_features.py \
		tests/test_fixtures.py \
		tests/test_prediction_math.py \
		tests/test_scoring_distribution.py \
		tests/test_threshold_monotonicity.py

lint:
	cd web && npm run lint

build-web:
	cd web && npm run build

smoke:
	cd web && npm run smoke

validate-artifacts:
	python3 scripts/validate_artifacts.py

audit-predictions:
	python3 scripts/audit_prediction_math.py

reconcile-predictions:
	python3 scripts/reconcile_prediction_exports.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down
