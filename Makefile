include .env
export $(shell sed 's/=.*//' .env)


core-build:
	docker-compose build aymurai-core
core-run:
	docker-compose run aymurai-core


dev-build: core-build
	docker-compose -f .devcontainer/docker-compose.yml build devcontainer-gpu
dev-build-cpu: core-build
	docker-compose -f .devcontainer/docker-compose.yml build devcontainer


jupyter-run: dev-build redis-run
	docker-compose -f .devcontainer/docker-compose.yml run devcontainer-gpu \
		jupyter-lab /workspace --no-browser
jupyter-run-cpu: dev-build-cpu redis-run
	docker-compose -f .devcontainer/docker-compose.yml run devcontainer \
		jupyter-lab /workspace --no-browser

redis-run:
	docker-compose -f .devcontainer/docker-compose.yml up -d redis

api-build: core-build
	docker-compose build aymurai-api-dev
api-run: redis-run
	docker-compose run --service-ports aymurai-api-dev
api-test:
	docker-compose run \
			-e TEST_MODE=1 \
			-e LOG_LEVEL=error \
		aymurai-api-dev

api-prod-build: api-build
	docker-compose build aymurai-api-prod
api-prod-run:
	docker run -p 8899:8000 --hostname=aymurai-api-prod registry.gitlab.com/collective.ai/datagenero-public/aymurai-api:prod
api-prod-push:
	docker-compose push aymurai-api-prod
api-prod-pull:
	docker-compose pull aymurai-api-prod
