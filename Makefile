export DATA_DIRECTORY=data

.PHONY: run
run:
	python web/manage.py runserver 8080


.PHONY: train
train:
	python web/manage.py train_embedder $(DATA)
