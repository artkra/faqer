export DATA_DIRECTORY=data

.PHONY: run
run:
	python manage.py runserver 8080


.PHONY: train
train:
	python manage train_classificator $(DATA)
