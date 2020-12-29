.PHONY: run
run:
	python web/manage.py runserver 8080


.PHONY: train
train:
	python web/manage.py train_embedder $(DATA)


.PHONY: notebook
notebook:
	python web/manage.py shell_plus --notebook


.PHONY: load-rdt
load-rdt:
	wget -O models/rdt/ http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz100-w10-cb0-it1-min100.w2v


.PHONY: load-navec-data
load-navec-data:
	wget -O data/navec/ https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar