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
	wget -O models/rdt/rdt.w2v http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz100-w10-cb0-it1-min100.w2v


.PHONY: load-navec-data
load-navec-data:
	wget -O data/navec/ https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar


.PHONY: check-tag
check-tag:
ifndef FAQER_TAG
	$(error FAQER_TAG is not defined)
endif


.PHONY: check-rep
check-rep:
ifndef FAQER_REP
	$(error FAQER_REP repository prefix is not defined)
endif


.PHONY: lock
lock:
	poetry lock


.PHONY: reqs
reqs: lock
	poetry export > requirements.txt

.PHONY: build-image
build-image: check-tag check-rep reqs
	docker build -t faqer:$(FAQER_TAG) -f build/Dockerfile . \
	&& docker tag faqer:$(FAQER_TAG) artkra/faqer:$(FAQER_TAG) \
	&& docker push $(FAQER_REP)/faqer:$(FAQER_TAG)
