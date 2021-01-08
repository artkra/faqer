FROM python:3.6

RUN mkdir /faqer

COPY . /faqer/

WORKDIR /faqer

RUN pip install -r requirements.txt

CMD [ "python", "web/manage.py", "runserver", "0.0.0.0:8000" ]