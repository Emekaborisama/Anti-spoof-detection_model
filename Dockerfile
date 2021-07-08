FROM python:3.8
RUN mkdir /app
WORKDIR /app
ADD . /app/

RUN pip install -r requirements.txt
CMD ["python", "main.py"]
RUN sudo apt-get install pkg-config
RUN sudo apt-get install libgtk2.0-de

EXPOSE 8080