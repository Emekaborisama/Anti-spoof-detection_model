FROM python:3.8
RUN mkdir /app
WORKDIR /app
ADD . /app/
RUN sudo apt-get install pkg-config
RUN sudo apt-get install libgtk2.0-de
RUN pip install -r requirements.txt
CMD ["python", "main.py"]

EXPOSE 8080