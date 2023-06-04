FROM cherishpf/python3-java8:1.0

MAINTAINER ashu

ADD . /app

WORKDIR /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
RUN pip install -r requirements.txt
RUN export PYTHONPATH="${PYTHONPATH}:yolov5_deepsort"

ENTRYPOINT ["python", "/app/main.py"]