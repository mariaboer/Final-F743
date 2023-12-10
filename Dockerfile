FROM tensorflow/tensorflow:latest-gpu

RUN apt update && apt install -y nano && apt clean autoclean && apt autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /tf
# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --upgrade -r requirements.txt

VOLUME /tf/outputs

COPY . .
CMD ["python3", "main.py"]