FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

# Create a writable directory for cache (fixing potential permission issues in HF Spaces)
RUN mkdir -p /code/cache && chmod 777 /code/cache
ENV XDG_CACHE_HOME=/code/cache

CMD ["python", "app.py"]
