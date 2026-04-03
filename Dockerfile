FROM python:3.10-slim

WORKDIR /app

# Added openai to the dependencies!
RUN pip install --no-cache-dir gymnasium pandas numpy openenv openenv-core openai

COPY . .

CMD python inference.py && tail -f /dev/null