FROM python:3.10-slim

WORKDIR /app

# Added fastapi and uvicorn to run the web server!
RUN pip install --no-cache-dir gymnasium pandas numpy openenv openenv-core openai fastapi uvicorn

COPY . .

# Expose the standard port Hugging Face Spaces expects
EXPOSE 7860

# Start the web server on boot
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
