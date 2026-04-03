FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies (Added openenv and openenv-core here!)
RUN pip install --no-cache-dir gymnasium pandas numpy openenv openenv-core

# Copy all your files into the container
COPY . .

# Keep the container running for the grader
CMD python inference.py && tail -f /dev/null