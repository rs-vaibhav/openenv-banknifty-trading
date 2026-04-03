FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir gymnasium pandas numpy

# Copy all your files into the container
COPY . .

# Hugging Face Spaces requires the container to stay alive. 
# We'll just run the inference script to prove it works on startup, 
# then tail a null device to keep the container running for the OpenEnv grader to connect.
CMD python inference.py && tail -f /dev/null