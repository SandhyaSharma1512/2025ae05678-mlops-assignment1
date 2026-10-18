# Use an official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY src/ src/
COPY models/ models/

ENV PYTHONPATH=/app/src
# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "heart_disease.api:app", "--host", "0.0.0.0", "--port", "8000"]