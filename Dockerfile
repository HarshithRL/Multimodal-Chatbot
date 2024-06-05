# Use an official base image with Python
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy your application code into the container
COPY . .

# Install any Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirments.txt

# Expose the port your app runs on (optional but recommended)
EXPOSE 8000

# Add a healthcheck to monitor the application status (optional but recommended)
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]