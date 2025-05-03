FROM python:3.13

# Update pip and install necessary packages
RUN pip install -U pip

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./
COPY reddit_model_pipeline.joblib /app/
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY redditApp.py ./

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using uvicorn
# Uses 0.0.0.0 to be accessible from outside the container
# Defaults to port 8000 if --port is not specified
ENTRYPOINT ["uvicorn", "redditApp:app", "--host", "0.0.0.0"]