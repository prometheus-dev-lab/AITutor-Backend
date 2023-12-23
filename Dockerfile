FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv

WORKDIR /AITUTOR-BACKEND

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of your application's code
COPY . /AITUTOR-BACKEND

# Make and apply migrations
RUN python3 manage.py makemigrations
RUN python3 manage.py migrate

# Expose port if your application is a web service
EXPOSE 8000

# Command to run your application, replace 'app.py' with your entrypoint script
CMD ["python3", "run.py", "0.0.0.0:8000"]

