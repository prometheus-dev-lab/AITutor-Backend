FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv

WORKDIR /AITUTOR-BACKEND

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --break-system-packages -r requirements.txt


# Copy the rest of your application's code
COPY . /AITUTOR-BACKEND

# Make migrations
RUN python3 manage.py makemigrations
RUN python3 manage.py migrate

# Expose port if your application is a web service
EXPOSE 8000
