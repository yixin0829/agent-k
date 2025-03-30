FROM python:3.12

RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m sandboxuser
USER sandboxuser
WORKDIR /home/sandboxuser

CMD ["python", "--version"]

# Build image:
# !docker build -t python_sandbox:latest ./docker

# Run the container in restricted mode. The container will run in the background.
# !docker run -d --name sandbox --network none --cap-drop all --pids-limit 64 --tmpfs /tmp:rw,size=64M   python_sandbox:latest sleep infinity

# Can execute code in the container:
# !docker exec -i sandbox python -c "print('hello world')"
