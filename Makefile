.PHONY: build run exec clean

# Build the Docker image
build:
	docker build -t python_sandbox:latest ./docker

# Run the Docker container
run:
	docker run -d --name sandbox --network none --cap-drop all --pids-limit 64 --tmpfs /tmp:rw,size=64M python_sandbox:latest sleep infinity

# Execute a command in the Docker container
exec:
	docker exec -i sandbox python -c "print('hello world')"

# Clean up the Docker container
clean:
	docker stop sandbox || true
	docker rm sandbox || true
