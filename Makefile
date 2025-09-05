.PHONY: build run exec clean

# Build the Docker image
build:
	docker build -t python_code_interpreter:latest ./src/utils

# Run the Docker container
run:
	docker run -d --name python_code_interpreter --network none --cap-drop all --pids-limit 64 --tmpfs /tmp:rw,size=64M python_code_interpreter:latest sleep infinity

# Execute a test python script in the Docker container
exec:
	docker exec -i python_code_interpreter python -c "print('hello world')"

# Clean up the Docker container
clean:
	docker stop python_code_interpreter || true
	docker rm python_code_interpreter || true
