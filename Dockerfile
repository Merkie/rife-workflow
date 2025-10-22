# 1. Use a stable RunPod base image with GPU drivers
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# 2. Set environment to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# 3. Set the path for the binary we will download
ENV RIFE_BIN=/app/rife-ncnn-vulkan

# 4. Install dependencies, download & set up the RIFE binary in ONE layer
RUN apt-get update && \
    apt-get install -y python3-pip wget unzip ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    \
    # Set up the app directory
    mkdir /app && \
    cd /app && \
    \
    # Download and unzip the pre-built binary
    wget https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip && \
    unzip rife-ncnn-vulkan-20221029-ubuntu.zip && \
    \
    # Clean up the zip file
    rm rife-ncnn-vulkan-20221029-ubuntu.zip && \
    \
    # Move the binary and models to the /app root
    mv /app/rife-ncnn-vulkan-20221029-ubuntu/* /app/ && \
    rmdir /app/rife-ncnn-vulkan-20221029-ubuntu && \
    \
    # Make the binary executable
    chmod +x $RIFE_BIN

# 5. Set the working directory
WORKDIR /app

# 6. Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy your handler code
COPY handler.py .

# 8. Set the command to run when the container starts
CMD ["python3", "-u", "handler.py"]