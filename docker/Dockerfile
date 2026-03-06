# ============================================================
# 3D Scene Graph - Docker Image
# Base: CUDA 12.6 + Ubuntu 24.04 (Blackwell / RTX 5000 series)
# Includes: ROS2 Jazzy, librealsense, Open3D, YOLOv8
# ============================================================
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ── System basics ────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl wget git nano build-essential cmake \
    python3 python3-pip python3-dev python3-venv \
    libssl-dev libffi-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libusb-1.0-0-dev udev \
    software-properties-common locales tzdata \
    lsb-release gnupg2 ca-certificates \
    libopencv-dev python3-opencv \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ── ROS2 Jazzy ───────────────────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y \
      ros-jazzy-ros-base \
      ros-jazzy-rviz2 \
      ros-jazzy-rqt \
      ros-jazzy-cv-bridge \
      ros-jazzy-image-transport \
      ros-jazzy-sensor-msgs \
      ros-jazzy-geometry-msgs \
      ros-jazzy-visualization-msgs \
      ros-jazzy-std-msgs \
      python3-colcon-common-extensions \
      python3-rosdep \
      python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

# ── Intel RealSense SDK (librealsense2) ──────────────────────
# Use Intel's official apt repo (supports L515)
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
      | tee /etc/apt/keyrings/librealsense.pgp > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
      https://librealsense.intel.com/Debian/apt-repo \
      $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y \
      librealsense2-utils \
      librealsense2-dev \
      librealsense2-dbg \
    && rm -rf /var/lib/apt/lists/*

# ── RealSense ROS2 wrapper ───────────────────────────────────
RUN apt-get update && apt-get install -y \
    ros-jazzy-realsense2-camera \
    ros-jazzy-realsense2-description \
    && rm -rf /var/lib/apt/lists/*

# ── Python virtual environment (avoids debian package conflicts) ──────────
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Python ML/3D packages ─────────────────────────────────────────────────
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/nightly/cu128

RUN /opt/venv/bin/pip install "numpy<2" "opencv-python<4.10" && \
    /opt/venv/bin/pip install \
      ultralytics \
      open3d \
      pyrealsense2 \
      numpy scipy \
      networkx \
      fastapi uvicorn \
      websockets \
      python-multipart \
      scikit-learn \
      Pillow \
      matplotlib \
      trimesh \
      rospkg \
      catkin_pkg \
      empy \
      lark

# ── ROS2 Python dependencies ─────────────────────────────────────────────
RUN /opt/venv/bin/pip install "setuptools>=68.0"

# ── rosdep init ──────────────────────────────────────────────
RUN PYTHONPATH="" rosdep init && PYTHONPATH=""

# ── Workspace setup ──────────────────────────────────────────
WORKDIR /ros2_ws

# Source ROS2 in every shell
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash 2>/dev/null || true" >> /root/.bashrc

# udev rules for RealSense
COPY udev/99-realsense-libusb.rules /etc/udev/rules.d/

EXPOSE 8080 8765

CMD ["/bin/bash"]
