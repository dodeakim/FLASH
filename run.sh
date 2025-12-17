xhost +local:root

ROS_IMAGE="flash"
ROS_CONTAINER="flash"

sudo rm -f /tmp/.docker.xauth
XAUTH=/tmp/.docker.xauth
touch "$XAUTH"
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge -

docker run --rm -it \
  --workdir="/root" \
  --gpus all \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --env XAUTHORITY=$XAUTH \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "$XAUTH:$XAUTH" \
  --volume "$(pwd)/modules:/root/FLASH" \
  --volume "$(pwd)/data:/root/data" \
  --privileged \
  --network host \
  --name "$ROS_CONTAINER" \
  "$ROS_IMAGE" \
