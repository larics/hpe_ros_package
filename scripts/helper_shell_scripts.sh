waitForRealsenseCamera() {
  until timeout 3s rostopic echo /camera/color/camera_info -n 1 --noarr > /dev/null 2>&1; do 
    echo "waiting for realsense camera" 
    sleep 1; 
  done
}

waitFor() {
  until timeout 3s rostopic echo $1 -n 1 --noarr > /dev/null 2>&1; do
    echo "waiting for $1"
    sleep 1;
  done
}


