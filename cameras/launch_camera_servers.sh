#!/bin/bash

tmux new-session -d -s camera_server "python camera_server.py; bash"
echo -e "\e[33mAttach to the camera server tmux with 'tmux a -t camera_server'. Please wait 15 seconds for the cameras to start.\e[0m"
