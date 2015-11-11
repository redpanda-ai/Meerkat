#/bin/bash

port=$1
cd ..
sudo python3 -m meerkat.web_service $port &
cd web_service_tester
