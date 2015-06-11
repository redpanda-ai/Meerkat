#/bin/bash

cd ..
sudo nohup python3 -m meerkat.web_service > async.log 2>&1 &
cd web_service_test
