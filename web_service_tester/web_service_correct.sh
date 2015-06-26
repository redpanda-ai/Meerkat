#/bin/bash

curl -s --insecure -X POST -d @$1 https://localhost:443/meerkat/v1.0.1 --header "Content-Type:application/json" | python3 -m json.tool