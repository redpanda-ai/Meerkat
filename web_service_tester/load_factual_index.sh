#/bin/bash

cd ..
python3.3 -m meerkat.bulk_loader config/factual_loader_a.json
python3.3 -m meerkat.bulk_loader config/factual_loader_b.json
python3.3 -m meerkat.bulk_loader config/factual_loader_c.json
python3.3 -m meerkat.bulk_loader config/factual_loader_d.json
cd web_service_tester
