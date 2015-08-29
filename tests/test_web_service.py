from plumbum import ProcessExecutionError
from plumbum import local, BG
from plumbum.cmd import sudo, kill, grep, python3, sleep
from requests.exceptions import ConnectionError
from meerkat.various_tools import load_params
from pprint import pprint
import requests
import unittest

def web_service_is_online():
        """Tests to see if the web service is already online."""
        try:
                no = sudo["ps"]["-ef"] | grep["python"] | \
                grep["root"]| grep ["meerkat.web_service"]
                web_service_pid = no().split()[1]
                return True, web_service_pid
        except ProcessExecutionError:
                return False, None

def start_web_service():
        """Starts the web service as a background process."""
        start = sudo[python3["-m"]["meerkat.web_service"]]
        with local.cwd("/home/ubuntu/git/Meerkat"):
                (start) & BG

def stop_linux_process(my_pid):
        """Stops any linux process with the provided process id (my_pid) """
        try:
                sudo[kill[my_pid]]()
        except ProcessExecutionError:
                print("Unable to kill, aborting")
                sys.exit()

def check_status():
        """Get a status code (e.g. 200) from the web service"""
        r = requests.get("https://localhost/status/index.html", verify=False)
        status = r.status_code
        print(r.content)
        r.connection.close()
        return (status)

def post_sample():
        """Get a status code (e.g. 200) from web service after posting a sample input for classification by meerkat"""

        one_ledger = '{ "container":"bank", "transaction_list":[ { "date":"2014-08-10T00:00:00", "description":"taco bell scarsdale, ny", "amount":59.0, "transaction_id":5024853, "ledger_entry":"debit" } ], "cobrand_id":99
