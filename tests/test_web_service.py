from plumbum import ProcessExecutionError
from plumbum import local, BG
from plumbum.cmd import sudo, kill, grep, python3, sleep
from requests.exceptions import ConnectionError
from multiprocessing.pool import ThreadPool
import json
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


def get_trans_text():
        """Get the json string of one_ledger.json"""
        transFile = open('./web_service_tester/one_ledger.json', 'rb')
        transText = transFile.read()
        transFile.close()
        return transText


def classify_one(self, transaction, max_retries=10, sleep_interval=2):
        """Send a single transaction to the web service for classification"""
        sleep_interval = 2
        count = 1
        while count <= max_retries:
                try:
                        sleep(sleep_interval)
                        r_post = requests.post(
                                "https://localhost:443/meerkat/v1.3",
                                data=transaction,
                                verify=False)

                        r_post.connection.close()
                        self.assertTrue(r_post.status_code == 200)
                        break
                except ConnectionError:
                        count += 1

        return r_post.content


class WebServiceTest(unittest.TestCase):
        """Our UnitTest class."""

        @classmethod
        def setUpClass(cls):
                online, web_service_pid = web_service_is_online()
                if online:
                        stop_linux_process(web_service_pid)
                start_web_service()

        @classmethod
        def tearDownClass(cls):
                online, web_service_pid = web_service_is_online()
                if online:
                        stop_linux_process(web_service_pid)

        def test_web_service_status(self):
                """Test starts meerkat, checks status of meerkat, and stops meerkat web service"""
                count = 1
                sleep_interval = 2
                max_retries = 10
                # Wait for sleep_interval seconds before trying up to
                # max_retries times
                while count <= max_retries:
                        try:
                                sleep(sleep_interval)
                                status = check_status()
                                self.assertTrue(status == 200)
                                return
                        except ConnectionError:
                                count += 1
                self.assertTrue(0 == 1, "Failed to connect to Meerkat service")
                return

        def test_web_service_races(self):
                """Test starts meerkat, runs 100 classifications, and stops meerkat"""
                samples = 100
                pool = ThreadPool(samples)
                transText = get_trans_text()
                tasks = []
                for i in range(samples):
                        tasks.append(pool.apply_async(
                                     classify_one,
                                     (self, transText)))
                classified = []
                for task in tasks:
                        classified.append(sorted(json.loads(
                                          task.get().decode("utf-8")
                                          )))
                for i in range(1, samples):
                        self.assertEqual(
                                         classified[0],
                                         classified[i],
                                         "Two results were not equal\n{}\n\n{}".format(classified[0], classified[i]))
                return

if __name__ == "__main__":
        unittest.main()
