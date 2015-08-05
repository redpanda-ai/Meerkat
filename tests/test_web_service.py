from plumbum import ProcessExecutionError
from plumbum import local, BG
from plumbum.cmd import sudo, kill, grep, python3
import requests

def start_web_service():
	start = sudo[python3["-m"]["meerkat.web_service"]]
	with local.cwd("/home/ubuntu/git/Meerkat"):
		(start) & BG


def stop_web_service():
	try:
		no = sudo["ps"]["-ef"] | grep["python"] | grep["root"]| grep ["meerkat.web_service"]
		number = no().split()[1] 
		print(number)
		kill_process = sudo[kill[number]]
		kill_process()
		print("web service stopped")
	except ProcessExecutionError:
		print("The service was not running, so it couldn't be stopped") 
	
def check_status():
	r = requests.get("https://localhost/status/index.html", verify=False)
	status = r.status_code
	return (status)

class WebServiceTest(unittest.TestCase):
	"""Our UnitTest class."""
	
	def test_web_service_status(self):
		"""Test starts, checks status of, and stops meerkat web service"""
		
		#stop_web_service()
		start_web_service()
		status = check_status()
		self.assertTrue(status == 200)
		self.stop_web_service()

if __name__ == "__main__":
	unittest.main()
	sys.exit()

	
