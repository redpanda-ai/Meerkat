import sys
import paramiko
import time

def load_stack_from_file(filename):
	my_stack = []
	with open(filename) as input_file:
		for line in input_file:
			line = line.strip()
			if line != "":
				my_stack.append(line)

	new_stack = []
	while my_stack:
		new_stack.append(my_stack.pop())
	return new_stack

def poll_clients(my_stack):
	clients = [ "172.31.11.39", "172.31.8.64", "172.31.1.103", "172.31.6.213", "172.31.6.214", "172.31.6.215",
		"172.31.42.132", "172.31.42.131", "172.31.42.140", "172.31.42.137", "172.31.42.139", "172.31.42.134",
		"172.31.42.136", "172.31.42.135", "172.31.42.138", "172.31.25.114", "172.31.25.115", "172.31.25.116"]
	rsa_private_key_file = "/root/.ssh/meerkat.pem"
	shell_command = "ps -ef|grep python3.3|grep -v grep|wc -l"
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	target_procs = 12
	for client in clients:
		ssh.connect(client, username="root", key_filename=rsa_private_key_file)
		print(client)
		stdin, stdout, stderr = ssh.exec_command(shell_command)
		x = stdout.readlines()
		y = stderr.readlines()
		if str(x) != "[]":
			for line in x:
				running_procs = int(line.strip())
				print("There are {0} running processes".format(running_procs))
				new_procs = target_procs - running_procs
				for proc in range(0,new_procs):
					command = my_stack.pop()
					print(command)
					stdin, stdout, stderr = ssh.exec_command(command)
				#print(line)
		if str(y) != "[]":
			print(y)
		time.sleep(1)
		stdin, stdout, stderr = ssh.exec_command(shell_command)
		x = stdout.readlines()
		y = stderr.readlines()
		if str(x) != "[]":
			running_procs = int(line.strip())
			print("Now there are {0} running processes".format(running_procs))
		ssh.close()
	ssh.close()

print("Hello")
my_stack = load_stack_from_file(sys.argv[1])
while my_stack:
	print("There are {0} files remaining".format(len(my_stack)))
	poll_clients(my_stack)
	time.sleep(60)


