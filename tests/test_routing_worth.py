from meerkat.classification.load_model import load_scikit_model
import csv

def main():
	classifier = load_scikit_model("bank_sws")
	with open("data/input/merchant_labels.csv", encoding = "latin-1") as f:
		data = csv.reader(f, delimiter = ",")
		for row in data:
			description = row[0]
			classification = int(classifier(description))
			if classification == 1:
				print(description)
				
if __name__ == "__main__":
	main()
