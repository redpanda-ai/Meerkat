### TASKS
#### v1.1.0 *(scheduled for 3/31)*
1.  **Meerkat Team**, Andy
	* Begins 3/1 - Work with Program Management
		* - [x] Assist with complete schedule for v1.1.0 release
			* - [x] thoughtful plan
			* - [x] list tasks
			* - [x] list dependencies
			* - [x] timeline with milestones

	* 3/4
		* - [x] Update the Meerkat daemon to provide
			* - [x] column re-ordering
			* - [x] handle multiple input paths
			* - [x] handle newer version of the input files
			* - [ ] provide alerting when problems occur
			* - [ ] produce a daily report for all input paths, sent to a distribution list
			* - [ ] :clock4: full deployment
	* 3/10
		* - [ ] Advertise the Jobvite for our new position in LinkedIn.
		* - [ ] Work with Richa on end-to-end integration strategy
		* - [ ] Speak with Sanjay and Nikhil about getting the gpanel_v2 up and running
	* 3/11
		* - [ ] Meet with Kirti on the architecture and design of our transaction data enrichment service
	* 3/15
		* - [ ] Speak with Sanjay and team about prototype for building a web service panel for Meerkat
	* 3/18
		* - [ ] Meet with Richa from Program Management to discuss architecture of Meerkat
2.  **Meerkat Team**, Matt
	* 3/11
		* - [x] Produce a comprehensive set of transactions from our labeling sample where the “choice_pair” was “Payment – Loan”
			* Place the results in S3
	* 3/7
		* - [x] Check and see if there are any types/subtypes with abnormally low rates of agreement between the labelers, provide useful metrics.
			* Report findings as a Fleiss' Kappa matrix
		* - [x] Set up an EC2 environment for the nolearn deep learning libraries that
			* uses the graphics processing unit
			* works quickly over large data sets


3.  **Meerkat Team**, Andy and Matt
	* 3/5
		* - [ ] Set up a call with Deepu to get the label choices and the samples to be labeled in a sensible place so that we can move forward.
		* - [ ] Determine why our labelers had trouble distinguishing subtypes of transfers, perhaps retry those
		* - [ ] Reclassify the ***other*** transactions and see if providing ***charges and fees*** gives us labels
		* - [ ] Add some E-trade data, or other cash management accounts to make sure we see some dividends and capital gains
	* 3/7
		* - [ ] Think of a way to integrate a series of per-merchant classifiers that take the form of a series of regex expressions.
			* Each bundle of regex expressions will serve as a per-merchant classifier
			* Nikhil is heading up a team of interns and data analysts to build and maintain at least dozens but possibly a hundered of them in Q1 and Q2.

### DEPENDENCIES
#### v1.1.0
1. **Data Warehouse Team**, Sanjay
	* - [x] Confirm the complete list of Meerkat S3 input and output locations
	* - [x] Provide a distribution list for everyone who needs Meerkat panel alerts
	* - [x] Provide a distribution list for everyone who wants the daily panel report
2. **Database Administrators**, Subha S., Arun
	* - [x] Provide the *sum_info_id* for e*trade (https://github.com/joeandrewkey/Meerkat/issues/120)

### RECENTLY COMPLETED
#### Matt
* 3/3
	* - [x] :clock4: Conduct analysis on first 5000 labeled transactions, report results 
	* - [x] :clock4: Review Ram's contribution to our labeling effort, report findings
* 3/2
	* - [x] Submit pull request for unit tests in meerkat.producer
* 2/28
	* - [x] Compare the lists of ***txn_type*** and ***txn_sub_type*** labels between Platform and Meerkat
	* - [x] Add semantic versioning for the v1.0.0 version of Meerkat
	* - [x] Provide thoughtful response to how to find and predict transaction types.

#### Andy
* 3/4
	* - [x] Speak with Ram and ask him to get back to work on the label resolution.
* 3/3
	* - [x] Give Rahul and Mandira a complete list of development and 3rd party libraries, with versions.
	* - [x] Reduce EC2 infrastructure cost and report results to Neil and Katie
		* Total instance count 69
			* Meerkat Development, 4 instances
			* ***Meerkat Panel Processing***, 52 instances
			* Meerkat Production Web Service, 9 instances
			* Meerkat Unknown, 4 instances
* 3/2
	* - [x] Review pull request for revised unit tests from Matt
	* - [x] Phone screen for Sivan, a new summer intern
* 2/28
	* - [x] Update the Meerkat daemon to give CT testing data
	* - [x] :+1: Deploy Meerkat v1.0.0 to production VPC 
	* - [x] Set up a [distribution list] (https://github.com/joeandrewkey/meerkat_webservice_schema/issues/21) or equivalent for labels
	* - [x] Obtain agreement on requiring ***ledger_entry*** in meerkat web service v1.0.0

