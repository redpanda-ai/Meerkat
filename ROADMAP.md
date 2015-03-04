### TASKS
#### v1.1.0 *(scheduled for 3/31)*
1.  **Meerkat Team**, Andy
	* Begins 3/1 - Work with Program Management
		* - [x] Assist with complete schedule for v1.1.0 release
			* - [x] thoughtful plan
			* - [x] list tasks
			* - [x] list dependencies
			* - [x] timeline with milestones
	* 3/2 
		* - [x] Review pull request for revised unit tests from Matt
	* 3/3
		* - [x] Reduce EC2 infrastructure cost and report results to Neil and Katie
			* Total instance count 69
				* Meerkat Development, 4 instances
				* ***Meerkat Panel Processing***, 52 instances
				* Meerkat Production Web Service, 9 instances
				* Meerkat Unknown, 4 instances
	* 3/4
		* - [ ] Update the Meerkat daemon to provide
			* - [ ] column re-ordering
			* - [ ] handle multiple input paths
			* - [ ] handle newer version of the input files
			* - [ ] provide alerting when problems occur
			* - [ ] produce a daily report for all input paths, sent to a distribution list
			* - [ ] :clock4: full deployment
		* - [ ] Meet with Richa from Program Management to discuss architecture of Meerkat
	* 3/5
		* - [ ] Speak with Ram and ask him to get back to work on the label resolution.
	* 3/6
		* - [ ] Speak with Sanjay and team about prototype for building a web service panel for Meerkat
2.  **Meerkat Team**, Matt
	* 3/7
		* - [ ] Set up an EC2 environment for the nolearn deep learning libraries that
			* uses the graphics processing unit
			* works quickly over large data sets

3.  **Meerkat Team**, Andy and Matt
	* 3/5
		* - [ ] Set up a call with Deepu to get the label choices and the samples to be labeled in a sensible place so that we can move forward.
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
* 3/3
	* - [x] Give Rahul and Mandira a complete list of development and 3rd party libraries, with versions.
	* - [x] Provide an update to Neil and Katie about progress on reducing EC2 cost
* 3/2
	* - [x] Review pull request for revised unit tests from Matt
	* - [x] Phone screen for Sivan, a new summer intern
* 2/28
	* - [x] Update the Meerkat daemon to give CT testing data
	* - [x] :+1: Deploy Meerkat v1.0.0 to production VPC 
	* - [x] Set up a [distribution list] (https://github.com/joeandrewkey/meerkat_webservice_schema/issues/21) or equivalent for labels
	* - [x] Obtain agreement on requiring ***ledger_entry*** in meerkat web service v1.0.0

#### Notes
	* Type x, Sub-type y, 5-> (81%) 4-> (9%) 3-> (12%)
	* Complete Consensus (all 5)
		Type x1, sub-type y1 -> 79%
		Type x1, sub-type y2 -> 88%
                     .....
        * 4 out of 5 
