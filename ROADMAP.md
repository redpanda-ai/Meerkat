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
		* - [ ] Give Rahul a complete list of development and 3rd party libraries, with versions.
		* - [ ] Update the Meerkat daemon to provide
			* - [ ] column re-ordering
			* - [ ] handle multiple input paths
			* - [ ] handle newer version of the input files
			* - [ ] provide alerting when problems occur
			* - [ ] full deployment
	* 3/3
		* - [ ] Reduce EC2 infrastructure cost and report results to Neil and Katie		
	* 3/7
		* - [ ] Think of a way to process set of ***deep clean*** regular expressions.
			* Each bundle of regex expressions will serve as a per-merchant classifier
			* Nikhil is heading up a team of interns and data analysts to build it in Q1 and Q2.
1.  **Meerkat Team**, Matt
	* - [ ] 3/3 Conduct analysis on first 5000 labeled transactions, report results
	* - [ ] 3/3 Review Ram's contribution to our labeling effort, report findings

### DEPENDENCIES
#### v1.1.0
1. **Data Warehouse Team**, Sanjay
	* - [ ] Confirm the complete list of Meerkat S3 input and output locations

### AGREEMENTS / DEADLINES
#### v1.1.0
* 3/2 
	* - [ ] Review pull request for revised unit tests from Matt
	* - [ ] Give Rahul a complete list of development and 3rd party libraries, with versions.
	* - [ ] Update the Meerkat daemon to provide
		* - [ ] column re-ordering
		* - [ ] handle multiple input paths
		* - [ ] handle newer version of the input files
		* - [ ] provide alerting when problems occur
		* - [ ] full deployment
* 3/3
	* - [ ] Provide an update to Neil and Katie about progress on reducing EC2 cost
	* - [ ] Meet with Richa from Program Management to discuss architecture of Meerkat
* 3/7
	* - [ ] Speak with Sanjay and team about prototype for building a web service panel for Meerkat
	* - [ ] Create plan to deal integrate Nikhil's per-merchant classifier program (uses bundles of regular expressions)
* 3/31
	* - [ ] Relesae shadow CT with end-to-end functionality, Meerkat v1.1.0

### RECENTLY COMPLETED
#### Matt
* 3/2
	* - [x] Submit pull request for unit tests in meerkat.producer
* 2/28
	* - [x] Compare the lists of ***txn_type*** and ***txn_sub_type*** labels between Platform and Meerkat
	* - [x] Add semantic versioning for the v1.0.0 version of Meerkat
	* - [x] Provide thoughtful response to how to find and predict transaction types.

#### Andy
* 3/2
	* - [x] Merge fix_unittests pull request 
* 2/28
	* - [x] Phone screen for Sivan, a new summer intern
	* - [x] Update the Meerkat daemon to give CT testing data
	* - [x] :+1: Deploy Meerkat v1.0.0 to production VPC 
	* - [x] Set up a [distribution list] (https://github.com/joeandrewkey/meerkat_webservice_schema/issues/21) or equivalent for labels
	* - [x] Obtain agreement on requiring ***ledger_entry*** in meerkat web service v1.0.0

