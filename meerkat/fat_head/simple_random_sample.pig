/* This sample uses Hive's HCatalog to tap into Hive tables
    You should run it this way: 
    [~]$ pig -useHCatalog -f my_script.pig
*/
--This jar file contains the User Defined Function for the Simple Random Sample.
register s3://s3yodlee/meerkat/cnn/datafu-pig-incubating-1.3.0.jar;

--We are asking for 1% of the population, without replacement
DEFINE SRS datafu.pig.sampling.SimpleRandomSample('0.01');

--Get the data just as Hive would, using HCatalog definitions
A = LOAD 'transactions' USING org.apache.hive.hcatalog.pig.HCatLoader();

--Do the sampling
sampled = FOREACH (GROUP A ALL) GENERATE FLATTEN(SRS(A));

--Store the result as a pipe-delimited file back into HDFS
STORE sampled INTO 'transaction_sample.csv' using PigStorage('|');
