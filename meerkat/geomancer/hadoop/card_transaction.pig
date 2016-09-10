SET mapreduce.job.queuename ad-hoc;

register /home/ssathiam/datafu-1.2.0.jar;

DEFINE SRS datafu.pig.sampling.SimpleRandomSample('0.0001');

A = LOAD 'goldrush.card_transaction' USING org.apache.hcatalog.pig.HCatLoader(); 
B = FILTER A BY c_date >= '2014-08-19';

sampled = FOREACH (GROUP B ALL) GENERATE FLATTEN(SRS(B));
-- explain sampled;
STORE sampled INTO 'card_transaction_sample' using PigStorage('\t');

