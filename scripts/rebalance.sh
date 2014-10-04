curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "cluster.routing.allocation.balance.primary": 1.00 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "cluster.routing.allocation.balance.threshold": 0.5 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "cluster.routing.allocation.balance.index": 1.0 } }'
curl -XPUT ${1}:9200/_cluster/settings -d '{ "transient" : { "cluster.routing.allocation.cluster_concurrent_rebalance": -1 } }'

#curl -XPUT s01:9200/user_index/_settings -d '{ "index.routing.allocation.include._id" : "bZ9Jie5gRIuOR6GjGrOM5A,vlBOkbe1R_m0LsGEYMaRIA,Lug5dJcNRfu75argZsygIA,82rU8C1tQz-NAvaypdiB0A,J9swJloIQ2Kkza5ZwzhWcw,m8b4Or4sQeKHDlZsQT5cyg,QWO7UUrlQ-2B2Jrd6j5-xw,sdUzW2vfTQ2PCQZe3b3-jQ,XK9HV45NT4Wy83XRPghz1g,zqD41G58Qdu4O1JL9mdIOw,NKb1zLkfTHCNXUvqUTWPFQ,OjoUa3FCTnCd50UAbQ8c8A,ncc_QuizSqm_g9mKzjcZuQ,lewpLcFyStmps35rN8JnEQ,jsCnZPpHT4mdJcvZmpCqYQ,q_wH_OWUSZOTABYBOtOTxQ,ctzPLErzScaRkfl4mlDMVw,NpteXbIQSx-pI_7FcaixKQ" }'
