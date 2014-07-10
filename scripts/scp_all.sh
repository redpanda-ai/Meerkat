#!/bin/bash

KEY="/root/.ssh/meerkat.pem"

scp -i ${KEY} ${1} s01:${1}
scp -i ${KEY} ${1} s02:${1}
scp -i ${KEY} ${1} s03:${1}
scp -i ${KEY} ${1} s04:${1}
scp -i ${KEY} ${1} s05:${1}
scp -i ${KEY} ${1} s06:${1}
scp -i ${KEY} ${1} s07:${1}
scp -i ${KEY} ${1} s08:${1}
scp -i ${KEY} ${1} s09:${1}
scp -i ${KEY} ${1} s10:${1}
scp -i ${KEY} ${1} s11:${1}
scp -i ${KEY} ${1} s12:${1}
scp -i ${KEY} ${1} s13:${1}
scp -i ${KEY} ${1} s14:${1}
scp -i ${KEY} ${1} s15:${1}
scp -i ${KEY} ${1} s16:${1}
scp -i ${KEY} ${1} s17:${1}
scp -i ${KEY} ${1} s18:${1}

