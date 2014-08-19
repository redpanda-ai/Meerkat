#!/bin/bash

KEY="/root/.ssh/meerkat.pem"

ssh -i ${KEY}  s01 ${1}
ssh -i ${KEY}  s02 ${1}
ssh -i ${KEY}  s03 ${1}
ssh -i ${KEY}  s04 ${1}
ssh -i ${KEY}  s05 ${1}
ssh -i ${KEY}  s06 ${1}
ssh -i ${KEY}  s07 ${1}
ssh -i ${KEY}  s08 ${1}
ssh -i ${KEY}  s09 ${1}
ssh -i ${KEY}  s10 ${1}
ssh -i ${KEY}  s11 ${1}
ssh -i ${KEY}  s12 ${1}
ssh -i ${KEY}  s13 ${1}
ssh -i ${KEY}  s14 ${1}
ssh -i ${KEY}  s15 ${1}
ssh -i ${KEY}  s16 ${1}
ssh -i ${KEY}  s17 ${1}
ssh -i ${KEY}  s18 ${1}

