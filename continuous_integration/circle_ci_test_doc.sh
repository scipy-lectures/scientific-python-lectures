#!bin/bash

# on circle ci, each command run with it's own execution context so we have to
# activate the conda testenv on a per command basis. That's why we put calls to
# python (conda) in a dedicated bash script and we activate the conda testenv
# here.
source activate testenv

# pipefail is necessary to propagate exit codes
set -o pipefail && make html 2>&1 | tee ~/log.txt

