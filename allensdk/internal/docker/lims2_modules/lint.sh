set -x
REPORT_DIR=/root/allensdk/htmlcov
cd /shared/bioapps/infoapps/lims2_modules/mousecelltypes
find . -name '*.py' -exec pylint --disable=C {} \; >> $REPORT_DIR/pylint.txt || exit 0
grep 'import-error' $REPORT_DIR/pylint.txt > $REPORT_DIR/pylint_imports.txt
cd /shared/bioapps/infoapps/lims2_modules/lib/allensdk
pylint --disable=C allensdk > $REPORT_DIR/pylint_allensdk.txt || exit 0
grep 'import-error' $REPORT_DIR/pylint_allensdk.txt > $REPORT_DIR/pylint_allensdk_imports.txt
