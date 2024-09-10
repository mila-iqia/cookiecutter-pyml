# exit at the first error
set -e

# go to the examples folder and run the example
cd $GITHUB_WORKSPACE/examples/local
sh run.sh
mv output output_OLD
# re-run the example to check reproducibility
sh run.sh
# check results are the same
echo "results are:"
cat output*/results.txt
DIFF_LINES=`cat output*/results.txt | uniq | wc -l`
if [ ${DIFF_LINES} -gt 1 ]; then
    echo "ERROR: two identical runs produced different output results - review seed implementation"
    exit 1
else
    echo "PASS: two identical runs produced the same output results."
fi