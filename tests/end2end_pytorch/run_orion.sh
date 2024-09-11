# exit at the first error
set -e

# go to the examples folder and run the example
cd $GITHUB_WORKSPACE/examples/local_orion
sh run.sh
mv orion_working_dir orion_working_dir_OLD
# re-run the example to check reproducibility
rm -fr orion_db*
sh run.sh
# check results are the same
echo "results are:"
cat orion_working_dir*/*/results.txt
DIFF_LINES=`grep "best_dev_metric" orion_working_dir*/*/results.txt | sed 's@^.*best_dev_metric: @@g' | sort | uniq | wc -l`
if [ ${DIFF_LINES} -gt 2 ]; then # note we have two trials per experiment, this is why we can have 2 different results - but not more
    echo "ERROR: two identical Orion runs produced different output results - review seed implementation"
    exit 1
else
    echo "PASS: two identical Orion runs produced the same output results."
fi
