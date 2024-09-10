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

# now run eval and store the results on valid in a variable
EVAL_RESULT=`sh eval.sh | grep "Validation Metrics"`
CLEANED_EVAL_RESULT=`echo $EVAL_RESULT | sed 's/.*: //g' | sed 's/}.*//g'`
TRAIN_RESULT=`cat output/results.txt`
CLEANED_TRAIN_RESULT=`echo ${TRAIN_RESULT} | sed 's/.*: //g'`

echo "train results: ${CLEANED_TRAIN_RESULT} / eval results: ${CLEANED_EVAL_RESULT}"

# Compare the two values, formatted to 5 decimal places
if ! [ "$(printf "%.5f" "$CLEANED_EVAL_RESULT")" = "$(printf "%.5f" "$CLEANED_TRAIN_RESULT")" ]; then
  echo "results are NOT equal up to 5 decimal places."
  exit 1
else
  echo "results are equal."
fi