rm -fr logs
mkdir logs
folder=`pwd`
name=`basename ${folder}`
sbatch --job-name ${name} to_submit.sh
