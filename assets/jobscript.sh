#!/bin/bash
#SBATCH -p cascade.p
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --mincpus=40
#SBATCH --exclusive
#SBATCH --cpus-per-task=1

echo "Job Name:"
echo "$SLURM_JOB_NAME"

source ./_modules.sh

JOB=$1
# max runtime on the cluster is 24 hours
# gromacs will initiate a stop at (CYCLE - 0.1) hours,
# attempting to run another 100 steps
# those 100 steps take about 1 to 2 hours.
# add a little buffer:
CYCLE=20
SUBMIT="jobscript.sh $JOB"

targetFinished="slurm-finished-${JOB}.info"

ntomp=1

START=$(date +"%s")
# get the name of the output file from the slurm job id
SLURMOUT="slurm-${SLURM_JOB_ID}.out"

np=40
# -np $np

mpirun -v gmx mdrun -s ${JOB}.tpr -cpi ${JOB}.cpt -x ${JOB}.xtc -o ${JOB}.trr -cpo ${JOB}.cpt -c ${JOB}.gro -g ${JOB}.log -e ${JOB}.edr -px ${JOB}_pullx.xvg -pf ${JOB}_pullf.xvg -ro ${JOB}-rotation.xvg -ra ${JOB}-rotangles.log -rs ${JOB}-rotslabs.log -rt ${JOB}-rottorque.log -maxh ${CYCLE} -dlb yes -ntomp $ntomp


END=$(date +"%s")
LEN=$((END-START))
HOURS=$((LEN/3600))
echo "$LEN seconds ran"
echo "$HOURS full hours ran"
let "CYCLE--"
if [ $HOURS -lt $CYCLE ]; then
  echo "last cycle was just $HOURS h long and therefore finito"
  touch $targetFinished
  # save latest slurm output file
  cp $SLURMOUT ${JOB}-slurm.out
  exit 3
else
  echo "cycle resubmitting"
  sbatch -J $SLURM_JOB_NAME ${SUBMIT}
  exit 2
fi


