#!/bin/bash

source ./_modules.sh

JOB="$1"

# -nt 16 -pin on 
# -ntmpi 1

gmx mdrun -nt 16 -pin on -s ${JOB}.tpr -cpi ${JOB}.cpt -x ${JOB}.xtc -o ${JOB}.trr -cpo ${JOB}.cpt -c ${JOB}.gro -g ${JOB}.log -e ${JOB}.edr -px ${JOB}_pullx.xvg -pf ${JOB}_pullf.xvg -ro ${JOB}-rotation.xvg -ra ${JOB}-rotangles.log -rs ${JOB}-rotslabs.log -rt ${JOB}-rottorque.log -dlb yes 


