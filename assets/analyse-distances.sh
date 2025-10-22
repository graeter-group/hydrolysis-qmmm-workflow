#!/bin/env bash
job=$1

# get timestep from mdp file
dt=$(grep dt ${job}.mdp | awk '{print $3}')
# get k from mdp file
k=$(grep pull-coord3-k ${job}.mdp | awk '{print $3}')
rate=$(grep pull-coord3-rate ${job}.mdp | awk '{print $3}')
echo $dt $k $rate > ${job}-info.txt

cp ${job}.ndx ${job}-mindist.ndx
# create group Prot_H_f0_t0.000
gmx select -s ${job}.tpr -f ${job}-start.gro -n ${job}.ndx  -on ${job}-tmp.ndx -select '"ProtH" group "Protein" and type "H*"'
echo ""  >> ${job}-mindist.ndx
cat ${job}-tmp.ndx >> ${job}-mindist.ndx
rm ${job}-tmp.ndx

# get distance from O of OH to C of peptide bond
echo -e 'O_OH\nC_CARBONYL' | gmx mindist -dt 0.01 -f ${job}.xtc -s ${job}.tpr -n ${job}-mindist.ndx -od ${job}-mindist-o-c.xvg

# get distance from O of OH to H of protein
echo -e 'O_OH\nProtH' | gmx mindist -dt 0.01 -f ${job}.xtc -s ${job}.tpr -n ${job}-mindist.ndx -od ${job}-mindist-o-h.xvg

# get distances from C of peptide bond to QM water and OH
echo -e 'C_CARBONYL\nQWOH' | gmx pairdist -dt 0.01 -f ${job}.xtc -s ${job}.tpr -n ${job}-mindist.ndx -o ${job}-mindist-c-w.xvg -selrpos mol_com -seltype mol_com -selgrouping mol


