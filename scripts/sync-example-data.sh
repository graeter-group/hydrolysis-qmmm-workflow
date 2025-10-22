#!/bin/env bash

d="run_6/f-603/ixs-178-180/single/frame-96"
source="zeus:/hits/fast/mbm/buhrjk/phd/col-hydrolysis/qmmm/hydrolysis/$d"
mkdir -p "$d"

rsync --stats -h --recursive --exclude='*.pdb' --exclude='*.out' --exclude='*.xvg' --exclude='*.xtc' --exclude='*.trr' --exclude='*.tpr' "${source}/*" "$d"
rsync --stats -h "${source}/wethyd.tpr" "$d"
rsync --stats -h "${source}/wetbreak.tpr" "$d"


d="run_6/f-603/ixs-178-180/triple/frame-96"
source="zeus:/hits/fast/mbm/buhrjk/phd/col-hydrolysis/qmmm/hydrolysis/$d"
mkdir -p "$d"

rsync --stats -h --recursive --exclude='*.pdb' --exclude='*.out' --exclude='*.xvg' --exclude='*.xtc' --exclude='*.trr' --exclude='*.tpr' "${source}/*" "$d"
rsync --stats -h "${source}/wethyd.tpr" "$d"
rsync --stats -h "${source}/wetbreak.tpr" "$d"


