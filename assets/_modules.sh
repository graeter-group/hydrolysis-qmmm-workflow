#!/bin/bash

if [[ "$(hostname)" == "cascade"* ]]; then
  ## cluster
  module load OpenMPI
  source /hits/fast/mbm/buhrjk/sw/cp2k/tools/toolchain/install/setup
  PATH="/hits/fast/mbm/buhrjk/sw/cp2k/exe/local:$PATH"
  PATH="/hits/fast/mbm/buhrjk/sw/gromacs/build/bin:$PATH"
elif [[ "$(hostname)" == "o05i15"* ]]; then
  module use /opt/bwhpc/common/modulefiles/Compiler/
  module load gnu/12.1/mpi/openmpi/4.1
  source $HOME/sw/cp2k/tools/toolchain/install/setup
  PATH="~/sw/cp2k/exe/local:$PATH"
  PATH="~/sw/gromacs/build/bin:$PATH"
elif [[ "$(hostname)" == "pop-desktop" ]]; then
  ## local workstation
  # source $HOME/sw/qm/cp2k/tools/toolchain/install/setup
  # PATH="$HOME/sw/qm/cp2k/exe/local:$PATH"
  # PATH="$HOME/sw/qm/gromacs/build/bin:$PATH"
  :
elif [[ "$(hostname)" == "pop-laptop" ]]; then
  # laptop
  :
elif [[ "$(hostname)" == "pop-os" ]]; then
  # laptop
  :
else
  ## workstation
  source /sw/mbm/qm/cp2k/tools/toolchain/install/setup
  PATH="/sw/mbm/qm/cp2k/exe/local:$PATH"
  PATH="/sw/mbm/qm/gromacs/build/bin:$PATH"
fi

export PATH

