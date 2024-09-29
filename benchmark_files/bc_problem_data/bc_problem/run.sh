#!/bin/bash

#cd bc_problem_optimization
rm -rf postProcess*
source /opt/openfoam10/etc/bashrc
laplacianFoam
cd ..
