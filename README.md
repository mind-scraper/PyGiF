# PyGiF
Simple ASE interface for GiF program.

GiF is a vibrational analysis program that calculates the vibrational frequencies of an atomic structure given a potential energy (and forces) calculator. 
One advantage of using GiF is that the dynamical matrix can be refined repeatedly by explicitly calculating forces along normal coordinates, instead of only Cartesian coordinates. 

GiF is part of STATE program developed by Morikawa-Hamada Group. 
Check STATE documentation at the following link
https://prec.eng.osaka-u.ac.jp/06/puki_state/

Instead of using GiF only as an extension of STATE program, this repository makes it possible to use any ASE calculator. 

You don't need to compile all STATE executables. Just compile GiF using Fortran compiler and include it in $PATH.
Make sure to include PyGiF in $PYTHONPATH.
