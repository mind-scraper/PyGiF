# PyGiF
Simple ASE interface for GiF program.

GiF is a vibrational analysis program that calculates the vibrational frequencies of an atomic structure given a potential energy (and forces) calculator. 
It is part of STATE program developed by Morikawa-Hamada Group. 
Check STATE documentation at the following link
https://prec.eng.osaka-u.ac.jp/06/puki_state/

Instead of using GiF only as an extension of STATE program, this repository makes it possible to use any ASE calculator. 

Yo don't need to install all STATE executable. Just compiled GiF using any fortran compiler and include it in $PATH.
Make sure to include PyGiF in $PYTHONPATH.
