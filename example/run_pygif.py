from ase.io import read
from PyGiF import PyGiF
from copy import copy
from my_vasp_calculator import VASP # Check https://github.com/mind-scraper/my_vasp_calculator

# Read atoms
atoms = read('structure.xyz') # Make sure this is relaxed structure

# Set calculator
calc = VASP(command='mpirun -np 8 vasp_std > vasp.log')

atoms.calc = copy(calc)

# Select the atoms to consider for vibrational analysis (C, O, and H)
indices = [atom.index for atom in atoms if atom.symbol in ['C', 'H']]

# Vibrational analysis
vib = PyGiF(atoms, N_Proc=9, calc=copy(calc), ASR=True)
vib.gif(2)
