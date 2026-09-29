""" 
PyGiF: ASE interface for GiF program

Written by
Samuel Eka Putra Payong Masan
Ph.D. student at Morikawa-Hamada Group
The University of Osaka
September 2026
######

Example
    >>> from ase.optimize import BFGS as relaxer
    >>> from ase import Atoms
    >>> from ase.calculators.emt import EMT
    >>> from PyGiF import PyGiF
    >>> from copy import copy
    >>> atoms = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)])
    >>> atoms.set_cell([30, 30, 30])
    >>> calc = EMT()
    >>> atoms.calc = copy(calc)
    >>> opt = relaxer(atoms)
    >>> opt.run(fmax=0.01)
    >>> vib = PyGiF(atoms, N_Proc=1, calc=copy(calc), ASR=False) # ASR can be used when all atoms in the system is included in vibratioal calculations. It will skip calculating the free translation in gif(2). Set this to ASR=False for example in calculating vibration of adsorbate where Surface vibration is not considered. 
    >>> vib.gif(1)
             =========             
              SUMMARY              
             =========             

 MODE  WR       : NU(meV)  NU(cm-1)
    1 -0.62D-26 :    0.00      0.00
    2  0.22D-19 :    0.00      0.00
    3  0.35D-17 :    0.00      0.00
    4  0.50D-05 :    1.42     11.47
    5  0.50D-05 :    1.42     11.47
    6  0.57D-01 :  152.11   1226.82
Zero point energy = 77.475 meV

Use with care ^_^
"""

import numpy as np
from ase.io import read
from ase import units
import os
from copy import copy
from ase.calculators.singlepoint import SinglePointCalculator
import re
from multiprocessing import Pool
from ase import Atoms

class PyGiF():
    def __init__(self, atoms, 
                 indices=None, 
                 delta=0.01, 
                 N_Proc=1, 
                 ASR=False, # Remove free translation mode
                 calc=None, 
                 ):
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        anew = Atoms(symbols=symbols, positions=positions)
        anew.set_cell(cell)
        anew.set_pbc([True, True, True])

        self.atoms = anew
        self.atoms.set_constraint() # Remove constrains
        self.calc = calc
        self.N_Proc = N_Proc
        self.ASR = ASR
        self.delta = delta

        if indices == None:
            self.indices = [atom.index for atom in self.atoms]
        else:
            self.indices = indices        
        
        Nmode = 3*len(self.indices)      
        self.Nmode = Nmode

    def gif(self, N):
        """
        Main part of the code.
        """       

        # Checking vib directory
        gif_iter = 0
        for dir_i in range(N):
            if os.path.isdir(f"vib{dir_i}"):
                print(f"Displacement directory excist vib{dir_i}.")
                print(f"Displacement files inside the directory will be used.")
                print(f"Please make sure this is intended!\n")

        if gif_iter == 0:
            print(f'\n###Step : {gif_iter}###')
            print('Displacement in cartesian coordinate.')
            print('Atomic positions, energy, and forces will be written inside vib0.')
            self.make_xyz_displacements(n_processes=self.N_Proc)
            self.write_nfforce("vib0", "nfforce_0.data")
            print("Running vibrational analysis with gif . . .")
            os.system("gif -f nfforce_0.data -o gif_0.out")
            os.system("mv vib.data vib_0.data")
            os.system(f"tail -{5+len(self.indices*3)} gif_0.out")

            os.system(f"tail -{len(self.indices*3)} gif_0.out > tmp.log")
            with open(f'tmp.log', 'r') as file:
                data = file.readlines()
                data = [line.split() for line in data]
                data = np.array(data)
            signs = data[:, 1]
            signs = np.array([float(i.replace('D', 'E')) for i in signs])
            hnus = np.array(data[:, 3].astype(float))
            zpe = 0
            for i in range(len(hnus)):
                if signs[i]>=0:
                    zpe += hnus[i]
            print(f"Zero point energy = {zpe/2:.3f} meV")        
            print("\nFull output of gif is written to gif_0.out.")
            print("Suggested displacement is written to vib_0.data")           
            self.write_normal_mode_animations("vib_0.data", foldername="anim0", temperature=300) 
            gif_iter +=1

        while gif_iter < N:             
            print(f'\n###Step : {gif_iter}###')
            print('Displacement in normal coordinate.')
            print(f'Atomic positions, energy, and forces will be written inside vib{gif_iter}.')
            self.read_vib_data_and_calculate_forces(
                f"vib{gif_iter}",
                f"vib_{gif_iter-1}.data",
                n_processes=self.N_Proc,
                ASR=self.ASR
                )
            self.write_nfforce(f"vib{gif_iter}", f"nfforce_{gif_iter}.data")
            print("Running vibrational analysis with gif . . .")
            os.system(f"gif -f nfforce_{gif_iter}.data -o gif_{gif_iter}.out")
            os.system(f"mv vib.data vib_{gif_iter}.data")
            os.system(f"tail -{5+len(self.indices*3)} gif_{gif_iter}.out")

            os.system(f"tail -{len(self.indices*3)} gif_{gif_iter}.out > tmp.log")
            with open(f'tmp.log', 'r') as file:
                data = file.readlines()
                data = [line.split() for line in data]
                data = np.array(data)
            signs = data[:, 1]
            signs = np.array([float(i.replace('D', 'E')) for i in signs])
            hnus = np.array(data[:, 3].astype(float))
            zpe = 0
            for i in range(len(hnus)):
                if signs[i]>=0:
                    zpe += hnus[i]
            print(f"Zero point energy = {zpe/2:.3f} meV")

            print(f"\nFull output of gif is written to gif_{gif_iter}.out.")
            print(f"Suggested displacement is written to vib_{gif_iter}.data")
            self.write_normal_mode_animations(f"vib_{gif_iter}.data", foldername=f"anim{gif_iter}", temperature=300)

            gif_iter +=1            
        
    def make_xyz_displacements(self, n_processes=1):

        os.makedirs("vib0", exist_ok=True)

        jobs = []

        for i in self.indices:
            for j in range(3):
                for sign in [1, -1]:
                    jobs.append(
                        (
                            self.atoms,
                            i,
                            j,
                            self.delta,
                            sign
                        )
                    )

        with Pool(processes=n_processes) as pool:
            pool.map(self.calculate_xyz_displacement, jobs)

    def calculate_xyz_displacement(self, args):

        atoms, i, j, delta, sign = args

        direction = ["x", "y", "z"][j]
        sign_string = "+" if sign > 0 else "-"

        filename = f"vib0/{i}{direction}{sign_string}.xyz"

        # Skip if calculation already exists
        if os.path.exists(filename):
            print(f"Skipping {filename}")
            return

        # Displace atoms
        positions = copy(atoms.get_positions())
        positions[i, j] += sign * delta

        disp_structure = atoms.copy()
        disp_structure.set_positions(positions)
        disp_structure.calc = copy(self.calc)

        # Calculate energy and forces
        disp_structure.get_potential_energy()
        disp_structure.get_forces()

        # Save structure + calculator results
        disp_structure.write(filename)

        print(f"Calculated {filename}")

    def write_nfforce(self, dir_name, file_name):        
        with open(file_name, "w") as f:
            f.write(" &ATOM\n")
            f.write(f"    {len(self.atoms)}\n")
            positions = self.atoms.get_positions() / units.Bohr
            atomic_numbers = self.atoms.get_atomic_numbers()
            masses = self.atoms.get_masses()
            for i in range(len(atomic_numbers)):
                positions_i = positions[i]
                f.write(f"    {i+1:.0f}   {positions_i[0]:.12f}    {positions_i[1]:.12f}    {positions_i[2]:.12f}  {atomic_numbers[i]:.0f} {masses[i]*1822.89:.4f}\n")
            f.write(" &END\n")
            f.write(" &FORCE\n")

        files = os.listdir(dir_name)
        if dir_name == "vib0":    
            files = sorted(files, key=self.sort_key_vib1)
        else:
            files = sorted(files, key=self.sort_key_vib_other)
        print("Making sure the structure files are sorted correctly")
        print(files)
        for i in range(len(files)):
            
            atoms = read(f"{dir_name}/{files[i]}")
            energy = atoms.get_potential_energy() / units.Hartree

            with open(file_name, "a") as f:
                f.write(f"     {i+1:.0f}    {energy:.12f}\n")
            
            positions = atoms.get_positions() / units.Bohr
            forces = atoms.get_forces() / (units.Hartree/units.Bohr)

            for j in range(len(positions)):
                positions_j = positions[j]
                forces_j = forces[j]

                with open(file_name, "a") as f:
                    f.write(f"     {j+1:.0f}    {positions_j[0]:.12f}    {positions_j[1]:.12f}    {positions_j[2]:.12f} {forces_j[0]:.12f}  {forces_j[1]:.12f}  {forces_j[2]:.12f} \n")

    def sort_key_vib1(self, s):
        match = re.match(r"(\d+)([xyz])([+-])\.xyz", s)
        if match:
            number = int(match.group(1))
            axis = match.group(2)
            sign = match.group(3)
            return (number, axis, 0 if sign == '+' else 1)  # '+' first, then '-'
        else:
            return (float('inf'), '', 0)                    

    def calculate_q_displacement(self, args):
        (
            i,
            lines,
            positions,
            atoms,
            calc,
            foldername,
            mode,
            ASR,
        ) = args

        natoms = len(atoms)

        # -----------------------------
        # Read displacement information
        # -----------------------------

        current_line = i * (natoms + 1)

        factor = lines[current_line].split()
        factor = float(factor[1].replace("D", "E")) * units.Bohr

        displacement = np.zeros((natoms, 3))

        for j in range(natoms):
            current_line += 1

            disp_values = lines[current_line].split()

            displacement[j] = np.array([
                float(disp_values[1].replace("D", "E")),
                float(disp_values[2].replace("D", "E")),
                float(disp_values[3].replace("D", "E")),
            ]) * factor

        # =============================
        # Check Acoustic Sum Rule
        # =============================

        apply_asr = ASR and i < 3

        if apply_asr:
            print(
                f"Mode {i + 1}: ASR enabled "
                f"-> energy and forces set to zero"
            )

        # =============================
        # Minus displacement
        # =============================

        filename_min = f"{foldername}/disp{i}_min.xyz"

        if os.path.exists(filename_min):

            print(f"Skipping {filename_min}")

        else:

            disp = positions - displacement

            disp_atoms = atoms.copy()
            disp_atoms.set_positions(disp)

            if apply_asr:

                # Artificially set energy and forces to zero
                E = 0.0
                F = np.zeros((natoms, 3))

                results = {
                    "energy": E,
                    "forces": F
                }

                calc_sp = SinglePointCalculator(
                    disp_atoms,
                    **results
                )

                disp_atoms.calc = calc_sp

            else:
                disp_atoms.calc = calc

                # Calculate energy and forces
                disp_atoms.get_potential_energy()
                disp_atoms.get_forces()

            disp_atoms.write(filename_min)

            print(f"Calculated {filename_min}")

        # =============================
        # Plus displacement
        # =============================

        filename_plus = f"{foldername}/disp{i}_plus.xyz"

        if os.path.exists(filename_plus):

            print(f"Skipping {filename_plus}")

        else:

            disp = positions + displacement

            disp_atoms = atoms.copy()
            disp_atoms.set_positions(disp)

            if apply_asr:

                # Artificially set energy and forces to zero
                E = 0.0
                F = np.zeros((natoms, 3))

                results = {
                    "energy": E,
                    "forces": F
                }

                calc_sp = SinglePointCalculator(
                    disp_atoms,
                    **results
                )

                disp_atoms.calc = calc_sp

            else:
                disp_atoms.calc = calc

                # Calculate energy and forces
                disp_atoms.get_potential_energy()
                disp_atoms.get_forces()

            disp_atoms.write(filename_plus)

            print(f"Calculated {filename_plus}")

        return i

    def read_vib_data_and_calculate_forces(
        self,
        foldername,
        filename,
        n_processes=1,
        ASR=False,
    ):

        os.makedirs(foldername, exist_ok=True)

        # Read vibration data
        with open(filename, "r") as f:
            lines = f.readlines()

        atoms = copy(self.atoms)
        positions = atoms.get_positions()

        mode = 3 * len(self.indices)

        # Create one job for each vibrational mode
        jobs = []

        for i in range(mode):
            jobs.append(
                (
                    i,
                    lines,
                    positions,
                    atoms,
                    copy(self.calc),
                    foldername,
                    mode,
                    ASR,
                )
            )

        # Parallel calculation
        with Pool(processes=n_processes) as pool:

            results = pool.map(
                self.calculate_q_displacement,
                jobs,
                chunksize=1
            )

        print(f"Finished {len(results)} vibrational modes.")    

    def sort_key_vib_other(self, s):        
        match = re.match(r"disp(\d+)_(plus|min)\.xyz", s)
        if match:
            number = int(match.group(1))
            sign = match.group(2)
            return (number, 0 if sign == 'plus' else 1)  # 'plus' before 'min'
        else:
            return (float('inf'), 0)        

    def write_normal_mode_animations(
        self,
        filename,
        foldername="anim",
        temperature=300.0,
    ):
        """
        Write temperature-dependent normal-mode animations.

        The maximum displacement is

            A = sqrt(kB * T / |frequency|)

        where frequency is given in cm^-1.

        Modes with frequency <= 1 cm^-1 are skipped.
        """

        os.makedirs(foldername, exist_ok=True)

        # --------------------------------
        # Read vibrational data
        # --------------------------------

        with open(filename, "r") as f:
            lines = f.readlines()

        natoms = len(self.atoms)
        mode = 3 * len(self.indices)

        positions = self.atoms.get_positions()

        # --------------------------------
        # Loop over all modes
        # --------------------------------

        for i in range(mode):

            print(f"\nWorking on mode {i + 1}/{mode}")

            # --------------------------------
            # Read mode header
            # --------------------------------

            current_line = i * (natoms + 1)

            header = lines[current_line].split()

            # Example:
            #
            # 6  0.15D-01  18  0.37D+00 : 385.98 3113.11
            #
            # header[5] = frequency in THz
            # header[6] = frequency in cm^-1

            frequency_THz = float(
                header[5].replace("D", "E")
            )

            frequency_cm1 = float(
                header[6].replace("D", "E")
            )

            # --------------------------------
            # Read displacement factor
            # --------------------------------

            factor = float(
                header[1].replace("D", "E")
            ) * units.Bohr

            # --------------------------------
            # Read normal-mode displacement
            # --------------------------------

            displacement = np.zeros((natoms, 3))

            for j in range(natoms):

                current_line += 1

                disp_values = lines[current_line].split()

                displacement[j] = np.array([
                    float(disp_values[1].replace("D", "E")),
                    float(disp_values[2].replace("D", "E")),
                    float(disp_values[3].replace("D", "E")),
                ]) * factor

            # --------------------------------
            # Normalize eigenvector
            # --------------------------------

            norm = np.linalg.norm(displacement)

            if norm < 1e-12:

                print(
                    f"Mode {i + 1}: "
                    "zero displacement. Skipping."
                )

                continue

            displacement /= norm

            # --------------------------------
            # Check frequency
            # --------------------------------

            if frequency_cm1 <= 1.0:

                print(
                    f"Mode {i + 1}: "
                    f"{frequency_cm1:.2f} cm^-1 "
                    "(zero/imaginary mode). Skipping."
                )

                continue

            # --------------------------------
            # Temperature-dependent amplitude
            # --------------------------------

            amplitude = np.sqrt(
                units.kB * temperature
                / abs(frequency_cm1 * units.invcm)
            )

            print(
                f"Frequency = {frequency_cm1:.2f} cm^-1 "
                f"({frequency_THz:.2f} THz)"
            )

            print(
                f"Maximum displacement at "
                f"{temperature:.1f} K = "
                f"{amplitude:.4f} Å"
            )

            # --------------------------------
            # Generate displacement amplitudes
            # --------------------------------

            dr = amplitude / 7.0

            all_r = np.array([
                -amplitude + dr * j
                for j in range(15)
            ])

            # Same forward/backward trajectory
            # as your original SSCHA code

            trajectory_r = list(all_r)

            for r_i in range(len(all_r) - 2, 0, -1):
                trajectory_r.append(all_r[r_i])

            # --------------------------------
            # Output file
            # --------------------------------

            animation_file = (
                f"{foldername}/mode_{i + 1:03d}.xyz"
            )

            # Remove existing animation

            if os.path.exists(animation_file):
                os.remove(animation_file)

            # --------------------------------
            # Write trajectory
            # --------------------------------

            for r in trajectory_r:

                disp_atoms = self.atoms.copy()

                pos = positions + r * displacement

                disp_atoms.set_positions(pos)

                disp_atoms.write(
                    animation_file,
                    append=True
                )

            print(
                f"Animation written to "
                f"{animation_file}"
            )

        print(
            f"\nAll normal-mode animations written "
            f"to '{foldername}/'."
        )