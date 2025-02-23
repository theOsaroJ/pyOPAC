from pyscf import gto, dft, solvent
import math
import os

# Constants
HARTREE_TO_EV = 27.211386  # Hartree to eV
xyz_file = 'train.xyz'  # Replace with your actual file

with open(xyz_file, 'r') as f:
    # Remove empty lines and strip whitespace
    lines = [l.strip() for l in f if l.strip()]

idx = 0
molecule_count = 0

# Open results in write mode with line-buffering
# buffering=1 means line-buffered if output is to a terminal. If not a terminal, it may still buffer.
with open("results_train.dat", "w", buffering=1) as f_out:
    # Optional header
    # f_out.write("Gas_Phase_E(eV),DipX(Debye),DipY(Debye),DipZ(Debye),Sol_Phase_E(eV),Solv_Free_E(eV)\n")

    while True:
        if idx >= len(lines):
            break

        # The line with the number of atoms should be a digit
        if not lines[idx].isdigit():
            # Not a well-formed XYZ block, stop parsing
            break

        natoms = int(lines[idx])
        idx += 1
        if natoms <= 0 or (idx + natoms) > len(lines):
            # Invalid block or not enough lines left
            break

        # Next line is a comment
        comment_line = lines[idx]
        idx += 1

        # Next natoms lines are coordinates
        atom_lines = lines[idx:idx+natoms]
        idx += natoms

        atoms = []
        for line in atom_lines:
            fields = line.split()
            if len(fields) < 4:
                continue
            atom_symbol = fields[0]
            x, y, z = fields[1], fields[2], fields[3]
            atoms.append(f"{atom_symbol} {x} {y} {z}")

        if len(atoms) == 0:
            break

        # Build and compute
        mol = gto.M(
            atom=atoms,
            basis='sto-3g',
            unit='Angstrom',
            charge=0,
            spin=0
        )

        mf_gas = dft.RKS(mol)
        mf_gas.xc = 'LDA'
        mf_gas.verbose = 0
        gas_phase_energy = mf_gas.kernel()

        # Dipole in Debye
        dipole = mf_gas.dip_moment(mol, unit='Debye')

        # Solvent calculation
        mf_solvent = solvent.ddCOSMO(mf_gas).run()
        solution_phase_energy = mf_solvent.e_tot

        # Solvation free energy
        solvation_free_energy = solution_phase_energy - gas_phase_energy

        # Convert to eV
        gas_phase_energy_eV = gas_phase_energy * HARTREE_TO_EV
        solution_phase_energy_eV = solution_phase_energy * HARTREE_TO_EV
        solvation_free_energy_eV = solvation_free_energy * HARTREE_TO_EV

        molecule_count += 1

        # Write results and flush
        f_out.write(f"{gas_phase_energy_eV:.6f},{dipole[0]:.6f},{dipole[1]:.6f},{dipole[2]:.6f},{solution_phase_energy_eV:.6f},{solvation_free_energy_eV:.6f}\n")
        f_out.flush()
        os.fsync(f_out.fileno())

        print(f"Processed molecule {molecule_count}")

print("All possible molecules processed. Results are in results_test.dat.")
