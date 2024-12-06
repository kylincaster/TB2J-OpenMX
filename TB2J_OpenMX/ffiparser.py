import os
import numpy as np
import scipy.linalg as sl
from ase.units import Ha, Bohr
from ase.io import read
from ase import Atoms
from TB2J.utils import kmesh_to_R, symbol_number
from TB2J.myTB import AbstractTB
import matplotlib.pyplot as plt
from TB2J_OpenMX.cmod._scfout_parser import ffi, lib
from ase.data import atomic_numbers

def validate_element(element_name: str) -> str:
    """Validate if the element name exists in the atomic numbers set."""
    element_name = element_name.strip()
    if element_name not in atomic_numbers:
        raise RuntimeError(f"Invalid element name: `{element_name}`")
    return element_name

def parse_integer(s: str) -> tuple[int, int]:
    """Parse leading integer from a string and return the integer and its length."""
    num_length = 0
    for char in s:
        if char.isnumeric():
            num_length += 1
        else:
            break
    
    if num_length == 0:
        raise RuntimeError(f"Cannot parse `{s}` as an integer")
    return int(s[:num_length]), num_length

ATOMIC_ORBITALS = {
    "s": (1, "s",),
    "p": (2, ("px", "py", "pz")),
    "d": (3, ("dz2", "dx2-y2", "dxy", "dxz", "dyz")),
    "f": (4, ("fz3", "fxz2", "fyz2", "fzx2", "fxyz", "fx3-3xy2", "f3yx2-y3")),
    "g": (5, ("g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9")),
    "h": (6, ("h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11")),
}

def parse_pao_config(pao_config: str) -> list[tuple[int, int, tuple[str, ...]]]:
    """Parse a PAO (Pseudo Atomic Orbital) configuration string."""
    pao_config = pao_config.strip()
    config_length = len(pao_config)
    position = 0
    parsed_orbitals = []

    while position < config_length:
        orbital_type = pao_config[position]
        if orbital_type in ATOMIC_ORBITALS:
            try:
                orbital_count, token_length = parse_integer(pao_config[position + 1:])
                parsed_orbitals.append((orbital_count, *ATOMIC_ORBITALS[orbital_type]))
                position += token_length + 1
            except (IndexError, RuntimeError):
                raise RuntimeError(
                    f"Invalid PAO configuration: `{pao_config}`. Expected format like `s2p1d1`.")
        else:
            raise RuntimeError(
                f"Invalid PAO configuration: `{pao_config}`. Unknown orbital type `{orbital_type}`.")

    return parsed_orbitals




# Create the dictionary mapping ctypes to np dtypes.
ctype2dtype = {'int': 'i4', 'double': 'f8'}

# Integer types
for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = "%s%d_t" % (prefix, 8 * (2**log_bytes))
        dtype = "%s%d" % (prefix[0], 2**log_bytes)
        ctype2dtype[ctype] = np.dtype(dtype)

# Floating point types
ctype2dtype['float'] = np.dtype('f4')
ctype2dtype['double'] = np.dtype('f8')


def asarray(ffi, ptr, length):
    # Get the canonical C type of the elements of ptr as a string.
    T = ffi.getctype(ffi.typeof(ptr).item)

    if T not in ctype2dtype:
        raise RuntimeError("Cannot create an array for element type: %s" % T)

    return np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(T)), ctype2dtype[T])


def reorder(Hk):
    n = np.shape(Hk)[0] // 2
    N = 2 * n
    Hk2 = np.zeros_like(Hk)
    Hk2[0:n, 0:n] = Hk[::2, ::2]
    Hk2[n:N, n:N] = Hk[1::2, 1::2]
    Hk2[0:n, n:N] = Hk[::2, 1::2]
    Hk2[n:N, 0:n] = Hk[1::2, ::2]
    return Hk2


def reorder_back(Hk2):
    n = np.shape(Hk)[0] // 2
    N = 2 * n
    Hk2 = np.zeros_like(Hk)
    Hk2[::2, ::2] = Hk[0:n, 0:n]
    Hk2[1::2, 1::2] = Hk[n:N, n:N]
    Hk2[::2, 1::2] = Hk[0:n, n:N]
    Hk2[1::2, ::2] = Hk[n:N, 0:n]
    return Hk2


def reorder_back_evecs(evecs):
    n = np.shape(evecs)[0] // 2
    N = 2 * n
    evecs2 = np.zeros_like(evecs)
    evecs2[0::2, :] = evecs[0:n, :]
    evecs2[1::2, :] = evecs[n:N, :]
    # evecs2[::2, 1::2] = evecs[0:n, n:N]
    # evecs2[1::2, ::2] = evecs[n:N, 0:n]
    return evecs2


def reorder_and_solve_and_back(Hk, Sk):
    Hk2 = reorder(Hk)
    Sk2 = reorder(Sk)
    evalue, evecs = sl.eigh(Hk2, Sk2)
    evecs = reorder_back_evecs(evecs)
    return evalue, evecs


class OpenmxWrapper(AbstractTB):
    def __init__(self, path, prefix="openmx", dat_file=None):
        self.is_siesta = False
        self.is_orthogonal = False
        xyz_fname = os.path.join(path, prefix + ".xyz")

        self.dat_file = None
        if dat_file is not None:
            if os.path.isfile(dat_file):
                self.dat_file = dat_file
            else:
                raise RuntimeError(f"Cannot find the OpenMX Script file: `{dat_file}`")
        else:
            dat_file = os.path.join(path, prefix + ".dat")
            if os.path.isfile(dat_file):
                self.dat_file = dat_file
        
        fname = os.path.join(path, prefix + ".scfout")
        self.fname = fname
        if not os.path.isfile(self.fname):
            raise RuntimeError(f"Cannot find the OpenMX Hamilton file: `{self.fname}`")


        self.R2kfactor = 2.0j * np.pi
        self.parse_scfoutput()
        self.Rdict = dict()
        for i, R in enumerate(self.R):
            self.Rdict[tuple(R)] = i
        atoms = read(xyz_fname)
        self.atoms = Atoms(
            atoms.get_chemical_symbols(), cell=self.cell, positions=self.positions
        )
        self.norbs_to_basis(self.atoms, self.norbs)
        self.nspin = 2
        self.nbasis = self.nspin * self.norb
        self._name = 'OpenMX'

    def solve(self, k):
        phase = np.exp(self.R2kfactor * (self.R @ k))
        Hk = np.einsum("rij, r->ij", self.H, phase)
        Sk = np.einsum("rij, r->ij", self.S, phase)
        return sl.eigh(Hk, Sk)

    def solve_all(self, kpts):
        nk = len(kpts)
        evals = np.zeros((nk, self.norb))
        evecs = np.zeros((nk, self.norb, self.norb), dtype=complex)
        for ik, k in enumerate(kpts):
            evals[ik], evecs[ik] = self.solve(k)
        return evals, evecs

    def HSE_k(self, k, convention=2):
        phase = np.exp(self.R2kfactor * (self.R @ k))
        Hk = np.einsum("rij, r->ij", self.H, phase)
        Sk = np.einsum("rij, r->ij", self.S, phase)
        # evalue, evec = sl.eigh(Hk, Sk)
        evalue, evec = reorder_and_solve_and_back(Hk, Sk)
        return Hk, Sk, evalue, evec

    def HS_and_eigen(self, kpts):
        nk = len(kpts)
        evals = np.zeros((nk, self.nbasis))
        evecs = np.zeros((nk, self.nbasis, self.nbasis), dtype=complex)
        Hk = np.zeros((nk, self.nbasis, self.nbasis), dtype=complex)
        Sk = np.zeros((nk, self.nbasis, self.nbasis), dtype=complex)
        for ik, k in enumerate(kpts):
            phase = np.exp(self.R2kfactor * (self.R @ k))
            Hk[ik] = np.einsum("rij, r->ij", self.H, phase)
            Sk[ik] = np.einsum("rij, r->ij", self.S, phase)
            evals[ik], evecs[ik] = sl.eigh(Hk[ik], Sk[ik])
        return Hk, Sk, evals, evecs

    def get_hamR(self, R):
        return self.H[self.Rdict[tuple(R)]]

    def norbs_to_basis(self, atoms: Atoms, norbs: list[int]):
        if self.dat_file:
            return self.basis_from_dat_file(atoms)
        else:
            return self._norbs_to_basis(atoms, norbs)

    def basis_from_dat_file(self, atoms: Atoms):
        """Parse the basis set configuration from a dat file and construct basis."""
        symbols = atoms.get_chemical_symbols()

        # Read lines from the dat file
        with open(self.dat_file, 'r') as file:
            lines = file.readlines()

        # Find the indices of the atomic species definition section
        start_index, end_index = -1, -1
        for index, line in enumerate(lines):
            if line.startswith('<Definition.of.Atomic.Species'):
                start_index = index + 1
            if line.startswith('Definition.of.Atomic.Species>'):
                end_index = index
                break

        if start_index == -1 or end_index == -1:
            raise RuntimeError("Cannot find paired tag `Definition.of.Atomic.Species`")

        if (end_index - start_index) < len(symbols):
            raise RuntimeError("Insufficient atomic species definitions in the file")

        # Parse PAOs for each element
        pao_definitions = {}
        pao_input_s = {}
        for line in lines[start_index:end_index]:
            line = line.strip()
            if line.startswith('#'):
                continue
            tokens = line.split()
            element = validate_element(tokens[2].split('_')[0])
            pao_input_s[element] = tokens[1].split('-')[1]
            pao_definitions[element] = parse_pao_config(pao_input_s[element])
        
        # Map atomic tags to their symbols
        atom_tags = list(symbol_number(symbols).keys())
        self.basis = []
        for tag, symbol in zip(atom_tags, symbols):
            basis = []
            if symbol not in pao_definitions:
                raise RuntimeError(f"No PAO definition found for symbol: {symbol}")
            
            paos_for_symbol = pao_definitions[symbol]
            # num_orbs = 0
            for num_paos, pao_order, pao_list in paos_for_symbol:
                # num_orbs += num_paos * len(pao_list)
                for index in range(num_paos):
                    for pao in pao_list:
                        basis.append((tag, f"{pao_order}{pao}N{index+1}", "up"))
                        basis.append((tag, f"{pao_order}{pao}N{index+1}", "down"))

            print(f"{tag} `{pao_input_s[symbol]}`[{len(basis)//2}]: ", end="")
            for i in range(0, len(basis), 2):
                print(basis[i][1], end=" ")
            print("\n")

            self.basis.extend(basis)

        return self.basis


    def _norbs_to_basis(self, atoms, norbs):
        self.basis = []
        symbols = atoms.get_chemical_symbols()
        
        sn = list(symbol_number(symbols).keys())
        print(sn, norbs)
        for i, n in enumerate(norbs):
            for x in range(n):
                self.basis.append((sn[i], f"orb{x+1}", "up"))
                self.basis.append((sn[i], f"orb{x+1}", "down"))
        return self.basis

    def parse_scfoutput(self):
        argv0 = ffi.new("char[]", b"")
        argv = ffi.new("char[]", bytes(self.fname, encoding="ascii"))
        lib.read_scfout([argv0, argv])
        lib.prepare_HSR()
        self.ncell = lib.TCpyCell + 1
        self.natom = lib.atomnum
        self.norbs = np.copy(asarray(ffi, lib.Total_NumOrbs, self.natom + 1)[1:])

        if lib.SpinP_switch == 3:
            self.non_collinear = True
        elif lib.SpinP_switch == 1:
            self.non_collinear = False
        else:
            raise ValueError(
                " The value of SpinP_switch is %s. Can only get J from collinear and non-collinear mode."
                % lib.SpinP_switch
            )

        fnan = asarray(ffi, lib.FNAN, self.natom + 1)

        natn = []
        for iatom in range(self.natom):
            natn.append(asarray(ffi, lib.natn[iatom + 1], fnan[iatom + 1] + 1))

        ncn = []
        for iatom in range(self.natom):
            ncn.append(asarray(ffi, lib.ncn[iatom + 1], fnan[iatom + 1] + 1))
        # atv
        #  x,y,and z-components of translation vector of
        # periodically copied cells
        # size: atv[TCpyCell+1][4];
        atv = []
        for icell in range(self.ncell):
            atv.append(asarray(ffi, lib.atv[icell], 4))
        atv = np.copy(np.array(atv))

        atv_ijk = []
        for icell in range(self.ncell):
            atv_ijk.append(asarray(ffi, lib.atv_ijk[icell], 4))
        atv_ijk = np.array(atv_ijk)
        self.R = atv_ijk[:, 1:]
        tv = []
        for i in range(4):
            tv.append(asarray(ffi, lib.tv[i], 4))
        tv = np.array(tv)
        self.cell = np.copy(tv[1:, 1:]) * Bohr

        rtv = []
        for i in range(4):
            rtv.append(asarray(ffi, lib.rtv[i], 4))
        rtv = np.array(rtv)
        self.rcell = np.copy(rtv[1:, 1:])

        Gxyz = []
        for iatom in range(self.natom):
            Gxyz.append(asarray(ffi, lib.Gxyz[iatom + 1], 60))
        self.positions = np.copy(np.array(Gxyz)[:, 1:4]) * Bohr

        self.MP = np.copy(asarray(ffi, lib.MP, self.natom + 1)[1:])

        self.norb = lib.T_NumOrbs
        norb = self.norb

        if self.non_collinear:
            HR = np.zeros([self.ncell, 4, lib.T_NumOrbs, lib.T_NumOrbs])
            for iR in range(0, self.ncell):
                for ispin in range(lib.SpinP_switch + 1):
                    for iorb in range(lib.T_NumOrbs):
                        HR[iR, ispin, iorb, :] = asarray(
                            ffi, lib.HR[iR][ispin][iorb], norb
                        )

            HR_imag = np.zeros([self.ncell, 3, lib.T_NumOrbs, lib.T_NumOrbs])
            for iR in range(0, self.ncell):
                for ispin in range(3):
                    for iorb in range(lib.T_NumOrbs):
                        HR_imag[iR, ispin, iorb, :] = asarray(
                            ffi, lib.HR_imag[iR][ispin][iorb], norb
                        )

            self.H = np.zeros(
                [self.ncell, lib.T_NumOrbs * 2, lib.T_NumOrbs * 2], dtype=complex
            )

            # up up
            for iR in range(self.ncell):
                self.H[iR, ::2, ::2] = HR[iR, 0, :, :] + \
                    1j * HR_imag[iR, 0, :, :]
                # up down
                self.H[iR, ::2, 1::2] = HR[iR, 2, :, :] + 1j * (
                    HR[iR, 3, :, :] + HR_imag[iR, 2, :, :]
                )
                # down up
                self.H[iR, 1::2, ::2] = HR[iR, 2, :, :] - 1j * (
                    HR[iR, 3, :, :] + HR_imag[iR, 2, :, :]
                )
                # down down
                self.H[iR, 1::2, 1::2] = HR[iR, 1, :, :] + \
                    1j * HR_imag[iR, 1, :, :]
        else:  # collinear
            HR = np.zeros([self.ncell, 4, lib.T_NumOrbs, lib.T_NumOrbs])
            for iR in range(0, self.ncell):
                for ispin in range(lib.SpinP_switch + 1):
                    for iorb in range(lib.T_NumOrbs):
                        HR[iR, ispin, iorb, :] = asarray(
                            ffi, lib.HR[iR][ispin][iorb], norb
                        )

            self.H = np.zeros(
                [self.ncell, lib.T_NumOrbs * 2, lib.T_NumOrbs * 2], dtype=complex
            )

            # up up
            for iR in range(self.ncell):
                self.H[iR, ::2, ::2] = HR[iR, 0, :, :]
                self.H[iR, 1::2, 1::2] = HR[iR, 1, :, :]
        self.efermi = lib.ChemP * Ha
        self.H *= Ha

        SR = np.zeros([self.ncell, lib.T_NumOrbs, lib.T_NumOrbs])
        for iR in range(0, self.ncell):
            for iorb in range(lib.T_NumOrbs):
                SR[iR, iorb, :] = asarray(ffi, lib.SR[iR][iorb], norb)
        self.S = np.kron(SR, np.eye(2))
        lib.free_HSR()
        lib.free_scfout()
        print("Loading from scfout file OK!")


def test():
    openmx = OpenmxWrapper(
        path="/home/hexu/projects/TB2J_example/OPENMX/SrMnO3_FM_SOC/"
    )
    # hsr = openmx.parse_scfoutput()
    # from banddownfolder.plot import plot_band
    # plot_band(hsr)
    # plt.savefig('band.pdf')
    # plt.show()


# test()
