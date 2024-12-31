import os
import numpy as np
import scipy.linalg as sl
from ase.units import Ha, Bohr
from ase.io import read
from ase import Atoms
from TB2J.utils import kmesh_to_R, symbol_number
from TB2J.myTB import AbstractTB
from TB2J_OpenMX.cmod._scfout_parser import ffi, lib
from ase.data import atomic_numbers
import pickle
from typing import Optional
from scipy.sparse import csr_array

def sizeof_fmt(num, suffix="B"):
    num = float(num)
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"

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
    "s": (
        1,
        "s",
    ),
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
                orbital_count, token_length = parse_integer(pao_config[position + 1 :])
                parsed_orbitals.append((orbital_count, *ATOMIC_ORBITALS[orbital_type]))
                position += token_length + 1
            except (IndexError, RuntimeError):
                raise RuntimeError(
                    f"Invalid PAO configuration: `{pao_config}`. Expected format like `s2p1d1`."
                )
        else:
            raise RuntimeError(
                f"Invalid PAO configuration: `{pao_config}`. Unknown orbital type `{orbital_type}`."
            )

    return parsed_orbitals


# Create the dictionary mapping ctypes to np dtypes.
ctype2dtype = {"int": "i4", "double": "f8"}

# Integer types
for prefix in ("int", "uint"):
    for log_bytes in range(4):
        ctype = "%s%d_t" % (prefix, 8 * (2**log_bytes))
        dtype = "%s%d" % (prefix[0], 2**log_bytes)
        ctype2dtype[ctype] = np.dtype(dtype)

# Floating point types
ctype2dtype["float"] = np.dtype("f4")
ctype2dtype["double"] = np.dtype("f8")


def asarray(ffi, ptr, length):
    # Get the canonical C type of the elements of ptr as a string.
    T = ffi.getctype(ffi.typeof(ptr).item)

    if T not in ctype2dtype:
        raise RuntimeError("Cannot create an array for element type: %s" % T)

    return np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(T)), ctype2dtype[T])


class OpenMXParser:
    """
    A Parser to read OpenMX Hamiltonian and overlap matrices.

    Attributes:
        H (np.ndarray[float64]): Hamiltonian matrices for each lattice vector R.
        S (np.ndarray[float64]): Overlap matrices for each lattice vector R.
        R (np.ndarray[int]): Array of lattice vectors R.
        atoms (ase.Atoms): ase.Atoms for OpenMX systems
        natom (int): number of atoms in the lattice
        HR_n_nonzeros (np.ndarray[int, 1]): The number of nonzeros PAO neighbors for each atoms
        HR_nonzeros (List[np.ndarray[int, 2]]): The list for nonzeros PAO neighbors for each atoms. 
            the 1st row is the nonzero index and 2st row is the length of consecutive sequence
        norbs (np.ndarray[int]): number of PAOs in for each atoms in OpenMX system
        non_collinear (bool): Whether a DFT calculation with non_collinear Hamiltonian
        output (Optional[str]): The path to write the Restart Hamiltonian file
        restart_style (str): The style to write down restart Hamiltonina file: full, sparse and auto
        basis (List[PAO]): a list to contain all atom basis, including both spin-up and spin-down
    """
    def __init__(
        self, 
        path: str, 
        prefix: str = "openmx", 
        output_path: str = "TB2J_results", 
        allow_non_spin_polarized: bool = False,
        restart_style: str = "auto",
    ) -> None:
        """
        Initialize the OpenMXParser class.

        Args:
            path (str): Path to the OpenMX files or a specific pickle dump.
            prefix (str): Prefix for OpenMX output files. Default is "openmx".
            output_path (Optional[str]): Path to dump Hamiltonian file. Default is None.
            allow_non_spin_polarized (bool): Whether to allow non-spin-polarized calculations. Default is False.
            restart_style (str): Switch to control the style of Hamiltonian file, using `full`, `sparse`, `auto` or `skip`. Default is `auto`
        """
        self.non_collinear: bool = False
        self.output_path: str = output_path
        self.restart_style: str = restart_style
        self.HR_nonzeros = None
        # If a specific file is provided, read the data and set up the models
        if path is None:
            return
        
        # Construct paths for required OpenMX output files
        fname = os.path.join(path, prefix + ".scfout")
        if not os.path.isfile(fname):
            print(f"Cannot find the OpenMX Hamilton file: `{fname}`")
            exit(1)

        fxyzname = os.path.join(path, prefix + ".xyz")
        if not os.path.isfile(fxyzname):
            print(f"Cannot find the OpenMX XYZ file: `{fxyzname}`")
            exit(1)

        self.openmx_outfile: Optional[str] = os.path.join(path, prefix + ".out")
        if not os.path.isfile(self.openmx_outfile):
            self.openmx_outfile = None

        # Parse the SCF output and atomic information
        self.parse_scfoutput(fname)
        atoms = read(fxyzname)
        self.atoms: Atoms = Atoms(
            symbols=atoms.get_chemical_symbols(), 
            cell=self.cell, 
            positions=self.positions
        )

        # Map orbitals to atomic basis functions
        self.norbs_to_basis(self.atoms, self.norbs)

        # Set up models based on parsed data
        self.set_models(allow_non_spin_polarized)

        # Dump data if an output path is provided
        if self.restart_style != "skip":
            self.dump_data(prefix)
    
    def read_data(self, restart: Optional[str], allow_non_spin_polarized: bool = False) -> None:
        """
        Load serialized data from a pickle file and update instance attributes.

        Args:
            restart (Optional[str, dict]): Path to the pickle file containing serialized data or a dictionary contained the data
            allow_non_spin_polarized (bool): Whether to allow non-spin-polarized calculations. Default is False.
        """
        if isinstance(restart, str):
            with open(restart, "rb") as f:
                data = pickle.load(f)
        elif isinstance(restart, dict):
            data = restart
        else:
            raise RuntimeError(f"Unknown restart type {type(restart)}: {restart}")
        
        for k, v in data.items():
            if not k.startswith("__"):
                self.__dict__[k] = v
        
        n_spin = self.SpinP_switch + 1
        self.norb = norb = len(self.basis) // 2
        self.natom = natom = len(self.atoms)
        
        if data.get("__sparse_flag", False):
            print(f"Read spare Hamiltonin file from {restart}")
            sp_S, sp_H = data["__sp_S"], data["__sp_H"]
            ncell = len(self.R)
            self.S = np.zeros((ncell, norb, norb))
            self.H = np.zeros((4, ncell, norb, norb))
            S = self.S.reshape(-1, norb)
            H = self.H.reshape(4, -1, norb)
            for i in range(natom):
                nonzeros = np.empty((self.HR_n_nonzeros[i],), dtype = int)
                jj = 0
                for j, n in self.HR_nonzeros[i]:
                    nonzeros[jj:jj+n] = range(j, j+n)
                    jj += n
                S[nonzeros, self.norbs_index[i]:self.norbs_index[i+1]] = sp_S[i]
                H[:n_spin, nonzeros, self.norbs_index[i]:self.norbs_index[i+1]] = sp_H[i]
        else:
            # full style
            print(f"Read full Hamiltonin file from {restart}")
            if self.SpinP_switch == 1:
                H = np.empty((4, *self.H[0].shape))
                H[0] = self.H[0]
                H[1] = self.H[1]
                H[2:,:] = 0
                self.H = H
            
        # Determine if the calculation is non-collinear
        if getattr(self, 'SpinP_switch', 0) == 3:
            self.non_collinear = True
        
        self.set_models(allow_non_spin_polarized)
    
    def set_models(self, allow_non_spin_polarized: bool) -> None:
        """
        Set up the models based on the SpinP_switch parameter.

        Args:
            allow_non_spin_polarized (bool): Whether to allow non-spin-polarized calculations.

        Raises:
            RuntimeError: If non-spin-polarized calculations are not allowed or if SpinP_switch is invalid.
        """

        if self.SpinP_switch == 0:
            if not allow_non_spin_polarized:
                raise RuntimeError(
                    "The non-spin polarized DFT calculation is not supported for TB2J"
                )
            tmodel = OpenmxWrapper(self.H[0, :, :, :], self.S, self.R, self.get_basis())
            tmodel.efermi = self.efermi
            tmodel.atoms = self.atoms
            self.tmodels: Tuple[OpenmxWrapper] = (tmodel,)
        elif self.SpinP_switch == 1:
            tmodel_up = OpenmxWrapper(
                self.H[0, :, :, :], self.S, self.R, self.get_basis(0)
            )
            tmodel_dn = OpenmxWrapper(
                self.H[1, :, :, :], self.S, self.R, self.get_basis(1)
            )
            tmodel_up.efermi = self.efermi
            tmodel_dn.efermi = self.efermi
            tmodel_up.atoms = self.atoms
            tmodel_dn.atoms = self.atoms
            self.tmodels: Tuple[OpenmxWrapper, OpenmxWrapper] = (tmodel_up, tmodel_dn)
        elif self.SpinP_switch == 3:
            tmodel = OpenmxWrapper(
                self.H, self.S, self.R, self.get_basis(), non_collinear=True
            )
            tmodel.efermi = self.efermi
            tmodel.atoms = self.atoms
            self.tmodels: Tuple[OpenmxWrapper] = (tmodel,)
        else:
            raise RuntimeError(f"Invalid SpinP_switch {self.SpinP_switch}")

    def get_models(self):
        """
        Retrieve the models set by set_models.

        Returns:
            Union[OpenmxWrapper, Tuple[OpenmxWrapper, OpenmxWrapper]]: The models depending on SpinP_switch.
        """
        if self.SpinP_switch == 1: # spin-polar case with two spin
            return self.tmodels
        else:
            return self.tmodels[0]

    def __getitem__(self, spin):
        """
        Access a specific model by spin index.

        Args:
            spin (int): Spin index (0 for spin-up, 1 for spin-down in spin-polarized cases).

        Returns:
            OpenmxWrapper: The requested model.

        Raises:
            IndexError: If the spin index is out of range.
        """
        return self.tmodels[spin]

    def norbs_to_basis(self, atoms: Atoms, norbs: np.ndarray[int]) -> None:
        """
        Generate basis set for the given atoms and number of orbitals.

        Parameters:
        atoms (Atoms): Atomic structure information.
        norbs (list[int]): Number of orbitals for each atomic species.
        """
        if self.openmx_outfile:
            self.basis_from_output_file(atoms)
        else:
            self.gen_basis_by_number(atoms, norbs)

        if self.SpinP_switch == 0:
            basis = []
            for i in range(0, len(self.basis), 2):
                i, j, _ = self.basis[i]
                basis.append((i, j, ""))
            self.basis = basis
        print(f"Generate total {len(self.basis)} basis")

    def gen_basis_by_number(self, atoms: Atoms, norbs: np.ndarray[int]) -> list[tuple[str, str, str]]:
        """
        Generate the basis set based on the number of orbitals.

        Parameters:
        atoms (Atoms): Atomic structure information.
        norbs (list[int]): Number of orbitals for each atomic species.

        Returns:
        list[tuple[str, str, str]]: Generated basis set with spin information.
        """
        self.basis = []
        symbols = atoms.get_chemical_symbols()

        sn = list(symbol_number(symbols).keys())
        for i, n in enumerate(norbs):
            for j in range(n):
                self.basis.append((sn[i], f"orb{j+1}", "up"))
                self.basis.append((sn[i], f"orb{j+1}", "down"))
        return self.basis

    def get_basis(self, spin: int | None = None) -> list[tuple[str, str, str]]:
        """
        Retrieve the basis set for the specified spin channel.

        Parameters:
        spin (int | None): Spin channel (0 for up, 1 for down, None for all spins).

        Returns:
        list[tuple[str, str, str]]: Basis set for the given spin channel.
        """
        if self.SpinP_switch == 0 or spin is None:
            return self.basis
        elif spin < 2:
            return self.basis[spin::2]
        else:
            raise RuntimeError(f"Invalid spin value: {spin}")

    def parse_scfoutput(self, fname: str):
        argv0 = ffi.new("char[]", b"")
        argv = ffi.new("char[]", bytes(fname, encoding="ascii"))
        lib.read_scfout([argv0, argv])
        lib.prepare_HSR()

        self.ncell = lib.TCpyCell + 1
        self.natom = lib.atomnum
        self.norbs = np.copy(asarray(ffi, lib.Total_NumOrbs, self.natom + 1)[1:])

        fnan = asarray(ffi, lib.FNAN, self.natom + 1)

        natn = [None]
        for iatom in range(self.natom):
            natn.append(asarray(ffi, lib.natn[iatom + 1], fnan[iatom + 1] + 1))

        ncn = [None]
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
        self.SpinP_switch = lib.SpinP_switch

        self.norb = lib.T_NumOrbs
        norb = self.norb
        
        # get the non-zero position for an matrix
        norbs = self.norbs
        self.norbs_index = norbs_index = np.cumsum(np.pad(norbs, (1, 0), "constant"))
        self.HR_nonzeros = []
        HR_n_nonzeros = []
        for II in range(self.natom):
            ct_AN = II + 1
            ct_natn = natn[ct_AN]
            ct_ncn  = ncn[ct_AN]
            nonzeros = []
            n_nonzeros = 0
            for h_AN in range(fnan[ct_AN] + 1):
                Gh_AN_minus = ct_natn[h_AN] - 1 #
                iR = ct_ncn[h_AN]
                nonzeros.append((iR*norb + norbs_index[Gh_AN_minus], norbs[Gh_AN_minus]))
                n_nonzeros += norbs[Gh_AN_minus]
            HR_n_nonzeros.append(n_nonzeros)
            self.HR_nonzeros.append(np.asarray(nonzeros))
        self.HR_n_nonzeros = np.asarray(HR_n_nonzeros)
        
        if self.SpinP_switch == 3:
            self.non_collinear = True
        
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
                self.H[iR, ::2, ::2] = HR[iR, 0, :, :] + 1j * HR_imag[iR, 0, :, :]
                # up down
                self.H[iR, ::2, 1::2] = HR[iR, 2, :, :] + 1j * (
                    HR[iR, 3, :, :] + HR_imag[iR, 2, :, :]
                )
                # down up
                self.H[iR, 1::2, ::2] = HR[iR, 2, :, :] - 1j * (
                    HR[iR, 3, :, :] + HR_imag[iR, 2, :, :]
                )
                # down down
                self.H[iR, 1::2, 1::2] = HR[iR, 1, :, :] + 1j * HR_imag[iR, 1, :, :]
        else:  # collinear
            HR = np.zeros([self.ncell, 4, lib.T_NumOrbs, lib.T_NumOrbs])
            for iR in range(0, self.ncell):
                for ispin in range(self.SpinP_switch + 1):
                    for iorb in range(lib.T_NumOrbs):
                        HR[iR, ispin, iorb, :] = asarray(
                            ffi, lib.HR[iR][ispin][iorb], norb
                        )

            self.H = np.swapaxes(HR, 0, 1)

        self.efermi = lib.ChemP * Ha
        self.H *= Ha

        SR = np.zeros([self.ncell, lib.T_NumOrbs, lib.T_NumOrbs])
        for iR in range(0, self.ncell):
            for iorb in range(lib.T_NumOrbs):
                SR[iR, iorb, :] = asarray(ffi, lib.SR[iR][iorb], norb)
        self.S = SR  # np.kron(SR, np.eye(2))
        print("Loading from scfout file OK!")
        lib.free_HSR()
        lib.free_scfout()

    def dump_data(self, name: str = "openmx") -> str:
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        pklfile = os.path.join(self.output_path, name + "_restart.pkl")
        
        if self.restart_style == "auto":
            n_elems = np.prod(self.S)
            n_nonzeros = np.count_nonzero(self.S)
            if float(n_nonzeros) / n_elems < 0.3 and n_nonzeros > 800000:
                self.dump_HRs_sparse(pklfile)
            else:
                self.dump_HRs_sparse(pklfile)
        elif self.restart_style == "full":
            self.dump_HRs_full(pklfile)
        elif self.restart_style == "sparse":
            self.dump_HRs_sparse(pklfile)
        else:
            raise ValueError(f"Invalid restart style {self.restart_style}")
        
        return pklfile
    
    def calc_HR_nonzeros(self):
        n_PAOs = self.norb
        S = self.S.reshape(-1, n_PAOs)
        self.HR_nonzeros = []
        self.HR_n_nonzeros = np.empty((self.natom,), dtype=int)
        for i in range(self.natom):
            a = S[:, self.norbs_index[i]:self.norbs_index[i+1]]
            nonzeros_index = np.nonzero(np.any(a != 0.0, axis=1))[0]
            idx = np.r_[0, np.where(np.diff(nonzeros_index) != 1)[0]+1, len(nonzeros_index)]
            HR_nonzeros = [(nonzeros_index[idx[i]], idx[i+1]-idx[i]) for i in range(len(idx)-1)]
            self.HR_n_nonzeros[i] = len(nonzeros_index)
            self.HR_nonzeros.append(np.asarray(HR_nonzeros))
            
    
    def dump_HRs_sparse(self, pklfile: str = "openmx.pkl") -> None:
        if self.HR_nonzeros is None:
            self.calc_HR_nonzeros()
        
        n_PAOs = self.norb
        n_spin = self.SpinP_switch + 1
        S = self.S.reshape(-1, n_PAOs)
        H = self.H.reshape(self.H.shape[0],-1,n_PAOs)
            
        sp_S = []
        sp_H = []
        n_total_nonzeros = 0
        for i in range(self.natom):
            n_total_nonzeros += self.HR_n_nonzeros[i]*(self.norbs[i])
            nonzeros = np.empty((self.HR_n_nonzeros[i],), dtype = int)
            jj = 0
            for j, n in self.HR_nonzeros[i]:
                nonzeros[jj:jj+n] = range(j, j+n)
                jj += n
            sp_S.append(S[nonzeros, self.norbs_index[i]:self.norbs_index[i+1]])
            sp_H.append(H[:n_spin, nonzeros, self.norbs_index[i]:self.norbs_index[i+1]])
            
            #not_Sel = ~np.isin(np.arange(S.shape[0]), nonzeros)
            #n_err = np.count_nonzero(S[not_Sel, self.norbs_index[i]:self.norbs_index[i+1]])
            #if n_err != 0:
            #    raise RuntimeError(f"invalid HR_nonzeros struct with missing nonzeros {n_err}")
        
        sparse_rate = 100*float(n_total_nonzeros) / np.prod(S.shape)
        
        data = {
            "__sparse_flag": True,
            "basis": self.basis,
            "efermi": self.efermi,
            "SpinP_switch": self.SpinP_switch,
            "__sp_H": sp_H,
            "R": self.R,
            "__sp_S": sp_S,
            "HR_nonzeros": self.HR_nonzeros,
            "HR_n_nonzeros": self.HR_n_nonzeros,
            "norbs_index": self.norbs_index,
            "atoms": self.atoms,
        }
        with open(pklfile, "wb") as f:
            pickle.dump(data, f)

        file_size = sizeof_fmt(os.stat(pklfile).st_size)
        print(f"write sparse HRs, S and R to {pklfile} with size: {file_size} sparse rate: {sparse_rate:.2f}%%")

    def dump_HRs_full(self, pklfile: str = "openmx.pkl") -> None:
        """
        Serialize the parsed data into a pickle file for later use.

        Parameters:
        name (str): The prefix for the output file name. Defaults to "openmx".

        Returns:
        str: Path to the serialized pickle file.
        """
        # save the memory usage
        if self.SpinP_switch == 1:
            HRs = (self.H[0], self.H[1])
            #print(np.count_nonzero(self.H[0]), np.count_nonzero(self.H[1]), np.count_nonzero(self.H[2]), np.count_nonzero(self.H[3]))
        else:
            HRs = self.H
        
        data = {
            "__sparse_flag": False,
            "basis": self.basis,
            "efermi": self.efermi,
            "SpinP_switch": self.SpinP_switch,
            "H": HRs,
            "R": self.R,
            "S": self.S,
            "atoms": self.atoms,
        }
        
        with open(pklfile, "wb") as f:
            pickle.dump(data, f)
            
        file_size = sizeof_fmt(os.stat(pklfile).st_size)
        print(f"write full HRs, S and R to {pklfile} with size: {file_size}")
        
    def basis_from_output_file(self, atoms: Atoms) -> list[tuple[str, str, str]]:
        """
        Parse the basis set configuration from a OpenMX  file and construct the basis set.

        Parameters:
        atoms (Atoms): Atomic structure information.

        Returns:
        list[tuple[str, str, str]]: The constructed basis set.

        Raises:
        RuntimeError: If the atomic species definition section is not found or insufficient.
        """
        symbols = atoms.get_chemical_symbols()

        # Read lines from the dat file
        with open(self.openmx_outfile, "r") as file:
            lines = file.readlines()

        # Find the indices of the atomic species definition section
        start_index, end_index = -1, -1
        for index, line in enumerate(lines):
            if line.startswith("<Definition.of.Atomic.Species"):
                start_index = index + 1
            if line.startswith("Definition.of.Atomic.Species>"):
                end_index = index
                break

        if start_index == -1 or end_index == -1:
            raise RuntimeError("Cannot find paired tag `Definition.of.Atomic.Species`")

        if (end_index - start_index) < len(set(symbols)):
            raise RuntimeError(f"Insufficient atomic species definitions in the file at {start_index}-{end_index} for {len(set(symbols))} different symbols")

        # Parse PAOs for each element
        pao_definitions = {}
        pao_input_s = {}
        for line in lines[start_index:end_index]:
            line = line.strip()
            if line.startswith("#"):
                continue
            tokens = line.split()
            element = validate_element(tokens[2].split("_")[0])
            pao_input_s[element] = tokens[1].split("-")[1]
            pao_definitions[element] = parse_pao_config(pao_input_s[element])

        # Map atomic tags to their symbols
        atom_tags = list(symbol_number(symbols).keys())
        self.basis = []
        for tag, symbol in zip(atom_tags, symbols):
            basis = []
            if symbol not in pao_definitions:
                raise RuntimeError(f"No PAO definition found for symbol: {symbol}")

            paos_for_symbol = pao_definitions[symbol]
            for num_paos, pao_order, pao_list in paos_for_symbol:
                for index in range(num_paos):
                    for pao in pao_list:
                        basis.append((tag, f"{pao_order}{pao}N{index+1}", "up"))
                        basis.append((tag, f"{pao_order}{pao}N{index+1}", "down"))

            print(f"{tag} `{pao_input_s[symbol]}`[{len(basis)//2}]: ", end="")
            for i in range(0, len(basis), 2):
                print(basis[i][1], end=" ")
            print("")

            self.basis.extend(basis)

        return self.basis

class OpenmxWrapper(AbstractTB):
    """
    A wrapper for handling OpenMX Hamiltonian and overlap matrices.

    Attributes:
        H (np.ndarray): Hamiltonian matrices for each lattice vector R.
        S (np.ndarray): Overlap matrices for each lattice vector R.
        R (np.ndarray): Array of lattice vectors R.
        basis (list): List of basis identifiers.
        non_collinear (bool): Whether the system is non-collinear.
        nbasis (int): Total number of basis functions.
        norb (int): Number of orbitals per spin channel.
        nspin (int): Number of spin channels (1 for collinear, 2 for non-collinear).
        Rdict (dict): Mapping from R tuples to indices in H and S.
        R2kfactor (complex): Factor for converting R vectors to k-space phase factors.
    """

    def __init__(self, H, S, R, basis, non_collinear=False):
        """
        Initialize the OpenmxWrapper instance.

        Args:
            H (np.ndarray): Hamiltonian matrices for each R.
            S (np.ndarray): Overlap matrices for each R.
            R (np.ndarray): Array of lattice vectors R.
            basis (list): List of basis identifiers.
            non_collinear (bool): Whether the system is non-collinear. Default is False.
        """
        self.is_siesta = False
        self.is_orthogonal = (S is None)
        self.non_collinear = non_collinear
        self.H = H
        self.S = S
        self.basis = basis
        self.nbasis = H.shape[-1]

        if self.nbasis != len(basis):
            raise RuntimeError(
                f"Invalid number of basis: {self.nbasis} from H, and {len(basis)} from basis list"
            )

        self.R2kfactor = 2.0j * np.pi
        self.nspin = 1 + non_collinear  # 1 for collinear, 2 for non-collinear
        self.norb = self.nbasis // self.nspin
        self._name = "OpenMX"

        self.Rdict = {tuple(R): i for i, R in enumerate(R)}
        self.R = R

    def solve(self, k, convention=2):
        """
        Solve the eigenvalue problem for a given k-point.

        Args:
            k (tuple): The k-point in reciprocal space.
            convention (int): Phase convention (1 or 2). Default is 2.

        Returns:
            tuple: Eigenvalues and eigenvectors.
        """
        Hk, Sk = self.gen_ham(k, convention=convention)
        return sl.eigh(Hk, Sk)

    def solve_all(self, kpts, convention=2):
        """
        Solve the eigenvalue problem for all k-points.

        Args:
            kpts (list): List of k-points.
            convention (int): Phase convention (1 or 2). Default is 2.

        Returns:
            tuple: Eigenvalues and eigenvectors for all k-points.
        """
        nk = len(kpts)
        evals = np.zeros((nk, self.nbasis), dtype=float)
        evecs = np.zeros((nk, self.nbasis, self.nbasis), dtype=complex)
        for ik, k in enumerate(kpts):
            evals[ik], evecs[ik] = self.solve(k, convention=convention)
        return evals, evecs

    def HSE_k(self, kpt, convention=2):
        """
        Generate Hamiltonian, overlap matrices, and solve the eigenvalue problem at a k-point.

        Args:
            kpt (tuple): The k-point in reciprocal space.
            convention (int): Phase convention (1 or 2). Default is 2.

        Returns:
            tuple: Hamiltonian, overlap matrix, eigenvalues, and eigenvectors.
        """
        Hk, Sk = self.gen_ham(tuple(kpt), convention=convention)
        evals, evecs = sl.eigh(Hk, Sk)
        return Hk, Sk, evals, evecs

    def gen_ham(self, k, convention=2):
        """
        Generate the Hamiltonian and overlap matrices at a given k-point.

        Args:
            k (tuple): The k-point in reciprocal space.
            convention (int): Phase convention (1 or 2). Default is 2.

        Returns:
            tuple: Hamiltonian and overlap matrices at the k-point.
        """
        Sk = None
        if convention == 2:
            phase = np.exp(self.R2kfactor * (self.R @ k))
            Hk = np.einsum("rij, r->ij", self.H, phase)
            if not self.is_orthogonal:
                Sk = np.einsum("rij, r->ij", self.S, phase)
        elif convention == 1:
            raise NotImplementedError("Convention 1 is not implemented yet.")
        else:
            raise ValueError("Convention should be either 1 or 2.")

        return Hk, Sk

    def HS_and_eigen(self, kpts, convention=2):
        """
        Calculate Hamiltonians, overlap matrices, and eigenvalues for all k-points.

        Args:
            kpts (list): List of k-points.
            convention (int): Phase convention (1 or 2). Default is 2.

        Returns:
            tuple: Hamiltonians, overlap matrices, eigenvalues, and eigenvectors for all k-points.
        """
        nk = len(kpts)
        hams = np.zeros((nk, self.nbasis, self.nbasis), dtype=complex)
        ovps = np.zeros((nk, self.nbasis, self.nbasis), dtype=complex)
        evals = np.zeros((nk, self.nbasis), dtype=float)
        evecs = np.zeros((nk, self.nbasis, self.nbasis), dtype=complex)

        for ik, k in enumerate(kpts):
            hams[ik], ovps[ik], evals[ik], evecs[ik] = self.HSE_k(
                tuple(k), convention=convention
            )
        return hams, ovps, evals, evecs

    def get_hamR(self, R):
        """
        Retrieve the Hamiltonian matrix for a given R vector.

        Args:
            R (tuple): Lattice vector R.

        Returns:
            np.ndarray: Hamiltonian matrix for the given R.
        """
        return self.H[self.Rdict[tuple(R)]]

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
