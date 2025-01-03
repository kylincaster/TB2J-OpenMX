#!/usr/bin/env python3
import argparse
from TB2J.versioninfo import print_license
from TB2J_OpenMX import gen_exchange_openmx
import sys
import numpy
def set_threads(n_threads = 1):
    from os import environ
    N_THREADS = str(n_threads)
    environ['OMP_NUM_THREADS'] = N_THREADS
    environ['OPENBLAS_NUM_THREADS'] = N_THREADS
    environ['MKL_NUM_THREADS'] = N_THREADS
    environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
    environ['NUMEXPR_NUM_THREADS'] = N_THREADS
    
def run_openmx2J():
    print_license()
    print("\n")
    parser = argparse.ArgumentParser(
        description=
        "openmx2J: Using magnetic force theorem to calculate exchange parameter J from openmx Hamiltonian"
    )

    parser.add_argument(
        '--prefix', help="prefix of the openmx files", default='openmx', type=str)

    parser.add_argument(
        '--elements',
        help="elements to be considered in Heisenberg model",
        default=None,
        type=str,
        nargs='*')
    parser.add_argument(
        '--rcut', help='range of R. The default is all the commesurate R to the kmesh', default=None, type=float)
    parser.add_argument(
        '--efermi', help='Fermi energy in eV', default=None, type=float)
    parser.add_argument(
        '--kmesh',
        help='kmesh in the format of kx ky kz. Monkhorst pack. If all the numbers are odd, it is Gamma cenetered. (strongly recommended)',
        type=int,
        nargs='*',
        default=[5, 5, 5])
    parser.add_argument(
        '--emin',
        help='energy minimum below efermi, default -14 eV',
        type=float,
        default=-14.0)
    parser.add_argument(
        '--emax',
        help='energy maximum above efermi, default 0.0 eV',
        type=float,
        default=0.05)
    parser.add_argument(
        '--use_cache',
        help="whether to use disk file for temporary storing wavefunctions and hamiltonian to reduce memory usage. Default: False",
        action='store_true',
        default=False)
    parser.add_argument(
        '--nz', help='number of integration steps, default: 50', default=50, type=int)

    parser.add_argument(
        '--exclude_orbs',
        help=
        "the indices of wannier functions to be excluded from magnetic site. counting start from 0",
        default=[],
        type=int,
        nargs='+')

    parser.add_argument(
        "--np",
        help="number of cpu cores to use in parallel, default: 1",
        default=1,
        type=int,
    )
    
    parser.add_argument(
        "--restart",
        help="read restart data from a pickle file, default: None",
        default=None,
    )
    
    parser.add_argument(
        "-t",
        "--threads", 
        help="number of threads to use in LAPACK and BLAS, default: 1",
        default=1,
        type=int,
    )
    
    parser.add_argument(
        "--description",
        help=
        "add description of the calculatiion to the xml file. Essential information, like the xc functional, U values, magnetic state should be given.",
        type=str,
        default="Calculated with TB2J.\n"
    )
    
    parser.add_argument(
        "--restart-style",
        help=
        "The style to store Hamiltonian in the restart file: full, sparse, auto or skip. Default: auto",
        type=str,
        default="auto"
    )

    parser.add_argument(
        "-D",
        "--orb_decomposition",
        default=False,
        action="store_true",
        help="whether to do orbital decomposition in the collinear and non-collinear mode. Default: False.",
    )

    parser.add_argument(
        "--fname",
        default='exchange.xml',
        type=str,
        help='exchange xml file name. default: exchange.xml')

    parser.add_argument(
        "--output_path",
        help="The path of the output directory, default is TB2J_results",
        type=str,
        default="TB2J_results",
    )

    args = parser.parse_args()
    set_threads(args.threads)
    if args.elements is None:
        print("Please input the magnetic elements, e.g. --elements Fe Ni")
        parser.print_help()
        exit(1)
    
    include_orbs = {}
    for element in args.elements:
        if "_" in element:
            elem = element.split("_")[0]
            orb = element.split("_")[1:]
            include_orbs[elem] = orb
        else:
            include_orbs[element] = None
    
    style = args.restart_style.lower()
    valid_styles =  ["full", "sparse", "auto"]
    if style not in valid_styles:
        print(f"invalid restart style: {style}, only valid is {valid_styles}")
        exit(1)

    gen_exchange_openmx(
        path='./',
        prefix=args.prefix,
        kmesh=args.kmesh,
        magnetic_elements=list(include_orbs.keys()),
        include_orbs=include_orbs,
        Rcut=args.rcut,
        emin=args.emin,
        emax=args.emax,
        nz=args.nz,
        description=args.description,
        output_path=args.output_path,
        use_cache=args.use_cache,
        np=args.np,
        exclude_orbs=args.exclude_orbs,
        orb_decomposition=args.orb_decomposition,
        restart = args.restart,
        restart_style = args.restart_style,
        )

if __name__ == "__main__":
    run_openmx2J()
