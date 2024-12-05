from TB2J_OpenMX.ffiparser import OpenmxWrapper
from TB2J.exchange import ExchangeNCL, ExchangeCL
import os


def gen_exchange(path,
                 prefix='openmx',
                 magnetic_elements=[],
                 include_orbs={},
                 dat_file = "openmx.dat",
                 kmesh=[5, 5, 5],
                 emin=-11.0,
                 emax=0.00,
                 nz=100,
                 exclude_orbs=[],
                 Rcut=None,
                 output_path="TB2J_results",
                 np=1,
                 use_cache=False,
                 orb_decomposition=True,
                 description=None):
    tbmodel=OpenmxWrapper(path, prefix, dat_file)
    if tbmodel.non_collinear:
        Exchange=ExchangeNCL
    else:
        Exchange=ExchangeCL
    print("Starting to calculate exchange.")
    description=f"""Using OpenMX data: 
path: {os.path.abspath(path)}
prefix: {prefix}
"""
    exchange = Exchange(
            tbmodels=tbmodel,
            atoms=tbmodel.atoms,
            basis=tbmodel.basis,
            efermi=tbmodel.efermi,
            magnetic_elements=magnetic_elements,
            include_orbs=include_orbs,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            np=np,
            output_path=output_path,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            use_cache=use_cache,
            orb_decomposition=orb_decomposition,
            description=description)
    exchange.run()
    print("\n")
    print("All calculation finsihed. The results are in TB2J_results directory.")

if __name__=='__main__':
    gen_exchange(
        path='/home/hexu/projects/TB2J_example/OPENMX/SrMnO3_FM_SOC/', magnetic_elements=['Mn'], nz=50, Rcut=8)
