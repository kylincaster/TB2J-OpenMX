from TB2J_OpenMX.ffiparser import OpenmxWrapper
from TB2J.exchange import ExchangeNCL


def gen_exchange(path,
                 prefix='openmx',
                 magnetic_elements=[],
                 kmesh=[5, 5, 5],
                 emin=-11.0,
                 emax=0.00,
                 nz=100,
                 exclude_orbs=[],
                 Rcut=None,
                 description=''):
    tbmodel=OpenmxWrapper(path, prefix)
    print(tbmodel.efermi)
    if tbmodel.non_collinear:
        print("Starting to calculate exchange.")
        exchange = ExchangeNCL(
            tbmodels=tbmodel,
            atoms=tbmodel.atoms,
            basis=tbmodel.basis,
            efermi=tbmodel.efermi,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            exclude_orbs=exclude_orbs,
            Rcut=Rcut,
            description=description)
        exchange.run()
        print("\n")
        print("All calculation finsihed. The results are in TB2J_results directory.")

if __name__=='__main__':
    gen_exchange(
        path='/home/hexu/projects/TB2J_example/OPENMX/SrMnO3_FM_SOC/', magnetic_elements=['Mn'], nz=50, Rcut=8)
