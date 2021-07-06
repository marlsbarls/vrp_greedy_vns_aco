
from vns.src.algorithm.simulation import VNSSimulation
from aco.execution_new import Execution
from greedy.surve_mobility.greedy_main import Greedy as Surve_Greedy
from greedy.solomon.solomon_greedy_main import Greedy as Solomon_Greedy
import sys

# test type: static or dynamic
# test_type = 'dynamic'
test_type = 'static'

# source: solomon or surve_mobility
source = 'surve_mobility'
# source = 'solomon' 

if source != 'surve_mobility' and source != 'solomon':
    sys.exit('Set variable source to "surve_mobility" or "solomon')
if test_type != 'dynamic' and test_type != 'static':
    sys.exit('Set variable source to "dynamic" or "static')

# test files surve_mobility
if source == 'surve_mobility':
    # test_files = ['2020-07-07', '2020-07-08']
    # test_files = ['2020-07-07']
    test_files = ['2020-07-08']
    # test_files = ['2020-08-15']
    # test_files = ['2020-08-14']

    # # run greedy
    # greedy = Surve_Greedy(test_files, test_type, source)
    # greedy.run_greedy()

    # run aco
    if __name__ == '__main__':
        aco = Execution(test_files, test_type, source)
        aco.run_macs()

    # # run vns
    # vns = VNSSimulation(test_files, test_type)
    # vns.run_simulation()


if source == 'solomon':
    # test_files = ['C101']
    # test_files = ['C201']
    test_files = ['R101']

    ## run greedy
    greedy = Solomon_Greedy(test_files, test_type, source)
    greedy.run_greedy()



## run experiments






