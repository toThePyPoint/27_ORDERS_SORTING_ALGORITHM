import sys
from scheduling_algorithm_p100 import ProductionOrderSchedulerP100
from scheduling_algorithm_m300 import ProductionOrderSchedulerM300


if __name__ == "__main__":

    assembly_line = sys.argv[1]

    if assembly_line == 'P100':
        scheduler = ProductionOrderSchedulerP100()
        scheduler.main_scheduling_function()
    elif assembly_line == 'M300':
        scheduler = ProductionOrderSchedulerM300()
        scheduler.main_scheduling_function()


# if __name__ == "__main__":
#
#     scheduler = ProductionOrderSchedulerP100()
#     scheduler.main_scheduling_function()
