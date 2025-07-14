import sys
from scheduling_algorithm_p100 import ProductionOrderSchedulerP100


# if __name__ == "__main__":
#
#     assembly_line = sys.argv[1]
#
#     if assembly_line == 'P100':
#         scheduler = ProductionOrderSchedulerP100()
#         scheduler.main_scheduling_function()

if __name__ == "__main__":

    scheduler = ProductionOrderSchedulerP100()
    scheduler.main_scheduling_function()
