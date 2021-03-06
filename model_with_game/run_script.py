import importlib
import logging
import sys
from functools import partial
from multiprocessing import Queue

from crowddynamics.functions import load_config
from crowddynamics.gui.graphics import MultiAgentPlot

configs = load_config("simulations.yaml")

# Simulation with multiprocessing.
queue = Queue(maxsize=4)
process = None

d = configs["simulations"]["room_evacuation_game"]
module = importlib.import_module(d["module"])
simulation = getattr(module, d["class"])
process = simulation(queue, **d["kwargs"])

args = [(("agent", "agent"),
         ["position", "active", "position_ls", "position_rs"])]

if process.game is not None:
    args.append((("game", "agent"), ["strategy"]))

process.configure_queuing(args)
# The simulation results are saved to the author's folders in the University server.
# If you are running the code, change the saving location something suitable.
process.configure_hdfstore("{}{}{}{}".format('/scratch/work/avonscha/', sys.argv[1], 'RoomEvacuationGame', sys.argv[2]))

process.start()

data = 0
while data is not None:
    data=queue.get()

process.stop()
