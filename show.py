from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

def show():
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("+0+0")
    plt.show(block=False)
    wm.window.attributes('-topmost', 1)
