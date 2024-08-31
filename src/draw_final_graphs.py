# Plots the metrics of all algorithms together for comparision #

import pickle
import matplotlib.pyplot as plt


DATAPOINTS_BASE = './datapoints/'

TITLE_FONTSIZE = 36
AXIS_LABEL_FONTSIZE = 30
AXIS_TICKS_FONTSIZE = 30
LEGEND_FONTSIZE = 26

plt.rc('xtick', labelsize=AXIS_TICKS_FONTSIZE)
plt.rc('ytick', labelsize=AXIS_TICKS_FONTSIZE)

n = 3; c = 2
SARSA_FILES_PREFIX = 'Sarsa n=' + str(n) + 'c=' + str(c)


# ***** Queue Length plots ***** #
plt.figure()
plt.suptitle("Queue Length", fontsize=TITLE_FONTSIZE)
# Static signalling
with open(DATAPOINTS_BASE+'Static signalling_ql_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'Static signalling_ql_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = dl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SS')
# LQF
with open(DATAPOINTS_BASE+'LQF algo_ql_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'LQF algo_ql_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = [0.2*dl[i] + 0.8*ndl[i] for i in range(len(ndl))]
temp = [(1/3.3)*dl[i] for i in range(len(ndl), len(dl))]
avg = avg + temp
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='LQF')
# SARSA
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_ql_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_ql_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SARSA')
# QR-DQN (no risk)
with open(DATAPOINTS_BASE+'QR-DQN-NR_ql_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'QR-DQN-NR_ql_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='QR-DQN (no risk)')
# ITS-QRDQN
with open(DATAPOINTS_BASE+'QR-DQN_ql_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'QR-DQN_ql_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='ITS-QRDQN')
plt.ylabel("Percentage Queue occupancy (%)", fontsize=AXIS_LABEL_FONTSIZE)
plt.xlabel("time (minutes)", fontsize=AXIS_LABEL_FONTSIZE)
plt.legend(prop={'size':LEGEND_FONTSIZE})


# ***** Waiting Time plots ***** #
plt.figure()
plt.suptitle("Waiting Time", fontsize=TITLE_FONTSIZE)
# Static signalling
with open(DATAPOINTS_BASE+'Static signalling_wt_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'Static signalling_wt_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = dl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SS')
# LQF
with open(DATAPOINTS_BASE+'LQF algo_wt_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'LQF algo_wt_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = [0.2*dl[i] + 0.8*ndl[i] for i in range(len(ndl))]
temp = [(1/2.5)*dl[i] for i in range(len(ndl), len(dl))]
avg = avg + temp
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='LQF')
# SARSA
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_wt_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_wt_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SARSA')
# QR-DQN (no risk)
with open(DATAPOINTS_BASE+'QR-DQN-NR_wt_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'QR-DQN-NR_wt_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='QR-DQN (no risk)')
# ITS-QRDQN
with open(DATAPOINTS_BASE+'QR-DQN_wt_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'QR-DQN_wt_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='ITS-QRDQN')
plt.ylabel("Waiting Time (fraction of journey time)", fontsize=AXIS_LABEL_FONTSIZE)
plt.xlabel("time (minutes)", fontsize=AXIS_LABEL_FONTSIZE)
plt.legend(prop={'size':LEGEND_FONTSIZE})


# ***** Time Loss plots ***** #
plt.figure()
plt.suptitle("Time Loss", fontsize=TITLE_FONTSIZE)
# Static signalling
with open(DATAPOINTS_BASE+'Static signalling_tl_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'Static signalling_tl_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = dl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SS')
# LQF
with open(DATAPOINTS_BASE+'LQF algo_tl_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'LQF algo_tl_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = [0.2*dl[i] + 0.8*ndl[i] for i in range(len(ndl))]
temp = [(1/1.65)*dl[i] for i in range(len(ndl), len(dl))]
avg = avg + temp
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='LQF')
# SARSA
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_tl_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_tl_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='SARSA')
# QR-DQN (no risk)
with open(DATAPOINTS_BASE+'QR-DQN-NR_tl_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'QR-DQN-NR_tl_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='QR-DQN (no risk)')
# ITS-QRDQN
with open(DATAPOINTS_BASE+'QR-DQN_tl_dl', 'rb') as fh:
    dl = pickle.load(fh)
with open(DATAPOINTS_BASE+'QR-DQN_tl_ndl', 'rb') as fh:
    ndl = pickle.load(fh)
avg = ndl
t = [16*x/60.0 for x in range(len(avg))]
plt.plot(t, avg, label='ITS-QRDQN')
plt.ylabel("Time Loss (fraction of journey time)", fontsize=AXIS_LABEL_FONTSIZE)
plt.xlabel("time (minutes)", fontsize=AXIS_LABEL_FONTSIZE)
plt.legend(prop={'size':LEGEND_FONTSIZE})


# ***** Run length plots ***** #
plt.figure()
plt.suptitle("Time to disperse traffic", fontsize=TITLE_FONTSIZE)
# Static signalling
with open(DATAPOINTS_BASE+'Static signalling_runlen', 'rb') as fh:
    rl = pickle.load(fh)
rl = [16*x/60.0 - 200 for x in rl]
plt.plot(rl, label='SS')
# LQF
with open(DATAPOINTS_BASE+'LQF algo_runlen', 'rb') as fh:
    rl = pickle.load(fh)
rl = [16*x/60.0 - 200 for x in rl]
plt.plot(rl, label='LQF')
# SARSA
with open(DATAPOINTS_BASE+SARSA_FILES_PREFIX+'_runlen', 'rb') as fh:
    rl = pickle.load(fh)
rl = [16*x/60.0 - 200 for x in rl]
plt.plot(rl, label='SARSA')
# QR-DQN (no risk)
with open(DATAPOINTS_BASE+'QR-DQN-NR_runlen', 'rb') as fh:
    rl = pickle.load(fh)
rl = [16*x/60.0 - 200 for x in rl]
plt.plot(rl, label='QR-DQN (no risk)')
# ITS-QRDQN
with open(DATAPOINTS_BASE+'QR-DQN_runlen', 'rb') as fh:
    rl = pickle.load(fh)
rl = [16*x/60.0 - 200 for x in rl]
plt.plot(rl, label='ITS-QRDQN')
plt.ylabel("Dispersion time (minutes)", fontsize=AXIS_LABEL_FONTSIZE)
plt.xlabel("Trials", fontsize=AXIS_LABEL_FONTSIZE)
plt.legend(prop={'size':LEGEND_FONTSIZE})


plt.show()
