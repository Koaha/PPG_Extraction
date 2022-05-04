import numpy as np
from scipy import signal
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("bidmc_csv/bidmc_01_Signals.csv",header=0)

fig = go.Figure()
fig.add_traces(go.Scatter(x=np.arange(len(df)),
                                  y=df["AVR"], mode="lines"))
fig.show()

# def impedance_peak_detection(inp):
#     min_local_indices = []
#     max_local_indices = []
#     zero_crossing = 0
#
#     while len(min_local_indices) > 0 and len(max_local_indices) > 0:
#
#         # flag the first crossing rate
#         if zero_crossing == 0:
#             zero_crossing = np.argmin([current_min_index,np.inf,curent_max_index])-1
#         else:


