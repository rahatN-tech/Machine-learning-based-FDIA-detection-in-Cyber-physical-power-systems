import pickle
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches



mean_orig_meas = 1.2852190097006362e-05
variance_mse = 4.3556439769492334e-11

file_path="mse_base_attk_std_scaler_neural.pkl"
with open(file_path, "rb") as file:
    mse = pickle.load(file)

mse = [x for x in mse]
font_legend = FontProperties(family='serif', size=10)
print((mse))
colors =["red" if e > (mean_orig_meas) else '#1f77b4' for e in mse]
x= list(range(50))

plt.bar(x, mse, color= colors)
plt.yscale('log')
# plt.gca().xaxis.set_visible(False)
red_patch = mpatches.Patch(color='red',label='Alarm raised')
blue_patch = mpatches.Patch(color = '#1f77b4', label='Normal operation')
font_properties = FontProperties(family='serif', size=12)
font_labels = FontProperties(family='serif', size=12)
plt.xticks(fontproperties=font_properties)
plt.yticks(fontproperties=font_properties)
plt.gca().xaxis.label.set_font_properties(font_labels)
plt.gca().yaxis.label.set_font_properties(font_labels)
plt.legend(handles =[red_patch, blue_patch],prop=font_legend, handlelength=0.7, handleheight=0.7,loc='upper right')
plt.xlabel("Samples")
plt.ylabel("Mean Square Errors (log scale)")

plt.show()
