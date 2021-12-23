import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# change working directory
os.chdir(sys.path[0])
# create event_accumulator instance


fig = plt.figure(dpi=200)
ax = fig.add_subplot()
#ax.set_ylim(top=10.0)
ax.plot(train_loss, label='Training Loss')
ax.plot(valid_loss, label='Validation Loss')
ax.set_yscale('log')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

#plt.show()

plt.savefig('./curve.png', dpi=200)