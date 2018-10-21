import os
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
# draw multiple losses

def save(path, ext='png', close=True, verbose=True):
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    #The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),
    # Actually save the figure
    plt.savefig(savepath)
    # Close it
    if close:
        plt.close()
    if verbose:
        print("Done")

if __name__ == '__main__':
    clip_num = 20
    rate = 10
    plts = []
    for idx_, filename in enumerate(sys.argv[1:]):
        losses = [[], []]

        re_str = [' loss: [^,]*', ' Loss: [^,]*']
        with open(filename) as f:
            for line in f.readlines():
                for idx in range(len(re_str)):
                    found = False
                    for match in re.findall(re_str[idx], line):
                        match = float(match.split(' ')[-1])
                        losses[idx].append(match)
                        found = True
                    if idx==1 and not found:
                        for match in re.findall(' Loss CTC: [^,]*', line):
                            match = float(match.split(' ')[-1])
                            losses[idx].append(match)


        loss1_mean = [np.mean(losses[1][_: _+rate]) for _ in range(0, len(losses[1]), rate)]
        losses[0] = losses[0][clip_num:]
        loss1_mean = loss1_mean[clip_num:]

        plts.append(plt.plot(range(0, len(losses[0])), losses[0], label='Test loss: %d'%idx_)[0])
        plts.append(plt.plot(range(0, len(loss1_mean)), loss1_mean, label='Train loss: %d'%idx_)[0])

    plt.legend(handles=plts)
    file_names = sys.argv[1:]
    file_dir = os.path.dirname(file_names[0])
    file_names = os.path.join(file_dir, ':'.join([os.path.basename(f) for f in file_names]))

    save(file_names, ext='png', close=True)
