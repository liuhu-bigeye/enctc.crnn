import os
import sys
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
# draw multiple accuracies

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
    plts = []
    for idx_, filename in enumerate(sys.argv[1:]):
        losses = []

        re_str = ' accuray: .*'
        with open(filename) as f:
            for line in f.readlines():
                for match in re.findall(re_str, line):
                    match = float(match.split(' ')[-1])
                    losses.append(match)

        plts.append(plt.plot(range(0, len(losses)), losses, label='Acc%d'%idx_)[0])

    plt.legend(handles=plts, loc=4)
    file_names = sys.argv[1:]
    file_dir = os.path.dirname(file_names[0])
    file_names = os.path.join(file_dir, ':'.join([os.path.basename(f) for f in file_names])) + '.acc'

    save(file_names, ext='png', close=True)
