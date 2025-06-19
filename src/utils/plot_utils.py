import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss_acc(loss, acc, run_save_dir,phase_names, current_time):
    for phase in phase_names:
        plt.figure(figsize=(10, 5))
        plt.title(f"Loss and Accuracy {phase}")
        plt.plot(loss[phase], label="Loss")
        plt.plot(acc[phase], label="Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc='best')
        # plt.savefig(f"{run_save_dir}/{current_time}_Loss_Acc_"+phase+".png")
        # plt.show()

