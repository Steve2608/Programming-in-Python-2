from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def plot(input, target, prediction, writer: SummaryWriter, epoch: int):
    fig, ax = plt.subplots(1, 3)

    for i in range(len(input)):
        ax[0].clear()
        ax[0].set_title(f'Input_{i}')
        ax[0].imshow(input[i].detach().cpu(), cmap=plt.cm.gray, interpolation='none',
                        origin='lower')
        ax[0].set_axis_off()

        ax[1].clear()
        ax[1].set_title(f'Target_{i}')
        ax[1].imshow(target[i], cmap=plt.cm.gray, interpolation='none', origin='lower')
        ax[1].set_axis_off()

        ax[2].clear()
        ax[2].set_title(f'Prediction_{i}')
        ax[2].imshow(prediction[i].detach().cpu().view(target[i].shape), cmap=plt.cm.gray,
                        interpolation='none',
                        origin='lower')
        ax[2].set_axis_off()

        fig.tight_layout()
        writer.add_figure(tag=f'train/samples_{i}', figure=fig, global_step=epoch)

    del fig
