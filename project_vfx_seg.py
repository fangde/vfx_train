
import tensorflow as tf


from input_vfx import train_input
from trainer import fit
from unet2d import model_fun_seg

from ETL import SetRun
import click

sess = tf.InteractiveSession()


@click.command()
@click.option('--rn', prompt='the runname',
              help='the run name.')
def StartTrain(rn):
    batch_size = 1
    grid_size = 512
    channel = 16

    


    model = model_fun_seg(batch_size, grid_size, '/gpu:0', channel)

    runname = 'vfx_seg_test'+rn
    SetRun(runname)

    data_g = train_input()

    fit(model, data_g, batch_size=batch_size, runname=runname, sess=sess)


if __name__ == '__main__':
    StartTrain()
