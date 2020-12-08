from boilr import Trainer

from git.lvae.experiment import LVAEExperiment

import argparse
args = argparse.Namespace()
args.test_imgs_every= 1000
args.test_log_every = 100
args.train_log_every= 100
args.max_grad_norm  = None
args.loglikelihood_every = 1000
args.z_dims = [32,32,32]
args.downsample = [1,1,1]
args.weight_decay = 0.01
args.dropout = 0.2
args.free_bits = 0.2
args.no_batch_norm = False
args.likelihood = "gaussian"
args.dataset_name = "used_sets"
args.blocks_per_layer = 2
args.n_filters = 64
args.skip_connections = False
args.gated = True
args.residual_type = "bacdbacd"
args.beta_anneal = 0
args.nonlin = "elu"
args.learn_top_prior = False
args.seed = 0
args.additional_descr = ""
args.resume = ""


args.checkpoint_every = 100000
args.no_cuda = False
args.dry_run = True
args.batch_size = 32
args.test_batch_size = 14
args.merge_layers = "residual"
args.no_initial_downscaling = True
args.analytical_kl = True
args.simple_data_dependent_init = True
args.lr = 3e-4
args.max_epochs = 5000
args.max_steps = 5000
args.have_tensorboard = True

def main():
    experiment = LVAEExperiment(args)
    trainer = Trainer(experiment)
    trainer.run()


if __name__ == "__main__":
    main()
