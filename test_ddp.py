import torch
import torch.nn as nn
import hydra
import logging
from mcunet.utils import executor
import socket
import os

logger = logging.getLogger(__name__)


def run(args):
    from mcunet.utils import distrib
    
    distrib.init(args, args.rendezvous_file)
    torch.manual_seed(args.seed)
    logger.info("Running on CUDA: %s", torch.cuda.current_device())
    logger.info("Running on world size: %d", args.world_size)
    logger.info("Running on rank: %d", args.rank)    
    return

def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("test_ddp").setLevel(logging.DEBUG)

    # Updating paths in config
    if args.continue_from:
        args.continue_from = os.path.join(
            os.getcwd(), "..", args.continue_from, args.checkpoint_file)
    args.db.root = hydra.utils.to_absolute_path(args.db.root + '/' + args.db.name)
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    run(args)
    

@hydra.main(config_path="conf", config_name="config")
def main(args):
    try:
        if args.ddp and args.rank is None:
            print("Distributed training with DDP")
            logger.info("Distributed training with DDP")
            executor.start_ddp_workers(args)
            return
        _main(args)
    except Exception:
        logger.exception("Exception occurred", Exception)
        logger.info("Distributed training without DDP")
        print(args)
        
if __name__ == "__main__":
    main()