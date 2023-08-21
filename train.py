import logging
import os
import socket
import hydra


logger = logging.getLogger(__name__)

def run(args):
    import src.distrib as distrib
    import src.dataset as dataset
    from src.trainer import Trainer
    from mcunet.model_zoo import build_model
    from mcunet.gumbel_module.gumbel_net import GumbelMCUNet
    import torch
    import torch.nn as nn
    from fvcore.nn import FlopCountAnalysis
    from mcunet.utils import rm_bn_from_net
    
    logger.info("Running on host %s", socket.gethostname())
    distrib.init(args, args.rendezvous_file)
    torch.manual_seed(args.seed)
    
    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size
    
    model, img_resize, info = build_model(args.model.lower(), pretrained=True)
    
    trainset, testset, num_classes = dataset.get_loader(args, img_resize)
    tr_loader = distrib.loader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    tt_loader = distrib.loader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    data = {"tr": tr_loader, "tt": tt_loader}
    
    logger.debug(model)
    model_size = sum(p.numel() for n, p in model.named_parameters()) * 4 / 2**20
    logger.info("Model size: %.2f MB", model_size)
    gumbel_config = args.gumbel
    gumbel_model = GumbelMCUNet.build_from_config(model.config, gumbel_config)
    logger.info("Change MCUModel to GumbelMCUNets")
    gumbel_model.load_pretrained_mcunet_param(model)
    logger.info("load pretrained MCUModel to GumbelMCUNets")
    
    if args.optim == 'adam':
        
        optimizer = torch.optim.Adam(gumbel_model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(gumbel_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    
    rm_bn_from_net(model)
    original_flops = FlopCountAnalysis(model, torch.randn(1, 3, img_resize, img_resize)).total()
    logger.info("Original Flops: %.2f M", original_flops / 1e6)

    del model
    
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        gumbel_model.cuda()
        criterion.cuda()
    
    trainer = Trainer(data, gumbel_model, criterion, optimizer, args, original_flops)
    trainer.train()
    

def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("gumbel_mcunet").setLevel(logging.DEBUG)
    
    if args.continue_from:
        args.continue_from = os.path.join(os.getcwd(), "..", args.continue_from, args.checkpoint_file)
    args.db.root = hydra.utils.to_absolute_path(args.db.root)
    
    logger.info("For logs, checkpoints and sample check %s", os.getcwd())
    logger.debug(args)
    run(args)
    
@hydra.main(config_path='conf', config_name='config_gumbel.yaml')
def main(args):
    try:
        if args.ddp and args.rank is None:
            from src.executor import start_ddp_workers
            start_ddp_workers(args)
            return
        _main(args)
    except Exception:
        logger.exception("Something went wrong")
        os._exit(1)
        

if __name__ == "__main__":
    main()