from train_tool import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from dataset import get_cifar10,get_tcga


def main():

    global best_prec1

    # Data
    print(f'==> Preparing data')
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    transform_train = transforms.Compose([
#        transforms.RandomHorizontalFlip(),
        dataset.ToTensor(),
#        transforms.Normalize(**channel_stats)
    ])

    transform_val = transforms.Compose([
        dataset.ToTensor(),
 #       transforms.Normalize(**channel_stats)
    ])

    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset = get_tcga('./data', args.n_labeled,  transform_train=transform_train, transform_val=transform_val)

    train_labeled_loader = data.DataLoader(train_labeled_dataset, batch_size=args.labeled_batch_size, shuffle=True, num_workers=args.num_workers,
                                          drop_last=True)
    train_unlabeled_loader = data.DataLoader(train_unlabeled_dataset, batch_size=args.unlabeled_batch_size, shuffle=True,
                                             num_workers=args.num_workers,
                                             drop_last=True)

    eval_loader = data.DataLoader(val_dataset, batch_size=args.unlabeled_batch_size, shuffle=False, num_workers=args.num_workers)
#    test_loader = data.DataLoader(test_dataset, batch_size=args.unlabeled_batch_size, shuffle=False, num_workers=args.num_workers)
    model = create_model()
    ema_model = create_model(ema=True)

    # LOG.info(parameters_string(model))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    cudnn.benchmark = True
    warmup_step = args.epoch_iteration* args.epochs//40
    totals = args.epoch_iteration* args.epochs//4
#    scheduler =  WarmupCosineSchedule(optimizer,warmup_step,totals)
    scheduler = None
    # optionally resume from a checkpoint
    title = 'noisy-cifar-10'
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        logger = Logger(os.path.join(args.resume, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out_path, 'log.txt'), title=title)
        logger.set_names(['epoch', 'Train_class_loss',  'Train_consistency_loss',  'Valid_Loss', 'Valid_Acc.', 'Test_Loss', 'Test_Acc.'])

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, criterion)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        class_loss, cons_loss = train(train_labeled_loader, train_unlabeled_loader, model, ema_model, optimizer,ema_optimizer, epoch, scheduler)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the  model:")
            val_loss, val_acc = validate(eval_loader, model, criterion)
            LOG.info("Test the  model:")
            if test_dataset is None:
                test_loss, test_acc = 0,0
 #           test_loss, test_acc = validate(test_loader, model, criterion)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, val_loss, val_acc, test_loss, test_acc])

            LOG.info("Evaluating the EMA model:")
            ema_val_loss, ema_val_acc = validate(eval_loader, ema_model,criterion)
            LOG.info("Test the EMA model:")
            if test_dataset is None:
                ema_test_loss, ema_test_acc = 0,0
#            ema_test_loss, ema_test_acc = validate(test_loader, ema_model, criterion)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, ema_val_loss, ema_val_acc, ema_test_loss, ema_test_acc])

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'checkpoint_path', epoch + 1)

if __name__ == '__main__':
    args = create_parser('tcga')
    main()
