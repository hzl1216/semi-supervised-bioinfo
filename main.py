from train_tool import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from dataset import get_cifar10


def main():

    global best_prec1

    # Data
    print(f'==> Preparing cifar10')
    transform_train = transforms.Compose([
        dataset.ToTensor(),
    ])

    transform_val = transforms.Compose([
        dataset.ToTensor(),
    ])

    train_dataset, val_dataset, test_dataset = get_cifar10('./data', args.n_labeled, train_repeated=5, transform_train=transform_train, transform_val=transform_val)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                          drop_last=True)


    eval_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = create_model()
    ema_model = create_model(ema=True)

    # LOG.info(parameters_string(model))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    cudnn.benchmark = True
 #   scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_step, t_total=totals)
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
        logger.set_names(['epoch','Model_type','Train_class_loss',  'Train_consistency_loss',  'Valid_Loss', 'Valid_Acc.', 'Test_Loss', 'Test_Acc.'])

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, criterion)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        class_loss, cons_loss = train(train_loader, model, ema_model, optimizer, epoch)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            val_loss,val_acc = validate(eval_loader, model, criterion)
            LOG.info("Test the primary model:")
            test_loss, test_acc = validate(test_loader, model, criterion)
            logger.append([epoch,0,class_loss, cons_loss, val_loss, val_acc, test_loss, test_acc])

            LOG.info("Evaluating the EMA model:")
            ema_val_loss, ema_val_acc = validate(eval_loader, ema_model,criterion)
            LOG.info("Test the EMA model:")
            ema_test_loss, ema_test_acc = validate(test_loader, ema_model, criterion)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch,1, class_loss, cons_loss, ema_val_loss, ema_val_acc, ema_test_loss, ema_test_acc])

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'checkpoint_path', epoch + 1)

if __name__ == '__main__':
    args = create_parser()
    main()
