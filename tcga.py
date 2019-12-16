from train_tool import *
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from util.dataset import get_cifar10,get_tcga,RandomErasing,GaussianNoise


def main():
    def create_model(ema=False):
        print("=> creating {ema}model ".format(
            ema='EMA ' if ema else ''))

#        model = TCN(input_size=1, output_size=33, num_channels=[32] *12, kernel_size=2)
        model = ResNet50(33)
        model = nn.DataParallel(model).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    global best_prec1

    # Data
    print(f'==> Preparing tcga data')

    transform_train = transforms.Compose([
        GaussianNoise(),
        dataset.ToTensor(),

    ])

    transform_val = transforms.Compose([
        dataset.ToTensor(),

    ])

    train_labeled_dataset, train_unlabeled_dataset, val_dataset = get_tcga('./data', args.n_labeled,  transform_train=transform_train, transform_val=transform_val)

    train_labeled_loader = data.DataLoader(train_labeled_dataset, batch_size=args.labeled_batch_size, shuffle=True,
                                           num_workers=args.num_workers,drop_last=True)
    train_unlabeled_loader = data.DataLoader(train_unlabeled_dataset, batch_size=args.unlabeled_batch_size, shuffle=True,
                                             num_workers=args.num_workers,drop_last=True)
    eval_loader = data.DataLoader(val_dataset, batch_size=args.unlabeled_batch_size, shuffle=False, num_workers=args.num_workers)

    model = create_model()
    ema_model = create_model(ema=True)
    tmp_model= create_model(ema=True)


    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, tmp_model, alpha=args.ema_decay)
    cudnn.benchmark = True
    if args.warmup_step>0:
        totals = args.epochs*args.epoch_iteration
        warmup_step = totals//20
        scheduler =  WarmupCosineSchedule(optimizer,warmup_step,totals)
    else:
        scheduler = None
    # optionally resume from a checkpoint
    title = 'tcga'
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        logger = Logger(os.path.join(args.resume, 'tcga_log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out_path, 'tcga_log.txt'), title=title)
        logger.set_names(['epoch', 'Train_class_loss',  'Train_consistency_loss', 'Val_Loss', 'Val_Acc.'])

    if args.evaluate:
        print("Evaluating the primary model:")
        validate(eval_loader, model, criterion)
        print("Evaluating the EMA model:")
        validate(eval_loader, ema_model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        if args.semi:
            if epoch<args.stage1:
                print (' train in semi-supervised stage1')
                class_loss = train(train_labeled_loader, model, ema_model, optimizer, ema_optimizer, epoch, scheduler)
                cons_loss = 0
            else:
                print (' train in semi-supervised stage2')
                class_loss, cons_loss = train_semi(train_labeled_loader, train_unlabeled_loader, model, ema_model, optimizer,ema_optimizer, epoch,scheduler)
        else:
            print (' train in supervised')
            class_loss = train(train_labeled_loader, model, ema_model, optimizer, ema_optimizer, epoch,scheduler)
            cons_loss =0
        print("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            print("Evaluating the  model:")
            val_loss, val_acc = validate(eval_loader, model, criterion,epoch)
            print("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, val_loss, val_acc])

            print("Evaluating the EMA model:")
            ema_val_loss, ema_val_acc = validate(eval_loader, ema_model, criterion,epoch)
            print("--- validation in %s seconds ---" % (time.time() - start_time))
            logger.append([epoch, class_loss, cons_loss, ema_val_loss, ema_val_acc])

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'checkpoint_path', epoch + 1)

if __name__ == '__main__':
    args = create_parser('tcga')
    set_args(args)
    main()
