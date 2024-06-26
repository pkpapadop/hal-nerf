import os.path as osp
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms
from pose_regressor.dataset_loaders.mega_nerf_data import mega_nerf_data
import resource
import torch.multiprocessing as mp

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# Set the sharing strategy to 'file_system'
mp.set_sharing_strategy('file_system')


def load_mega_nerf_dataloader(args, nerfacto_params):
    ''' Data loader for Pose Regression Network '''
    if args.pose_only:  # if train posenet is true
        pass
    else:
        raise Exception('wrong setting')
    data_dir, scene = osp.split(args.datadir)
    print('aaaaaaaaaaaaaaaaaaaaa')
    print(data_dir, scene)

    # transformer
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))

    ret_idx = False  # return frame index
    fix_idx = False  # return frame index=0 in training
    ret_hist = False

    if 'NeRFH' in args:
        if args.NeRFH:
            ret_idx = True
            if args.fix_index:
                fix_idx = True

    # encode hist experiment
    if args.encode_hist:
        ret_idx = False
        fix_idx = False
        ret_hist = True

    #coordinate = torch.load(osp.join(args.datadir, "coordinates.pt"))
    #args.map_scale = float(coordinate["pose_scale_factor"])
    kwargs = dict(scene=scene, data_path=data_dir,
                  transform=data_transform, target_transform=target_transform,
                  df=args.df, ret_idx=ret_idx, fix_idx=fix_idx,
                  ret_hist=ret_hist, hist_bin=args.hist_bin,
                  hwf=[nerfacto_params["height"], nerfacto_params["width"],
                       float(nerfacto_params["fx"])])

    train_set = mega_nerf_data(mode='train', trainskip=args.trainskip, **kwargs)
    val_set = mega_nerf_data(mode='val', testskip=args.testskip, **kwargs)
    test_set = mega_nerf_data(mode='test', testskip=1, **kwargs)

    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = test_set.gt_idx

    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val_dl = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    hwf = [train_set.H, train_set.W, train_set.focal]
    i_split = [i_train, i_val, i_test]

    return train_dl, val_dl, test_dl, hwf, i_split
