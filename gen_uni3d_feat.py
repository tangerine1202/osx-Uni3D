import logging
import argparse
from pathlib import Path
import random

from easydict import EasyDict
import torch
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from models.uni3d import create_uni3d, Uni3D

log = logging.getLogger()


def log_memory_usage():
    # Memory currently allocated by tensors
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    # Memory reserved by PyTorch's caching allocator
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB

    log.info(f"Allocated Memory: {allocated_memory:.2f} MB")
    log.info(f"Reserved Memory: {reserved_memory:.2f} MB")


def get_model_config(ckpt_path):
    cfg = EasyDict()
    cfg.pc_feat_dim = 768  # tiny: 192, small: 384, base: 768
    cfg.embed_dim = 1024
    cfg.num_group = 512
    cfg.group_size = 64
    cfg.pc_encoder_dim = 512
    cfg.pc_model = 'eva02_base_patch14_448'
    cfg.pretrained_pc = ''
    cfg.drop_path_rate = 0.2
    cfg.patch_dropout = 0
    cfg.ckpt_path = ckpt_path
    return cfg


def load_model_ckpt(model_cfg):
    model: Uni3D = create_uni3d(model_cfg).cpu()
    checkpoint = torch.load(model_cfg.ckpt_path, map_location='cpu', weights_only=False)
    # log.debug(f'Loaded checkpoint {model_cfg.ckpt_path}')

    sd = checkpoint['module']
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.cuda()


def generate_random_pcd_data(n_points, batch_size=1):
    pts = torch.rand(batch_size, n_points, 3)
    rgb = torch.rand(batch_size, n_points, 3)
    data = torch.cat([pts, rgb], dim=-1)
    return data


def preprocess_pc(pc: torch.Tensor) -> torch.Tensor:
    # pc = normalize_pc(pc) # numpy no batch version
    xyz = pc[..., :3]
    rgb = pc[..., 3:]
    # flip y and z seems to work better
    xyz = xyz[:, :, [0, 2, 1]]
    # centering
    pc = xyz - torch.mean(xyz, dim=1, keepdim=True)
    # normalize pc to [-1, 1]
    norm = torch.norm(pc, dim=2, keepdim=True)
    max_norm = torch.max(norm, dim=1, keepdim=True).values
    pc = pc / max_norm
    if any(max_norm < 1e-6):
        n = sum(max_norm < 1e-6)
        log.warning(f'Found {n} points with norm < 1e-6, setting to zero')
        pc[max_norm < 1e-6] = 0
    ret = torch.cat([pc, rgb], dim=-1)
    return ret.float()


def infer_uni3d_features(model, data):
    if data.device.type != 'cuda':
        data = data.cuda()
    data = preprocess_pc(data)
    features = model.encode_pc(data)  # [bs, n_pts]
    return features.detach().cpu()


def plot_PCA(feat_dict, output_dir=None, show=False, seed=42):
    def get_vertex_cnt(x): return int(x[4:].split('_')[0])
    keys = list(feat_dict.keys())
    features = np.stack(list(feat_dict.values()), axis=0)

    pca = PCA(n_components=2, random_state=seed)
    data_pca = pca.fit_transform(features)
    data_pca = (data_pca - data_pca.min()) / (data_pca.max() - data_pca.min())

    # Visualize for point with different color
    plt.figure(figsize=(10, 10))
    for i, key in enumerate(keys):
        v_cnt = get_vertex_cnt(key)
        plt.scatter(data_pca[i, 0], data_pca[i, 1], c=f'C{v_cnt}', label=f'{v_cnt} vertices')

    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())

    # Save the plot
    if output_dir:
        output_path = output_dir / '_PCA_plot.png'  # prefix "_" to move to top in directory
        plt.savefig(output_path)
        log.info(f'PCA plot saved to {output_path}')

    if show:
        plt.show()


def main(args):
    # Load Uni3D model
    log.info(f'Loading model from {args.ckpt_path}')
    model_cfg = get_model_config(args.ckpt_path)
    model = load_model_ckpt(model_cfg)
    log.info(f'Model loaded')

    log.info(f'Loading data from {args.input}')
    feat_dict = {}
    if args.is_input_dir:
        if not args.input.is_dir():
            raise ValueError(f'Input is not dir, please remove -d (--is_input_dir) flag: {args.input}')

        files = sorted(list(args.input.iterdir()))
        for i in tqdm(range(0, len(files), args.batch_size), desc='Inferencing features'):
            sub_files = files[i:i + args.batch_size]
            data = []

            # Load data
            for file in sub_files:
                data.append(torch.from_numpy(np.load(file)))
                log.debug(f'Loaded data from {file}')
            data = torch.stack(data, dim=0)
            assert len(data.shape) == 3 and data.shape[-1] == 6, f'Expect data shape (B, N, 6), Got {data.shape}'

            # infer features
            features = infer_uni3d_features(model, data)
            for j, file in enumerate(files[i:i + args.batch_size]):
                feat_dict[file.stem] = features[j].numpy()

    else:
        if args.input:
            if args.input.is_dir():
                log.warning('Input is a directory. Use -d (--is_input_dir) flag')

            data = torch.from_numpy(np.load(args.input))[None, ...]
            log.debug(f'Loaded data from {args.input}')
        else:
            log.warning('No input data provided. Generating random data')
            data = generate_random_pcd_data(args.num_points, batch_size=1)
        assert len(data.shape) == 3 and data.shape[-1] == 6, f'Expect data shape (B, N, 6), Got {data.shape}'

        features = infer_uni3d_features(model, data)
        feat_dict = {args.input.stem: features.numpy()}

    # log.debug(f'{feat_dict.keys()}')

    log_memory_usage()

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, feat in feat_dict.items():
            output_path = args.output_dir / f'{name}.npy'
            np.save(output_path, feat)
            log.debug(f'Features saved to {output_path}')
        log.info(f'Features saved to {args.output_dir}')

    if args.is_input_dir:
        plot_PCA(feat_dict, args.output_dir, show=args.viz, seed=args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path,
                        help='Path to the np file of the point cloud (shape [n_points, 6])', default=None)
    parser.add_argument('-d', '--is_input_dir', action='store_true', help='Flag to indicate input is a directory')
    parser.add_argument('-o', '--output_dir', type=Path, default='output_uni3d', help='Dir to save the point cloud')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('-n', '--num_points', type=int, default=1024, help='Number of points to sample')
    parser.add_argument('-ckpt', '--ckpt_path', type=Path,
                        default='Uni3D-B-ensembled.pt', help='Path to the checkpoint')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-viz', '--viz', action='store_true', help='Visualize the point cloud')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for PCA')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] - %(message)s',
    )

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)
