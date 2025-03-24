import logging
import argparse
from pathlib import Path

import trimesh
import numpy as np
import open3d as o3d
from tqdm.auto import tqdm

log = logging.getLogger()


def sample_pcd_from_mesh(mesh, num_points):
    points, _, rgba = trimesh.sample.sample_surface(mesh, num_points, sample_color=True)
    points = np.asarray(points)
    rgba = np.asarray(rgba) / 255
    return points, rgba


def load_stl_file(input_path):
    input_path = Path(input_path)
    mesh = None
    if input_path.is_dir():
        mesh = trimesh.Trimesh()
        for file in input_path.iterdir():
            if file.suffix == '.stl':
                part = trimesh.load_mesh(file)
                mesh = trimesh.util.concatenate([mesh, part])
                log.debug(f'Loaded {file}')
        if mesh.is_empty:
            raise ValueError('No valid STL files found in the directory')
    elif input_path.is_file() and args.path.suffix == '.stl':
        mesh = trimesh.load_mesh(input_path)
        log.debug(f'Loaded {input_path}')
    else:
        raise ValueError(f'Invalid path: {input_path}')
    return mesh


def process(mesh, name, args):
    # Sample points from the surface of the mesh
    pts, rgba = sample_pcd_from_mesh(mesh, args.num_points)

    # Concatenate the points and colors
    pcd = np.concatenate([pts, rgba[:, :3]], axis=-1)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f'{name}.npy'
        np.save(output_path, pcd)
        log.debug(f'Point cloud saved to {output_path}')

    return output_path, pcd


def main(args):
    if args.viz and args.is_input_dir:
        log.warning('Only visualizing the last mesh in the directory')

    log.info(f'Loading mesh from {args.input}')
    if args.is_input_dir:
        if not args.input.is_dir():
            raise ValueError(f'Input is not dir: {args.input}')

        dirs = sorted(list(args.input.iterdir()))
        for mesh_dir in tqdm(dirs, desc='Processing meshes'):
            mesh = load_stl_file(mesh_dir)
            output_path, pcd = process(mesh, mesh_dir.stem, args)
        output_path = output_path.parent
    else:
        mesh = load_stl_file(args.input)
        output_path, pcd = process(mesh, args.input.stem, args)
    log.info(f'Point cloud saved to {output_path}')

    if args.viz:
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3])
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
        o3d.visualization.draw_geometries([pcd_o3d])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path, help='Path to the STL file or dir')
    parser.add_argument('-d', '--is_input_dir', action='store_true', help='Flag to indicate input is a directory')
    parser.add_argument('-o', '--output_dir', type=Path, default='output_pcd', help='Dir to save the point cloud')
    parser.add_argument('-n', '--num_points', type=int, default=10000, help='Number of points to sample')
    parser.add_argument('-viz', '--viz', action='store_true', help='Visualize the point cloud')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] - %(message)s',
    )

    main(args)
