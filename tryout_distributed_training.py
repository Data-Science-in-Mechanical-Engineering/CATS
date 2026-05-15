# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler

# from dataset_loaders.utsdataset import UTSDataSource
# import torch

# import jax
# import jax.numpy as jnp
# from jax.sharding import NamedSharding, PartitionSpec as P
# import numpy as np

import sys
import os
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# proc_id = int(sys.argv[1])
# num_procs = int(sys.argv[2])
# print(f"Process ID: {proc_id}, Number of Processes: {num_procs}")

import logging
print("Hello")
logger = logging.getLogger(__name__)
try:
    logging.basicConfig(
        filename='example.log',
        encoding='utf-8',
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.addHandler(logging.StreamHandler())
    logger.debug('This message should go to the log file')
    logger.info('So should this')
    raise ValueError("This is a test error message")
    logger.warning('And this, too')
    logger.error('And non-ASCII stuff, too, like Øresund and Malmö')
except Exception as e:
    logger.exception("Exception occurred")
    print(f"An error occurred: {e}")
    sys.exit(1)

exit()

# def world_info_from_env():
#     local_rank = 0
#     for v in ('SLURM_LOCALID', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'LOCAL_RANK'):
#         if v in os.environ:
#             local_rank = int(os.environ[v])
#             break
#     global_rank = 0
#     for v in ('SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK'):
#         if v in os.environ:
#             global_rank = int(os.environ[v])
#             break
#     world_size = 1
#     for v in ('SLURM_NTASKS', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE'):
#         if v in os.environ:
#             world_size = int(os.environ[v])
#             break

#     return local_rank, global_rank, world_size

# # init torch and jax distributed
# local_rank, global_rank, world_size = world_info_from_env()
# print(f"Local Rank: {local_rank}, Global Rank: {global_rank}, World Size: {world_size}")

# with open(f"test_{global_rank}.txt", "w") as f:
#     f.write(f"Local Rank: {local_rank}, Global Rank: {global_rank}, World Size: {world_size}\n")

# print(f"Local Rank: {local_rank}, Global Rank: {global_rank}, World Size: {world_size}")
# torch.distributed.init_process_group(
#   world_size=world_size,
#   rank=global_rank,)

# with open(f"test2_{global_rank}.txt", "w") as f:
#     f.write(f"Local Rank: {local_rank}, Global Rank: {global_rank}, World Size: {world_size}\n")

# jax.distributed.initialize(f'{os.environ['MASTER_ADDR']}:10000', local_device_ids=[0, 1, 2, 3, 4])
# print("process id =", jax.process_index())
# print("global devices =", jax.devices())
# print("local devices =", jax.local_devices())
# print(f"Number devices: {jax.device_count()}")
# print(f"Number local devices: {jax.local_device_count()}")

# mesh = jax.make_mesh((jax.device_count(),), ('batch'))
# sharding = NamedSharding(mesh, P('batch'))

# data_source = UTSDataSource(dataset_base_path="/hpcwork/p0021919/datasets", 
#                                 subset_name=r'UTSD-1G', 
#                                 flag='train',
#                                 split=0.9, context_length=512, prediction_length=96,
#                                 scale=True, stride=1)

# dist_sampler = DistributedSampler(data_source, shuffle=True)

# data_loader = DataLoader(
#         data_source,
#         batch_size=1024,
#         sampler=dist_sampler,
#         drop_last=True
# )

# for i, batch in enumerate(data_loader):
#     print(f"Batch {i}: {batch['target'].shape}")
#     process_batch = np.array(batch['target'], dtype=np.float32)
#     # assemble a global array containing the per-process batches from all processes
#     global_batch = jax.make_array_from_process_local_data(sharding, process_batch)

#     print(global_batch.shape)

#     jax.debug.visualize_array_sharding(global_batch[:, :, 0])

#     if i == 10:  # Limit to 10 batches for demonstration
#         break