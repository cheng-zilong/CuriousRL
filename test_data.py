# %%
from torch._C import dtype
from CuriousRL.data import batch
from CuriousRL.data.data import Data
from CuriousRL.data.batch import Batch
from CuriousRL.data.dataset import Dataset
import torch
from CuriousRL.utils.config import global_config
import numpy as np
import time
if __name__ == "__main__":
    global_config.set_is_cuda(True)

    data2 = Data(state = np.random.randint((5, 5)), action = np.random.random(10), on_gpu=True)
    data3 = Data(state = np.random.randint((5, 5)), action = np.random.random(10), on_gpu=True)
    batch1 = Batch(True, data2,data2,data2).share_memmory_()
    batch2 = Batch(on_gpu=False, state = np.random.randint((5, 5)), action = np.random.random((5,10)))
    batch1[0] = data3
    print(batch1)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(batch2)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    dataset_test3 = Dataset(buffer_size = 5, on_gpu=False)
    dataset_test3.update(data3)
    dataset_test3.update(batch1)
    dataset_test3.update(batch1)
    dataset_test3.update(data3)
    dataset_test3 = dataset_test3.to_gpu()
    print(dataset_test3.fetch_random_data(3))

    # dataset1 = Dataset(10000)
    # time1 = time.time()
    # for i in range(len(dataset1)):
    #     # print(i)
    #     dataset1.update(Data(state = np.random.random((5, 5)), action = np.random.random(10), reward  = np.random.random(1)))
    # time2 = time.time()
    # for i in range(len(dataset1)):
    #     # print(i)
    #     dataset1.fetch_random_data(128)
    # time3= time.time()
    # print(time3-time2)
    # data3_from_gym = data1_from_gym.cat(data1_from_gym)
    # dataset1.update_dataset(data3_from_gym)
# %%
