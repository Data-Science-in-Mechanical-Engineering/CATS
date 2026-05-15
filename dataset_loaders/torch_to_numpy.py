# def prefetch_torch_to_numpy(batch_queue, dataloader):
#     while True:
#         dataloader_iter = iter(dataloader)
#         stop_iteration = False
#         while not stop_iteration:
#             # fill queue with batches
#             while not batch_queue.full():
#                 print(f"filling queue {batch_queue.qsize()}")
#                 try:
#                     batch = next(dataloader_iter)
#                     batch = {k: v.numpy() for k, v in batch.items()}
#                     batch_queue.put(batch)
#                 except StopIteration:
#                     stop_iteration = True
#                     break
#         # wait till consumer has consumed all the batches
#         while not batch_queue.empty():
#             pass

def prefetch_torch_to_numpy(dataloader, input_shared_memory_names, target_shared_memory_names, data_semaphore):
    input_shared_memories = [shared_memory.SharedMemory(name=n) for n in input_shared_memory_names]
    target_shared_memories = [shared_memory.SharedMemory(name=n) for n in target_shared_memory_names]

    used_shared_memory = shared_memory.ShareableList(name="used_shared_memory")

    inputs = [None for _ in used_shared_memory]
    targets = [None for _ in used_shared_memory]
    idx = 0
    while True:
        dataloader_iter = iter(dataloader)
        stop_iteration = False
        while not stop_iteration:
            # fill queue with batches
            data_semaphore.acquire()
            if not used_shared_memory[idx]:
                try:
                    batch = next(dataloader_iter)
                    batch = {k: np.array(v.numpy(), dtype=np.float32) for k, v in batch.items()}
                    
                    if inputs[idx] is None:
                        inputs[idx] = np.ndarray(batch["input"].shape, dtype=np.float32, buffer=input_shared_memories[idx].buf)
                        targets[idx] = np.ndarray(batch["target"].shape, dtype=np.float32, buffer=target_shared_memories[idx].buf)

                    inputs[idx][:] = batch["input"][:]
                    targets[idx][:] = batch["target"][:]
                    used_shared_memory[idx] = True
                    idx = (idx + 1) % len(input_shared_memory_names)
                    data_semaphore.release()
                except StopIteration:
                    stop_iteration = True
                    data_semaphore.release()
                    break
            else:
                data_semaphore.release()
            time.sleep(1.1e-6)
            

class TorchToNumpy:
    def __init__(self, dataloader, buffer_size=10):
        self.dataloader = dataloader
        self.buffer_size = buffer_size

        try:
            self.used_shared_memory = shared_memory.ShareableList([False for _ in range(buffer_size)], name="used_shared_memory")
        except FileExistsError:
            self.used_shared_memory = shared_memory.ShareableList(name="used_shared_memory")
        self.input_shared_memory_names = [f"xxinput_shared_memory_{i}" for i in range(buffer_size)]
        self.target_shared_memory_names = [f"xxtarget_shared_memory_{i}" for i in range(buffer_size)]

        self.input_shared_memories = []
        for n in self.input_shared_memory_names:
            try:
                self.input_shared_memories.append(shared_memory.SharedMemory(create=True, size=500_000_000, name=n))
            except FileExistsError:
                self.input_shared_memories.append(shared_memory.SharedMemory(name=n))

        self.target_shared_memories = []
        for n in self.target_shared_memory_names:
            try:
                self.target_shared_memories.append(shared_memory.SharedMemory(create=True, size=500_000_000, name=n))
            except FileExistsError:
                self.target_shared_memories.append(shared_memory.SharedMemory(name=n))


        self.inputs = [None for _ in self.used_shared_memory]
        self.targets = [None for _ in self.used_shared_memory]

        self.data_semaphore = multiprocessing.Semaphore()

        p = multiprocessing.Process(target=prefetch_torch_to_numpy, 
                                        args=(dataloader, self.input_shared_memory_names, self.target_shared_memory_names, self.data_semaphore))
        p.start()

        self.idx = 0

        self.idx_buffer = 0

    def __len__(self):
        return len(self.dataloader) - 1
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx < len(self.dataloader)-1:
            self.idx += 1
            start_time = time.time()
            self.data_semaphore.acquire()
            while not self.used_shared_memory[self.idx_buffer]:
                self.data_semaphore.release()
                time.sleep(1e-6)
                self.data_semaphore.acquire()
            
            self.data_semaphore.release()
            self.data_semaphore.acquire()
            if self.inputs[self.idx_buffer] is None:
                # context_length: 512 prediction_length: 128
                self.inputs[self.idx_buffer] = np.ndarray((1024, 512, 8), 
                                                            dtype=np.float32, buffer=self.input_shared_memories[self.idx_buffer].buf)
                self.targets[self.idx_buffer] = np.ndarray((1024, 512 + 128, 8), 
                                                            dtype=np.float32, buffer=self.target_shared_memories[self.idx_buffer].buf)
            
            input_ = self.inputs[self.idx_buffer][:]
            target_ = self.targets[self.idx_buffer][:]

            self.used_shared_memory[self.idx_buffer] = False
            self.idx_buffer = (self.idx_buffer + 1) % self.buffer_size
            self.data_semaphore.release()
            
            # print(f"Time to get batch: {time.time() - start_time}")
            return {"input": input_, "target": target_}
        else:   
            raise StopIteration