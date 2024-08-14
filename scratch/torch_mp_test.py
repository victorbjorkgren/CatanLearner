import torch
import torch.multiprocessing as mp
import time

def writer_shared_memory(shared_tensors, num_tensors):
    for i in range(num_tensors):
        tensor = torch.randn(shared_tensors[i].shape)
        shared_tensors[i].copy_(tensor)

def reader_shared_memory(shared_tensors, num_tensors):
    for i in range(num_tensors):
        tensor = shared_tensors[i]
        # Perform some operation to ensure tensor is accessed
        _ = tensor.sum()

def test_shared_memory():
    tensor_shape = (1000, 1000)
    num_tensors = 10

    # Create shared memory tensors
    shared_tensors = [torch.empty(tensor_shape).share_memory_() for _ in range(num_tensors)]

    writer_process = mp.Process(target=writer_shared_memory, args=(shared_tensors, num_tensors))
    reader_process = mp.Process(target=reader_shared_memory, args=(shared_tensors, num_tensors))

    start_time = time.time()
    writer_process.start()
    reader_process.start()

    writer_process.join()
    reader_process.join()
    end_time = time.time()

    return end_time - start_time

if __name__ == '__main__':
    shared_memory_time = test_shared_memory()
    print(f"Time taken using shared memory: {shared_memory_time:.4f} seconds")