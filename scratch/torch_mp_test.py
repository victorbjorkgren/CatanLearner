import torch.multiprocessing as mp
import torch
import time


def writer_shared_memory(shared_array, tensor_shape, num_tensors):
    for i in range(num_tensors):
        tensor = torch.randn(tensor_shape)
        shared_array[i] = tensor


def reader_shared_memory(shared_array, num_tensors):
    for i in range(num_tensors):
        tensor = shared_array[i]
        # Perform some operation to ensure tensor is accessed
        _ = tensor.sum()


def test_shared_memory():
    tensor_shape = (1000, 1000)
    num_tensors = 10

    # Create shared memory array
    shared_array = mp.Array('f', num_tensors * tensor_shape[0] * tensor_shape[1])

    # Convert shared_array to torch tensors
    shared_tensors = [
        torch.frombuffer(shared_array, dtype=torch.float32, count=tensor_shape[0] * tensor_shape[1]).reshape(
            tensor_shape) for _ in range(num_tensors)]

    writer_process = mp.Process(target=writer_shared_memory, args=(shared_tensors, tensor_shape, num_tensors))
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