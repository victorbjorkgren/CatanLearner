import multiprocessing
import torch
import time

# Define the size of the tensors and the number of transfers
tensor_size = (1000, 1000)
num_tensors = 10

# Function to create and push tensors to shared list
def writer_list(shared_list, num_tensors):
    for i in range(num_tensors):
        tensor = torch.randn(tensor_size)
        shared_list.append(tensor)

# Function to read tensors from shared list
def reader_list(shared_list, num_tensors):
    for i in range(num_tensors):
        if shared_list:  # Ensure list is not empty before popping
            tensor = shared_list.pop(0)
            # Perform some operation to ensure tensor is accessed
            _ = tensor.sum()

# Function to create and push tensors to queue
def writer_queue(queue, num_tensors):
    for i in range(num_tensors):
        tensor = torch.randn(tensor_size)
        queue.put(tensor)

# Function to read tensors from queue
def reader_queue(queue, num_tensors):
    for i in range(num_tensors):
        tensor = queue.get()
        # Perform some operation to ensure tensor is accessed
        _ = tensor.sum()

# Function for single-threaded tensor creation and processing
def single_threaded_test(num_tensors):
    for i in range(num_tensors):
        tensor = torch.randn(tensor_size)
        # Perform some operation to ensure tensor is accessed
        _ = tensor.sum()

def test_shared_list():
    manager = multiprocessing.Manager()
    shared_list = manager.list()

    writer_process = multiprocessing.Process(target=writer_list, args=(shared_list, num_tensors))
    reader_process = multiprocessing.Process(target=reader_list, args=(shared_list, num_tensors))

    start_time = time.time()
    writer_process.start()
    reader_process.start()

    writer_process.join()
    reader_process.join()
    end_time = time.time()

    return end_time - start_time

def test_queue():
    queue = multiprocessing.Queue()

    writer_process = multiprocessing.Process(target=writer_queue, args=(queue, num_tensors))
    reader_process = multiprocessing.Process(target=reader_queue, args=(queue, num_tensors))

    start_time = time.time()
    writer_process.start()
    reader_process.start()

    writer_process.join()
    reader_process.join()
    end_time = time.time()

    return end_time - start_time

def test_single_threaded():
    start_time = time.time()
    single_threaded_test(num_tensors)
    end_time = time.time()

    return end_time - start_time

if __name__ == '__main__':
    list_time = test_shared_list()
    queue_time = test_queue()
    single_threaded_time = test_single_threaded()

    print(f"Time taken using shared list: {list_time:.4f} seconds")
    print(f"Time taken using queue: {queue_time:.4f} seconds")
    print(f"Time taken using single-threaded approach: {single_threaded_time:.4f} seconds")