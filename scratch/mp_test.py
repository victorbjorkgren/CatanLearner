import multiprocessing
import torch
import time


def dummy_function():
    pass

# Function to create processes and immediately join them (no operation)
def measure_process_setup_overhead():


    start_time = time.time()

    process1 = multiprocessing.Process(target=dummy_function)
    process2 = multiprocessing.Process(target=dummy_function)

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    end_time = time.time()

    return end_time - start_time


# Original functions for comparison
def writer_list(shared_list, num_tensors):
    for i in range(num_tensors):
        tensor = torch.randn((10, 10))
        shared_list.append(tensor)


def reader_list(shared_list, num_tensors):
    for i in range(num_tensors):
        if shared_list:
            tensor = shared_list.pop(0)
            _ = tensor.sum()


def writer_queue(queue, num_tensors):
    for i in range(num_tensors):
        tensor = torch.randn((10, 10))
        queue.put(tensor)


def reader_queue(queue, num_tensors):
    for i in range(num_tensors):
        tensor = queue.get()
        _ = tensor.sum()


# Test functions
def t_shared_list():
    manager = multiprocessing.Manager()
    shared_list = manager.list()

    writer_process = multiprocessing.Process(target=writer_list, args=(shared_list, 10))
    reader_process = multiprocessing.Process(target=reader_list, args=(shared_list, 10))

    start_time = time.time()
    writer_process.start()
    reader_process.start()

    writer_process.join()
    reader_process.join()
    end_time = time.time()

    return end_time - start_time


def t_queue():
    queue = multiprocessing.Queue()

    writer_process = multiprocessing.Process(target=writer_queue, args=(queue, 10))
    reader_process = multiprocessing.Process(target=reader_queue, args=(queue, 10))

    start_time = time.time()
    writer_process.start()
    reader_process.start()

    writer_process.join()
    reader_process.join()
    end_time = time.time()

    return end_time - start_time


if __name__ == '__main__':
    setup_overhead = measure_process_setup_overhead()
    print(f"Process setup overhead: {setup_overhead:.4f} seconds")

    list_time = t_shared_list()
    queue_time = t_queue()

    adjusted_list_time = list_time - setup_overhead
    adjusted_queue_time = queue_time - setup_overhead

    print(f"Time taken using shared list (adjusted): {adjusted_list_time:.4f} seconds")
    print(f"Time taken using queue (adjusted): {adjusted_queue_time:.4f} seconds")