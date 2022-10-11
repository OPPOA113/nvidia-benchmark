import time
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed


def thread_func(data, pool_count, thread_count):
    # 线程函数
    time.sleep(3)
    print(f"{data}, {pool_count}, {thread_count}'\n'")
    return (data, data)


def pool_func(data_list, thread_num, pool_count):
    # 进程函数
    executor = ThreadPoolExecutor(max_workers=thread_num)  # 定义线程池，设置最大线程数量
    thread_list = []  # 储存线程
    thread_count = 0
    for dt in data_list:
        thrd = executor.submit(thread_func, dt, pool_count, thread_count)  # 将线程添加到线程池
        thread_list.append(thrd)  # 将当前线程存入列表，用于后面线程完成后，获取线程返回的结果
        thread_count += 1

    result_list = []  # 用于接收线程返回的结果
    # 因为线程开启后，默认就不管了，如果需要获取线程返回的结果，需要等待线程运行完成
    for task in as_completed(thread_list):  # 等待线程全部完成
        result_list.append(task.result())  # 获取线程的结果
    print(f"result===> pool_count:{pool_count}, result_list:{result_list}")

def demo_4process_10thread():
    thread_num = 10  # 线程最大数量
    pool_num = 4  # 进程数量

    pool = Pool(pool_num)  # 定义进程池
    batch_num = 10  # 一个进程处理的数据量
    pool_count = 0
    for batch_i in range(0, 40, batch_num):
        data_list = range(batch_i, batch_i+batch_num)
        pool.apply_async(pool_func, (data_list, thread_num, pool_count,))
        pool_count += 1
    pool.close()
    pool.join()

if __name__ == "__main__":
    demo_4process_10thread()