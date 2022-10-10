import multiprocessing as mp


class ParallelComputingWrapper:

    def __init__(self, num_cores, debug=False):
        self.debug = debug
        self.num_cores = num_cores
        self.queues = [mp.Queue() for d in range(num_cores)]

    def wrapper_for_parallel_process(self, method, args_tuple, queue):

        try:
            result = method(*args_tuple)
            queue.put(result)
        except BaseException as e:
            print('error in the subprocess (base exception)')
            print(e)
            queue.put(e)
        except OSError as ee:
            print('error in the subprocess (os error)')
            print(ee)
            queue.put(ee)

    def process_method_parallel(self,
                                method,
                                args_tuples_map
    ):
        processes = []
        for currCore in range(self.num_cores):
            curr_args_tuple = args_tuples_map[currCore]
            curr_process = mp.Process(
                            target=self.wrapper_for_parallel_process,
                            args=(
                                method,
                                curr_args_tuple,
                                self.queues[currCore]
                            )
                        )
            processes.append(curr_process)

        # start processes
        for process in processes:
            process.start()

        results_map = {}
        for currCore in range(self.num_cores):
            result = self.queues[currCore].get()

            if isinstance(result, BaseException) or isinstance(result, OSError):  # it means that subprocess has an error
                print('a child processed has thrown an exception. raising the exception in the parent process to terminate the program')
                print('one of the child processes failed, so killing all child processes')

                print(str(result))

                # kill all subprocesses
                for process in processes:
                    if process.is_alive():
                        process.terminate()  # assuming that the child process do not have its own children (those granchildren would be orphaned with terminate() if any)
                print('killed all child processes')
                print(result)
                raise(result)
            else:
                results_map[currCore] = result

            if self.debug:
                print('got results from core ', currCore)

        # wait for processes to complete
        for process in processes:
            process.join()

        return results_map
