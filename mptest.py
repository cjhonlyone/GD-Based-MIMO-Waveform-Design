# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:52:11 2023

@author: F520-CJH
"""
import os
import sys
import csv
import time
import platform
import psutil
import cpuinfo
import getpass
from time import perf_counter
from multiprocessing import Queue, Process, Event, Lock
import queue as q

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)

from datetime import datetime

from train import *

Counter = 0

class Worker(Process):
    
    def __init__(self, task_queue: Queue, 
                 stop_event: Event, **kwargs):
        super().__init__(**kwargs)
        self.task_queue = task_queue
        self.stop_event = stop_event
        self.name = kwargs['name']
    
    def run(self):
        global Counter
        logger.info(f'Intiailizing Worker - {self.name} ProcessID - {os.getpid()}')

        # While loop will run until both (Empty queue and stop event trigger by manager) is False      
        while not self.task_queue.empty() or not self.stop_event.is_set():

            try:
                job = self.task_queue.get() # This will get the jobs entered by manager into the queue
                
                # Perform operation on jobs
                logger.info(f"Starting process for JobID - {job['Layer']} on worker - {self.name}")
                ####################################################################
                #                  Processing tasks starts 
                ####################################################################
                start_job = perf_counter()
                if (['gpu'] == []):
                    logger.info(f"JobID - {job['Layer']} on Worker - Failed")
                    
                else:
                    
                    jobdir="{Name}_{Algorithm}_{Struct}_{Layer}_{Loss}_{LR}_{IC}_{M}_{N}_{Epochs}".format(
                                                                                            Name = 'Job'+str(Counter),
                                                                                            Algorithm = job['Algorithm'],
                                                                                            Struct = job['Struct'].__name__,
                                                                                            Layer = job['Layer'],
                                                                                            Loss = job['Loss'].__name__,
                                                                                            LR = job['Learning Rate'].__name__,
                                                                                            IC = job['Initial Code'],
                                                                                            M = job['M'],
                                                                                            N = job['N'],
                                                                                            Epochs = job['Epochs'])
                    Counter = Counter + 1
                    logger.info(jobdir)
                    os.mkdir(jobdir)
                    os.chdir(jobdir)
                    # System info

                    re = job["Opt Func"](job)
                    job["Eva Func"](job, re)
                    os.chdir("../")
                    
                    end_job = perf_counter()
        
                    with open(job["Log Name"],"a+") as f:
                        f.write(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+",")
                        f.write(f"{end_job - start_job},")
                        f.write("{M},{N},{t},{Algorithm},{Struct},{Layer},{Loss},{LR},{Epochs},{IC},{PC},".format(
                                                                                                M = job['M'],
                                                                                                N = job['N'],
                                                                                                t = job['t'],
                                                                                                Algorithm = job['Algorithm'],
                                                                                                Struct = job['Struct'].__name__,
                                                                                                Layer = job['Layer'],
                                                                                                Loss = job['Loss'].__name__,
                                                                                                LR = job['Learning Rate'].__name__,
                                                                                                Epochs = job['Epochs'],
                                                                                                IC = job['Initial Code'],
                                                                                                PC = job["Params Count"]))
                        f.write('{: .2f},{: .2f},{: .2f},{: .2f},{: .2f},{: .2f},'.format(job["Result"]["PSL"],job["Result"]["ISL"],job["Result"]["APSL"],job["Result"]["CPSL"],job["Result"]["AISL"],job["Result"]["CISL"]))
                        f.write('\n')
                        
                    logger.info(f"Time taken to execute JobID - {job['Layer']} on Worker - {self.name} is {end_job - start_job}")
                    logger.info(jobdir)
                    ####################################################################
                    #                  Processing tasks ends 
                    ####################################################################

            except q.Empty: pass
        logger.info(f"{self.name} - Process terminates")



class Manager:
    
    def __init__(self, n_workers: int = 1, 
                 max_tasks: int = 2000):
        self.stop_event = Event()
        self.task_queue = Queue(maxsize=max_tasks)

        n_workers = 1 if n_workers < 1 else n_workers
        logger.info(f"Starting {n_workers} workers in process mode")
        
        self.workers = [Worker(self.task_queue,
                            self.stop_event,
                            name=f"Worker{i}") 
                        for i in range(n_workers)]
        for worker in self.workers: worker.start()

    def add_jobs(self, joblist):
        '''Adds the jobs to the queue'''
        ####################################################################
        #                  Task addition on queue starts 
        ####################################################################

        for i in range(len(joblist)):
            self.task_queue.put(joblist[i]) # Assigns the job in queue
            logger.info(f"{joblist[i]['Layer']}")

        ####################################################################
        #                 Task addition on queue ends 
        ####################################################################
        
    def terminate(self):
        '''Sets termiate event when called'''
        self.stop_event.set()

def LogSysInfo():
    timestamp = time.time()
    username = getpass.getuser() + '\n'
    system_info = f"System: {platform.system()}\nRelease: {platform.release()}\nVersion: {platform.version()}\n"
    cpu_info0 = f"Physical cores: {psutil.cpu_count(logical=False)}\nTotal cores: {psutil.cpu_count(logical=True)}\n"
    cpu_info1 = cpuinfo.get_cpu_info()['brand_raw'] + '\n'
    memory_info = "Memory: {:.2f} GB\n".format(psutil.virtual_memory().total / (1024**3))
    disk_info = ""
    for partition in psutil.disk_partitions():
        partition_usage = psutil.disk_usage(partition.mountpoint)
        disk_info += "Device: {}, Mountpoint: {}, Total Size: {:.2f} GB, Used: {:.2f} GB, Free: {:.2f} GB, Usage: {:.2f}%\n".format(
            partition.device, partition.mountpoint, partition_usage.total / (1024**3), partition_usage.used / (1024**3),
            partition_usage.free / (1024**3), partition_usage.percent)
    with open("system_info.txt", "w") as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))+'\n')
        f.write(username)
        f.write(system_info)
        f.write(cpu_info0)
        f.write(cpu_info1)
        f.write(memory_info)
        f.write(disk_info)

if __name__ == "__main__":
    if (len(sys.argv) == 1):
        print("You forget workerlist.csv.")
        print("Generate a csv example.")
        with open("example.csv", "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["M",  "N","t","Algorithm","Struct","Layer","Loss","Learning Rate","Epochs","Initial Code","Opt Func","Eva Func","Result","Log Name", "Params Count"])
            writer.writerow(["4","256","0.000153"," NN"," NNGD","1"," NNGDLOSS"," lrstepscheduler","20000"," b","TrainNN","CalSL","0","0","0"])
        sys.exit(1)
        
    workerfile = sys.argv[1]
    ext = os.path.splitext(workerfile)[1]
    if ext != '.csv':
        print('Error: file is not a CSV')
        sys.exit(1)

    with open(workerfile, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        joblist = [row for row in reader]

    for row in joblist:
        for key, value in row.items():
            row[key] = value.strip()

    for job in joblist:
        print(job)
        job["M"] = int(job["M"])
        job["N"] = int(job["N"])
        job["t"] = float(job["t"])
        job["Layer"] = int(job["Layer"])
        job["Epochs"] = int(job["Epochs"])

        job["Struct"] = globals()[job["Struct"]]
        job["Learning Rate"] = globals()[job["Learning Rate"]]
        job["Loss"] = globals()[job["Loss"]]
        job["Opt Func"] = globals()[job["Opt Func"]]
        job["Eva Func"] = globals()[job["Eva Func"]]
        job["Params Count"] = 0

    DirName = 'WD'+ datetime.now().strftime("%Y%m%d%H%M%S")
    os.mkdir(DirName)
    os.chdir(DirName)

    LogSysInfo()

    cpfilelist = [workerfile, "NNGD.py","NNGDLOSS.py","Waveform_Evaluate.py","lrscheduler.py","mptest.py","train.py","colectresult.py"]
    for ff in cpfilelist:
        os.popen('cp ' + '../'+ff+' ' + ff)

    LogName = 'DL'+ datetime.now().strftime("%Y%m%d%H%M%S") + '.csv'
    LogList = ["Log Time", "Duration", "M", "N", "t", "Algorithm", "Struct", "Layer", "Loss", "Learnning Rate", "Epochs", "Initial Code", "Params Count", "PSL", "ISL", "APSL", "CPSL", "AISL", "CISL", "Remark"]
    with open(LogName, "w", newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(LogList)
    for job in joblist:
        job["Log Name"] = LogName

    n_workers =  os.cpu_count() # Gets no of cores present on the machine
    n_workers = 1#int(n_workers/4);
    process = Manager(n_workers=n_workers) # Intialize Manager Object
    process.add_jobs(joblist) # Adds the jobs to process for workers in queue
    process.terminate() # Triggers termiante event after add_jobs task is finished