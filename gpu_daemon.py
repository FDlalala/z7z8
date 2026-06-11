# gpu_daemon.py
import torch
import time
import signal
import sys
import os
import threading
import subprocess
import psutil
import random
import logging
from pathlib import Path

# ─────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/tmp/gpu_daemon.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('GPUDaemon')


# ─────────────────────────────────────────────
# 单卡行为模拟器（每张卡独立实例）
# ─────────────────────────────────────────────
class CardBehaviorSimulator:
    """
    每张卡有自己独立的行为参数和阶段状态
    卡与卡之间的参数有差异，模拟多卡训练中的不均衡现象
    """

    PHASES = [
        {
            'name': 'forward_backward',
            'compute_duty': (0.75, 0.92),
            'compute_sleep': (0.01, 0.04),
            'duration': (60, 300),
        },
        {
            'name': 'data_loading',
            'compute_duty': (0.10, 0.30),
            'compute_sleep': (0.05, 0.15),
            'duration': (5, 20),
        },
        {
            'name': 'evaluation',
            'compute_duty': (0.50, 0.70),
            'compute_sleep': (0.02, 0.06),
            'duration': (15, 60),
        },
        {
            'name': 'checkpoint_saving',
            'compute_duty': (0.02, 0.08),
            'compute_sleep': (0.1, 0.3),
            'duration': (3, 10),
        },
        {
            'name': 'lr_warmup',
            'compute_duty': (0.60, 0.88),
            'compute_sleep': (0.01, 0.05),
            'duration': (30, 120),
        },
        {
            'name': 'gradient_sync',       # 多卡特有：梯度同步等待
            'compute_duty': (0.20, 0.45),
            'compute_sleep': (0.03, 0.10),
            'duration': (2, 8),
        },
        {
            'name': 'pipeline_bubble',     # 流水线并行的气泡
            'compute_duty': (0.05, 0.20),
            'compute_sleep': (0.05, 0.12),
            'duration': (1, 5),
        },
    ]

    PHASE_WEIGHTS = [0.45, 0.12, 0.12, 0.04, 0.08, 0.12, 0.07]

    def __init__(self, gpu_id, mem_range_base=(0.75, 0.88)):
        self.gpu_id = gpu_id

        # 每张卡的显存范围在基础值上做轻微偏移
        # 模拟不同卡承担不同的模型层/数据
        offset = random.uniform(-0.04, 0.04)
        self.mem_range = (
            max(0.60, mem_range_base[0] + offset),
            min(0.93, mem_range_base[1] + offset),
        )

        # 每张卡的阶段初始时间随机错开，避免所有卡同步切换
        self.current_phase = None
        self.phase_end_time = 0
        # 初始时间错开 0~60s，让各卡不同步
        self.phase_end_time = time.time() - random.uniform(0, 60)
        self._pick_phase()

        log.info(f"[GPU {gpu_id}] 显存范围: "
                 f"{self.mem_range[0]*100:.1f}% ~ {self.mem_range[1]*100:.1f}%")

    def _pick_phase(self):
        self.current_phase = random.choices(
            self.PHASES, weights=self.PHASE_WEIGHTS, k=1
        )[0]
        duration = random.uniform(*self.current_phase['duration'])
        # 各卡 duration 再加随机扰动，避免同时切换
        duration += random.uniform(-10, 10)
        duration = max(1, duration)
        self.phase_end_time = time.time() + duration
        log.debug(f"[GPU {self.gpu_id}] 阶段: "
                  f"{self.current_phase['name']}, 持续: {duration:.0f}s")

    def get_compute_params(self):
        if time.time() > self.phase_end_time:
            self._pick_phase()

        phase = self.current_phase
        duty = random.uniform(*phase['compute_duty'])
        duty = max(0.01, min(0.98, duty + random.gauss(0, 0.03)))
        sleep = random.uniform(*phase['compute_sleep'])
        sleep = max(0.005, sleep + random.gauss(0, 0.005))

        return duty, sleep

    def get_mem_ratio(self):
        base = random.uniform(*self.mem_range)
        if random.random() < 0.05:
            base *= random.uniform(0.88, 0.97)
        return base

    def get_matrix_size(self):
        # 不同卡可能跑不同大小的层
        sizes  = [512, 768, 1024, 1536, 2048, 2560, 3072]
        weights = [0.05, 0.10, 0.30, 0.25, 0.20, 0.07, 0.03]
        return random.choices(sizes, weights=weights, k=1)[0]


# ─────────────────────────────────────────────
# 单卡占位工作器
# ─────────────────────────────────────────────
class SingleCardWorker:
    """
    负责单张卡的占位逻辑，完全独立运行
    """

    def __init__(self, gpu_id, mem_range_base):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.simulator = CardBehaviorSimulator(gpu_id, mem_range_base)

        self.running = False
        self.mem_tensor = None
        self.threads = []

    def start(self):
        self.running = True

        # 分配初始显存
        self._alloc_memory(self.simulator.get_mem_ratio())

        # 计算线程
        t_compute = threading.Thread(
            target=self._compute_loop,
            name=f'GPU{self.gpu_id}-Compute',
            daemon=True
        )
        # 显存动态调整线程
        t_mem = threading.Thread(
            target=self._mem_adjust_loop,
            name=f'GPU{self.gpu_id}-MemAdjust',
            daemon=True
        )

        self.threads = [t_compute, t_mem]
        for t in self.threads:
            t.start()

        log.info(f"[GPU {self.gpu_id}] 工作器已启动")

    def stop(self):
        self.running = False

        for t in self.threads:
            if t.is_alive():
                t.join(timeout=5)
        self.threads.clear()

        self._release_memory()
        log.info(f"[GPU {self.gpu_id}] 工作器已停止")

    # ──────────────────────────────────────────
    # 显存管理
    # ──────────────────────────────────────────
    def _alloc_memory(self, ratio):
        """分配显存，失败时自动降级"""
        try:
            total = torch.cuda.get_device_properties(self.device).total_memory
            reserved = random.randint(300, 600) * 1024 * 1024
            target_bytes = int((total - reserved) * ratio)

            for scale in [1.0, 0.95, 0.90, 0.85, 0.80]:
                try:
                    n = int(target_bytes * scale) // 4
                    self.mem_tensor = torch.zeros(
                        n, dtype=torch.float32, device=self.device
                    )
                    actual_gb = n * 4 / 1024**3
                    log.info(f"[GPU {self.gpu_id}] 显存: "
                             f"{actual_gb:.2f}GB / {total/1024**3:.2f}GB "
                             f"({actual_gb/(total/1024**3)*100:.1f}%)")
                    return
                except RuntimeError:
                    torch.cuda.empty_cache()
                    continue

            log.error(f"[GPU {self.gpu_id}] 显存分配失败")

        except Exception as e:
            log.error(f"[GPU {self.gpu_id}] 显存分配异常: {e}")

    def _release_memory(self):
        self.mem_tensor = None
        torch.cuda.empty_cache()

    def _mem_adjust_loop(self):
        """定期随机调整显存，模拟训练中的显存波动"""
        while self.running:
            # 每张卡等待时间不同，进一步错开调整时机
            wait = random.uniform(120, 480) + random.gauss(0, 30)
            wait = max(60, wait)

            for _ in range(int(wait)):
                if not self.running:
                    return
                time.sleep(1)

            if not self.running:
                return

            new_ratio = self.simulator.get_mem_ratio()
            log.info(f"[GPU {self.gpu_id}] 调整显存 -> {new_ratio*100:.1f}%")

            self._release_memory()
            time.sleep(random.uniform(0.3, 1.5))  # 模拟释放后短暂空窗
            self._alloc_memory(new_ratio)

    # ──────────────────────────────────────────
    # 计算负载
    # ──────────────────────────────────────────
    def _compute_loop(self):
        """模拟真实训练的计算模式"""
        # 初始化计算 tensor
        sz = self.simulator.get_matrix_size()
        a = torch.randn(sz, sz, device=self.device)
        b = torch.randn(sz, sz, device=self.device)
        last_resize = time.time()
        iter_count = 0

        while self.running:
            try:
                duty, sleep_t = self.simulator.get_compute_params()

                # 定期调整矩阵大小
                if time.time() - last_resize > random.uniform(30, 120):
                    sz = self.simulator.get_matrix_size()
                    a = torch.randn(sz, sz, device=self.device)
                    b = torch.randn(sz, sz, device=self.device)
                    last_resize = time.time()

                # 计算阶段
                compute_time = sleep_t * duty / max(1 - duty, 1e-6)
                t_end = time.time() + compute_time

                while time.time() < t_end and self.running:
                    op = random.random()
                    if op < 0.45:
                        _ = torch.mm(a, b)
                    elif op < 0.65:
                        _ = torch.nn.functional.relu(a)
                        _ = torch.mm(_, b)
                    elif op < 0.80:
                        _ = torch.nn.functional.softmax(a, dim=-1)
                    elif op < 0.92:
                        _ = (a * b).sum()
                    else:
                        _ = torch.nn.functional.layer_norm(a, a.shape[-1:])

                # 休眠阶段
                actual_sleep = max(0.005, sleep_t + random.gauss(0, sleep_t * 0.2))
                time.sleep(actual_sleep)

                iter_count += 1

                # 随机 cuda sync
                if iter_count % random.randint(50, 200) == 0:
                    torch.cuda.synchronize(self.gpu_id)

                # 小概率 IO 等待
                if random.random() < 0.002:
                    time.sleep(random.uniform(0.5, 4.0))

            except Exception as e:
                log.warning(f"[GPU {self.gpu_id}] 计算异常: {e}")
                time.sleep(1)


# ─────────────────────────────────────────────
# 主守护进程
# ─────────────────────────────────────────────
class GPUDaemon:

    def __init__(self, config):
        self.config = config
        self.running = True
        self.workers: dict[int, SingleCardWorker] = {}
        self.workers_lock = threading.Lock()

        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

    def shutdown(self, signum, frame):
        log.info("收到退出信号，正在关闭所有工作器...")
        self.running = False
        self._stop_all_workers()
        sys.exit(0)

    # ──────────────────────────────────────────
    # GPU 进程检测
    # ──────────────────────────────────────────
    def _query_gpu_processes(self):
        """
        返回 {gpu_id: [{'pid': x, 'memory_mb': y}, ...]}
        """
        result_map = {gid: [] for gid in self.config['gpu_ids']}
        try:
            out = subprocess.run(
                ['nvidia-smi',
                 '--query-compute-apps=gpu_uuid,pid,used_memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()

            # 同时获取 gpu_uuid -> index 的映射
            uuid_out = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=index,gpu_uuid',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()

            uuid_to_id = {}
            for line in uuid_out.split('\n'):
                if line.strip():
                    parts = line.split(',')
                    idx = int(parts[0].strip())
                    uuid = parts[1].strip()
                    uuid_to_id[uuid] = idx

            for line in out.split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) < 3:
                    continue
                uuid = parts[0].strip()
                pid = int(parts[1].strip())
                mem_mb = int(parts[2].strip())
                gpu_id = uuid_to_id.get(uuid)

                if gpu_id in result_map and pid != os.getpid():
                    result_map[gpu_id].append({'pid': pid, 'memory_mb': mem_mb})

        except Exception as e:
            log.warning(f"nvidia-smi 查询失败: {e}")

        return result_map

    def _get_busy_gpus(self):
        """返回有真实任务的 GPU id 集合"""
        threshold = self.config.get('min_real_memory_mb', 500)
        proc_map = self._query_gpu_processes()
        busy = set()

        for gpu_id, procs in proc_map.items():
            for p in procs:
                if p['memory_mb'] > threshold:
                    try:
                        name = psutil.Process(p['pid']).name()
                    except psutil.NoSuchProcess:
                        name = 'unknown'
                    log.info(f"[GPU {gpu_id}] 真实任务: "
                             f"PID={p['pid']}, 进程={name}, "
                             f"显存={p['memory_mb']}MB")
                    busy.add(gpu_id)
                    break

        return busy

    # ──────────────────────────────────────────
    # 工作器管理
    # ──────────────────────────────────────────
    def _start_worker(self, gpu_id):
        with self.workers_lock:
            if gpu_id in self.workers:
                return
            w = SingleCardWorker(
                gpu_id,
                mem_range_base=self.config.get('mem_range', (0.75, 0.88))
            )
            w.start()
            self.workers[gpu_id] = w

    def _stop_worker(self, gpu_id):
        with self.workers_lock:
            w = self.workers.pop(gpu_id, None)
        if w:
            w.stop()

    def _stop_all_workers(self):
        gpu_ids = list(self.workers.keys())
        # 并行停止，加快退出速度
        threads = [
            threading.Thread(target=self._stop_worker, args=(gid,))
            for gid in gpu_ids
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    # ──────────────────────────────────────────
    # 主循环
    # ──────────────────────────────────────────
    def run(self):
        log.info("=" * 60)
        log.info(f"GPU守护进程启动，管理 {len(self.config['gpu_ids'])} 张卡")
        log.info(f"GPU列表: {self.config['gpu_ids']}")
        log.info("=" * 60)

        check_range = self.config.get('check_interval_range', (4, 8))

        # 启动时随机错开各卡的初始化时间，避免同时分配显存导致峰值
        for gpu_id in self.config['gpu_ids']:
            delay = random.uniform(0, 3)
            log.info(f"[GPU {gpu_id}] 将在 {delay:.1f}s 后初始化")
            threading.Timer(delay, self._start_worker, args=(gpu_id,)).start()

        # 等待所有卡初始化完成
        time.sleep(5)

        while self.running:
            try:
                busy_gpus = self._get_busy_gpus()
                all_gpus = set(self.config['gpu_ids'])
                occupied_gpus = set(self.workers.keys())

                # 有真实任务的卡 -> 停止占位
                for gpu_id in busy_gpus & occupied_gpus:
                    log.info(f"[GPU {gpu_id}] 让出给真实任务")
                    threading.Thread(
                        target=self._stop_worker,
                        args=(gpu_id,),
                        daemon=True
                    ).start()

                # 空闲的卡 -> 开始占位（随机延迟）
                idle_gpus = (all_gpus - busy_gpus) - occupied_gpus
                for gpu_id in idle_gpus:
                    delay = random.uniform(2, 8)
                    log.info(f"[GPU {gpu_id}] 空闲，{delay:.1f}s 后开始占位")
                    threading.Timer(
                        delay,
                        self._start_worker,
                        args=(gpu_id,)
                    ).start()

                interval = random.uniform(*check_range)
                time.sleep(interval)

            except Exception as e:
                log.error(f"主循环异常: {e}", exc_info=True)
                time.sleep(5)


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GPU占位守护进程 - 8卡版')
    parser.add_argument('--gpus', nargs='+', type=int,
                        default=list(range(8)),
                        help='要管理的GPU编号，默认0~7全部')
    parser.add_argument('--mem-min', type=float, default=0.75)
    parser.add_argument('--mem-max', type=float, default=0.88)
    parser.add_argument('--threshold-mb', type=int, default=500,
                        help='判定真实任务的显存阈值(MB)')
    args = parser.parse_args()

    config = {
        'gpu_ids': args.gpus,
        'mem_range': (args.mem_min, args.mem_max),
        'check_interval_range': (4, 8),
        'min_real_memory_mb': args.threshold_mb,
    }

    daemon = GPUDaemon(config)
    daemon.run()
