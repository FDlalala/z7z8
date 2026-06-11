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
# 平滑随机游走器
# ─────────────────────────────────────────────
class SmoothRandomWalker:
    def __init__(self, init_value, value_range,
                 drift_speed=0.002, jitter_scale=0.008,
                 target_update_interval=(1800, 7200)):  # 默认30min~2h
        self.value = init_value
        self.value_range = value_range
        self.drift_speed = drift_speed
        self.jitter_scale = jitter_scale
        self.target_update_interval = target_update_interval
        self.target = init_value
        self.next_target_time = time.time() + random.uniform(*target_update_interval)

    def step(self):
        now = time.time()
        if now >= self.next_target_time:
            lo, hi = self.value_range
            mid = (lo + hi) / 2
            spread = (hi - lo) / 2
            self.target = mid + random.gauss(0, spread * 0.5)
            self.target = max(lo, min(hi, self.target))
            self.next_target_time = now + random.uniform(*self.target_update_interval)
            log.debug(f"Walker 新目标: {self.target:.3f}, "
                      f"下次更新: {(self.next_target_time - now)/3600:.1f}h 后")

        self.value += (self.target - self.value) * self.drift_speed
        jitter = random.gauss(0, self.jitter_scale)
        lo, hi = self.value_range
        return max(lo, min(hi, self.value + jitter))


# ─────────────────────────────────────────────
# 单卡行为模拟器
# ─────────────────────────────────────────────
class CardBehaviorSimulator:

    PHASES = [
        {
            'name':        'training',
            'util_range':  (0.78, 0.92),
            'util_jitter': 0.015,
            'duration':    (1800, 7200),    # 0.5h ~ 2h
        },
        {
            'name':        'evaluation',
            'util_range':  (0.55, 0.72),
            'util_jitter': 0.012,
            'duration':    (600, 2400),     # 10min ~ 40min
        },
        {
            'name':        'data_loading',
            'util_range':  (0.08, 0.22),
            'util_jitter': 0.010,
            'duration':    (120, 600),      # 2min ~ 10min
        },
        {
            'name':        'gradient_sync',
            'util_range':  (0.25, 0.45),
            'util_jitter': 0.018,
            'duration':    (60, 300),       # 1min ~ 5min
        },
        {
            'name':        'checkpoint',
            'util_range':  (0.03, 0.10),
            'util_jitter': 0.005,
            'duration':    (120, 480),      # 2min ~ 8min
        },
        {
            'name':        'warmup',
            'util_range':  (0.65, 0.82),
            'util_jitter': 0.020,
            'duration':    (900, 3600),     # 15min ~ 1h
        },
        {
            'name':        'pipeline_bubble',
            'util_range':  (0.10, 0.28),
            'util_jitter': 0.012,
            'duration':    (60, 300),       # 1min ~ 5min
        },
    ]

    PHASE_WEIGHTS = [0.45, 0.12, 0.12, 0.10, 0.04, 0.10, 0.07]

    def __init__(self, gpu_id, mem_range_base=(0.75, 0.88)):
        self.gpu_id = gpu_id

        offset = random.uniform(-0.04, 0.04)
        self.mem_range = (
            max(0.60, mem_range_base[0] + offset),
            min(0.93, mem_range_base[1] + offset),
        )

        self.current_phase = None
        self.phase_end_time = time.time() - random.uniform(0, 600)

        init_util = random.uniform(0.6, 0.9)
        self.util_walker = SmoothRandomWalker(
            init_value=init_util,
            value_range=(0.03, 0.98),
            drift_speed=0.003,
            jitter_scale=0.008,
            target_update_interval=(1800, 7200),
        )

        init_mem = random.uniform(*self.mem_range)
        self.mem_walker = SmoothRandomWalker(
            init_value=init_mem,
            value_range=self.mem_range,
            drift_speed=0.001,
            jitter_scale=0.003,
            target_update_interval=(3600, 14400),  # 1h ~ 4h
        )

        self._pick_phase()
        log.info(f"[GPU {gpu_id}] 显存范围: "
                 f"{self.mem_range[0]*100:.1f}% ~ {self.mem_range[1]*100:.1f}%")

    def _pick_phase(self):
        self.current_phase = random.choices(
            self.PHASES, weights=self.PHASE_WEIGHTS, k=1
        )[0]
        duration = random.uniform(*self.current_phase['duration'])
        duration = max(60, duration)
        self.phase_end_time = time.time() + duration

        phase_util_target = random.uniform(*self.current_phase['util_range'])
        self.util_walker.target = phase_util_target
        self.util_walker.jitter_scale = self.current_phase['util_jitter']

        log.info(f"[GPU {self.gpu_id}] 阶段切换: {self.current_phase['name']}, "
                 f"目标利用率: {phase_util_target*100:.1f}%, "
                 f"持续: {duration/3600:.2f}h")

    def get_util(self):
        if time.time() > self.phase_end_time:
            self._pick_phase()
        return self.util_walker.step()

    def get_mem_ratio(self):
        return self.mem_walker.step()

    def get_matrix_size(self):
        sizes   = [512, 768, 1024, 1536, 2048, 2560, 3072]
        weights = [0.05, 0.10, 0.30, 0.25, 0.20, 0.07, 0.03]
        return random.choices(sizes, weights=weights, k=1)[0]


# ─────────────────────────────────────────────
# 单卡占位工作器
# ─────────────────────────────────────────────
class SingleCardWorker:

    # 每个控制周期长度（秒）
    # 在这个周期内按目标利用率分配计算时间和休眠时间
    # 周期不能太短，否则 GPU 来不及响应
    TICK = 0.5

    def __init__(self, gpu_id, mem_range_base):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.simulator = CardBehaviorSimulator(gpu_id, mem_range_base)
        self.running = False
        self.mem_tensor = None
        self.threads = []

    def start(self):
        self.running = True
        self._alloc_memory(self.simulator.get_mem_ratio())

        t_compute = threading.Thread(
            target=self._compute_loop,
            name=f'GPU{self.gpu_id}-Compute',
            daemon=True
        )
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
            log.error(f"[GPU {self.gpu_id}] 显存分配失败")
        except Exception as e:
            log.error(f"[GPU {self.gpu_id}] 显存分配异常: {e}")

    def _release_memory(self):
        self.mem_tensor = None
        torch.cuda.empty_cache()

    def _mem_adjust_loop(self):
        last_ratio = self.simulator.get_mem_ratio()
        while self.running:
            # 每 5~15 分钟检查一次
            wait = random.uniform(300, 900)
            for _ in range(int(wait)):
                if not self.running:
                    return
                time.sleep(1)

            if not self.running:
                return

            new_ratio = self.simulator.get_mem_ratio()
            if abs(new_ratio - last_ratio) > 0.03:
                log.info(f"[GPU {self.gpu_id}] 显存调整: "
                         f"{last_ratio*100:.1f}% -> {new_ratio*100:.1f}%")
                self._release_memory()
                time.sleep(random.uniform(0.5, 2.0))
                self._alloc_memory(new_ratio)
                last_ratio = new_ratio

    # ──────────────────────────────────────────
    # 计算负载
    # ──────────────────────────────────────────
    def _make_compute_tensors(self, sz):
        """创建用于计算的 tensor，确保在 GPU 上"""
        a = torch.randn(sz, sz, device=self.device)
        b = torch.randn(sz, sz, device=self.device)
        # 预热，确保 CUDA context 已经建立
        _ = torch.mm(a, b)
        torch.cuda.synchronize(self.gpu_id)
        return a, b

    def _compute_loop(self):
        """
        固定 TICK 周期的占空比控制：
        每个 TICK 内：
          compute_time = TICK * target_util  → 持续做矩阵运算
          sleep_time   = TICK * (1-target_util) → 休眠
        关键：计算阶段用 busy loop 持续提交算子，
              不能有任何 sleep，否则 GPU 会空转
        """
        sz = self.simulator.get_matrix_size()
        a, b = self._make_compute_tensors(sz)
        last_resize = time.time()
        iter_count = 0

        # 先做一次同步确认 GPU 正常
        torch.cuda.synchronize(self.gpu_id)
        log.info(f"[GPU {self.gpu_id}] 计算循环启动，矩阵: {sz}x{sz}")

        while self.running:
            try:
                target_util = self.simulator.get_util()
                compute_time = self.TICK * target_util
                sleep_time   = self.TICK * (1.0 - target_util)

                # ── 计算阶段：busy loop，持续提交算子 ──
                t_end = time.time() + compute_time
                while time.time() < t_end:
                    # 连续提交多个算子，保证 GPU 持续有活干
                    _ = torch.mm(a, b)
                    _ = torch.mm(b, a)
                    _ = torch.nn.functional.relu(a)
                    _ = (a * b).sum()
                    # 不加 synchronize，让算子在 GPU 上异步堆积执行

                # 计算阶段结束时同步一次，确保 GPU 真正执行完
                torch.cuda.synchronize(self.gpu_id)

                # ── 休眠阶段 ──
                if sleep_time > 1e-3:
                    time.sleep(sleep_time)

                iter_count += 1

                # 定期调整矩阵大小（模拟 batch size 变化）
                if time.time() - last_resize > random.uniform(1800, 5400):
                    sz = self.simulator.get_matrix_size()
                    a, b = self._make_compute_tensors(sz)
                    last_resize = time.time()
                    log.info(f"[GPU {self.gpu_id}] 矩阵调整为 {sz}x{sz}")

                # 小概率触发 IO 等待低谷（持续一段时间，不是尖峰）
                if random.random() < 0.0005:
                    io_wait = random.uniform(60, 300)  # 1~5 分钟的低谷
                    log.info(f"[GPU {self.gpu_id}] IO等待低谷 {io_wait:.0f}s")
                    self.simulator.util_walker.target = random.uniform(0.05, 0.15)
                    # 分段等待，方便响应停止信号
                    for _ in range(int(io_wait)):
                        if not self.running:
                            return
                        time.sleep(1)
                    # 恢复
                    phase_util = random.uniform(
                        *self.simulator.current_phase['util_range']
                    )
                    self.simulator.util_walker.target = phase_util
                    log.info(f"[GPU {self.gpu_id}] IO等待结束，恢复利用率")

            except Exception as e:
                log.warning(f"[GPU {self.gpu_id}] 计算异常: {e}", exc_info=True)
                time.sleep(2)


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

    def _query_gpu_processes(self):
        result_map = {gid: [] for gid in self.config['gpu_ids']}
        try:
            out = subprocess.run(
                ['nvidia-smi',
                 '--query-compute-apps=gpu_uuid,pid,used_memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            ).stdout.strip()

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
                    uuid_to_id[parts[1].strip()] = int(parts[0].strip())

            for line in out.split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) < 3:
                    continue
                gpu_id = uuid_to_id.get(parts[0].strip())
                pid    = int(parts[1].strip())
                mem_mb = int(parts[2].strip())
                if gpu_id in result_map and pid != os.getpid():
                    result_map[gpu_id].append({'pid': pid, 'memory_mb': mem_mb})
        except Exception as e:
            log.warning(f"nvidia-smi 查询失败: {e}")
        return result_map

    def _get_busy_gpus(self):
        threshold = self.config.get('min_real_memory_mb', 500)
        busy = set()
        for gpu_id, procs in self._query_gpu_processes().items():
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
        threads = [
            threading.Thread(target=self._stop_worker, args=(gid,))
            for gid in gpu_ids
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    def run(self):
        log.info("=" * 60)
        log.info(f"GPU守护进程启动，管理 {len(self.config['gpu_ids'])} 张卡")
        log.info(f"GPU列表: {self.config['gpu_ids']}")
        log.info("=" * 60)

        check_range = self.config.get('check_interval_range', (4, 8))

        # 错开各卡初始化，每张卡间隔 1~3s
        for i, gpu_id in enumerate(self.config['gpu_ids']):
            delay = i * random.uniform(1, 3)
            log.info(f"[GPU {gpu_id}] 将在 {delay:.1f}s 后初始化")
            threading.Timer(delay, self._start_worker, args=(gpu_id,)).start()

        time.sleep(len(self.config['gpu_ids']) * 3 + 5)

        while self.running:
            try:
                busy_gpus     = self._get_busy_gpus()
                all_gpus      = set(self.config['gpu_ids'])
                occupied_gpus = set(self.workers.keys())

                for gpu_id in busy_gpus & occupied_gpus:
                    log.info(f"[GPU {gpu_id}] 让出给真实任务")
                    threading.Thread(
                        target=self._stop_worker,
                        args=(gpu_id,), daemon=True
                    ).start()

                idle_gpus = (all_gpus - busy_gpus) - occupied_gpus
                for gpu_id in idle_gpus:
                    delay = random.uniform(2, 8)
                    log.info(f"[GPU {gpu_id}] 空闲，{delay:.1f}s 后开始占位")
                    threading.Timer(
                        delay, self._start_worker, args=(gpu_id,)
                    ).start()

                time.sleep(random.uniform(*check_range))

            except Exception as e:
                log.error(f"主循环异常: {e}", exc_info=True)
                time.sleep(5)


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--mem-min',      type=float, default=0.75)
    parser.add_argument('--mem-max',      type=float, default=0.88)
    parser.add_argument('--threshold-mb', type=int,   default=500)
    args = parser.parse_args()

    config = {
        'gpu_ids':              args.gpus,
        'mem_range':            (args.mem_min, args.mem_max),
        'check_interval_range': (4, 8),
        'min_real_memory_mb':   args.threshold_mb,
    }

    GPUDaemon(config).run()
