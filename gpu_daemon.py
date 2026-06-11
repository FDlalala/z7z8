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
import math
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
# 平滑随机游走器
# 用于生成"有惯性"的随机值，避免尖峰
# ─────────────────────────────────────────────
class SmoothRandomWalker:
    """
    模拟带惯性的随机游走：
    - current_value 缓慢向 target 靠近
    - target 定期随机更新
    - 在 current_value 附近叠加极小的高频抖动
    """

    def __init__(self, init_value, value_range, 
                 drift_speed=0.002, jitter_scale=0.008,
                 target_update_interval=(30, 120)):
        """
        init_value:              初始值
        value_range:             (min, max) 允许范围
        drift_speed:             每次 step 向 target 靠近的比例（惯性）
        jitter_scale:            叠加在当前值上的高频抖动幅度
        target_update_interval:  多少秒后随机更新 target
        """
        self.value = init_value
        self.value_range = value_range
        self.drift_speed = drift_speed
        self.jitter_scale = jitter_scale
        self.target_update_interval = target_update_interval

        self.target = init_value
        self.next_target_time = time.time() + random.uniform(*target_update_interval)

    def step(self):
        """推进一步，返回当前值"""
        now = time.time()

        # 到时间了就随机选一个新 target
        if now >= self.next_target_time:
            lo, hi = self.value_range
            # target 在范围内随机，但偏向中间，避免长期贴边
            mid = (lo + hi) / 2
            spread = (hi - lo) / 2
            self.target = mid + random.gauss(0, spread * 0.5)
            self.target = max(lo, min(hi, self.target))
            self.next_target_time = now + random.uniform(*self.target_update_interval)

        # 当前值向 target 缓慢靠近（指数平滑）
        self.value += (self.target - self.value) * self.drift_speed

        # 叠加极小的高频抖动
        jitter = random.gauss(0, self.jitter_scale)
        result = self.value + jitter

        # 限制在范围内
        lo, hi = self.value_range
        return max(lo, min(hi, result))


# ─────────────────────────────────────────────
# 单卡行为模拟器
# ─────────────────────────────────────────────
class CardBehaviorSimulator:
    """
    每张卡独立的行为状态机：
    - 有若干"稳态阶段"，每个阶段有自己的利用率基线
    - 阶段内利用率在基线附近平稳波动
    - 阶段切换时利用率平滑过渡，不突变
    """

    # 各阶段的利用率基线范围和持续时间
    PHASES = [
        {
            'name':        'training',       # 正常训练
            'util_range':  (0.78, 0.92),     # 该阶段的利用率基线范围
            'util_jitter': 0.015,            # 基线内的抖动幅度
            'duration':    (120, 400),
        },
        {
            'name':        'evaluation',     # 验证推理，利用率略低
            'util_range':  (0.55, 0.72),
            'util_jitter': 0.012,
            'duration':    (20, 80),
        },
        {
            'name':        'data_loading',   # 数据加载，GPU 等待
            'util_range':  (0.08, 0.22),
            'util_jitter': 0.010,
            'duration':    (5, 25),
        },
        {
            'name':        'gradient_sync',  # 多卡梯度同步
            'util_range':  (0.25, 0.45),
            'util_jitter': 0.018,
            'duration':    (3, 12),
        },
        {
            'name':        'checkpoint',     # 保存 checkpoint
            'util_range':  (0.03, 0.10),
            'util_jitter': 0.005,
            'duration':    (4, 15),
        },
        {
            'name':        'warmup',         # 学习率 warmup
            'util_range':  (0.65, 0.82),
            'util_jitter': 0.020,
            'duration':    (40, 150),
        },
        {
            'name':        'pipeline_bubble', # 流水线气泡
            'util_range':  (0.10, 0.28),
            'util_jitter': 0.012,
            'duration':    (2, 8),
        },
    ]

    PHASE_WEIGHTS = [0.45, 0.12, 0.12, 0.10, 0.04, 0.10, 0.07]

    def __init__(self, gpu_id, mem_range_base=(0.75, 0.88)):
        self.gpu_id = gpu_id

        # 每张卡显存范围略有偏移
        offset = random.uniform(-0.04, 0.04)
        self.mem_range = (
            max(0.60, mem_range_base[0] + offset),
            min(0.93, mem_range_base[1] + offset),
        )

        # 当前阶段
        self.current_phase = None
        self.phase_end_time = 0

        # 利用率平滑游走器（核心）
        # 初始值随机，后续跟随阶段基线
        init_util = random.uniform(0.6, 0.9)
        self.util_walker = SmoothRandomWalker(
            init_value=init_util,
            value_range=(0.03, 0.98),
            drift_speed=0.005,       # 较慢的漂移，保证平稳
            jitter_scale=0.008,      # 极小抖动
            target_update_interval=(20, 60),
        )

        # 显存平滑游走器
        init_mem = random.uniform(*self.mem_range)
        self.mem_walker = SmoothRandomWalker(
            init_value=init_mem,
            value_range=self.mem_range,
            drift_speed=0.001,       # 显存变化更慢
            jitter_scale=0.003,
            target_update_interval=(120, 480),
        )

        # 初始时间错开
        self.phase_end_time = time.time() - random.uniform(0, 60)
        self._pick_phase()

        log.info(f"[GPU {gpu_id}] 显存范围: "
                 f"{self.mem_range[0]*100:.1f}% ~ {self.mem_range[1]*100:.1f}%")

    def _pick_phase(self):
        self.current_phase = random.choices(
            self.PHASES, weights=self.PHASE_WEIGHTS, k=1
        )[0]
        duration = random.uniform(*self.current_phase['duration'])
        duration += random.gauss(0, duration * 0.1)
        duration = max(2, duration)
        self.phase_end_time = time.time() + duration

        # 切换阶段时，把 util_walker 的 target 更新到新阶段的基线范围内
        # drift_speed 保证不会瞬间跳变，而是平滑过渡
        phase_util_target = random.uniform(*self.current_phase['util_range'])
        self.util_walker.target = phase_util_target
        self.util_walker.jitter_scale = self.current_phase['util_jitter']

        log.debug(f"[GPU {self.gpu_id}] 阶段: {self.current_phase['name']}, "
                  f"目标利用率: {phase_util_target*100:.1f}%, "
                  f"持续: {duration:.0f}s")

    def get_util(self):
        """获取当前利用率（0~1），平滑无尖峰"""
        if time.time() > self.phase_end_time:
            self._pick_phase()
        return self.util_walker.step()

    def get_mem_ratio(self):
        """获取当前显存占比，缓慢漂移"""
        return self.mem_walker.step()

    def get_matrix_size(self):
        sizes   = [512, 768, 1024, 1536, 2048, 2560, 3072]
        weights = [0.05, 0.10, 0.30, 0.25, 0.20, 0.07, 0.03]
        return random.choices(sizes, weights=weights, k=1)[0]


# ─────────────────────────────────────────────
# 单卡占位工作器
# ─────────────────────────────────────────────
class SingleCardWorker:

    # 计算循环的基础 tick 间隔（秒）
    # 每个 tick 内根据目标利用率决定算多久、停多久
    TICK = 0.05

    def __init__(self, gpu_id, mem_range_base):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.simulator = CardBehaviorSimulator(gpu_id, mem_range_base)

        self.running = False
        self.mem_tensor = None
        self.threads = []

        # 当前实际利用率目标（由 simulator 驱动，平滑更新）
        self._target_util = 0.8

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
        """
        显存调整：跟随 mem_walker 缓慢漂移
        不再是定时大幅重分配，而是周期性微调
        真正的大幅重分配概率很低
        """
        last_ratio = self.simulator.get_mem_ratio()

        while self.running:
            time.sleep(random.uniform(30, 60))  # 每 30~60s 检查一次
            if not self.running:
                return

            new_ratio = self.simulator.get_mem_ratio()
            delta = abs(new_ratio - last_ratio)

            # 变化超过 3% 才真正重新分配，避免频繁操作
            if delta > 0.03:
                log.info(f"[GPU {self.gpu_id}] 显存调整: "
                         f"{last_ratio*100:.1f}% -> {new_ratio*100:.1f}%")
                self._release_memory()
                time.sleep(random.uniform(0.2, 1.0))
                self._alloc_memory(new_ratio)
                last_ratio = new_ratio

    # ──────────────────────────────────────────
    # 计算负载（核心改动）
    # ──────────────────────────────────────────
    def _compute_loop(self):
        """
        基于固定 TICK 的占空比控制：
        每个 TICK 内：
          - 计算时间 = TICK * target_util
          - 休眠时间 = TICK * (1 - target_util)
        target_util 由 SmoothRandomWalker 驱动，平滑变化
        → 利用率曲线表现为：有惯性的平稳段 + 缓慢漂移 + 极小抖动
        """
        sz = self.simulator.get_matrix_size()
        a = torch.randn(sz, sz, device=self.device)
        b = torch.randn(sz, sz, device=self.device)
        last_resize = time.time()
        iter_count = 0

        while self.running:
            try:
                # 从 simulator 获取平滑后的目标利用率
                target_util = self.simulator.get_util()

                compute_time = self.TICK * target_util
                sleep_time   = self.TICK * (1.0 - target_util)

                # ── 计算阶段 ──────────────────────
                t_end = time.time() + compute_time
                while time.time() < t_end:
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
                        _ = torch.nn.functional.layer_norm(
                            a, a.shape[-1:]
                        )

                # ── 休眠阶段 ──────────────────────
                if sleep_time > 0:
                    time.sleep(sleep_time)

                iter_count += 1

                # 定期 resize 矩阵
                if time.time() - last_resize > random.uniform(30, 120):
                    sz = self.simulator.get_matrix_size()
                    a = torch.randn(sz, sz, device=self.device)
                    b = torch.randn(sz, sz, device=self.device)
                    last_resize = time.time()

                # 随机 cuda sync（模拟真实训练的同步点）
                if iter_count % random.randint(80, 250) == 0:
                    torch.cuda.synchronize(self.gpu_id)

                # 小概率触发较长 IO 等待（模拟磁盘读取）
                # 注意：这里用 sleep 而不是停止计算，
                # 让利用率有一段明显的低谷，而不是尖峰
                if random.random() < 0.001:
                    io_wait = random.uniform(1.0, 5.0)
                    log.debug(f"[GPU {self.gpu_id}] IO等待 {io_wait:.1f}s")
                    # 在 IO 等待期间，把 target 临时压低
                    self.simulator.util_walker.target = random.uniform(0.05, 0.15)
                    time.sleep(io_wait)
                    # IO 结束后恢复到当前阶段的正常范围
                    phase_util = random.uniform(
                        *self.simulator.current_phase['util_range']
                    )
                    self.simulator.util_walker.target = phase_util

            except Exception as e:
                log.warning(f"[GPU {self.gpu_id}] 计算异常: {e}")
                time.sleep(1)


# ─────────────────────────────────────────────
# 主守护进程（与之前相同，无改动）
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
                    result_map[gpu_id].append(
                        {'pid': pid, 'memory_mb': mem_mb}
                    )
        except Exception as e:
            log.warning(f"nvidia-smi 查询失败: {e}")
        return result_map

    def _get_busy_gpus(self):
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

        # 错开各卡初始化时间
        for gpu_id in self.config['gpu_ids']:
            delay = random.uniform(0, 3)
            log.info(f"[GPU {gpu_id}] 将在 {delay:.1f}s 后初始化")
            threading.Timer(delay, self._start_worker, args=(gpu_id,)).start()

        time.sleep(5)

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

    daemon = GPUDaemon(config)
    daemon.run()
