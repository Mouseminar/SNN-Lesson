"""
MNIST 项目的 SNN 输入编码器。

约定：
1. 输入为 batch-first 的静态图像特征 `[B, ...]`，取值范围应为 `[0, 1]`。
2. 编码输出统一为 `[B, T, F]`，可以直接送入 `tdLayer`。
3. 每个编码器自带 `decode(x)`，用于把网络输出 `[B, T, C]` 聚合成分类 logits。
4. 会扩展输入维度的编码器通过 `output_size(input_size)` 告诉模型第一层维度。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_time_steps(T: int) -> None:
    if T < 1:
        raise ValueError("T 必须大于等于 1")


def _check_unit_interval(x: torch.Tensor, name: str = "x") -> None:
    if not torch.all((x >= 0) & (x <= 1)):
        raise ValueError(f"输入 {name} 的范围必须是 [0, 1]")


def _flatten_feature(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("编码器输入必须包含 batch 维和至少 1 个特征维")
    return x.reshape(x.shape[0], -1)


def _check_sequence_input(x: torch.Tensor, name: str = "x") -> None:
    if x.ndim < 3:
        raise ValueError(f"输入 {name} 必须至少是 3 维，形状应为 [B, T, ...]")


class BaseEncoder(nn.Module):
    """编码器基类，默认把输出时间维取平均作为分类 logits。"""

    default_decode = "mean"

    def __init__(self, T: int, decode_mode: str | None = None):
        super().__init__()
        _check_time_steps(T)
        self.T = T
        self.decode_mode = decode_mode or self.default_decode

    def output_size(self, input_size: int) -> int:
        return input_size

    def input_firing_rate(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded.mean()

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        _check_sequence_input(x)
        if self.decode_mode == "mean":
            return x.mean(dim=1)
        if self.decode_mode == "sum":
            return x.sum(dim=1)
        if self.decode_mode == "last":
            return x[:, -1, ...]
        if self.decode_mode == "max":
            return x.max(dim=1).values
        if self.decode_mode == "first_spike":
            return self.first_spike_decode(x)
        if self.decode_mode == "time_weighted":
            return self.time_weighted_decode(x)
        raise ValueError(f"未知 decode 方式: {self.decode_mode}")

    def first_spike_decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        首次发放解码。

        对二值脉冲而言，某类越早第一次发放，得分越高。实现中使用
        exclusive cumulative product 构造 first-spike mask，保持对 surrogate
        spike 输出的可训练梯度。
        """

        _check_sequence_input(x)
        spike = x.clamp(0, 1)
        not_spike_before = torch.cumprod(1 - spike[:, :-1, ...], dim=1)
        not_spike_before = torch.cat(
            [torch.ones_like(spike[:, :1, ...]), not_spike_before],
            dim=1,
        )
        first_spike = spike * not_spike_before
        weights = torch.arange(self.T, 0, -1, device=x.device, dtype=x.dtype)
        weights = weights.reshape((1, self.T) + (1,) * (x.ndim - 2))
        return (first_spike * weights).sum(dim=1) / self.T

    def time_weighted_decode(self, x: torch.Tensor) -> torch.Tensor:
        """时间加权解码：早期时间步权重大，适合 TTFS/rank-order 类编码。"""

        _check_sequence_input(x)
        weights = torch.arange(self.T, 0, -1, device=x.device, dtype=x.dtype)
        weights = weights / weights.sum()
        weights = weights.reshape((1, self.T) + (1,) * (x.ndim - 2))
        return (x * weights).sum(dim=1)


class DirectEncoder(BaseEncoder):
    """直接编码：把连续输入复制到每个时间步，作为非脉冲基线。"""

    default_decode = "mean"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        return x.unsqueeze(1).repeat(1, self.T, 1)


class PoissonEncoder(BaseEncoder):
    """离散时间泊松近似：每个时间步按输入值作为概率独立采样。"""

    default_decode = "mean"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        random_numbers = torch.rand(
            (x.shape[0], self.T, x.shape[1]),
            device=x.device,
            dtype=x.dtype,
        )
        return (random_numbers < x.unsqueeze(1)).to(x.dtype)


class DeterministicRateEncoder(BaseEncoder):
    """确定性积分-发放频率编码，没有随机采样噪声。"""

    default_decode = "mean"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        voltage = torch.zeros_like(x)
        spikes = torch.zeros((x.shape[0], self.T, x.shape[1]), device=x.device, dtype=x.dtype)

        for t in range(self.T):
            voltage = voltage + x
            current_spike = (voltage >= 1.0).to(x.dtype)
            spikes[:, t, :] = current_spike
            voltage = voltage - current_spike

        return spikes


class LatencyEncoder(BaseEncoder):
    """延迟编码/TTFS：输入越大，唯一脉冲越早发放。"""

    default_decode = "first_spike"

    def __init__(self, T: int, mode: str = "linear", decode_mode: str | None = None):
        super().__init__(T, decode_mode=decode_mode)
        if mode not in ("linear", "log"):
            raise ValueError("mode 只能是 'linear' 或 'log'")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)

        if self.mode == "linear" or self.T == 1:
            spike_time = torch.round((self.T - 1) * (1 - x)).long()
        else:
            alpha = math.exp(self.T - 1) - 1
            spike_time = torch.round((self.T - 1) - torch.log(alpha * x + 1)).long()

        spikes = F.one_hot(spike_time, num_classes=self.T).to(x.dtype)
        spikes = spikes * (x > 0).unsqueeze(-1).to(x.dtype)
        return spikes.movedim(-1, 1)


class PhaseEncoder(BaseEncoder):
    """相位编码：输入决定每个参考周期中的发放相位。"""

    default_decode = "first_spike"

    def __init__(self, T: int, period: int, decode_mode: str | None = None):
        super().__init__(T, decode_mode=decode_mode)
        if period < 2:
            raise ValueError("period 必须大于等于 2")
        self.period = period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        phase = torch.round((1 - x) * (self.period - 1)).long()
        time = torch.arange(self.T, device=x.device).reshape(1, self.T, 1)
        spikes = ((time % self.period) == phase.unsqueeze(1)).to(x.dtype)
        return spikes * (x > 0).unsqueeze(1).to(x.dtype)


class BinaryTemporalEncoder(BaseEncoder):
    """
    二进制时间编码。

    原始版本只能表示 `[0, 1 - 2^-T]`。为了接收 MNIST 的 `[0, 1]`，
    这里先缩放到可表示区间，再按 1/2、1/4、... 展开。
    """

    default_decode = "sum"

    def time_weighted_decode(self, x: torch.Tensor) -> torch.Tensor:
        """二进制时间加权解码：第 0 步权重 1/2，第 1 步权重 1/4，依此类推。"""

        _check_sequence_input(x)
        exponents = torch.arange(1, self.T + 1, device=x.device, dtype=x.dtype)
        weights = 2.0 ** (-exponents)
        weights = weights / weights.sum()
        weights = weights.reshape((1, self.T) + (1,) * (x.ndim - 2))
        return (x * weights).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        max_value = 1 - 2 ** (-self.T)
        remaining = x * max_value
        spikes = torch.zeros((x.shape[0], self.T, x.shape[1]), device=x.device, dtype=x.dtype)

        weight = 0.5
        for t in range(self.T):
            spikes[:, t, :] = (remaining >= weight).to(x.dtype)
            remaining = remaining - weight * spikes[:, t, :]
            weight /= 2

        return spikes


WeightedPhaseEncoder = BinaryTemporalEncoder


class PopulationEncoder(BaseEncoder):
    """高斯群体响应，输出连续响应并在时间维复制。"""

    default_decode = "mean"

    def __init__(
        self,
        T: int,
        population_size: int = 8,
        std: float = 0.2,
        decode_mode: str | None = None,
    ):
        super().__init__(T, decode_mode=decode_mode)
        if population_size < 2 or std <= 0:
            raise ValueError("population_size >= 2 且 std > 0")
        self.population_size = population_size
        self.std = std
        self.register_buffer("centers", torch.linspace(0, 1, population_size))

    def output_size(self, input_size: int) -> int:
        return input_size * self.population_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        response = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.std) ** 2)
        response = response.reshape(x.shape[0], -1)
        return response.unsqueeze(1).repeat(1, self.T, 1)


class PopSpikeEncoderDeterministic(BaseEncoder):
    """高斯群体响应后，使用确定性积分-发放产生脉冲。"""

    default_decode = "mean"

    def __init__(
        self,
        T: int,
        pop_dim: int = 8,
        std: float = 0.2,
        decode_mode: str | None = None,
    ):
        super().__init__(T, decode_mode=decode_mode)
        if pop_dim < 2 or std <= 0:
            raise ValueError("pop_dim >= 2 且 std > 0")
        self.pop_dim = pop_dim
        self.std = std
        self.register_buffer("centers", torch.linspace(0, 1, pop_dim))

    def output_size(self, input_size: int) -> int:
        return input_size * self.pop_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        activation = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.std) ** 2)
        activation = activation.reshape(x.shape[0], -1)
        voltage = torch.zeros_like(activation)
        spikes = torch.zeros((x.shape[0], self.T, activation.shape[1]), device=x.device, dtype=x.dtype)

        for t in range(self.T):
            voltage = voltage + activation
            current_spike = (voltage >= 1.0).to(x.dtype)
            spikes[:, t, :] = current_spike
            voltage = voltage - current_spike

        return spikes


class PopSpikeEncoderRandom(BaseEncoder):
    """高斯群体响应后，将响应作为每个时间步的随机发放概率。"""

    default_decode = "mean"

    def __init__(
        self,
        T: int,
        pop_dim: int = 8,
        std: float = 0.2,
        decode_mode: str | None = None,
    ):
        super().__init__(T, decode_mode=decode_mode)
        if pop_dim < 2 or std <= 0:
            raise ValueError("pop_dim >= 2 且 std > 0")
        self.pop_dim = pop_dim
        self.std = std
        self.register_buffer("centers", torch.linspace(0, 1, pop_dim))

    def output_size(self, input_size: int) -> int:
        return input_size * self.pop_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        probability = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centers) / self.std) ** 2)
        probability = probability.reshape(x.shape[0], -1)
        random_numbers = torch.rand(
            (x.shape[0], self.T, probability.shape[1]),
            device=x.device,
            dtype=x.dtype,
        )
        return (random_numbers < probability.unsqueeze(1)).to(x.dtype)


class GaussianTuningEncoder(BaseEncoder):
    """高斯调谐曲线加延迟编码，每个输入维度扩展为 m 个调谐神经元。"""

    default_decode = "sum"

    def __init__(
        self,
        T: int,
        m: int = 8,
        beta: float = 1.5,
        decode_mode: str | None = None,
    ):
        super().__init__(T, decode_mode=decode_mode)
        if m <= 2 or beta <= 0:
            raise ValueError("m > 2 且 beta > 0")
        self.m = m
        indices = torch.arange(1, m + 1, dtype=torch.float32)
        self.register_buffer("centers", (2 * indices - 3) / (2 * (m - 2)))
        self.variance = (1 / (beta * (m - 2))) ** 2

    def output_size(self, input_size: int) -> int:
        return input_size * self.m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        response = torch.exp(-((x.unsqueeze(-1) - self.centers) ** 2) / (2 * self.variance))
        spike_time = torch.round((self.T - 1) * (1 - response)).long()
        spikes = F.one_hot(spike_time, num_classes=self.T).to(x.dtype)
        spikes = spikes.movedim(-1, 1)
        return spikes.reshape(x.shape[0], self.T, -1)


class RankOrderEncoder(BaseEncoder):
    """
    脉冲顺序编码：像素按强度排序后映射到 T 个时间桶。

    原始 rank-order 通常要求 `T >= 输入维度` 才能完整保留每个通道的严格
    顺序。MNIST 中 `T` 通常远小于 784，因此这里把 rank 分桶，保留整体
    强弱顺序，同时让所有非零像素都有一次发放机会。
    """

    default_decode = "sum"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        feature_size = x.shape[1]
        order = torch.argsort(x, dim=1, descending=True, stable=True)
        rank_time = (
            torch.arange(feature_size, device=x.device) * self.T // feature_size
        ).clamp(max=self.T - 1)
        spikes = torch.zeros((x.shape[0], self.T, feature_size), device=x.device, dtype=x.dtype)
        batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(order)
        time_indices = rank_time.unsqueeze(0).expand_as(order)
        spikes[batch_indices, time_indices, order] = (x.gather(1, order) > 0).to(x.dtype)
        return spikes


class ISIEncoder(BaseEncoder):
    """
    三脉冲 ISI Pattern Coding。

    每个有效输入固定产生 3 个脉冲：t1=0、t2 由输入 x 决定、t3=T-1。
    因此信息主要编码在两个脉冲间隔的模式中，而不是 firing rate 中。
    """

    default_decode = "time_weighted"

    def __init__(
        self,
        T: int,
        threshold: float = 0.0,
        min_interval: int = 1,
        decode_mode: str | None = None,
    ):
        super().__init__(T, decode_mode=decode_mode)
        if not 0 <= threshold < 1:
            raise ValueError("threshold 必须满足 0 <= threshold < 1")
        if min_interval < 1:
            raise ValueError("min_interval 必须大于等于 1")
        if T < 2 * min_interval + 2:
            raise ValueError(
                f"T 太小。当前 min_interval={min_interval} 时，"
                f"T 至少需要 {2 * min_interval + 2}"
            )

        self.threshold = threshold
        self.min_interval = min_interval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        batch_size, feature_size = x.shape

        active = x > self.threshold
        normalized_x = ((x - self.threshold) / (1.0 - self.threshold)).clamp(0.0, 1.0)

        min_isi1 = self.min_interval
        max_isi1 = self.T - 1 - self.min_interval
        isi1 = torch.round(max_isi1 - normalized_x * (max_isi1 - min_isi1)).long()

        spikes = torch.zeros(
            (batch_size, self.T, feature_size),
            device=x.device,
            dtype=x.dtype,
        )
        active_value = active.to(x.dtype)
        spikes[:, 0, :] = active_value
        spikes[:, self.T - 1, :] = active_value
        spikes.scatter_add_(
            dim=1,
            index=isi1.unsqueeze(1),
            src=active_value.unsqueeze(1),
        )
        return spikes.clamp_max(1.0)


class BurstEncoder(BaseEncoder):
    """突发编码：输入决定每个突发窗口开始处的连续脉冲数量。"""

    default_decode = "mean"

    def __init__(
        self,
        T: int,
        max_burst_size: int = 5,
        burst_interval: int = 10,
        decode_mode: str | None = None,
    ):
        super().__init__(T, decode_mode=decode_mode)
        if max_burst_size < 1 or burst_interval < 1:
            raise ValueError("max_burst_size 和 burst_interval 均需大于等于 1")
        self.max_burst_size = max_burst_size
        self.burst_interval = burst_interval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        time = torch.arange(self.T, device=x.device).reshape(1, self.T, 1)
        burst_size = torch.round(x * self.max_burst_size).long()
        offset = time % self.burst_interval
        return (offset < burst_size.unsqueeze(1)).to(x.dtype)


class BurstISIEncoder(BaseEncoder):
    """
    Burst + ISI 联合编码。

    输入 x 同时控制 burst 中的 spike 数量和 burst 内相邻 spike 的 ISI：
    x 越大，spike 数量越多且 ISI 越短。
    """

    default_decode = "mean"

    def __init__(
        self,
        T: int,
        max_burst_size: int = 5,
        burst_interval: int = 10,
        min_isi: int = 1,
        max_isi: int | None = None,
        threshold: float = 0.0,
        decode_mode: str | None = None,
    ):
        super().__init__(T, decode_mode=decode_mode)
        if max_burst_size < 2:
            raise ValueError("max_burst_size 至少需要为 2，否则无法利用 ISI 信息")
        if burst_interval < 1:
            raise ValueError("burst_interval 必须大于等于 1")
        if min_isi < 1:
            raise ValueError("min_isi 必须大于等于 1")
        if not 0 <= threshold < 1:
            raise ValueError("threshold 必须满足 0 <= threshold < 1")

        if max_isi is None:
            max_isi = burst_interval - 1
        if max_isi < min_isi:
            raise ValueError("max_isi 必须大于等于 min_isi")
        if max_isi >= burst_interval:
            raise ValueError("max_isi 必须小于 burst_interval，否则 spike 会跑到下一个 burst 中")

        minimum_required_window = 1 + (max_burst_size - 1) * min_isi
        if burst_interval < minimum_required_window:
            raise ValueError(
                f"burst_interval 太小。max_burst_size={max_burst_size}, "
                f"min_isi={min_isi} 时，burst_interval 至少需要 {minimum_required_window}"
            )

        self.max_burst_size = max_burst_size
        self.burst_interval = burst_interval
        self.min_isi = min_isi
        self.max_isi = max_isi
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        batch_size, feature_size = x.shape

        active = x > self.threshold
        normalized_x = ((x - self.threshold) / (1.0 - self.threshold)).clamp(0.0, 1.0)

        burst_size = torch.ceil(normalized_x * self.max_burst_size).long()
        burst_size = torch.where(active, burst_size, torch.zeros_like(burst_size))
        desired_isi = torch.round(
            self.max_isi - normalized_x * (self.max_isi - self.min_isi)
        ).long()

        num_gaps = (burst_size - 1).clamp_min(1)
        max_fit_isi = (self.burst_interval - 1) // num_gaps
        isi = torch.minimum(desired_isi, max_fit_isi).clamp(min=self.min_isi)

        spikes = torch.zeros(
            (batch_size, self.T, feature_size),
            device=x.device,
            dtype=x.dtype,
        )

        for burst_start in range(0, self.T, self.burst_interval):
            for k in range(self.max_burst_size):
                spike_time = burst_start + k * isi
                valid_spike = active & (k < burst_size) & (spike_time < self.T)
                safe_time = spike_time.clamp(max=self.T - 1)
                spikes.scatter_add_(
                    dim=1,
                    index=safe_time.unsqueeze(1),
                    src=valid_spike.to(x.dtype).unsqueeze(1),
                )

        return spikes.clamp_max(1.0)


class EventDeltaEncoder(BaseEncoder):
    """
    事件/差分编码的静态图像版本。

    对静态 MNIST 图像构造一条从 0 线性上升到 x 的短时间轨迹，并在估计值
    与轨迹差值超过阈值时发 ON 事件。输出包含 ON/OFF 两类通道，因此特征数
    扩展为 2 倍；静态输入通常只产生 ON 事件。
    """

    default_decode = "mean"

    def __init__(self, T: int, threshold: float = 0.1, decode_mode: str | None = None):
        super().__init__(T, decode_mode=decode_mode)
        if threshold <= 0:
            raise ValueError("threshold 必须大于 0")
        self.threshold = threshold

    def output_size(self, input_size: int) -> int:
        return input_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_feature(x)
        _check_unit_interval(x)
        estimate = torch.zeros_like(x)
        spikes = torch.zeros((x.shape[0], self.T, x.shape[1], 2), device=x.device, dtype=x.dtype)

        for t in range(self.T):
            value = x * ((t + 1) / self.T)
            error = value - estimate
            on_spike = (error >= self.threshold).to(x.dtype)
            off_spike = (error <= -self.threshold).to(x.dtype)
            spikes[:, t, :, 0] = on_spike
            spikes[:, t, :, 1] = off_spike
            estimate = estimate + self.threshold * (on_spike - off_spike)

        return spikes.reshape(x.shape[0], self.T, -1)


def build_encoder(
    encoder_type: str = "direct",
    T: int = 1,
    decode_mode: str | None = None,
    latency_mode: str = "linear",
    phase_period: int | None = None,
    population_size: int = 8,
    pop_dim: int = 8,
    population_std: float = 0.2,
    gaussian_m: int = 8,
    gaussian_beta: float = 1.5,
    min_interval: int = 1,
    isi_threshold: float = 0.0,
    max_burst_size: int = 5,
    burst_interval: int = 10,
    min_isi: int = 1,
    max_isi: int | None = None,
    burst_isi_threshold: float = 0.0,
    event_threshold: float = 0.1,
) -> BaseEncoder:
    """根据字符串名称构建编码器。"""

    encoder_type = encoder_type.lower()

    if encoder_type in ("direct", "repeat", "none", "identity", ""):
        return DirectEncoder(T=T, decode_mode=decode_mode)
    if encoder_type == "poisson":
        return PoissonEncoder(T=T, decode_mode=decode_mode)
    if encoder_type in ("rate", "deterministic_rate"):
        return DeterministicRateEncoder(T=T, decode_mode=decode_mode)
    if encoder_type in ("latency", "ttfs"):
        return LatencyEncoder(T=T, mode=latency_mode, decode_mode=decode_mode)
    if encoder_type == "phase":
        period = phase_period if phase_period is not None else max(2, T)
        return PhaseEncoder(T=T, period=period, decode_mode=decode_mode)
    if encoder_type in (
        "binary_temporal",
        "binarytemporal",
        "binary",
        "weighted_phase",
        "weightedphase",
    ):
        return BinaryTemporalEncoder(T=T, decode_mode=decode_mode)
    if encoder_type in ("population", "population_rate"):
        return PopulationEncoder(
            T=T,
            population_size=population_size,
            std=population_std,
            decode_mode=decode_mode,
        )
    if encoder_type in ("pop_spike_det", "pop_spike_deterministic", "population_spike_det"):
        return PopSpikeEncoderDeterministic(
            T=T,
            pop_dim=pop_dim,
            std=population_std,
            decode_mode=decode_mode,
        )
    if encoder_type in ("pop_spike_random", "population_spike_random"):
        return PopSpikeEncoderRandom(
            T=T,
            pop_dim=pop_dim,
            std=population_std,
            decode_mode=decode_mode,
        )
    if encoder_type in ("gaussian_tuning", "gaussian"):
        return GaussianTuningEncoder(
            T=T,
            m=gaussian_m,
            beta=gaussian_beta,
            decode_mode=decode_mode,
        )
    if encoder_type in ("rank_order", "rankorder"):
        return RankOrderEncoder(T=T, decode_mode=decode_mode)
    if encoder_type == "isi":
        return ISIEncoder(
            T=T,
            threshold=isi_threshold,
            min_interval=min_interval,
            decode_mode=decode_mode,
        )
    if encoder_type == "burst":
        return BurstEncoder(
            T=T,
            max_burst_size=max_burst_size,
            burst_interval=burst_interval,
            decode_mode=decode_mode,
        )
    if encoder_type in ("burst_isi", "burstisi"):
        return BurstISIEncoder(
            T=T,
            max_burst_size=max_burst_size,
            burst_interval=burst_interval,
            min_isi=min_isi,
            max_isi=max_isi,
            threshold=burst_isi_threshold,
            decode_mode=decode_mode,
        )
    if encoder_type in ("event_delta", "delta"):
        return EventDeltaEncoder(T=T, threshold=event_threshold, decode_mode=decode_mode)

    supported = [
        "direct",
        "poisson",
        "rate",
        "latency",
        "phase",
        "binary_temporal",
        "weighted_phase",
        "population",
        "pop_spike_det",
        "pop_spike_random",
        "gaussian_tuning",
        "rank_order",
        "isi",
        "burst",
        "burst_isi",
        "event_delta",
    ]
    raise ValueError(
        f"不支持的 encoder_type: '{encoder_type}'，可选值为: {', '.join(supported)}"
    )


__all__ = [
    "BaseEncoder",
    "DirectEncoder",
    "PoissonEncoder",
    "DeterministicRateEncoder",
    "LatencyEncoder",
    "PhaseEncoder",
    "BinaryTemporalEncoder",
    "WeightedPhaseEncoder",
    "PopulationEncoder",
    "PopSpikeEncoderDeterministic",
    "PopSpikeEncoderRandom",
    "GaussianTuningEncoder",
    "RankOrderEncoder",
    "ISIEncoder",
    "BurstEncoder",
    "BurstISIEncoder",
    "EventDeltaEncoder",
    "build_encoder",
]
