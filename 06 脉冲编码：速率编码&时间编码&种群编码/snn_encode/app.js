const $ = (selector) => document.querySelector(selector);

const defaultValues = '0, 0.2, 0.4, 0.6, 0.8, 1';
const encoders = {
  direct: { name: '直接编码', en: 'Direct', description: '将连续输入直接复制到每一个时间步，是非脉冲基线。', defaults: { T: 25, values: defaultValues } },
  rate: { name: '确定性速率编码', en: 'Deterministic rate', description: '通过积分—发放机制，以稳定的发放率表达输入大小。', defaults: { T: 25, values: defaultValues } },
  poisson: { name: '泊松编码', en: 'Poisson rate', description: '每个时间步按输入值作为概率独立随机采样。', defaults: { T: 25, values: defaultValues, seed: 42 } },
  latency: { name: '延迟编码', en: 'Latency / TTFS', description: '每个输入仅发放一次脉冲；输入越大，脉冲越早。', defaults: { T: 25, values: defaultValues, mode: 'linear' } },
  phase: { name: '相位编码', en: 'Phase', description: '输入决定脉冲在每个参考周期中的相位位置。', defaults: { T: 25, values: defaultValues, period: 5 } },
  binaryTemporal: { name: '二进制时间编码', en: 'Binary temporal', description: '先缩放到可表示范围，再以 1/2、1/4、… 的权重展开输入。', defaults: { T: 8, values: defaultValues } },
  population: { name: '高斯群体编码', en: 'Population response', description: '多个偏好中心不同的神经元共同以高斯响应表示输入，并在时间维复制。', defaults: { T: 15, values: defaultValues, pop: 7, std: 0.18 } },
  popDet: { name: '群体 + 确定性脉冲', en: 'Population deterministic', description: '先计算高斯群体响应，再使用积分—发放产生脉冲。', defaults: { T: 20, values: defaultValues, pop: 6, std: 0.18 } },
  popRandom: { name: '群体 + 随机脉冲', en: 'Population random', description: '将高斯群体响应作为每一步的随机发放概率。', defaults: { T: 20, values: defaultValues, pop: 6, std: 0.18, seed: 42 } },
  gaussian: { name: '高斯调谐 + 延迟', en: 'Gaussian tuning + latency', description: '调谐响应越高，延迟编码后的脉冲越早到达。', defaults: { T: 25, values: defaultValues, pop: 6, beta: 1.5 } },
  rank: { name: '脉冲顺序编码', en: 'Rank-order coding', description: '非零输入按数值从大到小依次在 t0、t1、t2… 发放；零输入不发放。', defaults: { T: 25, values: defaultValues } },
  isi: { name: '三脉冲 ISI 编码', en: '3-spike ISI pattern', description: '每个有效输入固定产生 3 个脉冲，信息主要由两段脉冲间隔模式携带。', defaults: { T: 25, values: defaultValues, minInterval: 1, threshold: 0 } },
  burst: { name: '突发编码', en: 'Burst coding', description: '输入决定每次突发中连续脉冲的数量，随后进入静默间隔。', defaults: { T: 25, values: defaultValues, maxBurst: 5, burstInterval: 10 } },
  burstIsi: { name: 'Burst + ISI 联合编码', en: 'Burst + ISI', description: '输入同时控制突发中的脉冲数量和突发内相邻脉冲的间隔。', defaults: { T: 25, values: defaultValues, maxBurst: 5, burstInterval: 10, minIsi: 1, maxIsi: 9, threshold: 0 } },
  delta: { name: '事件 / 差分编码', en: 'Event / delta coding', description: '事件编码表示的是变化：明显上升产生 ON，明显下降产生 OFF。', defaults: { eventMode: 'dynamic', signal: '0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.70, 0.55, 0.35, 0.20', T: 25, values: defaultValues, threshold: 0.1, selectedEventTime: 0 } },
};

let currentKey = 'direct';
let state = {};
const EVENT_EPSILON = 1e-10;
const eventPresets = {
  rising: { label: '单调上升', signal: '0.10, 0.18, 0.26, 0.38, 0.52, 0.68, 0.84' },
  riseFall: { label: '上升后下降', signal: '0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 0.70, 0.55, 0.35, 0.20' },
  step: { label: '阶跃信号', signal: '0.10, 0.10, 0.10, 0.75, 0.75, 0.75, 0.25, 0.25' },
  sine: { label: '正弦波', signal: '0.50, 0.71, 0.86, 0.92, 0.86, 0.71, 0.50, 0.29, 0.14, 0.08, 0.14, 0.29, 0.50' },
  constant: { label: '恒定高值', signal: '0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80' },
};

function format(value) { return Number.isInteger(+value) ? String(value) : (+value).toFixed(3).replace(/0+$/, '').replace(/\.$/, ''); }
function parseValues(text, name = '输入') {
  const values = text.split(/[,，\s]+/).filter(Boolean).map(Number);
  if (!values.length || values.some(v => !Number.isFinite(v) || v < 0 || v > 1)) throw Error(`${name}需要是一组 0 到 1 之间的数字。`);
  return values;
}
function rng(seed) { let value = (seed >>> 0) || 1; return () => ((value = (value * 1664525 + 1013904223) >>> 0) / 4294967296); }
function matrix(rows, columns) { return Array.from({ length: rows }, () => Array(columns).fill(0)); }
function gaussian(value, center, std) { return Math.exp(-.5 * ((value - center) / std) ** 2); }
function torchRound(value) {
  const lower = Math.floor(value), fraction = value - lower;
  if (fraction < .5) return lower;
  if (fraction > .5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function rangeControl(key, label, options = {}) {
  const { min = 0, max = 1, step = .01, hint = '' } = options;
  const value = state[key];
  const fill = ((value - min) / (max - min)) * 100;
  return `<div class='control-group'><div class='control-label-row'><label class='control-label' for='${key}'>${label}</label><output class='control-value' for='${key}'>${format(value)}</output></div><input class='range-input' id='${key}' type='range' min='${min}' max='${max}' step='${step}' value='${value}' style='--fill:${fill}%'><p class='control-hint'>${hint}</p></div>`;
}
function textControl(key, label, hint) { return `<div class='control-group'><label class='control-label' for='${key}'>${label}</label><input class='text-input' id='${key}' type='text' value='${state[key]}'><p class='control-hint'>${hint}</p></div>`; }
function selectControl(key, label, options) { return `<div class='control-group'><label class='control-label' for='${key}'>${label}</label><select id='${key}'>${options.map(([value, text]) => `<option value='${value}' ${state[key] === value ? 'selected' : ''}>${text}</option>`).join('')}</select></div>`; }

function renderControls() {
  state = { ...encoders[currentKey].defaults };
  $('#encoder-summary').textContent = encoders[currentKey].description;
  if (currentKey === 'delta') {
    renderEventControls();
    return;
  }
  const hasT = true;
  const hasValues = true;
  let html = hasT ? rangeControl('T', '时间步 T', { min: 1, max: 50, step: 1, hint: '离散时间窗口长度' }) : '';
  if (hasValues) html += textControl('values', '原始浮点输入', '用逗号分隔多个 0~1 数值');
  if (currentKey === 'latency') html += selectControl('mode', '时间映射', [['linear', '线性映射'], ['log', '对数映射']]);
  if (['poisson', 'popRandom'].includes(currentKey)) html += rangeControl('seed', '随机种子', { min: 0, max: 999, step: 1, hint: '固定种子便于重复观察' });
  if (currentKey === 'phase') html += rangeControl('period', '参考周期', { min: 2, max: 12, step: 1, hint: '每个周期内的相位位置数量' });
  if (['population', 'popDet', 'popRandom', 'gaussian'].includes(currentKey)) html += rangeControl('pop', currentKey === 'gaussian' ? '调谐神经元数 m' : '群体神经元数', { min: currentKey === 'gaussian' ? 3 : 2, max: 12, step: 1, hint: '神经元偏好中心均匀分布在输入范围内' });
  if (['population', 'popDet', 'popRandom'].includes(currentKey)) html += rangeControl('std', '高斯宽度 σ', { min: .05, max: .5, step: .01, hint: '越小表示神经元选择性越强' });
  if (currentKey === 'gaussian') html += rangeControl('beta', '调谐锐度 β', { min: .5, max: 3, step: .1, hint: '越大，调谐曲线越窄' });
  if (currentKey === 'isi') html += rangeControl('minInterval', '最短 ISI', { min: 1, max: 8, step: 1, hint: 't1-t2 和 t2-t3 都至少相隔这么多步' }) + rangeControl('threshold', '有效输入阈值', { min: 0, max: .95, step: .01, hint: '小于等于阈值的输入不产生脉冲' });
  if (currentKey === 'burst') html += rangeControl('maxBurst', '最大突发脉冲数', { min: 1, max: 10, step: 1, hint: '输入为 1 时每次突发包含的脉冲数' }) + rangeControl('burstInterval', '突发间隔', { min: 2, max: 20, step: 1, hint: '相邻两次突发起点之间的时间步数' });
  if (currentKey === 'burstIsi') html += rangeControl('maxBurst', '最大突发脉冲数', { min: 2, max: 10, step: 1, hint: '输入为 1 时每个 burst 最多包含的脉冲数' }) + rangeControl('burstInterval', '突发窗口', { min: 2, max: 20, step: 1, hint: '相邻两次 burst 起点之间的时间步数' }) + rangeControl('minIsi', '最短 burst 内 ISI', { min: 1, max: 8, step: 1, hint: '高强度输入对应的最短间隔' }) + rangeControl('maxIsi', '最长 burst 内 ISI', { min: 1, max: 19, step: 1, hint: '低强度输入对应的最长间隔，必须小于突发窗口' }) + rangeControl('threshold', '有效输入阈值', { min: 0, max: .95, step: .01, hint: '小于等于阈值的输入不产生 burst' });
  $('#controls').innerHTML = html;
  $('#controls').querySelectorAll('input, select').forEach(input => input.addEventListener('input', () => {
    state[input.id] = input.type === 'range' ? Number(input.value) : input.value;
    if (input.type === 'range') { input.style.setProperty('--fill', `${((+input.value - +input.min) / (+input.max - +input.min)) * 100}%`); input.parentElement.querySelector('output').textContent = format(input.value); }
    run();
  }));
}

function renderEventControls() {
  const dynamic = state.eventMode === 'dynamic';
  let html = `<div class="event-tabs"><button class="event-tab ${dynamic ? 'active' : ''}" data-event-mode="dynamic">动态信号事件编码</button><button class="event-tab ${dynamic ? '' : 'active'}" data-event-mode="static">静态输入适配</button></div>`;
  if (dynamic) {
    html += textControl('signal', '动态输入信号 x(t)', '逗号分隔；一个数对应同一通道的一个时间步。');
    html += `<div class="preset-label">预设信号</div><div class="event-presets">${Object.entries(eventPresets).map(([key, item]) => `<button class="preset-button ${state.signal === item.signal ? 'selected' : ''}" data-preset="${key}">${item.label}</button>`).join('')}</div>`;
    html += rangeControl('threshold', '事件阈值 θ', { min: .02, max: .5, step: .01, hint: 'θ 小：更敏感、事件更多；θ 大：只响应明显变化、事件更稀疏。' });
  } else {
    html += `<div class="static-adaptation-note"><strong>静态输入适配</strong><br>静态图像本身没有时间变化。这里人为构造从 0 上升到 x 的轨迹，因此不等同于真正的动态事件信号编码。</div>`;
    html += rangeControl('T', '时间步 T', { min: 1, max: 50, step: 1, hint: '构造轨迹 0 → x 的离散步数。' });
    html += textControl('values', '静态浮点输入', '逗号分隔多个 0~1 特征。每个特征会扩展为 ON/OFF 两个通道。');
    html += rangeControl('threshold', '事件阈值 θ', { min: .02, max: .5, step: .01, hint: '默认 0.05。过小会使较大输入进入事件饱和区。' });
  }
  html += '<div id="event-threshold-help" class="threshold-teaching"></div>';
  $('#controls').innerHTML = html;
  $('#controls').querySelectorAll('input, select').forEach(input => input.addEventListener('input', () => {
    state[input.id] = input.type === 'range' ? Number(input.value) : input.value;
    if (input.type === 'range') { input.style.setProperty('--fill', `${((+input.value - +input.min) / (+input.max - +input.min)) * 100}%`); input.parentElement.querySelector('output').textContent = format(input.value); }
    state.selectedEventTime = 0;
    run();
  }));
  $('#controls').querySelectorAll('[data-event-mode]').forEach(button => button.addEventListener('click', () => {
    state.eventMode = button.dataset.eventMode;
    state.threshold = state.eventMode === 'static' ? .05 : .1;
    state.selectedEventTime = 0;
    renderEventControls();
    run();
  }));
  $('#controls').querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => {
    state.signal = eventPresets[button.dataset.preset].signal;
    state.selectedEventTime = 0;
    renderEventControls();
    run();
  }));
}

function encode() {
  const values = () => parseValues(state.values);
  if (currentKey === 'direct') {
    const x = values();
    return { kind: 'continuousTemporal', values: matrix(state.T, x.length).map(() => [...x]), labels: x.map((v, i) => `x${i + 1} = ${v}`), title: '连续输入在时间维的复制', note: '与 DirectEncoder 一致：静态输入被复制到每个时间步，不进行二值化。' };
  }
  if (currentKey === 'rate' || currentKey === 'poisson') {
    const x = values(), spikes = matrix(state.T, x.length), random = rng(state.seed || 1), voltage = x.map(() => 0);
    for (let t = 0; t < state.T; t++) x.forEach((v, i) => { if (currentKey === 'poisson') spikes[t][i] = random() < v ? 1 : 0; else { voltage[i] += v; spikes[t][i] = voltage[i] >= 1 ? 1 : 0; voltage[i] -= spikes[t][i]; } });
    return spikeData(spikes, x.map((v, i) => `x${i + 1} = ${v}`), currentKey === 'rate' ? '积分—发放的确定性速率序列' : '按概率随机采样的脉冲序列', currentKey === 'rate' ? '在相同 T 内，输入越大，脉冲数越多；相同输入总会得到相同结果。' : '每一步独立采样，因此发放率在足够长的时间窗内趋近于输入值。');
  }
  if (currentKey === 'latency') {
    const x = values(), spikes = matrix(state.T, x.length), times = x.map(v => torchRound(state.mode === 'log' && state.T > 1 ? (state.T - 1) - Math.log((Math.exp(state.T - 1) - 1) * v + 1) : (state.T - 1) * (1 - v)));
    times.forEach((t, i) => { if (x[i] > 0) spikes[t][i] = 1; });
    return spikeData(spikes, x.map((v, i) => `x${i + 1} = ${v}`), '唯一脉冲的发放时刻', '与 LatencyEncoder 一致：零输入不发放；其余输入越大，脉冲越早。');
  }
  if (currentKey === 'phase') {
    const x = values(), spikes = matrix(state.T, x.length), phase = x.map(v => torchRound((1 - v) * (state.period - 1)));
    for (let t = 0; t < state.T; t++) phase.forEach((p, i) => { if (x[i] > 0 && t % state.period === p) spikes[t][i] = 1; });
    return spikeData(spikes, x.map((v, i) => `x${i + 1} · φ=${phase[i]}`), `每 ${state.period} 步重复的相位`, '与 PhaseEncoder 一致：零输入不发放，非零输入越大，相位越接近周期开始。');
  }
  if (currentKey === 'binaryTemporal') {
    const x = values(), maxValue = 1 - 2 ** (-state.T), rest = x.map(v => v * maxValue), spikes = matrix(state.T, x.length);
    for (let t = 0, weight = .5; t < state.T; t++, weight /= 2) rest.forEach((v, i) => { spikes[t][i] = v >= weight ? 1 : 0; rest[i] -= weight * spikes[t][i]; });
    return spikeData(spikes, x.map((v, i) => `x${i + 1} = ${v}`), '二进制时序权重展开', '与 BinaryTemporalEncoder 一致：先乘以 1 − 2⁻ᵀ，因此可直接接收 [0, 1] 输入。');
  }
  if (currentKey === 'population') {
    const x = values(), centers = Array.from({ length: state.pop }, (_, i) => i / (state.pop - 1)), response = x.flatMap(v => centers.map(c => gaussian(v, c, state.std))), labels = x.flatMap((v, i) => centers.map((_, j) => `x${i + 1} · n${j + 1}`));
    return { kind: 'continuousTemporal', values: matrix(state.T, response.length).map(() => [...response]), labels, title: '随时间重复的高斯群体响应', note: '与 PopulationEncoder 一致：先展平为 F × population_size 个特征，再沿时间维复制。' };
  }
  if (currentKey === 'popDet' || currentKey === 'popRandom') {
    const x = values(), centers = Array.from({ length: state.pop }, (_, i) => i / (state.pop - 1)), activation = x.flatMap(v => centers.map(c => gaussian(v, c, state.std))), spikes = matrix(state.T, activation.length), voltage = activation.map(() => 0), random = rng(state.seed || 1), labels = x.flatMap((v, i) => centers.map((_, j) => `x${i + 1} · n${j + 1}`));
    for (let t = 0; t < state.T; t++) activation.forEach((a, i) => { if (currentKey === 'popRandom') spikes[t][i] = random() < a ? 1 : 0; else { voltage[i] += a; spikes[t][i] = voltage[i] >= 1 ? 1 : 0; voltage[i] -= spikes[t][i]; } });
    return spikeData(spikes, labels, '群体神经元的脉冲序列', currentKey === 'popRandom' ? '每个神经元的高斯响应作为随机发放概率。' : '每个神经元的高斯响应作为积分—发放过程的输入电流。');
  }
  if (currentKey === 'gaussian') {
    const x = values(), centers = Array.from({ length: state.pop }, (_, i) => (2 * (i + 1) - 3) / (2 * (state.pop - 2))), variance = (1 / (state.beta * (state.pop - 2))) ** 2, spikes = matrix(state.T, x.length * state.pop), labels = x.flatMap((v, i) => centers.map((_, j) => `x${i + 1} · n${j + 1}`));
    x.forEach((v, i) => centers.forEach((center, j) => { const response = Math.exp(-((v - center) ** 2) / (2 * variance)); spikes[torchRound((state.T - 1) * (1 - response))][i * state.pop + j] = 1; }));
    return spikeData(spikes, labels, '调谐响应映射到发放时刻', '调谐神经元对输入越敏感，响应越高、发放越早。');
  }
  if (currentKey === 'rank') {
    const x = values();
    const spikes = matrix(state.T, x.length), order = x.map((_, i) => i).filter(i => x[i] > 0).sort((a, b) => x[b] - x[a] || a - b);
    if (state.T < order.length) throw Error(`当前有 ${order.length} 个非零输入，顺序编码需要 T 至少为 ${order.length}，才能让每个输入占用独立时间步。`);
    const times = Array(x.length).fill(null);
    order.forEach((channel, rank) => { spikes[rank][channel] = 1; times[channel] = rank; });
    return spikeData(spikes, x.map((v, i) => times[i] === null ? `n${i + 1} · x=0 · 不发放` : `n${i + 1} · x=${v} · t${times[i]}`), '按排名逐时间步发放', `非零输入按数值从大到小依次映射为 t0、t1、t2…；相同数值按输入顺序决定先后。零输入不参与排序，也不发放。顺序：${order.map(i => `n${i + 1}`).join(' → ')}。`);
  }
  if (currentKey === 'isi') {
    if (state.T < 2 * state.minInterval + 2) throw Error(`当前最短 ISI=${state.minInterval} 时，T 至少需要 ${2 * state.minInterval + 2}。`);
    const x = values(), spikes = matrix(state.T, x.length), minIsi1 = state.minInterval, maxIsi1 = state.T - 1 - state.minInterval;
    const middleTimes = x.map(v => torchRound(maxIsi1 - Math.max(0, Math.min(1, (v - state.threshold) / (1 - state.threshold))) * (maxIsi1 - minIsi1)));
    x.forEach((v, i) => {
      if (v <= state.threshold) return;
      spikes[0][i] = 1;
      spikes[middleTimes[i]][i] = 1;
      spikes[state.T - 1][i] = 1;
    });
    return spikeData(spikes, x.map((v, i) => v > state.threshold ? `x${i + 1} = ${v} · t2=${middleTimes[i]}` : `x${i + 1} = ${v} · inactive`), '三脉冲 ISI pattern', '所有有效输入固定为 3 个脉冲：t1=0、t2=f(x)、t3=T-1。输入越大，第一段 ISI 越短、第二段 ISI 越长。');
  }
  if (currentKey === 'burst') {
    const x = values(), spikes = matrix(state.T, x.length), sizes = x.map(v => torchRound(v * state.maxBurst));
    sizes.forEach((size, i) => { for (let start = 0; start < state.T; start += state.burstInterval) for (let t = start; t < Math.min(start + size, state.T); t++) spikes[t][i] = 1; });
    return spikeData(spikes, x.map((v, i) => `x${i + 1} = ${v} · ${sizes[i]} 脉冲/突发`), '连续脉冲组成的突发', '输入越大，每次突发内连续的脉冲越多；突发之间保持静默。');
  }
  if (currentKey === 'burstIsi') {
    if (state.maxIsi < state.minIsi) throw Error('最长 burst 内 ISI 必须大于等于最短 burst 内 ISI。');
    if (state.maxIsi >= state.burstInterval) throw Error('最长 burst 内 ISI 必须小于突发窗口。');
    if (state.burstInterval < 1 + (state.maxBurst - 1) * state.minIsi) throw Error(`当前 maxBurst=${state.maxBurst}, minIsi=${state.minIsi} 时，突发窗口至少需要 ${1 + (state.maxBurst - 1) * state.minIsi}。`);
    const x = values(), spikes = matrix(state.T, x.length), sizes = [], intervals = [];
    x.forEach((v, i) => {
      const active = v > state.threshold;
      const normalized = Math.max(0, Math.min(1, (v - state.threshold) / (1 - state.threshold)));
      const burstSize = active ? Math.ceil(normalized * state.maxBurst) : 0;
      const desiredIsi = torchRound(state.maxIsi - normalized * (state.maxIsi - state.minIsi));
      const gaps = Math.max(1, burstSize - 1);
      const fitIsi = Math.floor((state.burstInterval - 1) / gaps);
      const isi = Math.max(state.minIsi, Math.min(desiredIsi, fitIsi));
      sizes[i] = burstSize;
      intervals[i] = isi;
      for (let start = 0; start < state.T; start += state.burstInterval) {
        for (let k = 0; k < state.maxBurst; k++) {
          const t = start + k * isi;
          if (active && k < burstSize && t < state.T) spikes[t][i] = 1;
        }
      }
    });
    return spikeData(spikes, x.map((v, i) => sizes[i] ? `x${i + 1} = ${v} · size=${sizes[i]} · ISI=${intervals[i]}` : `x${i + 1} = ${v} · inactive`), 'Burst count + burst 内 ISI', '输入越大，burst 中 spike 数量越多、相邻 spike 间隔越短；同时利用 spike-count 和 temporal information。');
  }
  if (currentKey === 'delta') {
    return state.eventMode === 'dynamic' ? encodeDynamicEvent(parseValues(state.signal, '动态信号'), state.threshold) : encodeStaticEvent(values(), state.T, state.threshold);
  }
}

function encodeDynamicEvent(signal, threshold) {
  // Send-on-Delta：reference 只在报告事件时更新，记录上一次已发送的信号水平。
  let reference = signal[0];
  const referenceBefore = [], referenceAfter = [], errors = [], on = [], off = [];
  signal.forEach((value) => {
    const previous = reference, error = value - previous;
    let onEvent = 0, offEvent = 0;
    if (error >= threshold - EVENT_EPSILON) { onEvent = 1; reference = value; }
    else if (error <= -threshold + EVENT_EPSILON) { offEvent = 1; reference = value; }
    referenceBefore.push(previous); errors.push(error); on.push(onEvent); off.push(offEvent); referenceAfter.push(reference);
  });
  return { kind: 'eventDynamic', signal, threshold, referenceBefore, referenceAfter, errors, on, off, note: 'Send-on-Delta 关注相对于上一次报告值的变化：达到阈值时发送 ON/OFF，并把 reference 直接更新为当前 x(t)。' };
}

function encodeStaticEvent(values, T, threshold) {
  const spikes = matrix(T, values.length * 2), estimate = values.map(() => 0);
  for (let t = 0; t < T; t++) values.forEach((value, i) => {
    const trajectory = value * ((t + 1) / T), error = trajectory - estimate[i];
    if (error >= threshold - EVENT_EPSILON) { spikes[t][2 * i] = 1; estimate[i] += threshold; }
    else if (error <= -threshold + EVENT_EPSILON) { spikes[t][2 * i + 1] = 1; estimate[i] -= threshold; }
  });
  return { kind: 'eventStatic', spikes, values, threshold, labels: values.flatMap((_, i) => [`x${i + 1} ON ↑`, `x${i + 1} OFF ↓`]), note: '静态适配模式人为构造 0 → x 的上升轨迹，所以通常主要产生 ON 事件；真正的 OFF 事件需要动态信号下降。' };
}

function spikeData(spikes, labels, title, note) { return { kind: 'spikes', spikes, labels, title, note }; }
function metric(label, value, unit = '') { return `<div class='metric-card'><span class='metric-label'>${label}</span><span class='metric-value'>${value}${unit ? `<small> ${unit}</small>` : ''}</span></div>`; }

function renderEventDynamic(data) {
  const T = data.signal.length, onCount = data.on.reduce((sum, value) => sum + value, 0), offCount = data.off.reduce((sum, value) => sum + value, 0), selected = Math.min(state.selectedEventTime || 0, T - 1);
  return `<div class="event-core-message"><strong>Event Coding 关注“相对于上一次报告值发生了多大变化”，而不是当前数值本身有多大。</strong><span>reference 表示上一次产生事件时记录下来的信号水平；只有当前信号相对它的变化达到阈值 θ，才会产生新的 ON 或 OFF 事件。</span><span>明显上升 → ON；明显下降 → OFF；变化不足 θ → 不发事件。</span></div><section class="event-chart-card"><h3>连续信号 x(t) 与阶梯 reference / estimate(t)</h3>${renderEventChart(data)}</section><section class="event-raster-card"><div><h3>ON / OFF Event Raster</h3><p>↑ 表示检测到明显上升；↓ 表示检测到明显下降。</p></div>${renderEventRaster(data)}</section><section class="event-step-card"><div class="event-step-heading"><h3>当前时间步解释器</h3><label>选择时间步 <input id="event-step-slider" type="range" min="0" max="${T - 1}" step="1" value="${selected}"> <output id="event-step-value">t${selected}</output></label></div><div id="event-step-explainer">${renderEventStepExplanation(data, selected)}</div></section><div class="event-count-line">ON ${onCount} 次　·　OFF ${offCount} 次　·　总事件 ${onCount + offCount} 次　·　事件密度 ${((onCount + offCount) / (2 * T)).toFixed(3)}</div>`;
}

function renderEventChart(data) {
  const width = 760, height = 250, left = 42, right = 18, top = 18, bottom = 35, T = data.signal.length;
  const x = (i) => left + i * (width - left - right) / Math.max(1, T - 1);
  const y = (value) => height - bottom - value * (height - top - bottom);
  const line = data.signal.map((value, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(value).toFixed(1)}`).join(' ');
  const step = data.referenceAfter.reduce((path, value, i) => i === 0 ? `M${x(0).toFixed(1)},${y(value).toFixed(1)}` : `${path} L${x(i).toFixed(1)},${y(data.referenceAfter[i - 1]).toFixed(1)} L${x(i).toFixed(1)},${y(value).toFixed(1)}`, '');
  const grid = [0, .25, .5, .75, 1].map(value => `<line class="event-gridline" x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}"></line><text x="5" y="${y(value) + 4}">${value.toFixed(2)}</text>`).join('');
  const dots = data.signal.map((value, i) => `<circle class="signal-dot" cx="${x(i)}" cy="${y(value)}" r="3"></circle><text x="${x(i)}" y="${height - 11}" text-anchor="middle">t${i}</text>`).join('');
  const eventMarks = data.on.map((value, i) => value ? `<text class="on-mark" x="${x(i)}" y="${top + 14}" text-anchor="middle">↑</text>` : data.off[i] ? `<text class="off-mark" x="${x(i)}" y="${top + 14}" text-anchor="middle">↓</text>` : '').join('');
  return `<svg class="event-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="输入信号与 reference 阶梯曲线">${grid}<path class="signal-path" d="${line}"></path><path class="estimate-path" d="${step}"></path>${dots}${eventMarks}<text class="chart-legend-input" x="${width - 210}" y="20">● x(t) 输入信号</text><text class="chart-legend-estimate" x="${width - 210}" y="36">━ reference / estimate(t)</text></svg>`;
}

function renderEventRaster(data) {
  const T = data.signal.length;
  const header = Array.from({ length: T }, (_, t) => `<span class="event-time-label">t${t}</span>`).join('');
  const row = (name, values, direction) => `<span class="event-row-label">${name} ${direction}</span>${values.map((value, t) => `<button class="event-cell ${value ? (direction === '↑' ? 'on' : 'off') : ''}" data-event-time="${t}" title="选择 t${t}">${value ? direction : ''}</button>`).join('')}`;
  return `<div class="event-raster-grid" style="--columns:${T}"><span></span>${header}${row('ON', data.on, '↑')}${row('OFF', data.off, '↓')}</div>`;
}

function renderEventStepExplanation(data, t) {
  const value = data.signal[t], before = data.referenceBefore[t], after = data.referenceAfter[t], error = data.errors[t], on = data.on[t], off = data.off[t];
  const decision = on ? `<strong class="event-on-text">${format(error)} ≥ ${format(data.threshold)} → ON = 1，OFF = 0</strong>` : off ? `<strong class="event-off-text">${format(error)} ≤ −${format(data.threshold)} → ON = 0，OFF = 1</strong>` : `<strong>|${format(error)}| &lt; ${format(data.threshold)} → 本时间步不产生事件</strong>`;
  return `<p>当前时间步：<strong>t = ${t}</strong></p><div class="event-formula-grid"><span>当前输入</span><code>x(t) = ${format(value)}</code><span>当前 reference</span><code>${format(before)}</code><span>误差</span><code>x(t) − reference = ${format(value)} − ${format(before)} = ${format(error)}</code><span>事件阈值</span><code>θ = ${format(data.threshold)}</code></div><p>${decision}</p><p>reference 更新：<code>${format(before)} → ${format(after)}</code></p>`;
}

function renderStaticEvent(data) {
  const countEvents = (parity) => data.spikes.reduce((sum, row) => sum + row.reduce((rowSum, value, index) => rowSum + (index % 2 === parity ? value : 0), 0), 0);
  const onCount = countEvents(0), offCount = countEvents(1), saturated = data.values.filter(value => value / data.spikes.length >= data.threshold - EVENT_EPSILON);
  const warning = saturated.length ? `<div class="saturation-warning"><strong>饱和提示：</strong>当前 θ 较小，${saturated.length} 个输入满足 x / T ≥ θ，可能接近每一步都产生 ON 事件，较大输入间的区分度会下降。</div>` : '';
  return `<div class="static-adaptation-banner"><strong>注意：这是静态输入适配版。</strong> 静态图像并没有真实时间变化；页面人为构造 0 → x 的轨迹后才执行事件编码。</div>${warning}${renderSpikes({ spikes: data.spikes, labels: data.labels, title: '静态输入适配的 ON / OFF 事件', note: data.note })}<div class="event-contrast"><div><strong>Rate Coding</strong><br>高值持续存在 → 可以持续高频发放</div><div><strong>Event Coding</strong><br>高值保持不变 → 不产生新的事件</div></div><div class="event-count-line">ON 平均发放率 ${(onCount / (data.spikes.length * data.values.length)).toFixed(3)}　·　OFF 平均发放率 ${(offCount / (data.spikes.length * data.values.length)).toFixed(3)}　·　整体事件密度 ${((onCount + offCount) / data.spikes.flat().length).toFixed(3)}</div>`;
}

function wireEventDynamicControls(data) {
  const setStep = (time) => {
    state.selectedEventTime = Number(time);
    const slider = $('#event-step-slider'), output = $('#event-step-value'), explainer = $('#event-step-explainer');
    if (slider) slider.value = state.selectedEventTime;
    if (output) output.textContent = `t${state.selectedEventTime}`;
    if (explainer) explainer.innerHTML = renderEventStepExplanation(data, state.selectedEventTime);
  };
  $('#event-step-slider')?.addEventListener('input', (event) => setStep(event.target.value));
  $('#visualization').querySelectorAll('[data-event-time]').forEach(cell => cell.addEventListener('click', () => setStep(cell.dataset.eventTime)));
}

function updateEventThresholdHelp(data) {
  const target = $('#event-threshold-help');
  if (!target) return;
  if (data.kind === 'eventDynamic') {
    const onCount = data.on.reduce((sum, value) => sum + value, 0), offCount = data.off.reduce((sum, value) => sum + value, 0);
    target.innerHTML = `<strong>θ = ${format(data.threshold)}</strong><br>ON ${onCount} 次 · OFF ${offCount} 次 · 总事件 ${onCount + offCount} 次。reference 记录上一次产生事件时的信号水平；θ 较小更敏感、事件更多，θ 较大会忽略小变化。`;
  } else {
    const countEvents = (parity) => data.spikes.reduce((sum, row) => sum + row.reduce((rowSum, value, index) => rowSum + (index % 2 === parity ? value : 0), 0), 0);
    const onCount = countEvents(0), offCount = countEvents(1);
    target.innerHTML = `<strong>θ = ${format(data.threshold)}</strong><br>ON ${onCount} 次 · OFF ${offCount} 次。由于轨迹只会上升，OFF 通道通常为 0；θ 较小时 ON 事件会更密集。`;
  }
}

function renderSpikes(data) {
  const T = data.spikes.length, rows = data.spikes[0].length, rates = Array.from({ length: rows }, (_, i) => data.spikes.reduce((sum, step) => sum + step[i], 0) / T);
  const header = Array.from({ length: T }, (_, t) => `<span class='time-label'>t${t}</span>`).join('') + `<span class='time-label rate-heading'>发放率</span>`;
  const row = (i) => `<span class='row-label'>${data.labels[i] || `n${i + 1}`}</span>${data.spikes.map((step, t) => `<i class='spike-cell ${step[i] ? 'active' : ''}' title='t=${t}，值=${step[i]}'></i>`).join('')}<span class='row-rate'>${rates[i].toFixed(3)}</span>`;
  return `<h3 class='chart-title'>${data.title}</h3><p class='chart-subtitle'>行：输入通道／神经元　·　列：离散时间步　·　最右侧：发放率（脉冲数 / T）</p><div class='spike-grid' style='--columns:${T}'><span></span>${header}${Array.from({ length: rows }, (_, i) => row(i)).join('')}</div><div class='legend'><span><i></i>脉冲 = 1</span><span><i class='empty'></i>无脉冲 = 0</span></div>`;
}

function renderContinuousTemporal(data) {
  const T = data.values.length, rows = data.values[0].length, averages = Array.from({ length: rows }, (_, i) => data.values.reduce((sum, step) => sum + step[i], 0) / T);
  const header = Array.from({ length: T }, (_, t) => `<span class='time-label'>t${t}</span>`).join('') + `<span class='time-label rate-heading'>平均值</span>`;
  const row = (i) => `<span class='row-label'>${data.labels[i] || `n${i + 1}`}</span>${data.values.map((step, t) => `<i class='spike-cell continuous' style='--opacity:${Math.max(.08, step[i])}' title='t=${t}，值=${step[i].toFixed(3)}'></i>`).join('')}<span class='row-rate'>${averages[i].toFixed(3)}</span>`;
  return `<h3 class='chart-title'>${data.title}</h3><p class='chart-subtitle'>单样本输出形状为 [1, T, F]；行是特征，列是时间步，最右侧为时间平均值。</p><div class='spike-grid' style='--columns:${T}'><span></span>${header}${Array.from({ length: rows }, (_, i) => row(i)).join('')}</div><div class='legend'><span><i></i>高连续响应</span><span><i class='empty'></i>低连续响应</span></div>`;
}

function renderContinuous(data) {
  const columns = data.matrix[0].length;
  return `<h3 class='chart-title'>${data.title}</h3><p class='chart-subtitle'>每行是一个输入，各列是偏好中心不同的群体神经元</p><div class='spike-grid continuous-grid' style='--columns:${columns}'><span></span>${data.columns.map(label => `<span class='time-label'>${label}</span>`).join('')}<span class='time-label rate-heading'>最大响应</span>${data.matrix.map((row, i) => `<span class='row-label'>${data.rows[i]}</span>${row.map(value => `<i class='spike-cell continuous' style='--opacity:${Math.max(.08, value)}' title='响应 ${value.toFixed(3)}'></i>`).join('')}<span class='row-rate'>${Math.max(...row).toFixed(3)}</span>`).join('')}</div><div class='legend'><span><i></i>高响应</span><span><i class='empty'></i>低响应</span></div>`;
}

function run() {
  try {
    const data = encode();
    $('#result-title').textContent = `${encoders[currentKey].name} · ${encoders[currentKey].en}`;
    if (data.kind === 'eventDynamic') {
      $('#metric-cards').className = 'metric-cards event-metric-cards';
      const onCount = data.on.reduce((sum, value) => sum + value, 0), offCount = data.off.reduce((sum, value) => sum + value, 0), total = onCount + offCount;
      $('#result-badge').textContent = `[1, ${data.signal.length}, 2]`;
      $('#metric-cards').innerHTML = metric('时间步 T', data.signal.length, '步') + metric('ON 事件', onCount, '次') + metric('OFF 事件', offCount, '次') + metric('总事件', total, '次') + metric('事件密度', (total / (2 * data.signal.length)).toFixed(3));
      $('#visualization').innerHTML = renderEventDynamic(data);
      $('#explanation-text').textContent = data.note;
      $('#raw-output').textContent = JSON.stringify({ signal: data.signal, reference: data.referenceAfter, on: data.on, off: data.off }, null, 2);
      updateEventThresholdHelp(data);
      wireEventDynamicControls(data);
      return;
    }
    if (data.kind === 'eventStatic') {
      $('#metric-cards').className = 'metric-cards event-metric-cards';
      const countEvents = (parity) => data.spikes.reduce((sum, row) => sum + row.reduce((rowSum, value, index) => rowSum + (index % 2 === parity ? value : 0), 0), 0);
      const onCount = countEvents(0), offCount = countEvents(1), total = onCount + offCount;
      $('#result-badge').textContent = `[1, ${data.spikes.length}, ${data.spikes[0].length}]`;
      $('#metric-cards').innerHTML = metric('输入通道', data.values.length, '个') + metric('ON 平均发放率', (onCount / (data.spikes.length * data.values.length)).toFixed(3)) + metric('OFF 平均发放率', (offCount / (data.spikes.length * data.values.length)).toFixed(3)) + metric('事件密度', (total / data.spikes.flat().length).toFixed(3));
      $('#visualization').innerHTML = renderStaticEvent(data);
      $('#explanation-text').textContent = data.note;
      $('#raw-output').textContent = JSON.stringify(data.spikes, null, 2);
      updateEventThresholdHelp(data);
      return;
    }
    $('#metric-cards').className = 'metric-cards';
    $('#result-badge').textContent = data.kind === 'spikes' ? `T = ${data.spikes.length}` : data.kind === 'continuousTemporal' ? `[1, ${data.values.length}, ${data.values[0].length}]` : `${data.matrix[0].length} 个神经元`;
    if (data.kind === 'spikes') {
      const total = data.spikes.flat().reduce((sum, value) => sum + value, 0), rows = data.spikes[0].length;
      $('#metric-cards').innerHTML = metric('时间步', data.spikes.length, '步') + metric('输出通道', rows, '个') + metric('平均发放率', (total / (data.spikes.length * rows)).toFixed(3));
      $('#visualization').innerHTML = renderSpikes(data);
      $('#raw-output').textContent = JSON.stringify(data.spikes, null, 2);
    } else if (data.kind === 'continuousTemporal') {
      const total = data.values.flat().reduce((sum, value) => sum + value, 0), rows = data.values[0].length;
      $('#metric-cards').innerHTML = metric('时间步', data.values.length, '步') + metric('输出通道', rows, '个') + metric('平均特征值', (total / (data.values.length * rows)).toFixed(3));
      $('#visualization').innerHTML = renderContinuousTemporal(data);
      $('#raw-output').textContent = JSON.stringify(data.values.map(row => row.map(value => +value.toFixed(4))), null, 2);
    } else {
      $('#metric-cards').innerHTML = metric('输入通道', data.matrix.length, '个') + metric('群体神经元', data.matrix[0].length, '个') + metric('最大响应', Math.max(...data.matrix.flat()).toFixed(3));
      $('#visualization').innerHTML = renderContinuous(data);
      $('#raw-output').textContent = JSON.stringify(data.matrix.map(row => row.map(value => +value.toFixed(4))), null, 2);
    }
    $('#explanation-text').textContent = data.note;
  } catch (error) {
    $('#metric-cards').innerHTML = metric('状态', '等待有效输入');
    $('#visualization').innerHTML = `<h3 class='chart-title'>请检查输入</h3><p class='chart-subtitle'>${error.message}</p>`;
    $('#explanation-text').textContent = '修改左侧参数后会自动重新计算。';
    $('#raw-output').textContent = '';
  }
}

function init() {
  const select = $('#encoder-select');
  select.innerHTML = Object.entries(encoders).map(([key, item]) => `<option value='${key}'>${item.name} · ${item.en}</option>`).join('');
  select.value = currentKey;
  select.addEventListener('change', () => { currentKey = select.value; renderControls(); run(); });
  $('#run-button').addEventListener('click', run);
  $('#reset-button').addEventListener('click', () => { renderControls(); run(); });
  renderControls();
  run();
}

init();
