import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, qmc, wasserstein_distance
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# 设置全局默认字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Song Times'
plt.rcParams['font.size'] = 12
# 设置公式字体
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Song Times'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 参数设置 (Configuration)
# ==========================================
class SimConfig:
    WARMUP_HOURS = 24 
    SIM_HOURS = 168 
    TOTAL_SIM_HOURS = 192 
    DT = 1.0
    AES_PENETRATION = 0.2 
    NUM_BERTHS = 5
    LAMBDA_NORMAL = 0.5
    LAMBDA_PEAK = 2.5 
    PEAK_HOURS = [(9, 17)]
    DURATION_MU = 2.5 
    DURATION_SIGMA = 0.5
    DURATION_MIN = 4 
    DURATION_MAX = 24 
    FUEL_POWER_MU = 22.5
    FUEL_POWER_SIGMA = 8.75
    FUEL_POWER_MIN = 5 
    FUEL_POWER_MAX = 40 
    AES_CHARGE_POWER_MU = 600 
    AES_CHARGE_POWER_SIGMA = 70  
    AES_CHARGE_POWER_MIN = 400  
    AES_CHARGE_POWER_MAX = 800 
    AES_SMALL_CHARGE_POWER_MU = 100
    AES_SMALL_CHARGE_POWER_SIGMA = 15
    AES_SMALL_CHARGE_POWER_MIN = 60
    AES_SMALL_CHARGE_POWER_MAX = 150
    NUM_SCENARIOS = 500 
    NUM_CLUSTERS = 4 
    PARALLEL_JOBS = -1 
    REDUCTION_METHOD = 'SBR' 
    DISTANCE_METRIC = 'wasserstein' 

# ==========================================
# 2. Wasserstein距离计算
# ==========================================

def wasserstein_dist(ts1, ts2):
    return wasserstein_distance(ts1, ts2)

def _compute_wasserstein_row(i, time_series_list, n):
    row = np.zeros(n)
    for j in range(i+1, n):
        row[j] = wasserstein_distance(time_series_list[i], time_series_list[j])
    return row

def compute_wasserstein_distance_matrix(time_series_list, n_jobs=-1):
    n = len(time_series_list)
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs <= 0:
        n_jobs = 1
    
    print(f"  使用Wasserstein距离 + joblib并行计算 ({n_jobs} 个进程)...")
    print(f"  计算 {n} 个场景的距离矩阵 (共 {n*(n-1)//2} 对)...")
    rows = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(_compute_wasserstein_row)(i, time_series_list, n) 
        for i in range(n)
    )

    distance_matrix = np.zeros((n, n))
    for i, row in enumerate(rows):
        distance_matrix[i, :] = row
        for j in range(i+1, n):
            distance_matrix[j, i] = row[j]
    
    print(f"  距离矩阵计算完成!")
    return distance_matrix

# ==========================================
# 3. 基于Wasserstein距离的场景削减 (SBR)
# ==========================================

class SBRScenarioReduction:
    def __init__(self, n_scenarios_target, random_state=None, diversity_weight=0.5):
        self.n_scenarios_target = n_scenarios_target
        self.random_state = random_state
        self.diversity_weight = diversity_weight
        self.selected_indices_ = None  # 保留的场景索引
        self.deleted_indices_ = None   # 删除的场景索引
        self.probabilities_ = None      # 场景概率（重分配后）
        self.labels_ = None             # 每个原始场景对应的保留场景
        
    def fit(self, distance_matrix, probabilities=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_scenarios = distance_matrix.shape[0]
        if probabilities is None:
            probabilities = np.ones(n_scenarios) / n_scenarios
        else:
            probabilities = probabilities.copy()
        remaining_indices = list(range(n_scenarios))
        deleted_indices = []
        
        n_to_delete = n_scenarios - self.n_scenarios_target
        for iteration in range(n_to_delete):
            if iteration % max(1, n_to_delete // 10) == 0:
                progress = (iteration / n_to_delete) * 100
                print(f"    进度: {iteration}/{n_to_delete} ({progress:.1f}%)")
            n_remaining = len(remaining_indices)
            deletion_costs = np.zeros(n_remaining)
            
            for i, idx_i in enumerate(remaining_indices):
                distances_to_others = []
                for j, idx_j in enumerate(remaining_indices):
                    if i != j:
                        distances_to_others.append(distance_matrix[idx_i, idx_j])
                
                if distances_to_others:
                    min_distance = min(distances_to_others)
                    avg_distance = np.mean(distances_to_others)
                    w = self.diversity_weight
                    deletion_costs[i] = probabilities[idx_i] * (
                        (1 - w) * min_distance - w * avg_distance
                    )
                else:
                    deletion_costs[i] = np.inf
            idx_to_delete_local = np.argmin(deletion_costs)
            idx_to_delete_global = remaining_indices[idx_to_delete_local]
            remaining_except_deleted = [idx for idx in remaining_indices if idx != idx_to_delete_global]
            
            if remaining_except_deleted:
                distances_to_remaining = [distance_matrix[idx_to_delete_global, idx] 
                                         for idx in remaining_except_deleted]
                nearest_neighbor_local = np.argmin(distances_to_remaining)
                nearest_neighbor_global = remaining_except_deleted[nearest_neighbor_local]
                probabilities[nearest_neighbor_global] += probabilities[idx_to_delete_global]
                probabilities[idx_to_delete_global] = 0
            deleted_indices.append(idx_to_delete_global)
            remaining_indices.remove(idx_to_delete_global)
        
        print(f"    进度: {n_to_delete}/{n_to_delete} (100.0%)")
        print(f"  场景削减完成!")
        self.selected_indices_ = np.array(sorted(remaining_indices))
        self.deleted_indices_ = np.array(deleted_indices)
        self.probabilities_ = probabilities[self.selected_indices_]
        self.probabilities_ = self.probabilities_ / self.probabilities_.sum()
        self.labels_ = np.zeros(n_scenarios, dtype=int)
        for i in range(n_scenarios):
            distances_to_selected = distance_matrix[i, self.selected_indices_]
            nearest_selected = np.argmin(distances_to_selected)
            self.labels_[i] = nearest_selected
        return self

# ==========================================
# 4. 拉丁超立方采样
# ==========================================

def generate_lhs_scenarios(n_scenarios, random_state=None):

    sampler = qmc.LatinHypercube(d=1, seed=random_state)
    sample = sampler.random(n=n_scenarios)
    seeds = (sample[:, 0] * 1000000).astype(int)
    
    scenarios = []
    progress_interval = max(1, n_scenarios // 20)  # 减少进度输出频率
    for i, seed in enumerate(seeds):
        if i % progress_interval == 0 and i > 0:
            print(f"  已生成 {i}/{n_scenarios} 个场景 ({i/n_scenarios*100:.1f}%)...")
        scenario = generate_scenario(seed=int(seed))
        scenarios.append(scenario)
    
    print(f"  已生成 {n_scenarios}/{n_scenarios} 个场景 (100.0%)")
    print(f"场景生成完成!")
    return scenarios

# ==========================================
# 3. 核心类与函数
# ==========================================

class Ship:
    def __init__(self, arrival_time, ship_type):
        self.arrival_time = arrival_time
        self.ship_type = ship_type 
        
        if self.ship_type == 'FUEL':
            self.duration, self.load_profile = self._calculate_fuel()
            self.departure_time = self.arrival_time + self.duration
        else:
            self._calculate_aes_profile()

    def _generate_duration(self):
        for _ in range(100): 
            d = np.random.lognormal(SimConfig.DURATION_MU, SimConfig.DURATION_SIGMA)
            d = round(d)
            if SimConfig.DURATION_MIN <= d <= SimConfig.DURATION_MAX:
                return d
        return (SimConfig.DURATION_MIN + SimConfig.DURATION_MAX) // 2

    def _calculate_fuel(self):
        power = np.random.normal(SimConfig.FUEL_POWER_MU, SimConfig.FUEL_POWER_SIGMA)
        power = np.clip(power, SimConfig.FUEL_POWER_MIN, SimConfig.FUEL_POWER_MAX)
        duration = self._generate_duration()
        return duration, power

    def _calculate_fuel_load(self):
        p = np.random.normal(SimConfig.FUEL_POWER_MU, SimConfig.FUEL_POWER_SIGMA)
        return np.clip(p, SimConfig.FUEL_POWER_MIN, SimConfig.FUEL_POWER_MAX)

    def _calculate_aes_profile(self):
        is_small = (self.ship_type == 'AES_SMALL')
        p_mu = SimConfig.AES_SMALL_CHARGE_POWER_MU if is_small else SimConfig.AES_CHARGE_POWER_MU
        p_sigma = SimConfig.AES_SMALL_CHARGE_POWER_SIGMA if is_small else SimConfig.AES_CHARGE_POWER_SIGMA
        p_min = SimConfig.AES_SMALL_CHARGE_POWER_MIN if is_small else SimConfig.AES_CHARGE_POWER_MIN
        p_max = SimConfig.AES_SMALL_CHARGE_POWER_MAX if is_small else SimConfig.AES_CHARGE_POWER_MAX
        rated_power = np.random.normal(p_mu, p_sigma)
        rated_power = np.clip(rated_power, p_min, p_max)
        duration = self._generate_duration()
        self.duration = duration
        self.load_profile = rated_power
        self.departure_time = self.arrival_time + self.duration

def is_peak_hour(hour):
    hour_of_day = hour % 24
    for start, end in SimConfig.PEAK_HOURS:
        if start <= hour_of_day < end:
            return True
    return False

def get_lambda(hour):
    return SimConfig.LAMBDA_PEAK if is_peak_hour(hour) else SimConfig.LAMBDA_NORMAL

def generate_scenario(seed=None):
    if seed:
        np.random.seed(seed)
    total_time_steps = int(SimConfig.TOTAL_SIM_HOURS / SimConfig.DT)
    warmup_steps = int(SimConfig.WARMUP_HOURS / SimConfig.DT)
    timeline = np.zeros(total_time_steps)
    aes_load_curve = np.zeros(total_time_steps)
    fuel_load_curve = np.zeros(total_time_steps)
    berth_occupancy = np.zeros(total_time_steps) 
    medium_aes_connect = np.zeros(total_time_steps, dtype=bool)
    ships = [] 
    ships_in_port = [] 
    rejected_ships = 0 
    
    for t in range(total_time_steps):
        if ships_in_port:
            i = 0
            while i < len(ships_in_port):
                if ships_in_port[i][1] <= t:
                    ships_in_port.pop(i)
                else:
                    i += 1
        current_occupancy = len(ships_in_port)
        berth_occupancy[t] = current_occupancy
        lambda_t = get_lambda(t)
        num_arrivals = np.random.poisson(lambda_t)
        
        for _ in range(num_arrivals):
            if current_occupancy >= SimConfig.NUM_BERTHS:
                rejected_ships += 1
                continue 
            if np.random.rand() < SimConfig.AES_PENETRATION:
                s_type = 'AES_SMALL' if np.random.rand() < 0.5 else 'AES_MEDIUM'
            else:
                s_type = 'FUEL'
            ship = Ship(arrival_time=t, ship_type=s_type)
            ships.append(ship) 
            ships_in_port.append((ship, ship.departure_time))
            current_occupancy += 1 
            start_idx = t
            end_idx = min(total_time_steps, int(t + ship.duration))
            
            if start_idx < total_time_steps:
                if ship.ship_type != 'FUEL':
                    aes_load_curve[start_idx:end_idx] += ship.load_profile
                    if ship.ship_type == 'AES_MEDIUM':
                        medium_aes_connect[start_idx:end_idx] = True
                else:
                    fuel_load_curve[start_idx:end_idx] += ship.load_profile
                    
    total_load = aes_load_curve + fuel_load_curve
    actual_start_idx = warmup_steps
    actual_end_idx = total_time_steps
    
    result = pd.DataFrame({
        'Time_Hour': np.arange(actual_end_idx - actual_start_idx) * SimConfig.DT,
        'Total_Load_kW': total_load[actual_start_idx:actual_end_idx],
        'AES_Load_kW': aes_load_curve[actual_start_idx:actual_end_idx],
        'Fuel_Load_kW': fuel_load_curve[actual_start_idx:actual_end_idx],
        'Berth_Occupancy': berth_occupancy[actual_start_idx:actual_end_idx],
        'Medium_AES_Connect': medium_aes_connect[actual_start_idx:actual_end_idx].astype(int)
    })

    ships_after_warmup = [s for s in ships if s.arrival_time >= warmup_steps]
    result.attrs['total_ships'] = len(ships_after_warmup)
    result.attrs['rejected_ships'] = rejected_ships  # 注意:这是整个模拟期间的拒绝数
    result.attrs['warmup_hours'] = SimConfig.WARMUP_HOURS
    
    return result

# ==========================================
# 6. 场景聚类与评估
# ==========================================

def compute_risk_probs_for_scenario(scenario_df, grid_threshold_kw=200.0):
    n = len(scenario_df)
    if n == 0:
        return 0.0, 0.0, 0.0
    interop_mask = scenario_df['Medium_AES_Connect'].astype(bool).values
    interop_p = interop_mask.sum() / n
    load = scenario_df['Total_Load_kW'].values
    grid_mask = (~interop_mask) & (load > grid_threshold_kw)
    grid_p = grid_mask.sum() / n
    expected_shortage_p = interop_p + grid_p
    return float(interop_p), float(grid_p), float(expected_shortage_p)

def evaluate_risks_all_scenarios(scenarios, grid_threshold_kw=200.0):
    per = []
    for df in scenarios:
        interop_p, grid_p, expected_p = compute_risk_probs_for_scenario(df, grid_threshold_kw)
        per.append({'interop': interop_p, 'grid': grid_p, 'expected': expected_p})

    if len(per) == 0:
        avgs = {'interop': 0.0, 'grid': 0.0, 'expected': 0.0}
    else:
        avgs = {
            'interop': float(np.mean([x['interop'] for x in per])),
            'grid': float(np.mean([x['grid'] for x in per])),
            'expected': float(np.mean([x['expected'] for x in per]))
        }
    return per, avgs

def reduce_scenarios_sbr(scenarios, n_clusters=10, random_state=42):

    load_curves = [scenario['Total_Load_kW'].values for scenario in scenarios]
    expected_length = SimConfig.SIM_HOURS
    actual_length = len(load_curves[0])

    if actual_length != expected_length:
        print(f"警告：场景数据长度与预期不符！")
    distance_matrix = compute_wasserstein_distance_matrix(load_curves, n_jobs=-1)

    sbr_model = SBRScenarioReduction(n_scenarios_target=n_clusters, random_state=random_state)
    sbr_model.fit(distance_matrix)
    
    print(f"\n场景削减完成!")
    print(f"保留的场景索引: {sbr_model.selected_indices_}")
    print(f"场景概率分布: {sbr_model.probabilities_}")
    unique_labels, counts = np.unique(sbr_model.labels_, return_counts=True)
    print(f"\n各保留场景对应的原始场景数:")
    for label, count in zip(unique_labels, counts):
        actual_idx = sbr_model.selected_indices_[label]
        prob = sbr_model.probabilities_[label]
        print(f"  场景 {actual_idx} (概率={prob:.4f}): {count} 个原始场景")
    
    return sbr_model, distance_matrix

def evaluate_reduction(sbr_model, distance_matrix):    
    n_original = distance_matrix.shape[0]
    n_selected = len(sbr_model.selected_indices_)   
    representation_errors = []
    for i in range(n_original):

        nearest_selected_idx = sbr_model.labels_[i]
        actual_selected_scenario = sbr_model.selected_indices_[nearest_selected_idx]
        error = distance_matrix[i, actual_selected_scenario]
        representation_errors.append(error)
    
    avg_representation_error = np.mean(representation_errors)
    max_representation_error = np.max(representation_errors)
    weighted_errors = []
    for i in range(n_original):
        nearest_selected_idx = sbr_model.labels_[i]
        actual_selected_scenario = sbr_model.selected_indices_[nearest_selected_idx]
        error = distance_matrix[i, actual_selected_scenario]
        weight = sbr_model.probabilities_[nearest_selected_idx]
        weighted_errors.append(error * weight)
    
    weighted_avg_error = np.sum(weighted_errors)
    if n_selected > 1:
        inter_distances = []
        for i in range(n_selected):
            for j in range(i+1, n_selected):
                idx_i = sbr_model.selected_indices_[i]
                idx_j = sbr_model.selected_indices_[j]
                inter_distances.append(distance_matrix[idx_i, idx_j])
        avg_inter_distance = np.mean(inter_distances)
        min_inter_distance = np.min(inter_distances)
    else:
        avg_inter_distance = 0
        min_inter_distance = 0
    silhouette_avg = None
    n_unique_labels = len(np.unique(sbr_model.labels_))
    if n_unique_labels >= 2 and n_selected >= 2:
        try:
            silhouette_avg = silhouette_score(distance_matrix, sbr_model.labels_, metric='precomputed')
        except:
            silhouette_avg = None
    
    # 输出结果
    print(f"\n场景削减质量评估:")
    print(f"  原始场景数: {n_original}")
    print(f"  保留场景数: {n_selected}")
    print(f"  削减率: {(1 - n_selected/n_original)*100:.1f}%")
    
    # 返回评估指标
    metrics = {
        'n_original': n_original,
        'n_selected': n_selected,
        'reduction_rate': (1 - n_selected/n_original),
        'avg_representation_error': avg_representation_error,
        'max_representation_error': max_representation_error,
        'weighted_avg_error': weighted_avg_error,
        'avg_inter_distance': avg_inter_distance,
        'min_inter_distance': min_inter_distance,
        'silhouette_score': silhouette_avg
    }
    
    return metrics

def plot_reduction_results(scenarios, sbr_model, n_display=4):
    n_scenarios = len(sbr_model.selected_indices_)
    hours = scenarios[0]['Time_Hour'].values
    selected_curves = []
    for idx in sbr_model.selected_indices_:
        selected_curves.append(scenarios[idx]['Total_Load_kW'].values)

    palette = plt.get_cmap('Set1')

    if n_scenarios > 4:
        _plot_single_figure(scenarios, sbr_model, selected_curves, hours, palette, 
                           range(0, 4))
        
        _plot_single_figure(scenarios, sbr_model, selected_curves, hours, palette, 
                           range(4, min(8, n_scenarios)))
    else:
        _plot_single_figure(scenarios, sbr_model, selected_curves, hours, palette, 
                           range(n_scenarios))


def _plot_single_figure(scenarios, sbr_model, selected_curves, hours, palette, 
                        scenario_range):
    scenarios_to_plot = list(scenario_range)
    n_scenarios_plot = len(scenarios_to_plot)
    n_scenarios_total = len(sbr_model.selected_indices_)
    
    num_cols = 2
    num_rows = (n_scenarios_plot + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, scenario_idx in enumerate(scenarios_to_plot):
        ax = axes[idx]

        n_represented = (sbr_model.labels_ == scenario_idx).sum()
        prob = sbr_model.probabilities_[scenario_idx]
        actual_scenario_idx = sbr_model.selected_indices_[scenario_idx]

        for s in scenarios_to_plot:
            if s == scenario_idx:
                ax.plot(hours, selected_curves[s],
                       color=palette(scenario_idx % 9),
                       linewidth=1,
                       marker='o',
                       markersize=2,
                       label=f'Scenario {idx+1} (n={n_represented})')
            else:
                ax.plot(hours, selected_curves[s],
                       color='grey',
                       linewidth=0.6,
                       alpha=0.8)
        ax.set_title(f'Scenario {idx+1}',
                    loc='left',
                    fontsize=18 if n_scenarios_total > 4 else 24,
                    color=palette(scenario_idx % 9))
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, len(hours) - 0.5)
        ax.set_xticks(np.arange(0, len(hours), 24))
        ax.tick_params(axis='both', labelsize=14 if n_scenarios_total > 4 else 20)
        if idx < n_scenarios_plot - num_cols: 
            ax.tick_params(labelbottom=False)
        if idx % num_cols != 0:
            ax.tick_params(labelleft=False)
        ax.legend(loc='upper right', fontsize=10 if n_scenarios_total > 4 else 14)
    for i in range(n_scenarios_plot, num_rows * num_cols):
        fig.delaxes(axes[i])
    fig.text(0.5, 0.02, 'Time (h)', ha='center', fontsize=20 if n_scenarios_total > 4 else 28)
    fig.text(0.02, 0.5, 'Power (kW)', va='center', rotation='vertical', 
             fontsize=20 if n_scenarios_total > 4 else 28)
    plt.subplots_adjust(left=0.09, right=0.975, bottom=0.1, top=0.92, wspace=0.05, hspace=0.2)
    plt.show()

# ==========================================
# 7. 运行模拟与SBR场景削减分析
# ==========================================

if __name__ == "__main__":
    RANDOM_SEED = 42
    scenarios = generate_lhs_scenarios(
        n_scenarios=SimConfig.NUM_SCENARIOS, 
        random_state=RANDOM_SEED
    )
    print(f"\n示例场景统计信息 (场景0):")
    print(f"  时长: {len(scenarios[0])} 小时")
    print(f"  峰值负荷: {scenarios[0]['Total_Load_kW'].max():.2f} kW")
    print(f"  平均负荷: {scenarios[0]['Total_Load_kW'].mean():.2f} kW")
    print(f"  AES负荷占比: {scenarios[0]['AES_Load_kW'].sum() / scenarios[0]['Total_Load_kW'].sum() * 100:.2f}%")
    sbr_model, distance_matrix = reduce_scenarios_sbr(
        scenarios, 
        n_clusters=SimConfig.NUM_CLUSTERS,
        random_state=RANDOM_SEED
    )
    metrics = evaluate_reduction(sbr_model, distance_matrix)
    plot_reduction_results(scenarios, sbr_model)
    per_scn_risks, avg_risks = evaluate_risks_all_scenarios(scenarios, grid_threshold_kw=200.0)
    print("\n风险概率评估（基于定义）：")
    print(f"  平均互操作性风险概率: {avg_risks['interop']*100:.2f}%")
    print(f"  平均电网运行风险概率: {avg_risks['grid']*100:.2f}%")
    print(f"  平均期望供电不足风险: {avg_risks['expected']*100:.2f}%")
    
    print("\n削减后典型场景的单场景风险概率：")
    for k, idx in enumerate(sbr_model.selected_indices_):
        r = per_scn_risks[idx]
        print(f"  场景 {idx}: 互操作性 {r['interop']*100:.2f}%, 电网运行 {r['grid']*100:.2f}%, 期望供电不足 {r['expected']*100:.2f}%")

    # 7. 输出最终统计
    print(f"\n" + "="*60)
    print(f"SBR场景削减分析完成!")
    print(f"="*60)
    print(f"原始场景数: {SimConfig.NUM_SCENARIOS}")
    print(f"削减后场景数: {SimConfig.NUM_CLUSTERS}")
    print(f"削减率: {metrics['reduction_rate']*100:.1f}%")
    for k, idx in enumerate(sbr_model.selected_indices_):
        prob = sbr_model.probabilities_[k]
        n_represented = (sbr_model.labels_ == k).sum()
        print(f"  场景 {idx}: 概率={prob:.4f}, 代表{n_represented}个原始场景")
    print(f"="*60)
    print("\n当前渗透率下风险统计（全场景平均）:")
    print(f"  平均互操作性风险概率: {avg_risks['interop']*100:.2f}%")
    print(f"  平均电网运行风险概率: {avg_risks['grid']*100:.2f}%")
    print(f"  平均期望供电不足风险: {avg_risks['expected']*100:.2f}%")
