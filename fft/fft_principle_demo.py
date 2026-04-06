import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def demonstrate_fft_principle():
    """
    演示FFT的基本原理
    展示如何将复杂信号分解为简单的正弦波
    """
    # 创建时间轴
    t = np.linspace(0, 1, 1000)
    
    # 创建复合信号
    fundamental_freq = 5  # 基频
    signal = (np.sin(2 * np.pi * fundamental_freq * t) +              # 基频
              0.5 * np.sin(2 * np.pi * 2 * fundamental_freq * t) +    # 二次谐波
              0.3 * np.sin(2 * np.pi * 3 * fundamental_freq * t) +    # 三次谐波
              0.2 * np.sin(2 * np.pi * 5 * fundamental_freq * t))     # 五次谐波
    
    # 添加一些噪声
    signal += 0.1 * np.random.randn(len(signal))
    
    # 执行FFT
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(t), t[1] - t[0])
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制原始信号
    ax1.plot(t, signal, 'b-', linewidth=1, label='Composite Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time Domain Signal')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制频谱
    ax2.plot(frequencies[:len(frequencies)//2], 
             np.abs(fft_result[:len(frequencies)//2]), 'r-', linewidth=1)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Frequency Domain Representation (FFT)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 输出分析结果
    print("Signal Components Analysis:")
    print("=" * 40)
    
    # 找出主要频率成分
    magnitudes = np.abs(fft_result[:len(frequencies)//2])
    main_freq_indices = np.where(magnitudes > np.max(magnitudes) * 0.1)[0]  # 幅度大于最大值10%的频率
    
    for i in main_freq_indices[:5]:  # 只显示前5个主要频率
        freq = frequencies[i]
        mag = magnitudes[i]
        # 计算相位
        phase = np.angle(fft_result[i])
        print(f"Frequency: {freq:6.2f} Hz | Magnitude: {mag:6.2f} | Phase: {phase:6.2f} rad")

def reconstruct_signal():
    """
    演示如何从频域重建时域信号
    """
    # 创建时间轴
    t = np.linspace(0, 1, 1000)
    
    # 原始信号
    original = (np.sin(2 * np.pi * 5 * t) + 
                0.5 * np.sin(2 * np.pi * 10 * t) + 
                0.3 * np.sin(2 * np.pi * 15 * t))
    
    # 执行FFT
    fft_coeffs = np.fft.fft(original)
    frequencies = np.fft.fftfreq(len(t), t[1] - t[0])
    
    # 逐步重建信号
    plt.figure(figsize=(12, 10))
    
    # 显示原始信号
    plt.subplot(3, 2, 1)
    plt.plot(t, original, 'k-', linewidth=1)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # 显示频谱
    plt.subplot(3, 2, 2)
    plt.plot(frequencies[:len(frequencies)//2], 
             np.abs(fft_coeffs[:len(frequencies)//2]), 'r-', linewidth=1)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    # 选择要重建的频率数量
    num_components = [1, 3, 5, 10, 20]
    
    for i, n_components in enumerate(num_components):
        # 找到幅度最大的n_components个频率成分
        magnitudes = np.abs(fft_coeffs)
        indices = np.argsort(magnitudes)[-n_components:]
        
        # 创建只包含这些频率成分的系数数组
        filtered_coeffs = np.zeros_like(fft_coeffs)
        filtered_coeffs[indices] = fft_coeffs[indices]
        
        # 反变换重建信号
        reconstructed = np.real(np.fft.ifft(filtered_coeffs))
        
        # 绘制重建信号
        plt.subplot(3, 2, i+3)
        plt.plot(t, original, 'k--', alpha=0.5, label='Original', linewidth=1)
        plt.plot(t, reconstructed, 'b-', label=f'Reconstructed ({n_components} components)', linewidth=1)
        plt.title(f'Signal Reconstruction with {n_components} Components')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def windowing_effect():
    """
    演示窗函数对FFT结果的影响
    """
    # 创建信号
    t = np.linspace(0, 2, 2000)
    signal = np.sin(2 * np.pi * 10 * t)  # 简单正弦波
    
    # 截取一部分信号（非整数周期）
    t_truncated = t[:1234]  # 非周期截断
    signal_truncated = signal[:1234]
    
    # 应用不同的窗函数
    window_rect = np.ones(len(signal_truncated))  # 矩形窗（无窗）
    window_hann = np.hanning(len(signal_truncated))  # Hann窗
    window_hamming = np.hamming(len(signal_truncated))  # Hamming窗
    
    signals_windowed = [
        signal_truncated * window_rect,
        signal_truncated * window_hann,
        signal_truncated * window_hamming
    ]
    
    window_names = ['Rectangular Window', 'Hann Window', 'Hamming Window']
    
    # 计算各自的FFT
    plt.figure(figsize=(15, 5))
    
    for i, (sig_win, win_name) in enumerate(zip(signals_windowed, window_names)):
        fft_result = np.fft.fft(sig_win)
        frequencies = np.fft.fftfreq(len(sig_win), t_truncated[1] - t_truncated[0])
        
        plt.subplot(1, 3, i+1)
        plt.plot(frequencies[:len(frequencies)//2], 
                 20 * np.log10(np.abs(fft_result[:len(frequencies)//2]) + 1e-10))
        plt.title(f'FFT with {win_name}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Demonstrating FFT Principle...")
    demonstrate_fft_principle()
    
    print("\nReconstructing Signals from Frequency Components...")
    reconstruct_signal()
    
    print("\nShowing Effect of Window Functions...")
    windowing_effect()

if __name__ == "__main__":
    main()