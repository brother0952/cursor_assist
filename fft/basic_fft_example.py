import numpy as np
import matplotlib.pyplot as plt

def basic_signal_analysis():
    """
    基础信号分析示例
    展示如何对简单正弦波进行FFT分析
    """
    # 创建时间轴
    t = np.linspace(0, 1, 1000)
    
    # 创建复合信号（包含多个频率成分）
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 15 * t)
    
    # 执行快速傅里叶变换
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(t), t[1] - t[0])
    
    # 绘制原始信号
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t[:100], signal[:100])  # 只显示前100个点
    plt.title('Original Signal (Time Domain)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # 绘制频谱图（完整）
    plt.subplot(2, 2, 2)
    plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result[:len(frequencies)//2]))
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    # 绘制频谱图（放大低频部分）
    plt.subplot(2, 2, 3)
    low_freq_indices = frequencies > 0
    low_freq_indices &= frequencies < 20
    plt.plot(frequencies[low_freq_indices], np.abs(fft_result[low_freq_indices]))
    plt.title('Frequency Spectrum (Low Frequency Detail)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    # 绘制相位图
    plt.subplot(2, 2, 4)
    plt.plot(frequencies[low_freq_indices], np.angle(fft_result[low_freq_indices]))
    plt.title('Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    basic_signal_analysis()