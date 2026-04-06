import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def create_sample_audio():
    """
    创建一个示例音频文件用于演示
    """
    # 参数设置
    sample_rate = 44100  # CD质量采样率
    duration = 2.0       # 持续时间（秒）
    
    # 生成时间轴
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 创建复合音频信号（包含几个不同的音符）
    # C4 (261.63 Hz), E4 (329.63 Hz), G4 (392.00 Hz) 组成C大调和弦
    audio_signal = (np.sin(2 * np.pi * 261.63 * t) +      # C4
                    0.7 * np.sin(2 * np.pi * 329.63 * t) + # E4
                    0.5 * np.sin(2 * np.pi * 392.00 * t) + # G4
                    0.1 * np.random.randn(len(t)))         # 添加少量噪声
    
    # 归一化并转换为16位整数
    audio_signal = audio_signal / np.max(np.abs(audio_signal))
    audio_signal = (audio_signal * 32767).astype(np.int16)
    
    # 保存为WAV文件
    wavfile.write('sample_audio.wav', sample_rate, audio_signal)
    print("Sample audio saved as 'sample_audio.wav'")
    return sample_rate, audio_signal

def analyze_audio_spectrum(sample_rate, audio_signal, filename=None):
    """
    分析音频信号的频谱
    """
    if filename:
        sample_rate, audio_signal = wavfile.read(filename)
    
    # 如果是立体声，只取一个声道
    if len(audio_signal.shape) > 1:
        audio_signal = audio_signal[:, 0]
    
    # 取一部分信号进行分析（避免处理太大的数据）
    segment_length = min(44100, len(audio_signal))  # 最多取1秒的数据
    signal_segment = audio_signal[:segment_length]
    
    # 执行FFT
    fft_result = np.fft.fft(signal_segment)
    frequencies = np.fft.fftfreq(len(signal_segment), 1/sample_rate)
    
    # 只取正频率部分
    positive_freq_indices = frequencies >= 0
    frequencies = frequencies[positive_freq_indices]
    magnitude_spectrum = np.abs(fft_result[positive_freq_indices])
    
    # 转换为dB标度
    magnitude_db = 20 * np.log10(magnitude_spectrum + 1e-10)  # 加小值防止log(0)
    
    # 绘制结果
    plt.figure(figsize=(12, 10))
    
    # 时域信号
    time_axis = np.linspace(0, len(signal_segment)/sample_rate, len(signal_segment))
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, signal_segment)
    plt.title('Audio Signal in Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # 频域信号（线性标度）
    plt.subplot(3, 1, 2)
    plt.plot(frequencies, magnitude_spectrum)
    plt.title('Frequency Spectrum (Linear Scale)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    # 频域信号（对数标度，dB）
    plt.subplot(3, 1, 3)
    plt.plot(frequencies, magnitude_db)
    plt.title('Frequency Spectrum (Logarithmic Scale - dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    
    plt.tight_layout()
    plt.show()
    
    # 找到幅度最大的几个频率
    top_indices = np.argsort(magnitude_spectrum)[-5:][::-1]  # 获取前5个最大幅度的索引
    print("\nTop 5 frequency components:")
    for i, idx in enumerate(top_indices):
        freq = frequencies[idx]
        mag = magnitude_spectrum[idx]
        print(f"  {i+1}. Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")

def main():
    # 创建示例音频
    sample_rate, audio_signal = create_sample_audio()
    
    # 分析频谱
    analyze_audio_spectrum(sample_rate, audio_signal)

if __name__ == "__main__":
    main()