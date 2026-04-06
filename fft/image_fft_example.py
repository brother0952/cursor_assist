import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def image_frequency_analysis(image_path=None):
    """
    对图像进行频域分析
    """
    # 如果没有提供图像，则创建一个示例图像
    if image_path is None:
        # 创建一个带有一些几何形状的示例图像
        img = np.zeros((256, 256), dtype=np.uint8)
        
        # 添加一些形状
        cv2.rectangle(img, (50, 50), (100, 100), 255, -1)  # 白色方块
        cv2.circle(img, (150, 150), 30, 255, -1)           # 白色圆圈
        cv2.line(img, (200, 50), (200, 150), 255, 5)       # 白线
        
        # 保存示例图像
        Image.fromarray(img).save('sample_image.png')
        print("Sample image saved as 'sample_image.png'")
    else:
        # 读取提供的图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
    
    # 执行二维FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # 将零频率分量移到中心
    
    # 计算频谱幅度
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # 创建理想低通滤波器
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  # 中心点
    
    # 创建掩码（理想低通滤波器）
    mask = np.zeros((rows, cols), np.uint8)
    r = 30  # 半径
    cv2.circle(mask, (ccol, crow), r, 1, -1)
    
    # 应用掩码
    fshift_filtered = fshift * mask
    magnitude_spectrum_filtered = 20 * np.log(np.abs(fshift_filtered) + 1)
    
    # 反变换回空间域
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Ideal Low Pass Filter Mask')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(magnitude_spectrum_filtered, cmap='gray')
    plt.title('Filtered Magnitude Spectrum')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(img_back, cmap='gray')
    plt.title('Image after Filtering')
    plt.axis('off')
    
    # 显示水平方向上的频谱剖面
    plt.subplot(2, 3, 6)
    middle_row = magnitude_spectrum.shape[0] // 2
    plt.plot(magnitude_spectrum[middle_row, :])
    plt.title('Horizontal Profile of Magnitude Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude (dB)')
    
    plt.tight_layout()
    plt.show()

def compare_filters():
    """
    比较不同类型的滤波器效果
    """
    # 创建示例图像
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (100, 100), 255, -1)
    cv2.circle(img, (150, 150), 30, 255, -1)
    cv2.line(img, (200, 50), (200, 150), 255, 5)
    
    # 执行FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # 创建不同类型的滤波器
    # 1. 理想低通滤波器
    mask_ideal = np.zeros((rows, cols), np.float32)
    cv2.circle(mask_ideal, (ccol, crow), 30, 1, -1)
    
    # 2. 巴特沃斯低通滤波器
    def butterworth_lowpass_filter(shape, cutoff, order):
        P, Q = shape
        U, V = np.meshgrid(range(P), range(Q), indexing='ij')
        D = np.sqrt((U - P//2)**2 + (V - Q//2)**2)
        H = 1 / (1 + (D/cutoff)**(2*order))
        return H
    
    mask_butterworth = butterworth_lowpass_filter((rows, cols), 30, 2)
    
    # 3. 高斯低通滤波器
    def gaussian_lowpass_filter(shape, sigma):
        P, Q = shape
        U, V = np.meshgrid(range(P), range(Q), indexing='ij')
        D = (U - P//2)**2 + (V - Q//2)**2
        H = np.exp(-D/(2*sigma**2))
        return H
    
    mask_gaussian = gaussian_lowpass_filter((rows, cols), 30)
    
    # 应用滤波器
    filtered_ideal = fshift * mask_ideal
    filtered_butterworth = fshift * mask_butterworth
    filtered_gaussian = fshift * mask_gaussian
    
    # 反变换
    img_ideal = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_ideal)))
    img_butterworth = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_butterworth)))
    img_gaussian = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_gaussian)))
    
    # 显示结果
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(mask_ideal, cmap='gray')
    plt.title('Ideal LPF')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(img_ideal, cmap='gray')
    plt.title('Ideal LPF Result')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(np.log(1 + np.abs(filtered_ideal)), cmap='gray')
    plt.title('Ideal LPF Spectrum')
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(mask_butterworth, cmap='gray')
    plt.title('Butterworth LPF')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(img_butterworth, cmap='gray')
    plt.title('Butterworth LPF Result')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(np.log(1 + np.abs(filtered_butterworth)), cmap='gray')
    plt.title('Butterworth LPF Spectrum')
    plt.axis('off')
    
    plt.subplot(3, 4, 9)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(mask_gaussian, cmap='gray')
    plt.title('Gaussian LPF')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.imshow(img_gaussian, cmap='gray')
    plt.title('Gaussian LPF Result')
    plt.axis('off')
    
    plt.subplot(3, 4, 12)
    plt.imshow(np.log(1 + np.abs(filtered_gaussian)), cmap='gray')
    plt.title('Gaussian LPF Spectrum')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Running image frequency analysis...")
    image_frequency_analysis()
    
    print("\nComparing different filters...")
    compare_filters()

if __name__ == "__main__":
    main()