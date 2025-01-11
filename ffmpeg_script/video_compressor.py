import os
import subprocess
import shutil
from typing import Dict, Optional

class VideoCompressor:
    def __init__(self):
        self.ffmpeg_path = r"F:\ziliao\ffmpeg\ffmpeg-20181208-6b1c4ce-win32-shared\ffmpeg-20181208-6b1c4ce-win32-shared\bin\ffmpeg.exe"
        self.input_dir = r"E:\memory\新格式\2024-12-28-清远佛冈\ffmpcom"
        self.output_dir = "output"
        self.control_fps = False
        self.has_nvidia = self._check_nvidia_support()
        self.profiles = self._init_profiles()

    def _check_nvidia_support(self) -> bool:
        """检查是否支持NVIDIA编码器"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True
            )
            return "h264_nvenc" in result.stdout
        except Exception:
            return False

    def _init_profiles(self) -> Dict:
        """初始化压缩配置文件"""
        profiles = {
            # CPU配置
            1: {
                "name": "Ultra High Quality",
                "quality": 18,
                "preset": "veryslow",
                "suffix": "_uhq",
                "encoder": "libx264"
            },
            2: {
                "name": "High Quality",
                "quality": 23,
                "preset": "slow",
                "suffix": "_hq",
                "encoder": "libx264"
            },
            3: {
                "name": "Medium Quality",
                "quality": 28,
                "preset": "medium",
                "suffix": "_mq",
                "encoder": "libx264"
            },
            4: {
                "name": "Low Quality High Compression",
                "quality": 33,
                "preset": "veryfast",
                "suffix": "_lq",
                "encoder": "libx264"
            },
            5: {
                "name": "Maximum Compression",
                "quality": 40,
                "preset": "ultrafast",
                "suffix": "_min",
                "encoder": "libx264"
            }
        }

        if self.has_nvidia:
            # GPU配置
            gpu_profiles = {
                1: {
                    "name": "Ultra High Quality (GPU)",
                    "quality": 18,
                    "preset": "medium",
                    "suffix": "_uhq_gpu",
                    "encoder": "h264_nvenc"
                },
                2: {
                    "name": "High Quality (GPU)",
                    "quality": 23,
                    "preset": "medium",
                    "suffix": "_hq_gpu",
                    "encoder": "h264_nvenc"
                },
                3: {
                    "name": "Medium Quality (GPU)",
                    "quality": 28,
                    "preset": "medium",
                    "suffix": "_mq_gpu",
                    "encoder": "h264_nvenc"
                },
                4: {
                    "name": "Low Quality High Compression (GPU)",
                    "quality": 33,
                    "preset": "fast",
                    "suffix": "_lq_gpu",
                    "encoder": "h264_nvenc"
                },
                5: {
                    "name": "Maximum Compression (GPU)",
                    "quality": 40,
                    "preset": "fast",
                    "suffix": "_min_gpu",
                    "encoder": "h264_nvenc"
                }
            }
            # 添加CPU配置作为额外选项
            gpu_profiles[6] = profiles[1].copy()
            gpu_profiles[6]["name"] = "Ultra High Quality (CPU)"
            gpu_profiles[6]["suffix"] = "_uhq_cpu"
            
            gpu_profiles[7] = profiles[2].copy()
            gpu_profiles[7]["name"] = "High Quality (CPU)"
            gpu_profiles[7]["suffix"] = "_hq_cpu"
            
            return gpu_profiles
        return profiles

    def _get_video_info(self, video_path: str) -> Optional[Dict]:
        """获取视频信息"""
        try:
            # 获取FPS和比特率
            result = subprocess.run(
                [self.ffmpeg_path, "-i", video_path],
                capture_output=True,
                text=True
            )
            
            info = {}
            output = result.stderr
            
            # 解析FPS和比特率
            for line in output.split('\n'):
                # 查找包含 fps 的行
                if "fps" in line:
                    # 更健壮的FPS提取
                    try:
                        parts = line.split(',')
                        for part in parts:
                            if 'fps' in part:
                                fps_str = ''.join(filter(lambda x: x.isdigit() or x == '.', part))
                                info['fps'] = float(fps_str)
                                break
                    except (ValueError, IndexError):
                        continue

                # 查找包含 bitrate 的行
                if "bitrate" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "kb/s":
                                info['bitrate'] = int(parts[i-1])
                                break
                    except (ValueError, IndexError):
                        continue
            
            # 获取文件大小
            info['size'] = os.path.getsize(video_path)
            
            # 确保所有必要的信息都存在
            if 'fps' not in info or 'bitrate' not in info:
                print(f"Warning: Could not extract complete video info from {video_path}")
                info['fps'] = info.get('fps', 30)  # 默认30fps
                info['bitrate'] = info.get('bitrate', 5000)  # 默认5Mbps
            
            print(f"Debug - Extracted info: FPS={info.get('fps')}, Bitrate={info.get('bitrate')}kb/s")
            return info
        except Exception as e:
            print(f"Error getting video info: {str(e)}")
            return None

    def _compress_video(self, input_path: str, output_path: str, profile: Dict, video_info: Dict):
        """压缩视频"""
        target_bitrate = min(video_info['bitrate'] // 2000, 15)  # Convert to Mbps and cap at 15
        
        command = []

        if profile["encoder"] == "h264_nvenc":
            # GPU encoding - hwaccel must come before input
            command.extend([
                self.ffmpeg_path,
                "-hwaccel", "cuda",
                "-i", input_path,
                "-c:v", "h264_nvenc",
                "-preset", profile["preset"],
                "-rc:v", "vbr_hq",
                "-cq:v", str(profile["quality"]),
                "-b:v", f"{target_bitrate}M",
                "-maxrate:v", f"{target_bitrate}M",
                "-rc-lookahead", "32",
                "-spatial-aq", "1",
                "-aq-strength", "8"
            ])
        else:
            # CPU encoding
            command.extend([
                self.ffmpeg_path,
                "-i", input_path,
                "-c:v", "libx264",
                "-crf", str(profile["quality"]),
                "-maxrate:v", f"{target_bitrate}M",
                "-bufsize", f"{target_bitrate}M",
                "-preset", profile["preset"]
            ])

        if self.control_fps and video_info['fps'] >= 50:
            command.extend(["-r", "30"])

        command.extend([
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-y",
            output_path
        ])

        try:
            print("Debug - Running command:", " ".join(command))  # 添加调试输出
            subprocess.run(command, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error compressing video: {e}")
            if e.stderr:
                print(f"FFmpeg error output: {e.stderr}")  # 添加详细错误信息
            return False

    def show_menu(self):
        """显示菜单"""
        print("\nFPS Control Settings")
        print("=====================================")
        print(f"Current Status: {'On' if self.control_fps else 'Off'}")
        print("[0] Off - Keep Original FPS")
        print("[1] On - Auto reduce 60FPS to 30FPS")
        
        choice = input("\nSelect FPS control mode (0/1): ")
        self.control_fps = choice == "1"

        print("\nVideo Compression Tool")
        print("=====================================")
        print("Select compression profile:\n")

        if self.has_nvidia:
            print("=== GPU Acceleration Mode (NVIDIA) ===")
        else:
            print("=== CPU Mode ===")

        for i, profile in self.profiles.items():
            print(f"{i}. {profile['name']}")
            print(f"   - Quality: {'Best' if profile['quality'] == 18 else 'Very Good' if profile['quality'] == 23 else 'Good' if profile['quality'] == 28 else 'Fair' if profile['quality'] == 33 else 'Poor'}")
            print(f"   - Preset: {profile['preset']}")
            print(f"   - Compression ratio: {0.6 if profile['quality'] == 18 else 0.4 if profile['quality'] == 23 else 0.3 if profile['quality'] == 28 else 0.2 if profile['quality'] == 33 else 0.1}-{0.7 if profile['quality'] == 18 else 0.5 if profile['quality'] == 23 else 0.4 if profile['quality'] == 28 else 0.3 if profile['quality'] == 33 else 0.2}x")
            print()

        print("0. Exit")
        print("=====================================")

        while True:
            try:
                choice = int(input(f"Enter your choice (0-{len(self.profiles)}): "))
                if choice == 0:
                    return None
                if choice in self.profiles:
                    return self.profiles[choice]
                print("Invalid choice, please try again")
            except ValueError:
                print("Invalid input, please enter a number")

    def process_videos(self):
        """处理视频"""
        if not os.path.exists(self.input_dir):
            print(f"Error: Input directory '{self.input_dir}' does not exist.")
            return

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        video_files = []
        for ext in ['.mp4', '.avi', '.mov']:
            video_files.extend(glob.glob(os.path.join(self.input_dir, f"*{ext}")))

        if not video_files:
            print("No video files found in input directory.")
            return

        profile = self.show_menu()
        if not profile:
            return

        print(f"\nSelected: {profile['name']}")
        print("Starting video processing...\n")

        for video_path in video_files:
            filename = os.path.basename(video_path)
            print(f"Processing: {filename}")

            video_info = self._get_video_info(video_path)
            if not video_info:
                continue

            output_path = os.path.join(
                self.output_dir,
                f"{os.path.splitext(filename)[0]}{profile['suffix']}{os.path.splitext(filename)[1]}"
            )

            if video_info['bitrate'] <= 2000:  # 2 Mbps
                print("Original bitrate too low, skipping compression")
                continue

            print(f"Original FPS: {video_info['fps']}")
            print(f"Original bitrate: {video_info['bitrate'] // 1000} Mbps")
            print(f"Original size: {video_info['size']} bytes")

            if self._compress_video(video_path, output_path, profile, video_info):
                if os.path.exists(output_path):
                    new_size = os.path.getsize(output_path)
                    if new_size >= video_info['size']:
                        print("Warning: Compressed file is larger, keeping original")
                        os.remove(output_path)
                    else:
                        saved_space = video_info['size'] - new_size
                        saved_ratio = saved_space * 100 / video_info['size']
                        print(f"Space saved ratio: {saved_ratio/100:.2f}x")

            print(f"Completed: {filename}\n")

        print("All videos processed!")

if __name__ == "__main__":
    import glob
    compressor = VideoCompressor()
    while True:
        compressor.process_videos()
        if input("\nPress Enter to continue or type 'exit' to quit: ").lower() == 'exit':
            break 