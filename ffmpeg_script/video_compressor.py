def __init__(self):
    self.config = self._load_config()
    self.ffmpeg_path = self.config.get('FFmpeg', 'ffmpeg_path')
    self.input_dir = self._get_input_dir()
    self.control_fps = False
    self.has_nvidia = self._check_nvidia_support()
    self.profiles = self._init_profiles()
    self.delete_original = self.config.getboolean('Compression', 'delete_original', fallback=False)

def process_videos(self):
    """处理视频"""
    if not os.path.exists(self.input_dir):
        print(f"Error: Input directory '{self.input_dir}' does not exist.")
        return

    # 获取所有可能的压缩后缀
    all_suffixes = set()
    for profile in self.profiles.values():
        all_suffixes.add(profile['suffix'])

    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        files = glob.glob(os.path.join(self.input_dir, f"*{ext}"))
        # 过滤掉已经压缩过的文件（文件名中包含任何压缩后缀的文件）
        for file in files:
            filename = os.path.basename(file)
            if not any(suffix in filename for suffix in all_suffixes):
                video_files.append(file)

    if not video_files:
        print("No uncompressed video files found in input directory.")
        return

    # 使用配置文件中的默认配置
    default_profile_num = int(self.config.get('Compression', 'default_profile'))
    if default_profile_num not in self.profiles:
        print(f"Error: Invalid default profile number {default_profile_num}")
        return
            
    profile = self.profiles[default_profile_num]
    print(f"\nUsing default profile: {profile['name']}")
    if self.delete_original:
        print("Warning: Original files will be deleted after successful compression")
    print("Starting video processing...\n")

    for video_path in video_files:
        filename = os.path.basename(video_path)
        dirname = os.path.dirname(video_path)
        base_name, ext = os.path.splitext(filename)
        
        # 检查是否已经存在压缩过的文件
        compressed_filename = f"{base_name}{profile['suffix']}{ext}"
        output_path = os.path.join(dirname, compressed_filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {filename} - compressed version already exists")
            continue

        print(f"Processing: {filename}")

        video_info = self._get_video_info(video_path)
        if not video_info:
            continue

        if video_info['bitrate'] <= 2000:  # 2 Mbps
            print("Original bitrate too low, skipping compression")
            continue

        print(f"Original FPS: {video_info['fps']}")
        print(f"Original bitrate: {video_info['bitrate'] // 1000} Mbps")
        print(f"Original size: {video_info['size']} bytes")

        compression_successful = False
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
                    print(f"Compressed file saved as: {compressed_filename}")
                    compression_successful = True

        # 如果压缩成功且配置为删除原文件
        if compression_successful and self.delete_original:
            try:
                os.remove(video_path)
                print(f"Original file deleted: {filename}")
            except Exception as e:
                print(f"Warning: Could not delete original file: {str(e)}")

        print(f"Completed: {filename}\n") 

    def _get_video_info(self, video_path: str) -> Optional[Dict]:
        """获取视频信息"""
        try:
            # 获取FPS和比特率
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            result = subprocess.run(
                [self.ffmpeg_path, "-i", video_path],
                capture_output=True,
                encoding='utf-8',  # 使用UTF-8编码
                errors='replace',   # 处理无法解码的字符
                startupinfo=startupinfo
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
                                if fps_str:
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
                                bitrate_str = parts[i-1]
                                if bitrate_str.isdigit():
                                    info['bitrate'] = int(bitrate_str)
                                    break
                    except (ValueError, IndexError):
                        continue
            
            # 获取文件大小
            try:
                info['size'] = os.path.getsize(video_path)
            except OSError as e:
                print(f"Error getting file size: {e}")
                return None
            
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
        # 从配置文件读取质量因子
        quality_factor = {
            18: float(self.config.get('Quality', 'uhq_factor')),
            23: float(self.config.get('Quality', 'hq_factor')),
            28: float(self.config.get('Quality', 'mq_factor')),
            33: float(self.config.get('Quality', 'lq_factor')),
            40: float(self.config.get('Quality', 'min_factor')),
        }
        
        # 从配置文件读取码率限制
        min_bitrate = int(self.config.get('Bitrate', 'min_bitrate'))
        max_bitrate = int(self.config.get('Bitrate', 'max_bitrate'))
        
        # 计算目标码率，但设置上下限
        target_bitrate = int(video_info['bitrate'] * quality_factor.get(profile["quality"], 0.5) / 1000)
        target_bitrate = max(min(target_bitrate, max_bitrate), min_bitrate)
        
        command = []

        if profile["encoder"] == "h264_nvenc":
            # GPU encoding with quality-focused parameters
            command.extend([
                self.ffmpeg_path,
                "-hwaccel", "cuda",
                "-i", input_path,
                "-c:v", "h264_nvenc",
                "-preset", profile["preset"],
                "-rc:v", "vbr_hq",
                "-cq:v", str(profile["quality"]),  # 使用固定质量参数
                "-b:v", f"{target_bitrate}M",
                "-maxrate:v", f"{int(target_bitrate * 1.2)}M",  # 稍微提高峰值码率
                "-bufsize", f"{target_bitrate}M",
                "-profile:v", "high",
                "-rc-lookahead", "32",
                "-spatial-aq", "1",
                "-temporal-aq", "1",
                "-aq-strength", "8",
                "-refs", "4"  # 减少参考帧数以降低文件大小
            ])
        else:
            # CPU encoding with quality-focused parameters
            command.extend([
                self.ffmpeg_path,
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", profile["preset"],
                "-crf", str(profile["quality"]),
                "-maxrate:v", f"{target_bitrate}M",
                "-bufsize", f"{target_bitrate}M",
                "-profile:v", "high",
                "-level", "4.1",
                "-movflags", "+faststart",
                "-x264-params", "ref=4:me=hex:subme=6:rc-lookahead=30:me-range=16:chroma-qp-offset=0:bframes=3:b-adapt=1"
            ])

        # 不再强制更改帧率，保持原始帧率
        if self.control_fps and video_info['fps'] >= 50:
            print(f"High FPS detected ({video_info['fps']}), but keeping original frame rate as requested")

        command.extend([
            "-c:a", "copy",  # 复制音频流，不重新编码
            "-y",
            output_path
        ])

        try:
            print("\nDebug - Compression settings:")
            print(f"Original bitrate: {video_info['bitrate']} kb/s ({video_info['bitrate']/1000:.1f} Mbps)")
            print(f"Target bitrate: {target_bitrate} Mbps")
            print(f"Quality factor: {quality_factor.get(profile['quality'], 0.5)}")
            print(f"Original FPS: {video_info['fps']}")
            print("Debug - Running command:", " ".join(command))
            
            # 在Windows上使用startupinfo来隐藏命令窗口
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            result = subprocess.run(
                command,
                capture_output=True,
                encoding='utf-8',
                errors='replace',
                startupinfo=startupinfo
            )
            
            if result.returncode != 0:
                print(f"FFmpeg error output: {result.stderr}")
                return False
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error compressing video: {e}")
            if e.stderr:
                print(f"FFmpeg error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error during compression: {str(e)}")
            return False 