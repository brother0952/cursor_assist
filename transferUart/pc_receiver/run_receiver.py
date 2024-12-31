from receiver import main
import sys

if __name__ == "__main__":
    try:
        print("启动文件接收程序...")
        main()
    except Exception as e:
        print(f"程序异常: {str(e)}")
        sys.exit(1) 