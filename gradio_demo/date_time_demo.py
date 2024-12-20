import gradio as gr
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px

def process_date_range(start_date, end_date):
    """处理日期范围"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        delta = end - start
        days = delta.days
        return f"选择的日期范围包含 {days} 天"
    except:
        return "请选择有效的日期范围"

def create_schedule(date, time, event):
    """创建日程"""
    if date and time and event:
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            datetime_str = f"{date_obj.strftime('%Y-%m-%d')} {time}"
            return f"已创建日程：{event} 于 {datetime_str}"
        except:
            return "日期格式无效"
    return "请填写完整的日程信息"

def generate_time_series(start_date, end_date, data_type):
    """生成时间序列数据可视化"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # 生成日期范围
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        # 生成示例数据
        if data_type == "温度":
            values = [20 + pd.np.random.normal(0, 5) for _ in range(len(date_range))]
            title = "每日温度变化"
            y_label = "温度 (°C)"
        else:
            values = [100 + pd.np.random.normal(0, 20) for _ in range(len(date_range))]
            title = "每日湿度变化"
            y_label = "湿度 (%)"
        
        # 创建数据框
        df = pd.DataFrame({
            '日期': date_range,
            y_label: values
        })
        
        # 创建图表
        fig = px.line(df, x='日期', y=y_label, title=title)
        return fig
    except:
        return None

def create_demo():
    # 自定义CSS样式
    custom_css = """
        /* 日期选择器样式 */
        input[type="date"] {
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 16px;
            width: 100%;
        }
        
        /* 信息窗口样式 */
        .date-info-popup {
            display: none;
            position: absolute;
            background: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            z-index: 1000;
        }
        
        .date-info-popup .morning {
            color: #4CAF50;
            margin-bottom: 5px;
        }
        
        .date-info-popup .afternoon {
            color: #2196F3;
            margin-bottom: 5px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .color-box {
            width: 15px;
            height: 15px;
            margin-right: 5px;
            border: 1px solid #ccc;
        }
    """
    
    # 添加日期信息弹窗的HTML和JavaScript
    date_info_html = """
        <div id="dateInfoPopup" class="date-info-popup">
            <div class="morning">
                <div class="legend-item">
                    <div class="color-box" style="background: #4CAF50;"></div>
                    <span>上午：可预约时段</span>
                </div>
            </div>
            <div class="afternoon">
                <div class="legend-item">
                    <div class="color-box" style="background: #2196F3;"></div>
                    <span>下午：会议时段</span>
                </div>
            </div>
            <div class="legend-item">
                <div class="color-box" style="background: #FFC107;"></div>
                <span>特殊日期</span>
            </div>
        </div>
        
        <script>
            function showDateInfo(event, inputId) {
                const popup = document.getElementById('dateInfoPopup');
                const input = document.getElementById(inputId);
                
                // 计算弹窗位置
                const rect = input.getBoundingClientRect();
                popup.style.left = rect.left + 'px';
                popup.style.top = (rect.bottom + 5) + 'px';
                
                // 显示弹窗
                popup.style.display = 'block';
                
                // 阻止事件冒泡
                event.stopPropagation();
            }
            
            function hideDateInfo() {
                const popup = document.getElementById('dateInfoPopup');
                popup.style.display = 'none';
            }
            
            // 为每个日期输入框添加事件监听
            function setupDateInfoEvents(inputId) {
                const input = document.getElementById(inputId);
                if (input) {
                    input.addEventListener('mousedown', (e) => showDateInfo(e, inputId));
                    input.addEventListener('touchstart', (e) => showDateInfo(e, inputId));
                    input.addEventListener('mouseup', hideDateInfo);
                    input.addEventListener('touchend', hideDateInfo);
                    input.addEventListener('mouseleave', hideDateInfo);
                }
            }
            
            // 页面加载完成后设置事件
            document.addEventListener('DOMContentLoaded', function() {
                setupDateInfoEvents('start_date');
                setupDateInfoEvents('end_date');
                setupDateInfoEvents('event_date');
                setupDateInfoEvents('ts_start_date');
                setupDateInfoEvents('ts_end_date');
                
                // 点击页面其他地方关闭弹窗
                document.addEventListener('click', function(e) {
                    if (!e.target.closest('.date-info-popup') && 
                        !e.target.closest('input[type="date"]')) {
                        hideDateInfo();
                    }
                });
            });
        </script>
    """
    
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("# 日期和时间组件演示")
        
        # 添加日期信息弹窗
        gr.HTML(date_info_html)
        
        with gr.Tab("日期范围选择"):
            with gr.Row():
                start_date = gr.Textbox(
                    label="开始日期",
                    value=datetime.now().strftime("%Y-%m-%d"),
                    elem_classes="date-input",
                    elem_id="start_date"
                )
                end_date = gr.Textbox(
                    label="结束日期",
                    value=(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    elem_classes="date-input",
                    elem_id="end_date"
                )
            
            # 添加日期选择器转换脚本
            gr.HTML("""
                <script>
                    function convertToDateInput(textboxId) {
                        const textbox = document.getElementById(textboxId);
                        if (textbox) {
                            textbox.type = 'date';
                        }
                    }
                    
                    document.addEventListener('DOMContentLoaded', function() {
                        convertToDateInput('start_date');
                        convertToDateInput('end_date');
                    });
                </script>
            """)
            
            result = gr.Textbox(label="结果")
            
            gr.Button("计算日期范围").click(
                fn=process_date_range,
                inputs=[start_date, end_date],
                outputs=result
            )
        
        with gr.Tab("日程安排"):
            with gr.Row():
                event_date = gr.Textbox(
                    label="选择日期",
                    value=datetime.now().strftime("%Y-%m-%d"),
                    elem_classes="date-input",
                    elem_id="event_date"
                )
                event_time = gr.Dropdown(
                    choices=[f"{i:02d}:00" for i in range(24)],
                    label="选择时间",
                    value="09:00"
                )
            
            # 为日程日期添加日期选择器
            gr.HTML("""
                <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        convertToDateInput('event_date');
                    });
                </script>
            """)
            
            event_name = gr.Textbox(label="事件名称", placeholder="请输入事件名称")
            schedule_result = gr.Textbox(label="日程结果")
            
            gr.Button("创建日程").click(
                fn=create_schedule,
                inputs=[event_date, event_time, event_name],
                outputs=schedule_result
            )
        
        with gr.Tab("时间序列数据"):
            with gr.Row():
                ts_start_date = gr.Textbox(
                    label="开始日期",
                    value=datetime.now().strftime("%Y-%m-%d"),
                    elem_classes="date-input",
                    elem_id="ts_start_date"
                )
                ts_end_date = gr.Textbox(
                    label="结束日期",
                    value=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    elem_classes="date-input",
                    elem_id="ts_end_date"
                )
                data_type = gr.Radio(
                    choices=["温度", "湿度"],
                    label="数据类型",
                    value="温度"
                )
            
            # 为时间序列日期添加日期选择器
            gr.HTML("""
                <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        convertToDateInput('ts_start_date');
                        convertToDateInput('ts_end_date');
                    });
                </script>
            """)
            
            plot_output = gr.Plot(label="数据可视化")
            
            gr.Button("生成图表").click(
                fn=generate_time_series,
                inputs=[ts_start_date, ts_end_date, data_type],
                outputs=plot_output
            )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch() 