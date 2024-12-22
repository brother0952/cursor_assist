// index.js
Page({
  data: {
    weekdays: ['日', '一', '二', '三', '四', '五', '六'],
    days: [],
    startDate: null,
    endDate: null,
    currentMonth: '',
    balance: 0,
    plan: {
      meals11: 0,
      meals15: 0,
      remaining: 0,
      totalMeals: 0
    },
    meals: [], // 存储日期范围和消费计划的数组
    todayIndex: null, // 今天在数组中的索引
    remainingWorkdays: 0, // 到下一个结算日的工作日数量
    debug: true, // 调试开关
    useCustomTime: false,  // 是否使用自定义时间
    customTime: 13,       // 自定义时间（小时）
    showPopup: false,
    popupInfo: {
      date: '',
      morning: '',
      afternoon: '',
      x: 0,
      y: 0
    },
    useCustomDate: false,  // 使用自定义日期
    customDate: '',        // 自定义日期
    holidays: [
      '2024-01-01', // 元旦
      '2024-02-10', // 春节
      '2024-02-11',
      '2024-02-12',
      '2024-02-13',
      '2024-02-14',
      '2024-02-15',
      '2024-02-16',
      '2024-02-17',
      '2024-04-04', // 清明节
      '2024-05-01', // 劳动节
      '2024-06-10', // 端午节
      '2025-01-01', // 元旦
    ]
  },

  onLoad() {
    // 1. 初始化日期范围和消费计划数组
    this.initMealsArray();
  },

  // 1. 初始化日期范围和消费计划数组
  initMealsArray() {
    const today = this.getCurrentDate();
    const currentHour = this.getCurrentHour();
    
    let startDate, endDate;

    // 判断今天的日期
    if (today.getDate() > 20) {
      startDate = new Date(today.getFullYear(), today.getMonth(), 21);
      endDate = new Date(today.getFullYear(), today.getMonth() + 1, 20);
    } else {
      startDate = new Date(today.getFullYear(), today.getMonth() - 1, 21);
      endDate = new Date(today.getFullYear(), today.getMonth(), 20);
    }

    if (this.data.debug) {
      console.log('日期范围计算 >>>>>>>>>>>>');
      console.log(JSON.stringify({
        今天: {
          日期: today.toLocaleDateString(),
          小时: currentHour,
          日期数字: today.getDate()
        },
        开始日期: startDate.toLocaleDateString(),
        结束日期: endDate.toLocaleDateString(),
        计算过程: {
          是否大于20日: today.getDate() > 20,
          月份处理: today.getDate() > 20 ? '使用本月21日到下月20日' : '使用上月21日到本月20日'
        }
      }, null, 2));
    }

    // 计算天数
    const days = Math.floor((endDate - startDate) / (24 * 60 * 60 * 1000)) + 1;
    
    // 初始化消费计划数组
    const meals = new Array(days).fill(null).map(() => [0, 0]);

    // 先计算今天的索引
    const todayIndex = Math.floor((today - startDate) / (24 * 60 * 60 * 1000));
    let workdayCount = 0;

    let currentDate = new Date(startDate.getTime());
    while (currentDate <= endDate) {
      const index = Math.floor((currentDate - startDate) / (24 * 60 * 60 * 1000));
      
      if (index >= 0 && index < days) {
        const dayOfWeek = currentDate.getDay();
        const dateStr = this.formatDate(currentDate);
        
        // 修改日期比较逻辑
        const isPast = this.compareDates(currentDate, today) < 0;
        
        if (isPast) {
          meals[index] = [-1, -1]; // 过去的日期
        } else if (dayOfWeek === 0 || dayOfWeek === 6 || this.data.holidays.includes(dateStr)) {
          meals[index] = [-2, -2]; // 节假日
        } else if (this.isSameDay(currentDate, today)) {
          meals[index] = this.getTodayMealStatus(currentHour);
        } else {
          meals[index] = [0, 0]; // 未来工作日
          workdayCount++;
        }
      }
      
      currentDate.setDate(currentDate.getDate() + 1);
    }

    if (this.data.debug) {
      console.log('初始化消费计划数组:', JSON.stringify({
        meals: meals,
        todayIndex: todayIndex,
        workdayCount: workdayCount
      }, null, 2));
    }

    this.setData({ 
      meals, 
      todayIndex, 
      remainingWorkdays: workdayCount 
    });

    // 初始化日历显示
    this.initCalendar();
  },

  // 2. 处理用户输入金额，计算最优方案
  onBalanceInput(e) {
    const balance = Number(e.detail.value);
    this.setData({ balance });

    if (balance > 0) {
      this.calculateOptimalPlan(balance);
    } else {
      this.resetMealsArray();
    }
  },

  // 计算最优消费方案
  calculateOptimalPlan(balance) {
    // 在计算最优方案前，先打印当前状态
    if (this.data.debug) {
      console.log('计算最优方案开始 >>>>>>>>>>>>');
      console.log(JSON.stringify({
        balance: balance,
        todayIndex: this.data.todayIndex,
        meals数组长度: this.data.meals.length,
        可用时间槽统计: this.data.meals.reduce((slots, day) => {
          return {
            morning: slots.morning + (day[0] >= 0 ? 1 : 0),
            afternoon: slots.afternoon + (day[1] >= 0 ? 1 : 0)
          };
        }, { morning: 0, afternoon: 0 })
      }, null, 2));
    }

    let bestPlan = {
      meals11: 0,
      meals15: 0,
      remaining: balance,
      totalMeals: 0
    };
    
    // 计算可用的时间槽（分别计算上午和下午的可用槽）
    const availableSlots = this.data.meals.reduce((slots, day) => {
      return {
        morning: slots.morning + (day[0] >= 0 ? 1 : 0),
        afternoon: slots.afternoon + (day[1] >= 0 ? 1 : 0)
      };
    }, { morning: 0, afternoon: 0 });

    // 只有在有可用时间槽时才计算
    if (availableSlots.morning === 0 && availableSlots.afternoon === 0) {
      console.warn("没有可用的消费计划时间槽，详细信息：", JSON.stringify({
        todayIndex: this.data.todayIndex,
        meals: this.data.meals,
        availableSlots: availableSlots
      }, null, 2));
      return;
    }

    // 计算最优方案
    for (let meals15 = 0; meals15 <= availableSlots.morning + availableSlots.afternoon; meals15++) {
      for (let meals11 = 0; meals11 <= availableSlots.morning + availableSlots.afternoon - meals15; meals11++) {
        const total = meals15 * 15 + meals11 * 11;
        const remaining = balance - total;
        
        if (remaining >= 0 && remaining < bestPlan.remaining) {
          bestPlan = {
            meals15,
            meals11,
            remaining,
            totalMeals: meals15 + meals11
          };
        }
      }
    }

    this.setData({ plan: bestPlan });
    this.updateMealsWithPlan(bestPlan);
  },

  // 3. 更新日历显示
  updateCalendarDisplay() {
    const today = this.getCurrentDate();
    let startDate, endDate;

    if (today.getDate() > 20) {
      startDate = new Date(today.getFullYear(), today.getMonth(), 21);
      endDate = new Date(today.getFullYear(), today.getMonth() + 1, 20);
    } else {
      startDate = new Date(today.getFullYear(), today.getMonth() - 1, 21);
      endDate = new Date(today.getFullYear(), today.getMonth(), 20);
    }

    const days = this.data.days.map((day, index) => {
      if (!day.isInRange || !day.day) {
        return day;
      }

      const currentDate = new Date(startDate.getFullYear(), startDate.getMonth(), day.day);
      if (day.day < 21) {
        currentDate.setMonth(currentDate.getMonth() + 1);
      }
      
      const dayIndex = Math.floor((currentDate - startDate) / (24 * 60 * 60 * 1000));
      const mealDay = this.data.meals[dayIndex];
      
      if (!mealDay) return day;

      const newDay = { ...day };
      newDay.morning = this.getMealStatus(mealDay[0]);
      newDay.afternoon = this.getMealStatus(mealDay[1]);

      if (this.data.debug) {
        console.log(JSON.stringify({
          日期: day.day,
          meals值: mealDay,
          显示状态: {
            morning: newDay.morning,
            afternoon: newDay.afternoon
          }
        }, null, 2));
      }
      
      return newDay;
    });
    
    this.setData({ days });
  },

  // 获取餐食状态的显示类型
  getMealStatus(value) {
    switch (value) {
      case -2: return 'holiday';  // 节假日
      case -1: return 'past';     // 过去日期
      case 11: return 'meal11';   // 11元餐
      case 15: return 'meal15';   // 15元餐
      case 0: return 'off';       // 未计划
      default: return 'off';
    }
  },

  // 添加新方法：根据当前时间判断今天的餐食状态
  getTodayMealStatus(currentHour) {
    if (currentHour >= 17) {
      // 下午5点后，全天结束
      return [-1, -1];
    } else if (currentHour >= 11) {
      // 上午11点后，上午结束，下午可用
      return [-1, 0];
    } else {
      // 上午11点前，全天可用
      return [0, 0];
    }
  },

  // 重置消费计划数组
  resetMealsArray() {
    const meals = this.data.meals.map(day => {
      // 如果是过去的日期(-1)或节假日(-2)，保持不变
      if (day[0] === -1 || day[0] === -2) {
        return [...day]
      }
      // 其他日期重置为0
      return [0, 0]
    })
    this.setData({ meals })
    this.updateCalendarDisplay()
  },

  // 根据计算结果更新消费计划数组
  updateMealsWithPlan(plan) {
    let remainingMeals15 = plan.meals15;
    let remainingMeals11 = plan.meals11;
    const meals = [...this.data.meals];
    
    if (this.data.debug) {
      console.log('更新消费计划开始 >>>>>>>>>>>>');
      console.log(JSON.stringify({
        plan: plan,
        todayIndex: this.data.todayIndex,
        meals: meals
      }, null, 2));
    }

    // 第一轮：优先安排15元餐到上午时段
    for (let i = this.data.todayIndex; i < meals.length && remainingMeals15 > 0; i++) {
      if (meals[i] && meals[i][0] >= 0) { // 只处理上午可用的时段
        meals[i][0] = 15;
        remainingMeals15--;
        continue; // 找到一个位置后，继续下一个日期
      }
    }
    
    // 第二轮：剩余的15元餐安排到下午时段
    for (let i = this.data.todayIndex; i < meals.length && remainingMeals15 > 0; i++) {
      if (meals[i] && meals[i][1] >= 0) {
        meals[i][1] = 15;
        remainingMeals15--;
        continue;
      }
    }
    
    // 第三轮：安排11元餐到上午剩余时段
    for (let i = this.data.todayIndex; i < meals.length && remainingMeals11 > 0; i++) {
      if (meals[i] && meals[i][0] >= 0 && meals[i][0] !== 15) { // 确保不覆盖15元餐
        meals[i][0] = 11;
        remainingMeals11--;
      }
    }
    
    // 第四轮：安排剩余的11元餐到下午时段
    for (let i = this.data.todayIndex; i < meals.length && remainingMeals11 > 0; i++) {
      if (meals[i] && meals[i][1] >= 0 && meals[i][1] !== 15) { // 确保不覆盖15元餐
        meals[i][1] = 11;
        remainingMeals11--;
      }
    }

    if (this.data.debug) {
      console.log('更新消费计划结束 >>>>>>>>>>>>');
      console.log(JSON.stringify({
        meals: meals
      }, null, 2));
    }
    
    this.setData({ meals });
    this.updateCalendarDisplay();
  },

  initCalendar() {
    const today = this.getCurrentDate();
    today.setHours(this.getCurrentHour(), 0, 0, 0);

    let startDate, endDate;

    // 判断今天的日期
    if (today.getDate() > 20) {
      // 如果今天的日期大于20日，范围是本月21日到下个月20日
      startDate = new Date(today.getFullYear(), today.getMonth(), 21);
      endDate = new Date(today.getFullYear(), today.getMonth() + 1, 20);
    } else {
      // 如果今天的日期小于等于20日，范围是上个月21日到本月20日
      startDate = new Date(today.getFullYear(), today.getMonth() - 1, 21);
      endDate = new Date(today.getFullYear(), today.getMonth(), 20);
    }

    // 获取起始日期是周几（0-6，0代表周日）
    const firstDayOfWeek = startDate.getDay();

    let days = [];
    
    // 填充起始日期之前的空白
    for (let i = 0; i < firstDayOfWeek; i++) {
      days.push({
        day: '',
        isInRange: false
      });
    }

    // 填充日期直到结束日期
    let currentDate = new Date(startDate.getTime());
    while (currentDate <= endDate) {
      const isPast = currentDate < today;
      const dateInfo = this.getDateInfo(currentDate, isPast);
      days.push({
        day: currentDate.getDate(),
        isToday: this.isSameDay(currentDate, today),
        isInRange: true,
        isPast: isPast,
        morning: dateInfo.morning,
        afternoon: dateInfo.afternoon
      });
      currentDate.setDate(currentDate.getDate() + 1);
    }

    if (this.data.debug) {
      console.log('日历初始化:', {
        开始日期: startDate.toLocaleDateString(),
        结束日期: endDate.toLocaleDateString(),
        总天数: days.length,
        第一天星期: firstDayOfWeek
      });
    }
    console.log(days);
    this.setData({ days });
  },

  isSameDay(date1, date2) {
    //console.log(date1.getDate(),date1.getMonth(),date1.getFullYear(),date2.getDate(),date2.getMonth(),date2.getFullYear());
    return date1.getDate() === date2.getDate() &&
           date1.getMonth() === date2.getMonth() &&
           date1.getFullYear() === date2.getFullYear()
  },

  getDateInfo(date, isPast) {
    if (isPast) {
      return {
        morning: 'past',
        afternoon: 'past'
      }
    }
    
    // 末接返回holiday状态
    const dayOfWeek = date.getDay()
    if (dayOfWeek === 0 || dayOfWeek === 6) {
      return {
        morning: 'holiday',
        afternoon: 'holiday'
      }
    }
    
    return {
      morning: 'off',
      afternoon: 'off'
    }
  },

  getTimeSlotStatus(date, timeSlot) {
    // 周末休息
    if (date.getDay() === 0 || date.getDay() === 6) {
      return 'off'
    }
    // 工作日随机状态
    const statuses = ['available', 'busy']
    return statuses[Math.floor(Math.random() * statuses.length)]
  },

  isHoliday(date) {
    const dateStr = this.formatDate(date)
    return this.data.holidays.includes(dateStr)
  },

  formatDate(date) {
    const year = date.getFullYear()
    const month = String(date.getMonth() + 1).padStart(2, '0')
    const day = String(date.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  },

  // 新增获取当前时间的方法
  getCurrentHour() {
    if (this.data.useCustomTime) {
      return this.data.customTime;
    }
    return new Date().getHours();
  },

  // 添加长按处理方法
  handleLongPress(e) {
    const { date, morning, afternoon } = e.currentTarget.dataset;
    // 获点击位置
    const { clientX, clientY } = e.touches[0];
    
    this.setData({
      showPopup: true,
      popupInfo: {
        date: date,
        morning: this.getMealDescription(morning),
        afternoon: this.getMealDescription(afternoon),
        x: clientX,
        y: clientY - 100  // 向上偏移100px，避免被手指遮挡
      }
    });
  },

  // 添加手指松开处理方法
  handleTouchEnd() {
    this.setData({
      showPopup: false
    });
  },

  // 添加获取餐食描述的方法
  getMealDescription(status) {
    const statusMap = {
      'meal11': '11元标准餐',
      'meal15': '15元营养餐',
      'holiday': '节假日休息',
      'past': '已过去',
      'off': '未安排'
    };
    const colorMap = {
      'meal11': '浅绿色',
      'meal15': '深绿色',
      'holiday': '橙色',
      'past': '灰色',
      'off': '红色'
    };
    return `${statusMap[status] || status}(${colorMap[status] || '未知'})`;
  },

  onCustomTimeSwitch(e) {
    this.setData({
      useCustomTime: e.detail.value
    });
    this.initMealsArray(); // 重新初始化日历
  },

  onCustomTimeInput(e) {
    let value = parseInt(e.detail.value);
    // 确保输入值在0-23之间
    value = Math.min(23, Math.max(0, value));
    this.setData({
      customTime: value
    });
    this.initMealsArray(); // 重新初始化日历
  },

  onCustomDateSwitch(e) {
    this.setData({
      useCustomDate: e.detail.value
    });
    this.initMealsArray(); // 重新初始化日历
  },

  onCustomDateChange(e) {
    this.setData({
      customDate: e.detail.value
    });
    this.initMealsArray(); // 重新初始化日历
  },

  // 修改获取当前时间的方法
  getCurrentDate() {
    if (this.data.useCustomDate && this.data.customDate) {
      return new Date(this.data.customDate);
    }
    return new Date();
  },

  getCurrentHour() {
    if (this.data.useCustomTime) {
      return this.data.customTime;
    }
    return new Date().getHours();
  },

  // 添加日期比较方法
  compareDates(date1, date2) {
    const d1 = new Date(date1.getFullYear(), date1.getMonth(), date1.getDate());
    const d2 = new Date(date2.getFullYear(), date2.getMonth(), date2.getDate());
    return d1 - d2;
  },

  // 修改日期相等判断方法
  isSameDay(date1, date2) {
    return this.compareDates(date1, date2) === 0;
  }
})
