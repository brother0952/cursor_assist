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
    debug: false, // 调试开关
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
      '2025-01-01', // 元旦
      '2025-01-28', // 除夕
      '2025-01-29', // 春节
      '2025-01-30',
      '2025-01-31',
      '2025-02-01',
      '2025-02-02',
      '2025-02-03',
      '2025-02-04',
      '2025-04-04',
      '2025-04-05',
      '2025-04-06',
      '2025-05-01',
      '2025-05-02',
      '2025-05-03',
      '2025-05-04',
      '2025-05-05',
      '2025-05-31',
      '2025-06-01',
      '2025-06-02',
      '2025-10-01',
      '2025-10-02',
      '2025-10-03',
      '2025-10-04',
      '2025-10-05',
      '2025-10-06',
      '2025-10-07',
      '2025-10-08'
    ],
    workdays: [
      '2025-01-26', // 春节调休
      '2025-02-08',
      '2025-04-27', // 劳动节调休
      '2025-09-28', // 国庆节调休
      '2025-10-11'
    ],
    planType: 0, // 默认选择方案一
    planTypes: ['方案一：标准餐(11)+营养餐(15)', '方案二：仅营养餐(15)']
  },

  onLoad() {
    // 从本地存储读取之前的方案选择
    try {
      const planType = wx.getStorageSync('planType');
      if (planType !== '') {
        this.setData({ planType });
      }
    } catch (e) {
      console.error('读取方案选择失败:', e);
    }

    // 初始化日期范围和消费计划数组
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
        const isWorkday = this.isWorkday(currentDate);
        const dateStr = this.formatDate(currentDate);
        const isPast = this.compareDates(currentDate, today) < 0;
        
        if (isPast) {
          meals[index] = [-1, -1]; // 过去的日期
        } else if (!isWorkday) {
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
    const balance = parseFloat(e.detail.value) || 0;
    
    // 先更新余额
    this.setData({
      balance: balance,
      // 同时重置计划
      plan: {
        meals11: 0,
        meals15: 0,
        remaining: 0,
        totalMeals: 0
      }
    });

    // 重置meals数组
    const meals = this.data.meals.map(day => {
      // 保持过期和节假日的状态不变
      if (day[0] === -1 || day[0] === -2) {
        return [...day];
      }
      // 重置其他日期的状态
      return [0, 0];
    });
    
    this.setData({ meals });
    
    // 更新日历显示
    this.updateCalendarDisplay();

    // 如果有金额，计算新的方案
    if (balance > 0) {
      this.calculateOptimalPlan(balance);
    }
  },

  // 计算最优消费方案
  calculateOptimalPlan(balance) {
    let bestPlan = {
      meals11: 0,
      meals15: 0,
      remaining: balance,
      totalMeals: 0
    };
    
    // 计算可用的时间槽
    const availableSlots = this.data.meals.reduce((slots, day) => {
      return {
        morning: slots.morning + (day[0] >= 0 ? 1 : 0),
        afternoon: slots.afternoon + (day[1] >= 0 ? 1 : 0)
      };
    }, { morning: 0, afternoon: 0 });

    if (availableSlots.morning === 0 && availableSlots.afternoon === 0) {
      console.warn("没有可用的消费计划时间槽");
      return;
    }

    if (this.data.planType === 0) {
      // 方案一：标准餐+营养餐
      // 先计算用11元能覆盖多少个上午
      const maxMorning11 = Math.min(Math.floor(balance / 11), availableSlots.morning);
      let bestRemaining = balance;
      let bestMeals11 = 0;
      let bestMeals15 = 0;

      // 从最大可能的11元早餐数开始尝试
      for (let morning11 = maxMorning11; morning11 >= 0; morning11--) {
        const remaining11 = balance - (morning11 * 11);
        
        // 尝试用剩余金额安排15元餐
        const maxMeals15 = Math.floor(remaining11 / 15);
        const meals15 = Math.min(maxMeals15, 
          availableSlots.morning - morning11 + availableSlots.afternoon);
        
        const remaining = remaining11 - (meals15 * 15);
        
        // 如果这个方案的剩余金额更小，且保证了更多的上午有餐
        if (remaining >= 0 && 
            (morning11 + meals15 >= bestMeals11 + bestMeals15) && 
            remaining < bestRemaining) {
          bestMeals11 = morning11;
          bestMeals15 = meals15;
          bestRemaining = remaining;
        }
      }

      bestPlan = {
        meals11: bestMeals11,
        meals15: bestMeals15,
        remaining: bestRemaining,
        totalMeals: bestMeals11 + bestMeals15
      };
    } else {
      // 方案二：仅营养餐
      const maxMeals15 = Math.floor(balance / 15);
      const meals15 = Math.min(maxMeals15, 
        availableSlots.morning + availableSlots.afternoon);
      
      bestPlan = {
        meals15: meals15,
        meals11: 0,
        remaining: balance - (meals15 * 15),
        totalMeals: meals15
      };
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
    
    // 重置可用日期的状态
    const meals = this.data.meals.map(day => {
      // 保持过期和节假日的状态不变
      if (day[0] === -1 || day[0] === -2) {
        return [...day];
      }
      // 重置其他日期的状态
      return [0, 0];
    });

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
    let startDate, endDate;

    if (today.getDate() > 20) {
      startDate = new Date(today.getFullYear(), today.getMonth(), 21);
      endDate = new Date(today.getFullYear(), today.getMonth() + 1, 20);
    } else {
      startDate = new Date(today.getFullYear(), today.getMonth() - 1, 21);
      endDate = new Date(today.getFullYear(), today.getMonth(), 20);
    }

    // 计算第一天是星期几
    const firstDayWeek = startDate.getDay();
    
    // 计算总天数
    const totalDays = Math.floor((endDate - startDate) / (24 * 60 * 60 * 1000)) + 1;

    // 初始化日历数组
    const days = new Array(42).fill(null).map(() => ({
      day: '',
      isInRange: false,
      isToday: false,
      isPast: false,
      morning: 'off',
      afternoon: 'off',
      month: '' // 添加月份信息
    }));

    // 填充日期
    let currentDate = new Date(startDate.getTime());
    for (let i = 0; i < totalDays; i++) {
      const index = firstDayWeek + i;
      if (index < days.length) {
        days[index] = {
          day: currentDate.getDate(),
          month: currentDate.getMonth() + 1, // 添加月份信息
          isInRange: true,
          isToday: this.isSameDay(currentDate, today),
          isPast: this.compareDates(currentDate, today) < 0,
          morning: 'off',
          afternoon: 'off'
        };
      }
      currentDate.setDate(currentDate.getDate() + 1);
    }

    this.setData({
      days,
      currentMonth: today.getMonth() + 1 // 设置当前月份
    });

    this.updateCalendarDisplay();
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
    const dayInfo = this.data.days.find(d => d.day === date);
    
    if (!dayInfo || !dayInfo.isInRange) return;

    // 获取点击位置
    const { clientX, clientY } = e.touches[0];
    
    // 获取系统信息
    const systemInfo = wx.getSystemInfoSync();
    const screenWidth = systemInfo.windowWidth;
    const screenHeight = systemInfo.windowHeight;
    
    // 计算弹窗位置
    let x = clientX;
    let y = clientY;
    
    // 确保弹窗不会超出屏幕左右边界
    const popupWidth = 120; // 弹窗宽度的一半（rpx）
    const minX = popupWidth;
    const maxX = screenWidth - popupWidth;
    
    x = Math.max(minX, Math.min(maxX, x));
    
    // 确保弹窗不会超出屏幕上下边界
    const popupHeight = 180; // 弹窗高度（rpx）
    if (y - popupHeight < 0) {
      // 如果上方空间不足，则显示在下方
      y = y + 100;
    }

    this.setData({
      showPopup: true,
      popupInfo: {
        date: date,
        month: dayInfo.month,
        morning: this.getMealStatusText(morning),
        afternoon: this.getMealStatusText(afternoon),
        x: x,
        y: y
      }
    });
  },

  // 添加手指松开处理方法
  handleTouchEnd() {
    this.setData({
      showPopup: false
    });
  },

  // 获取餐食状态的文字描述
  getMealStatusText(status) {
    switch (status) {
      case 'meal11': return '标准餐 (¥11)';
      case 'meal15': return '营养餐 (¥15)';
      case 'holiday': return '节假日';
      case 'past': return '已过期';
      case 'off': return '未安排';
      default: return '未知状态';
    }
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
  },

  // 修改方案选择处理方法
  onPlanTypeChange(e) {
    const planType = parseInt(e.detail.value);
    this.setData({ planType });
    
    // 保存选择到本地存储
    try {
      wx.setStorageSync('planType', planType);
    } catch (e) {
      console.error('保存方案选择失败:', e);
    }
    
    // 如果已经输入金额，重新计算方案
    if (this.data.balance > 0) {
      this.calculateOptimalPlan(this.data.balance);
    }
  },

  // 判断是否为工作��（需要考虑调休）
  isWorkday(date) {
    const dayOfWeek = date.getDay();
    const dateStr = this.formatDate(date);
    
    // 如果是调休上班日，则返回true
    if (this.data.workdays.includes(dateStr)) {
      return true;
    }
    
    // 如果是法定节假日，则返回false
    if (this.data.holidays.includes(dateStr)) {
      return false;
    }
    
    // 普通工作日判断（周一至周五）
    return dayOfWeek !== 0 && dayOfWeek !== 6;
  }
})
