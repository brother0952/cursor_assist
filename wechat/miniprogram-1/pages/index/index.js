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
    meals: [], // 二维数组存储消费计划
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
      '2025-01-10', // 春节
      '2025-01-11',
      '2025-01-12',
      '2025-01-13',
      '2025-01-14',
      '2025-01-15',
      '2025-01-16',
      '2025-01-17',
      '2025-04-05', // 清明节
      '2025-05-01', // 劳动节
      '2025-06-01', // 端午节
    ],
    todayIndex: null, // 今天在meals数组中的索引
    remainingWorkdays: 0, // 到下一个结算日的工作日数量
    debug: false, // 调试开关，设置为false以���蔽调试输出
    useCustomTime: false,  // 是否使用自定义时间
    customTime: 13,       // 自定义时间（小时）
    showPopup: false,
    popupInfo: {
      date: '',
      morning: '',
      afternoon: '',
      x: 0,
      y: 0
    }
  },

  onLoad() {
    this.initMealsArray()
    this.initCalendar()
  },

  // 初始化消费计划数组
  initMealsArray() {
    const today = new Date()
    const currentHour = this.getCurrentHour()
    
    // 计算起始日期（上一个21号）
    let startDate = new Date()
    startDate.setDate(21)
    if (today.getDate() <= 20) {
      startDate.setMonth(startDate.getMonth() - 1)
    }

    // 计算结束日期（下一个20号）
    let endDate = new Date()
    if (today.getDate() > 20) {
      endDate.setMonth(endDate.getMonth() + 1)
    }
    endDate.setDate(20)

    // 计算天数
    const days = Math.floor((endDate - startDate) / (24 * 60 * 60 * 1000)) + 1
    
    // 初始化二维数组
    const meals = new Array(days).fill(null).map(() => [-1, -1])

    // 设置未来日期的初始值
    let currentDate = new Date(startDate.getTime())
    let todayIndex = null; // 用于记录今天在meals数组中的索引
    let workdayCount = 0; // 记录工作日数量

    while (currentDate <= endDate) {
      const index = Math.floor((currentDate - startDate) / (24 * 60 * 60 * 1000))
      
      if (index >= 0 && index < days) {
        const dayOfWeek = currentDate.getDay()
        const dateStr = this.formatDate(currentDate)
        const isPast = currentDate < today
        
        // 周末或节假日设置为-2
        if (isPast) {
          // 过去的日期设置为-1
          meals[index] = [-1, -1]
        } else if (dayOfWeek === 0 || dayOfWeek === 6 || this.data.holidays.includes(dateStr)) {
          meals[index] = [-2, -2]
        } else if (this.isSameDay(currentDate, today)) {
          // 今天的特殊处理
          meals[index] = this.getTodayMealStatus(currentHour)
          todayIndex = index; // 记录今天的索引
        } else {
          // 未来日期设置为0
          meals[index] = [0, 0]
          workdayCount++; // 计数工作日
        }
      }
      
      currentDate.setDate(currentDate.getDate() + 1)
    }

    if (this.data.debug) {
      console.log('初始化消费计划数组:', {
        startDate: startDate.toLocaleDateString(),
        endDate: endDate.toLocaleDateString(),
        days: days,
        meals: meals,
        todayIndex: todayIndex,
        remainingWorkdays: workdayCount
      })
    }
    
    this.setData({ meals, todayIndex, remainingWorkdays: workdayCount })
    this.resetMealsArray()
    this.updateCalendarDisplay()
    return { startDate, endDate, meals }
  },

  // 添加新方法：根据当前时间判断今天的餐食状态
  getTodayMealStatus(currentHour) {
    if (currentHour >= 17) {
      // 下午5点后，全天结束
      return [-1, -1]
    } else if (currentHour >= 11) {
      // 上午11点后，上午结束
      return [-1, 0]
    } else {
      // 上午11点前，全天可用
      return [0, 0]
    }
  },

  // 处理输入金额变化
  onBalanceInput(e) {
    const balance = Number(e.detail.value)
    this.setData({ balance })
    const currentHour = this.getCurrentHour()
    // console.log('当前时间:', currentHour, '时')
    // console.log('当前输入金额:', balance)
    
    // 每次输入金额后重新初始化meals数组
    this.initMealsArray()

    if (balance > 0) {
      this.calculateOptimalPlan(balance)
    } else {
      this.resetMealsArray()
      this.updateCalendarDisplay()
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

  // 计算最优消费方案
  calculateOptimalPlan(balance) {
    let bestPlan = {
      meals11: 0,
      meals15: 0,
      remaining: balance,
      totalMeals: 0
    }
    
    // 计算实际可用的时段数量（值为0的时段）
    const availableSlots = this.data.meals.reduce((count, day) => {
      return count + (day[0] === 0 ? 1 : 0) + (day[1] === 0 ? 1 : 0)
    }, 0)

    if (this.data.debug) {
      console.log('可用时段数量:', availableSlots)
    }

    // 尝试所有可能的组合
    for (let meals15 = 0; meals15 <= availableSlots; meals15++) {
      for (let meals11 = 0; meals11 <= availableSlots - meals15; meals11++) {
        const total = meals15 * 15 + meals11 * 11
        const remaining = balance - total
        
        // 确保不超出可用时段
        if (remaining >= 0 && remaining < bestPlan.remaining) {
          bestPlan = {
            meals15,
            meals11,
            remaining,
            totalMeals: meals15 + meals11
          }
        }
      }
    }

    // 处理大金额时的特殊情况
    const maxMeals11 = Math.floor(balance / 11);
    const maxMeals15 = Math.floor(balance / 15);

    if (maxMeals11 <= availableSlots) {
      const remaining11 = balance - (maxMeals11 * 11);
      if (remaining11 < bestPlan.remaining) {
        bestPlan = {
          meals11: maxMeals11,
          meals15: 0,
          remaining: remaining11,
          totalMeals: maxMeals11
        };
      }
    }

    if (maxMeals15 <= availableSlots) {
      const remaining15 = balance - (maxMeals15 * 15);
      if (remaining15 < bestPlan.remaining) {
        bestPlan = {
          meals11: 0,
          meals15: maxMeals15,
          remaining: remaining15,
          totalMeals: maxMeals15
        };
      }
    }

    if (this.data.debug) {
      console.log('计算结果:', {
        总金额: balance,
        '15元餐数': bestPlan.meals15,
        '11元餐数': bestPlan.meals11,
        剩余金额: bestPlan.remaining,
        总餐数: bestPlan.totalMeals,
        可用时段: availableSlots
      })
    }

    this.setData({ plan: bestPlan })
    this.updateMealsWithPlan(bestPlan)
  },

  // 根据计算结果更新消费计划数组
  updateMealsWithPlan(plan) {
    let remainingMeals15 = plan.meals15
    let remainingMeals11 = plan.meals11
    const meals = [...this.data.meals]
    
    // 优先安排所有11元餐
    for (let i = this.data.todayIndex; i < meals.length && remainingMeals11 > 0; i++) {
        // 跳过不可用的日期（节假日等）
        if (meals[i][0] === -2) continue
        
        // 上午时段
        if (meals[i][0] === 0) {
            meals[i][0] = 11
            remainingMeals11--
        }
        
        // 下午时段
        if (meals[i][1] === 0 && remainingMeals11 > 0) {
            meals[i][1] = 11
            remainingMeals11--
        }
    }
    
    // 然后安排15元餐
    for (let i = this.data.todayIndex; i < meals.length && remainingMeals15 > 0; i++) {
        if (meals[i][0] === -2) continue
        
        if (meals[i][0] === 0) {
            meals[i][0] = 15
            remainingMeals15--
        }
        
        if (meals[i][1] === 0 && remainingMeals15 > 0) {
            meals[i][1] = 15
            remainingMeals15--
        }
    }
    this.setData({ meals })
    this.updateCalendarDisplay()
  },

  // 更新日历显示
  updateCalendarDisplay() {
    // 获取开始日期（上一个21号）
    const today = new Date()
    let startDate = new Date(today.getFullYear(), today.getMonth() - 1, 21)
    
    const days = this.data.days.map((day, index) => {
      if (!day.isInRange || !day.day) {
        return day
      }

      // 计算当前日期
      const currentDate = new Date(startDate.getFullYear(), startDate.getMonth(), day.day)
      // 如果日期小于startDate，说明是下个月的日期
      if (day.day < 21) {
        currentDate.setMonth(currentDate.getMonth() + 1)
      }
      
      // 计算在meals数组中的索引
      const dayIndex = Math.floor((currentDate - startDate) / (24 * 60 * 60 * 1000))
      
      const mealDay = this.data.meals[dayIndex]
      if (!mealDay) return day

      const newDay = Object.assign({}, day)
      
      // 根据消费计划数组的值设置显示状态
      newDay.morning = this.getMealStatus(mealDay[0])
      newDay.afternoon = this.getMealStatus(mealDay[1])
      // 调试输出
      if(this.data.debug) {
        console.log('日期映射:', {
          日期: currentDate.toLocaleDateString(),
          索引: dayIndex,
          状态: mealDay,
          显示: {
            morning: newDay.morning,
            afternoon: newDay.afternoon
          }
        })
      }
      
      return newDay
    })
    
    this.setData({ days })
  },

  // 获取餐食状态对应的显示类型
  getMealStatus(value) {
    switch (value) {
      case -2: return 'holiday'  // 节假日橙色
      case -1: return 'past'     // 过去日期灰色
      case 11: return 'meal11'   // 11元餐浅绿色
      case 15: return 'meal15'   // 15元餐深绿色
      case 0: return 'off'       // 未计划消费红色
      default: return 'off'      // 默认也是红色
    }
  },

  initCalendar() {
    const today = new Date()
    today.setHours(this.getCurrentHour(), 0, 0, 0)

    // 设置当前月份（显示本月）
    this.setData({
      currentMonth: today.getMonth() + 1
    })

    // 计算起始日期（上月21日）
    const startDate = new Date(today.getFullYear(), today.getMonth() - 1, 21)
    
    // 计算结束日期（本月20日）
    const endDate = new Date(today.getFullYear(), today.getMonth(), 20)

    // 获取本月1号是周几
    const firstDayOfMonth = new Date(startDate.getFullYear(), startDate.getMonth(), 21).getDay()

    let days = []
    
    // 填充上月剩余日期
    for (let i = 0; i < firstDayOfMonth; i++) {
      days.push({
        day: '',
        isInRange: false
      })
    }

    // 填充日期直到结束日期
    let currentDate = new Date(startDate.getTime())
    while (currentDate <= endDate) {
      const isPast = currentDate < today
      const dateInfo = this.getDateInfo(currentDate, isPast)
      days.push({
        day: currentDate.getDate(),
        isToday: this.isSameDay(currentDate, today),
        isInRange: true,
        isPast: isPast,
        morning: dateInfo.morning,
        afternoon: dateInfo.afternoon
      })
      currentDate.setDate(currentDate.getDate() + 1)
    }

    // console.log('日历初始化:', {
    //   开始日期: startDate.toLocaleDateString(),
    //   结束日期: endDate.toLocaleDateString(),
    //   总天数: days.length,
    //   第一天星期: firstDayOfMonth
    // })

    this.setData({ days })
  },

  isSameDay(date1, date2) {
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
    
    // 周末接返回holiday状态
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
    // 获取点击位置
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
    return `${statusMap[status] || status}（${colorMap[status] || '未知'}）`;
  }
})
