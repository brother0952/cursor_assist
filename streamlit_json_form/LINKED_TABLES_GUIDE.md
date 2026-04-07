# 表格联动功能使用说明

## 功能概述

本功能允许在同一个 Tab 内实现两个表格之间的数据联动。表格1的内容会动态影响表格2的下拉选项。

## 使用场景示例

**场景**: 商品分类管理
- **表格1**: 定义商品分类 (如: 电子产品、服装、食品)
- **表格2**: 添加具体商品,选择商品所属分类时,下拉选项来自表格1的分类列表

## Schema 配置说明

### 1. 源表格 (Table 1) - 提供选项数据

```json
{
  "field_id": "table1_categories",
  "field_type": "table",
  "label": "Table 1: Categories",
  "columns": [
    {
      "key": "category_id",
      "label": "Category ID",
      "type": "text"
    },
    {
      "key": "category_name",
      "label": "Category Name",
      "type": "text"
    }
  ]
}
```

### 2. 联动表格 (Table 2) - 使用选项数据

```json
{
  "field_id": "table2_items",
  "field_type": "table_linked",
  "label": "Table 2: Items (Linked to Table 1)",
  "link_to_field": "table1_categories",
  "link_key": "category_id",
  "link_label_key": "category_name",
  "columns": [
    {
      "key": "item_name",
      "label": "Item Name",
      "type": "text"
    },
    {
      "key": "category",
      "label": "Category",
      "type": "select_linked",
      "source_field": "table1_categories",
      "value_key": "category_id",
      "label_key": "category_name"
    },
    {
      "key": "price",
      "label": "Price",
      "type": "number"
    }
  ]
}
```

## 关键配置参数

### 联动表格特有参数:

| 参数 | 说明 | 示例 |
|------|------|------|
| `field_type` | 必须设置为 `"table_linked"` | `"table_linked"` |
| `link_to_field` | 源表格的 field_id | `"table1_categories"` |
| `link_key` | 源表格中用作值的列名 | `"category_id"` |
| `link_label_key` | 源表格中用作显示标签的列名 | `"category_name"` |

### 联动列配置 (在 columns 数组中):

| 参数 | 说明 | 示例 |
|------|------|------|
| `type` | 必须设置为 `"select_linked"` | `"select_linked"` |
| `source_field` | 源表格的 field_id | `"table1_categories"` |
| `value_key` | 源表格的值列名 | `"category_id"` |
| `label_key` | 源表格的标签列名 | `"category_name"` |

## 工作流程

1. **用户操作**: 在表格1中添加/编辑/删除数据
2. **系统响应**: 保存表格1的数据到 current_data
3. **渲染表格2**: 
   - 读取表格1的最新数据
   - 动态生成下拉选项
   - 用户在表格2中选择时,只能看到表格1中已定义的选项
4. **数据保存**: 表格2存储的是值(value),但界面显示的是标签(label)

## 注意事项

⚠️ **重要提示**:

1. **渲染顺序**: 源表格必须在联动表格之前渲染(系统会自动处理)
2. **空数据处理**: 如果源表格为空,联动表格会显示警告信息
3. **数据一致性**: 如果删除了源表格中的某条记录,联动表格中引用该记录的行可能显示异常
4. **同一 Section**: 联动的两个表格必须在同一个 section 内

## 完整示例

参考 `schema.json` 中的 "Linked Tables Demo" section,该示例展示了:
- 表格1: 定义分类 (category_id, category_name)
- 表格2: 添加商品 (item_name, category[从表格1选择], price)

## 扩展应用

这种联动机制可以应用于多种场景:
- **部门-员工**: 表格1定义部门,表格2添加员工并选择部门
- **项目-任务**: 表格1定义项目,表格2添加任务并关联项目
- **课程-学生**: 表格1定义课程,表格2注册学生并选择课程
- **地区-门店**: 表格1定义地区,表格2添加门店并选择地区

## 技术实现要点

1. **Session State**: 利用 Streamlit 的表单机制和 current_data 字典保持状态
2. **动态选项**: 每次渲染时从 current_data 读取源表格数据生成选项
3. **值-标签映射**: 界面显示标签,数据存储值,自动转换
4. **渲染顺序控制**: render_section 函数检测联动表格并调整渲染顺序
