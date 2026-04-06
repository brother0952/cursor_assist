# Streamlit JSON Form Configuration Tool

A web-based configuration tool built with Streamlit that provides an intuitive GUI for managing application configurations.

## Features

- 📝 **Dynamic Form Generation**: Forms are automatically generated based on `schema.json`
- 📊 **Excel Data Integration**: Pulls dropdown options from Excel files defined in `data_source.json`
- 💾 **Persistent Storage**: Saves configurations to `data.json` for future editing
- ✅ **Validation**: Validates required fields before saving
- 📤 **Export**: Export configurations as JSON files
- 🎨 **Multiple Field Types**: Supports text, textarea, select, multiselect, number, checkbox, and table fields

## Project Structure

```
streamlit_json_form/
├── app.py                  # Main Streamlit application
├── schema.json             # Form structure definition
├── data_source.json        # Excel data source configuration
├── data.json               # User configuration data (auto-generated)
├── requirements.txt        # Python dependencies
└── readme.md              # This file
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Create sample Excel files:**
   Create the Excel files referenced in `data_source.json` with appropriate data:
   - `data/products.xlsx` - Product categories
   - `data/users.xlsx` - User roles
   - `data/regions.xlsx` - Region information

## Usage

1. **Start the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Configure your settings:**
   - Fill in the form fields
   - Select options from dropdowns (populated from Excel files)
   - Add custom parameters in the table section

3. **Save your configuration:**
   - Click "💾 Save Configuration" to save to `data.json`
   - Your settings will persist between sessions

4. **Export configuration:**
   - Click "📤 Export JSON" to download the configuration file

## Configuration Files

### schema.json
Defines the form structure with sections and fields:

```json
{
  "form_title": "Configuration Tool",
  "sections": [
    {
      "section_id": "basic_info",
      "section_title": "Basic Information",
      "fields": [
        {
          "field_id": "project_name",
          "field_type": "text",
          "label": "Project Name",
          "required": true
        }
      ]
    }
  ]
}
```

**Supported field types:**
- `text` - Single-line text input
- `textarea` - Multi-line text input
- `select` - Dropdown selection
- `multiselect` - Multiple selection dropdown
- `number` - Numeric input
- `checkbox` - Boolean checkbox
- `table` - Editable data table

### data_source.json
Defines Excel file sources for populating dropdown options:

```json
{
  "excel_sources": [
    {
      "id": "source_1",
      "name": "Product Categories",
      "file_path": "data/products.xlsx",
      "sheet_name": "Categories",
      "columns": {
        "value": "category_id",
        "label": "category_name"
      }
    }
  ]
}
```

### data.json
Stores user configurations (automatically managed by the application):

```json
{
  "project_name": "My Project",
  "category": "cat_001",
  "priority": 5,
  "_last_modified": "2026-04-06T18:00:00",
  "_version": 1
}
```

## Customization

### Adding New Fields
Edit `schema.json` to add new fields to your form:

```json
{
  "field_id": "my_field",
  "field_type": "text",
  "label": "My Field",
  "placeholder": "Enter value",
  "required": false,
  "help_text": "Helpful description"
}
```

### Adding New Data Sources
1. Create an Excel file with your data
2. Add a new entry to `data_source.json`:

```json
{
  "id": "new_source",
  "name": "My Data Source",
  "file_path": "path/to/file.xlsx",
  "sheet_name": "Sheet1",
  "columns": {
    "value": "id_column",
    "label": "name_column"
  }
}
```

3. Reference it in your schema:
```json
{
  "field_id": "my_select",
  "field_type": "select",
  "label": "My Select",
  "data_source": "new_source"
}
```

## Troubleshooting

- **Excel file not found**: Ensure the file path in `data_source.json` is correct
- **Invalid JSON**: Check JSON syntax in configuration files
- **Streamlit not starting**: Verify all dependencies are installed

## License

This project is open source and available for modification and distribution.
