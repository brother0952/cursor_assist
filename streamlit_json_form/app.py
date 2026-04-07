import streamlit as st
import json
import pandas as pd
import os
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="Configuration Tool",
    page_icon="⚙️",
    layout="wide"
)


def load_json_file(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        st.error(f"Invalid JSON format in: {file_path}")
        return None


def save_json_file(file_path, data):
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False


def load_excel_data(source_config):
    """Load data from Excel file based on source configuration."""
    try:
        file_path = source_config['file_path']
        sheet_name = source_config.get('sheet_name', 0)
        
        if not os.path.exists(file_path):
            st.warning(f"Excel file not found: {file_path}. Using empty options.")
            return []
        
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        value_col = source_config['columns']['value']
        label_col = source_config['columns']['label']
        
        # Create list of tuples (value, label)
        options = []
        for _, row in df.iterrows():
            options.append({
                'value': str(row[value_col]),
                'label': str(row[label_col])
            })
        
        return options
    except Exception as e:
        st.error(f"Error loading Excel data: {str(e)}")
        return []


def get_data_source_options(data_sources, source_id):
    """Get options for a specific data source."""
    for source in data_sources.get('excel_sources', []):
        if source['id'] == source_id:
            # Check if static options are provided (for debugging)
            if 'options' in source:
                return source['options']
            # Otherwise load from Excel
            return load_excel_data(source)
    return []


def render_text_input(field, value):
    """Render a text input field."""
    return st.text_input(
        label=field['label'],
        value=value if value else "",
        placeholder=field.get('placeholder', ''),
        help=field.get('help_text', '')
    )


def render_textarea_field(field, value):
    """Render a textarea field."""
    return st.text_area(
        label=field['label'],
        value=value if value else "",
        placeholder=field.get('placeholder', ''),
        help=field.get('help_text', ''),
        height=100
    )


def render_select_field(field, value, data_sources):
    """Render a select/dropdown field."""
    options = get_data_source_options(data_sources, field['data_source'])
    option_labels = [opt['label'] for opt in options]
    option_values = [opt['value'] for opt in options]
    
    # Find current selection index
    default_index = 0
    if value and value in option_values:
        default_index = option_values.index(value)
    
    selected_label = st.selectbox(
        label=field['label'],
        options=option_labels,
        index=default_index if option_labels else 0,
        help=field.get('help_text', '')
    )
    
    # Return the corresponding value
    if option_labels and selected_label:
        idx = option_labels.index(selected_label)
        return option_values[idx]
    return None


def render_multiselect_field(field, value, data_sources):
    """Render a multiselect field."""
    options = get_data_source_options(data_sources, field['data_source'])
    option_labels = [opt['label'] for opt in options]
    option_values = [opt['value'] for opt in options]
    
    # Find current selections
    default_indices = []
    if value:
        for i, val in enumerate(option_values):
            if val in value:
                default_indices.append(i)
    
    selected_labels = st.multiselect(
        label=field['label'],
        options=option_labels,
        default=[option_labels[i] for i in default_indices] if default_indices else [],
        help=field.get('help_text', '')
    )
    
    # Return corresponding values
    selected_values = []
    for label in selected_labels:
        if label in option_labels:
            idx = option_labels.index(label)
            selected_values.append(option_values[idx])
    
    return selected_values


def render_number_field(field, value):
    """Render a number input field."""
    min_val = field.get('min_value', None)
    max_val = field.get('max_value', None)
    step = field.get('step', 1)
    default = field.get('default_value', 0)
    
    return st.number_input(
        label=field['label'],
        min_value=min_val,
        max_value=max_val,
        value=value if value is not None else default,
        step=step,
        help=field.get('help_text', '')
    )


def render_checkbox_field(field, value):
    """Render a checkbox field."""
    default = field.get('default_value', False)
    return st.checkbox(
        label=field['label'],
        value=value if value is not None else default,
        help=field.get('help_text', '')
    )


def render_table_field(field, value):
    """Render an editable table field."""
    st.subheader(field['label'])
    if field.get('help_text'):
        st.caption(field['help_text'])
    
    columns = field['columns']
    
    # Initialize or load existing data
    if value and len(value) > 0:
        df = pd.DataFrame(value)
    else:
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=[col['key'] for col in columns])
    
    # Display editable table
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            col['key']: st.column_config.Column(
                label=col['label'],
                required=field.get('required', False)
            )
            for col in columns
        }
    )
    
    # Convert back to list of dictionaries
    result = edited_df.to_dict('records')
    # Remove NaN values
    cleaned_result = []
    for row in result:
        cleaned_row = {k: v for k, v in row.items() if pd.notna(v)}
        if cleaned_row:
            cleaned_result.append(cleaned_row)
    
    return cleaned_result


def render_linked_table_field(field, current_data, data_sources):
    """Render a table field with dynamic select options linked to another table."""
    st.subheader(field['label'])
    if field.get('help_text'):
        st.caption(field['help_text'])
    
    # Get the source table data
    source_field_id = field.get('link_to_field')
    source_table_data = current_data.get(source_field_id, []) if current_data else []
    
    # Build options from source table
    value_key = field.get('link_key', 'id')
    label_key = field.get('link_label_key', 'name')
    
    options = []
    for row in source_table_data:
        if value_key in row and label_key in row:
            options.append({
                'value': str(row[value_key]),
                'label': str(row[label_key])
            })
    
    if not options:
        st.warning(f"⚠️ No options available. Please add data to '{source_field_id}' first.")
        options = [{'value': '', 'label': '(No options available)'}]
    
    columns = field['columns']
    
    # Initialize or load existing data
    if current_data and field['field_id'] in current_data and len(current_data[field['field_id']]) > 0:
        df = pd.DataFrame(current_data[field['field_id']])
    else:
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=[col['key'] for col in columns])
    
    # Build column config with dynamic select options
    column_config = {}
    for col in columns:
        if col.get('type') == 'select_linked':
            # This is a linked select column
            option_values = [opt['value'] for opt in options]
            option_labels = [opt['label'] for opt in options]
            
            column_config[col['key']] = st.column_config.SelectboxColumn(
                label=col['label'],
                options=option_labels,
                required=field.get('required', False),
                help=f"Select from {source_field_id}"
            )
        elif col.get('type') == 'number':
            column_config[col['key']] = st.column_config.NumberColumn(
                label=col['label'],
                required=field.get('required', False)
            )
        else:
            column_config[col['key']] = st.column_config.TextColumn(
                label=col['label'],
                required=field.get('required', False)
            )
    
    # Display editable table
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )
    
    # Convert back to list of dictionaries
    result = edited_df.to_dict('records')
    # Remove NaN values and map labels back to values for select_linked columns
    cleaned_result = []
    for row in result:
        cleaned_row = {}
        for key, value in row.items():
            if pd.notna(value):
                # Check if this is a select_linked column and convert label back to value
                col_def = next((c for c in columns if c['key'] == key), None)
                if col_def and col_def.get('type') == 'select_linked':
                    # Find the corresponding value for this label
                    option_match = next((opt for opt in options if opt['label'] == value), None)
                    if option_match:
                        cleaned_row[key] = option_match['value']
                    else:
                        cleaned_row[key] = value
                else:
                    cleaned_row[key] = value
        
        if cleaned_row:
            cleaned_result.append(cleaned_row)
    
    return cleaned_result


def render_object_field(field, current_data, data_sources):
    """Render an object field containing nested fields."""
    field_id = field['field_id']
    st.subheader(field['label'])
    
    if field.get('help_text'):
        st.caption(field['help_text'])
    
    # Initialize or get existing data for this object
    obj_data = current_data.get(field_id, {}) if current_data else {}
    
    # Create columns for layout
    cols = st.columns(2)
    col_idx = 0
    
    # Render each nested field
    for nested_field in field.get('fields', []):
        with cols[col_idx % 2]:
            result = render_field(nested_field, obj_data, data_sources)
            if result is not None:
                obj_data[nested_field['field_id']] = result
        col_idx += 1
    
    return obj_data


def render_section(section, current_data, data_sources):
    """Render all fields within a section."""
    # Check if this section has linked tables that need special ordering
    has_linked_tables = any(field.get('field_type') == 'table_linked' for field in section.get('fields', []))
    
    if has_linked_tables:
        # For sections with linked tables, we need to render them in order and handle dependencies
        regular_fields = [f for f in section.get('fields', []) if f.get('field_type') != 'table_linked']
        linked_fields = [f for f in section.get('fields', []) if f.get('field_type') == 'table_linked']
        
        # Create columns for better layout
        cols = st.columns(2)
        col_idx = 0
        
        # First render regular fields (including source tables)
        for field in regular_fields:
            with cols[col_idx % 2]:
                result = render_field(field, current_data, data_sources)
                if result is not None:
                    current_data[field['field_id']] = result
            col_idx += 1
        
        # Then render linked tables (they depend on source tables being rendered first)
        for field in linked_fields:
            with cols[col_idx % 2]:
                result = render_field(field, current_data, data_sources)
                if result is not None:
                    current_data[field['field_id']] = result
            col_idx += 1
    else:
        # Standard rendering for sections without linked tables
        # Create columns for better layout
        cols = st.columns(2)
        col_idx = 0
        
        for field in section.get('fields', []):
            with cols[col_idx % 2]:
                result = render_field(field, current_data, data_sources)
                if result is not None:
                    current_data[field['field_id']] = result
            col_idx += 1


def render_field(field, current_data, data_sources):
    """Render a form field based on its type."""
    field_id = field['field_id']
    field_type = field['field_type']
    value = current_data.get(field_id) if current_data else None
    
    if field_type == 'text':
        return render_text_input(field, value)
    elif field_type == 'textarea':
        return render_textarea_field(field, value)
    elif field_type == 'select':
        return render_select_field(field, value, data_sources)
    elif field_type == 'multiselect':
        return render_multiselect_field(field, value, data_sources)
    elif field_type == 'number':
        return render_number_field(field, value)
    elif field_type == 'checkbox':
        return render_checkbox_field(field, value)
    elif field_type == 'table':
        return render_table_field(field, value)
    elif field_type == 'table_linked':
        return render_linked_table_field(field, current_data, data_sources)
    elif field_type == 'object':
        return render_object_field(field, current_data, data_sources)
    else:
        st.warning(f"Unknown field type: {field_type}")
        return None


def validate_required_fields(schema, data):
    """Validate that all required fields are filled."""
    errors = []
    
    for section in schema.get('sections', []):
        for field in section.get('fields', []):
            if field.get('required', False):
                field_id = field['field_id']
                value = data.get(field_id)
                
                # Check if value is empty
                if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
                    errors.append(f"Required field '{field['label']}' is empty")
    
    return errors


def main():
    """Main application function."""
    st.title("⚙️ Configuration Tool")
    
    # Load configuration files
    schema = load_json_file('schema.json')
    data_sources = load_json_file('data_source.json')
    
    if not schema:
        st.error("Failed to load schema.json. Please check the file.")
        return
    
    if not data_sources:
        st.warning("Failed to load data_source.json. Dropdown options may not work.")
        data_sources = {'excel_sources': []}
    
    # Load existing data or initialize empty
    if os.path.exists('data.json'):
        current_data = load_json_file('data.json')
        if current_data is None:
            current_data = {}
    else:
        current_data = {}

    sections = schema.get('sections', [])
    
    # Create tabs based on sections
    tab_labels = [section.get('section_title', f'Section {i+1}') for i, section in enumerate(sections)]
    tabs = st.tabs(tab_labels)
    
    # Render form inside tabs
    with st.form("config_form"):
        for i, (tab, section) in enumerate(zip(tabs, sections)):
            with tab:
                render_section(section, current_data, data_sources)
        
        st.divider()
        
        # Form submission buttons (visible in all tabs but at the bottom)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            submit_button = st.form_submit_button("💾 Save Configuration", type="primary")
        
        with col2:
            reset_button = st.form_submit_button("🔄 Reset Form")
        
        with col3:
            export_button = st.form_submit_button("📤 Export JSON")
    
    # Handle save
    if submit_button:
        # Validate required fields
        errors = validate_required_fields(schema, current_data)
        
        if errors:
            st.error("Please fix the following errors:")
            for error in errors:
                st.warning(f"• {error}")
        else:
            # Add metadata
            current_data['_last_modified'] = datetime.now().isoformat()
            current_data['_version'] = current_data.get('_version', 0) + 1
            
            # Save to file
            if save_json_file('data.json', current_data):
                st.success("✅ Configuration saved successfully!")
                st.json(current_data)
            else:
                st.error("❌ Failed to save configuration.")
    
    # Handle reset
    if reset_button:
        st.session_state.clear()
        st.rerun()
    
    # Handle export
    if export_button:
        json_str = json.dumps(current_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Display current configuration
    with st.expander("📋 View Current Configuration"):
        st.json(current_data)
    
    # Display data sources info
    with st.expander("ℹ️ Data Sources Information"):
        st.write("**Available Data Sources:**")
        for source in data_sources.get('excel_sources', []):
            st.write(f"• **{source['name']}** (ID: {source['id']})")
            # st.write(f"  - File: {source['file_path']}")
            st.write(f"  - Sheet: {source.get('sheet_name', 'Default')}")

if __name__ == "__main__":
    main()
