//! Utility functions

/// Extract code snippet from source lines given a span
pub fn extract_code(
    source_lines: &[String],
    span: (u32, u32, u32, u32), // (start_line, start_col, end_line, end_col)
) -> Option<String> {
    let (start_line, start_col, end_line, end_col) = span;
    
    // Convert to 0-based indexing
    let start_line_idx = (start_line - 1) as usize;
    let end_line_idx = (end_line - 1) as usize;
    
    if start_line_idx >= source_lines.len() || end_line_idx >= source_lines.len() {
        return None;
    }
    
    if start_line_idx == end_line_idx {
        // Single line
        let line = &source_lines[start_line_idx];
        if start_col as usize <= line.len() && end_col as usize <= line.len() {
            return Some(line[start_col as usize..end_col as usize].to_string());
        }
    } else {
        // Multiple lines
        let mut result = Vec::new();
        
        // First line
        let first_line = &source_lines[start_line_idx];
        if (start_col as usize) < first_line.len() {
            result.push(first_line[start_col as usize..].to_string());
        }
        
        // Middle lines
        for i in (start_line_idx + 1)..end_line_idx {
            if i < source_lines.len() {
                result.push(source_lines[i].clone());
            }
        }
        
        // Last line
        if end_line_idx < source_lines.len() {
            let last_line = &source_lines[end_line_idx];
            if end_col as usize <= last_line.len() {
                result.push(last_line[..end_col as usize].to_string());
            }
        }
        
        return Some(result.join("\n"));
    }
    
    None
}
