
import re

input_file = 'scientific_failure_presentation.md'
output_file = 'scientific_failure_presentation_v2.md'

with open(input_file, 'r') as f:
    content = f.read()

# Regex to find level 2 headers and append {.allowframebreaks}
# We look for lines starting with ## and not followed by #
# We assume the header text is the rest of the line.
# We append {.allowframebreaks} at the end of the line.

def add_allowframebreaks(match):
    header = match.group(0)
    # Check if it already has attributes
    if '{' in header:
        return header # Skip if complex
    return f"{header} {{.allowframebreaks}}"

# Pattern: Start of line, ##, space, anything, end of line
pattern = re.compile(r'^##\s+.*$', re.MULTILINE)

new_content = pattern.sub(add_allowframebreaks, content)

with open(output_file, 'w') as f:
    f.write(new_content)

print(f"Created {output_file} with allowframebreaks.")
