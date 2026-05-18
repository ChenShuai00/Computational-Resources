import re
import os
import json
import argparse

def extract_sections(content):
    # Match heading lines and capture the heading text.
    pattern = r'^#\s*(.*)$'
    matches = list(re.finditer(pattern, content, flags=re.MULTILINE))
    
    if not matches:
        return [("Document", content)]  # Return the full text when no headings exist.
    
    sections = []
    
    for i, match in enumerate(matches):
        title = match.group(1).strip()  # Extract the heading text.
        start_pos = match.end()  # Content starts after the heading line.
        
        # End at the next heading or the end of the document.
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        # Extract content and trim surrounding whitespace.
        section_content = content[start_pos:end_pos].strip()
        sections.append((title, section_content))
    
    return sections

def extract_title_level(title):
    title = title.strip()
    
    # Level 3+: at least two dots, such as 1.2.3 or 2.10.5.1.
    level3_pattern = r'^\d+(\.\d+){2,}'
    level3_match = re.match(level3_pattern, title)
    
    # Level 2: exactly one dot, such as 1.1 or 2.10.
    level2_pattern = r'^\d+\.\d+'
    level2_match = re.match(level2_pattern, title)
    
    # Level 1: starts with digits and is not followed by a dot or another digit.
    level1_pattern = r'^\d+(?![\d.])'
    level1_match = re.match(level1_pattern, title)

    if level3_match:
        num_part = level3_match.group(0)
        text_part = title[len(num_part):].strip().lower()
        return 3, num_part, text_part
    elif level2_match:
        num_part = level2_match.group(0)
        text_part = title[len(num_part):].strip().lower()
        return 2, num_part, text_part
    elif level1_match:
        num_part = level1_match.group(0)
        text_part = title[len(num_part):].strip().lower()
        return 1, num_part, text_part
    else:
        return 0, "", title.lower()

def extract_start_with_letter_dot_number(text):
    pattern = r'^[a-zA-Z](\.[0-9]+)+'
    match = re.search(pattern, text)
    return match.group() if match else None

def is_subsection(parent, child):
    """Return whether child is a subsection of parent."""
    if not child.startswith(parent):
        return False
    if len(child) == len(parent):
        return False  # Equal section numbers are not subsections.
    if child[len(parent)] != '.':
        return False  # Require a following dot so "10" does not match "1".
    return True

def merge_article_section(section_dict_list):
    i = 0

    while i < len(section_dict_list):
        current = section_dict_list[i]
        current_num = current["section_number"]
        content = current["content"]
        sub_section = []
        j = i + 1
        while j < len(section_dict_list):
            next_num = section_dict_list[j]["section_number"]
            if is_subsection(current_num, next_num):
                sub_section.append({
                    "section_number":section_dict_list[j]["section_number"],
                    "section_title" :section_dict_list[j]["section_title"],
                    "content":section_dict_list[j]["content"]
                })
                j += 1
            else:
                break
        current["sub_section"] = sub_section
        i = j
    return section_dict_list

def main():
    parser = argparse.ArgumentParser(description="Split Markdown parse results into section JSON")
    parser.add_argument("--conference", default="acl2023")
    parser.add_argument("--parse-root", default="papers_pdf_parse")
    parser.add_argument("--save-root", default="paper_section")
    args = parser.parse_args()

    dir_name = f"{args.parse_root}/{args.conference}"
    save_dir = f"{args.save_root}/{args.conference}"
    os.makedirs(save_dir, exist_ok=True)
    dir_list = os.listdir(dir_name)
    md_file_path = sorted([os.path.join(dir_name, d, "auto",f"{d}.md") for d in dir_list])

    for d in dir_list:
        path = os.path.join(dir_name, d, "auto",f"{d}.md")
        print(f"Processing file: {path}")
        with open(path, 'r') as file:
            doc_content = file.read()

        section_dict = {}
        raw_sections = extract_sections(doc_content)
        paper_title, paper_author_institution = raw_sections[0]
        abstract,abstract_content = raw_sections[1]
        section_dict["title"] = paper_title
        section_dict["authors_institution"] = paper_author_institution
        section_dict["abstract"] = abstract_content
        section_dict["sections"] = []
        for index in range(2,len(raw_sections)):
            raw_section_title, section_content = raw_sections[index]
            level, section_num, section_title = extract_title_level(raw_section_title)
            section_dict["sections"].append({
                "level": level,
                "section_number": section_num,
                "raw_section_title": raw_section_title,
                "section_title": section_title,
                "content": section_content
            })
            if section_title.lower() == "references":
                section_dict["appendix"] = raw_sections[index+1:]
                break
            
        appendix = section_dict["appendix"]
        appendix_section = []
        appendix_total_content = ""
        for appendix_title, appendix_content in appendix:
            appendix_total_content += appendix_content + "\n\n"
            appendix_dict = {}

            start_with_letter_dot_number = extract_start_with_letter_dot_number(appendix_title)  # Starts with letter.number.

            first_letter = appendix_title[0]
            if len(appendix_title) == 1:
                second_letter = "A"
            else:
                second_letter = appendix_title[1]

            if start_with_letter_dot_number:
                appendix_dict["section_number"] = start_with_letter_dot_number
                appendix_dict["section_title"] = appendix_title[len(start_with_letter_dot_number):].lstrip()
                appendix_dict["raw_section_title"] = appendix_title
                appendix_dict["content"] = appendix_content
                appendix_section.append(appendix_dict)

            elif appendix_section and (not first_letter.isalpha() or second_letter.islower()):
                appendix_section[-1]["content"] += f"{appendix_title} {appendix_content}"

            else:
                appendix_dict["section_number"] = first_letter
                appendix_dict["section_title"] = appendix_title[1:].lstrip()
                appendix_dict["raw_section_title"] = appendix_title
                appendix_dict["content"] = appendix_content
                appendix_section.append(appendix_dict)

        appendix_section.append({
            "section_number" : "-1",
            "section_title" : "appendix",
            "raw_section_title": "appendix",
            "content": appendix_total_content
        }
        )
        section_dict["appendix"] = appendix_section

        section_dict_list = section_dict["sections"]

        appendix_dict_list = section_dict["appendix"]

        section_dict["sections"] = merge_article_section(section_dict_list)
        section_dict["appendix"] = merge_article_section(appendix_dict_list)

        with open(os.path.join(save_dir, f"{d}_sectioned.json"), 'w', encoding='utf-8') as f:
            json.dump(section_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
