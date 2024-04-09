import re
import os
import glob

def extract_title(content):
    pattern = r"(?P<field>title)={(?P<value>.*)}"
    match = re.search(pattern, content)
    if match:
        return match.group('value')
    else:
        return None


def get_pdf_path(citation_file):
    parent_dir = os.path.dirname(citation_file)
    pdf_files = glob.glob(f'{parent_dir}/*.pdf')
    if len(pdf_files) == 1:
        return pdf_files[0]
    else:
        return None

def main():
    writer = open('README.md', 'w')
    writer.write('# References\n\n')

    citation_files = glob.glob('papers/**/citation.txt')
    for file in citation_files:
        with open(file, 'r') as f:
            content = f.read()
            title = extract_title(content)
            pdf_path = get_pdf_path(file)
            summary = open(os.path.dirname(file) + '/summary.txt').read()
            if title and pdf_path and summary:
                writer.write(f'## {title}\n\n')
                writer.write(f'![pdf]({pdf_path})\n\n')
                writer.write(f'{summary}\n\n')

    writer.close()

if __name__ == '__main__':
    main()