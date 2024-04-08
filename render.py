import glob

def main():
    writer = open('renders/citations.bib', 'w')

    citation_files = glob.glob('papers/**/citation.txt')
    for file in citation_files:
        with open(file, 'r') as f:
            content = f.read()
            writer.write(content + '\n\n')

    writer.close()

if __name__ == '__main__':
    main()