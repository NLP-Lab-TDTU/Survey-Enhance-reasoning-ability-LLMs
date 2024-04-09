PYTHON = python3

generate_bib:
	${PYTHON} render.py


generate_summary:
	${PYTHON} summary.py


generate_readme:
	${PYTHON} generate.py


all: generate_bib generate_summary generate_readme