TEX_FILES=$(shell find . -name '*.tex')
OUTPUT_FILE=proposal.pdf
BIB=references.bib

all: $(OUTPUT_FILE)

$(OUTPUT_FILE): $(TEX_FILES) $(BIB)
	pdflatex proposal.tex
	bibtex proposal
	pdflatex proposal.tex
	pdflatex proposal.tex

clean:
	rm -vf *.log
	rm -vf *.toc
	rm -vf *.lof
	rm -vf *.lot
	rm -vf *.blg
	rm -vf *.bbl
	rm -vf $(shell find . -name '*.aux')
	rm -vf $(OUTPUT_FILE)

.PHONY: clean
