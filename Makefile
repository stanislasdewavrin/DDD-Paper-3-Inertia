PAPER = paper
LATEX = pdflatex -interaction=nonstopmode
BIBTEX = bibtex

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PAPER).pdf

.PHONY: all clean distclean
