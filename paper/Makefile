PDF=sidmkit_sparc_submission.pdf
TEX=sidmkit_sparc_submission.tex

all: $(PDF)

$(PDF): $(TEX)
\tpdflatex -interaction=nonstopmode $(TEX)
\tpdflatex -interaction=nonstopmode $(TEX)

clean:
\trm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk
