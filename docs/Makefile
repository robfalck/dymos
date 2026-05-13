# Makefile for Sphinx documentation

.PHONY: help clean html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest gettext

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

ifeq ($(OS),Windows_NT)
	RMDIR = rmdir /s /q
	MKDIR = mkdir
else
	RMDIR = rm -rf
	MKDIR = mkdir -p
endif

# Target: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation"
	@echo "  clean      to remove all build files"
	@echo ""
	@echo "First install requirements: pip install -r requirements-sphinx.txt"

# Target: clean
clean:
	-$(RMDIR) $(BUILDDIR)
	-$(RMDIR) .jupytercache

# Target: html
html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) . $(BUILDDIR)/html
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html/"

# Target: dirhtml
dirhtml:
	$(SPHINXBUILD) -b dirhtml $(SPHINXOPTS) . $(BUILDDIR)/dirhtml
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml/"

# Target: singlehtml
singlehtml:
	$(SPHINXBUILD) -b singlehtml $(SPHINXOPTS) . $(BUILDDIR)/singlehtml
	@echo "Build finished. The single HTML page is in $(BUILDDIR)/singlehtml/"

# Target: pickle
pickle:
	$(SPHINXBUILD) -b pickle $(SPHINXOPTS) . $(BUILDDIR)/pickle
	@echo "Build finished. The pickle files are in $(BUILDDIR)/pickle/"

# Target: json
json:
	$(SPHINXBUILD) -b json $(SPHINXOPTS) . $(BUILDDIR)/json
	@echo "Build finished. The JSON files are in $(BUILDDIR)/json/"

# Target: htmlhelp
htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(SPHINXOPTS) . $(BUILDDIR)/htmlhelp
	@echo "Build finished. The HTML help files are in $(BUILDDIR)/htmlhelp/"

# Target: qthelp
qthelp:
	$(SPHINXBUILD) -b qthelp $(SPHINXOPTS) . $(BUILDDIR)/qthelp
	@echo "Build finished. The qthelp files are in $(BUILDDIR)/qthelp/"

# Target: devhelp
devhelp:
	$(SPHINXBUILD) -b devhelp $(SPHINXOPTS) . $(BUILDDIR)/devhelp
	@echo "Build finished. The devhelp files are in $(BUILDDIR)/devhelp/"

# Target: epub
epub:
	$(SPHINXBUILD) -b epub $(SPHINXOPTS) . $(BUILDDIR)/epub
	@echo "Build finished. The epub files are in $(BUILDDIR)/epub/"

# Target: latex
latex:
	$(SPHINXBUILD) -b latex $(SPHINXOPTS) . $(BUILDDIR)/latex
	@echo "Build finished. The latex files are in $(BUILDDIR)/latex/"

# Target: latexpdf
latexpdf:
	$(SPHINXBUILD) -b latex $(SPHINXOPTS) . $(BUILDDIR)/latex
	cd $(BUILDDIR)/latex && make all-pdf
	@echo "Build finished. The PDF files are in $(BUILDDIR)/latex/"

# Target: latexpsdfs
latexpsdfs:
	$(SPHINXBUILD) -b latex $(SPHINXOPTS) . $(BUILDDIR)/latex
	cd $(BUILDDIR)/latex && make all-ps
	@echo "Build finished. The PS files are in $(BUILDDIR)/latex/"

# Target: text
text:
	$(SPHINXBUILD) -b text $(SPHINXOPTS) . $(BUILDDIR)/text
	@echo "Build finished. The text files are in $(BUILDDIR)/text/"

# Target: man
man:
	$(SPHINXBUILD) -b man $(SPHINXOPTS) . $(BUILDDIR)/man
	@echo "Build finished. The man pages are in $(BUILDDIR)/man/"

# Target: changes
changes:
	$(SPHINXBUILD) -b changes $(SPHINXOPTS) . $(BUILDDIR)/changes
	@echo "The overview file is in $(BUILDDIR)/changes/"

# Target: linkcheck
linkcheck:
	$(SPHINXBUILD) -b linkcheck $(SPHINXOPTS) . $(BUILDDIR)/linkcheck
	@echo "Link check complete; look at $(BUILDDIR)/linkcheck/index.html"

# Target: doctest
doctest:
	$(SPHINXBUILD) -b doctest $(SPHINXOPTS) . $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the results in $(BUILDDIR)/doctest/index.html"

# Target: gettext
gettext:
	$(SPHINXBUILD) -b gettext $(SPHINXOPTS) . $(BUILDDIR)/gettext
	@echo "Build finished. The pot files are in $(BUILDDIR)/gettext/"

.PHONY: serve
serve: html
	@echo "Serving HTML documentation at http://localhost:8000/markdown/"
	@echo "Or visit: http://localhost:8000"
	cd $(BUILDDIR)/html && python -m http.server 8000
