PROJECTNAME = allen_wrench
DISTDIR = dist
BUILDDIR = build
VERSION = 0.2.0
RELEASE = dev123
RELEASEDIR = $(PROJECTNAME)-$(VERSION).$(RELEASE)
EGGINFODIR = $(PROJECTNAME).egg-info
DOCDIR = doc

build:
	mkdir -p $(DISTDIR)/$(PROJECTNAME)
	cp -r allen_wrench setup.py README.md $(DISTDIR)/$(PROJECTNAME)/
	cd $(DISTDIR); tar czvf $(PROJECTNAME).tgz $(PROJECTNAME)
	

distutils_build: clean
	python setup.py build

sdist: distutils_build
	python setup.py sdist
	
doc: clean
	sphinx-apidoc -d 4 -H "Allen Wrench" -A "Allen Institute for Brain Science" -V $(VERSION) -R $(VERSION)$(RELEASE) --full -o doc $(PROJECTNAME)
	cp doc_template/*.rst doc_template/conf.py doc
	cd doc && make html || true

clean:
	rm -rf $(DISTDIR)
	rm -rf $(BUILDDIR)
	rm -rf $(RELEASEDIR)
	rm -rf $(EGGINFODIR)
	rm -rf $(DOCDIR)
