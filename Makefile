PROJECTNAME = allen_wrench
DISTDIR = dist
BUILDDIR = build
export RELEASE=dev$(BUILD_NUMBER)
RELEASEDIR = $(PROJECTNAME)-$(VERSION).$(RELEASE)
EGGINFODIR = $(PROJECTNAME).egg-info
DOCDIR = doc

build:
	mkdir -p $(DISTDIR)/$(PROJECTNAME)
	cp -r allen_wrench setup.py README.md $(DISTDIR)/$(PROJECTNAME)/
	cd $(DISTDIR); tar czvf $(PROJECTNAME).tgz $(PROJECTNAME)
	

distutils_build: clean
	python setup.py build
	
setversion:
	sed -ie 's/'\''[0-9]\+.[0-9]\+.[0-9]\+.dev[0-9]\+'\''/'\''${VERSION}.${RELEASE}'\''/g' allen_wrench/__init__.py

sdist: distutils_build
	python setup.py sdist
	
doc: clean
	sphinx-apidoc -d 4 -H "Allen Wrench" -A "Allen Institute for Brain Science" -V $(VERSION) -R $(VERSION).dev$(RELEASE) --full -o doc $(PROJECTNAME)
	cp doc_template/*.rst doc_template/conf.py doc_template/logo.jpg doc
	mkdir -p doc/_static/stylesheets
	cp -R doc_template/aibs_sphinx/static/* doc/_static
	cp -R doc_template/aibs_sphinx/templates/* doc/_templates
	sed -ie "s/|version|/${VERSION}.${RELEASE}/g" doc/user.rst
	sed -ie "s/|version|/${VERSION}.${RELEASE}/g" doc/developer.rst
	sed -ie "s/|version|/${VERSION}.${RELEASE}/g" doc/links.rst
	cd doc && make html || true

clean:
	rm -rf $(DISTDIR)
	rm -rf $(BUILDDIR)
	rm -rf $(RELEASEDIR)
	rm -rf $(EGGINFODIR)
	rm -rf $(DOCDIR)
